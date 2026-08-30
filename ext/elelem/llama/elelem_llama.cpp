// Thin, stable C façade over llama.cpp + its common_chat layer, for in-process
// inference from Ruby (Fiddle). All volatile C++/by-value types are handled here,
// compiled against the vendored headers, so Ruby only sees simple signatures:
// JSON in (OpenAI-style messages + tools), JSON out (content + tool_calls). The
// model + chat templates are opened once and kept resident on an opaque handle; a
// fresh context per generate keeps each request statelessly isolated.
#include "llama.h"
#include "chat.h"
#include <nlohmann/json.hpp>
#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

using json = nlohmann::ordered_json;

extern "C" {

struct el_handle {
    llama_model *model;
    common_chat_templates_ptr tmpls;
    int n_ctx;
    int n_threads;
    float temp;       // <= 0 => greedy (deterministic); used by evals
    uint32_t seed;
};

static bool g_backend = false;

// llama.cpp/ggml log to stderr by default; gate that behind LOG_LEVEL so
// GGUF model loads aren't noisy unless a caller opts in, matching the
// verbosity Elelem.logger is configured with in lib/elelem.rb.
static enum ggml_log_level el_log_threshold() {
    const char *level = std::getenv("LOG_LEVEL");
    if (!level) return GGML_LOG_LEVEL_WARN;
    std::string v(level);
    std::transform(v.begin(), v.end(), v.begin(), ::tolower);
    if (v == "debug") return GGML_LOG_LEVEL_DEBUG;
    if (v == "info") return GGML_LOG_LEVEL_INFO;
    if (v == "error") return GGML_LOG_LEVEL_ERROR;
    return GGML_LOG_LEVEL_WARN;
}

static void el_log_callback(enum ggml_log_level level, const char *text, void *user_data) {
    if (level < el_log_threshold()) return;
    fputs(text, stderr);
}

// temp <= 0 selects greedy/deterministic sampling; seed is the RNG seed for the
// sampled path (both surfaced so callers -- notably the eval harness -- can pin them).
void *el_open(const char *path, int n_gpu_layers, int n_ctx, int n_threads, float temp, int seed) {
    if (!g_backend) {
        llama_log_set(el_log_callback, nullptr);
        llama_backend_init();
        g_backend = true;
    }

    llama_model_params mp = llama_model_default_params();
    mp.n_gpu_layers = n_gpu_layers;
    llama_model *model = llama_model_load_from_file(path, mp);
    if (!model) return nullptr;

    return new el_handle{model, common_chat_templates_init(model, ""), n_ctx, n_threads, temp, (uint32_t) seed};
}

// Tool-call arguments arrive as either a JSON string or an object; llama.cpp's
// common_chat wants a string either way.
static std::string as_json_string(const json &v) {
    return v.is_string() ? v.get<std::string>() : v.dump();
}

static std::vector<common_chat_msg> build_msgs(const json &arr) {
    std::vector<common_chat_msg> out;
    for (const auto &m : arr) {
        common_chat_msg cm;
        cm.role = m.value("role", "user");
        cm.content = m.value("content", "");
        cm.tool_name = m.value("tool_name", "");
        cm.tool_call_id = m.value("tool_call_id", "");
        if (m.contains("tool_calls")) {
            for (const auto &tc : m["tool_calls"]) {
                common_chat_tool_call c;
                c.id = tc.value("id", "");
                c.name = tc.value("name", "");
                if (tc.contains("arguments")) {
                    const auto &a = tc["arguments"];
                    c.arguments = as_json_string(a);
                }
                cm.tool_calls.push_back(c);
            }
        }
        out.push_back(cm);
    }
    return out;
}

static std::vector<common_chat_tool> build_tools(const json &arr) {
    std::vector<common_chat_tool> out;
    for (const auto &t : arr) {
        const json &fn = t.contains("function") ? t["function"] : t;
        common_chat_tool ct;
        ct.name = fn.value("name", "");
        ct.description = fn.value("description", "");
        ct.parameters = fn.contains("parameters") ? fn["parameters"].dump() : "{}";
        out.push_back(ct);
    }
    return out;
}

// Returns a JSON string {"content": "...", "tool_calls": [{id,name,arguments}]}.
// The buffer is valid until the next el_generate call on this thread.
const char *el_generate(void *handle, const char *messages_json, const char *tools_json, int max_tokens) {
    auto *h = (el_handle *) handle;
    const llama_vocab *vocab = llama_model_get_vocab(h->model);
    static thread_local std::string buf; // result JSON; valid until the next call on this thread

    common_chat_templates_inputs inputs;
    inputs.messages = build_msgs(json::parse(messages_json));
    if (tools_json && *tools_json) inputs.tools = build_tools(json::parse(tools_json));
    inputs.add_generation_prompt = true;
    inputs.use_jinja = true;
    common_chat_params cparams = common_chat_templates_apply(h->tmpls.get(), inputs);

    json result;
    result["content"] = "";
    result["tool_calls"] = json::array();

    llama_context_params cp = llama_context_default_params();
    cp.n_ctx = (uint32_t) h->n_ctx;
    cp.n_threads = h->n_threads;
    cp.n_threads_batch = h->n_threads;
    llama_context *ctx = llama_init_from_model(h->model, cp);
    if (!ctx) { buf = result.dump(); return buf.c_str(); }

    llama_sampler *smpl = llama_sampler_chain_init(llama_sampler_chain_default_params());
    if (h->temp <= 0.0f) {
        llama_sampler_chain_add(smpl, llama_sampler_init_greedy());
    } else {
        llama_sampler_chain_add(smpl, llama_sampler_init_top_k(40));
        llama_sampler_chain_add(smpl, llama_sampler_init_top_p(0.95f, 1));
        llama_sampler_chain_add(smpl, llama_sampler_init_temp(h->temp));
        llama_sampler_chain_add(smpl, llama_sampler_init_dist(h->seed));
    }

    const std::string &prompt = cparams.prompt;
    int n_prompt = -llama_tokenize(vocab, prompt.c_str(), (int32_t) prompt.size(), nullptr, 0, true, true);
    std::vector<llama_token> tokens(n_prompt);
    llama_tokenize(vocab, prompt.c_str(), (int32_t) prompt.size(), tokens.data(), (int32_t) tokens.size(), true, true);

    // Ingest the prompt in n_batch-sized chunks (a full prompt commonly exceeds one
    // batch, which llama_decode asserts against).
    const int n_batch = (int) llama_n_batch(ctx);
    for (int i = 0; i < n_prompt; i += n_batch) {
        int n = std::min(n_batch, n_prompt - i);
        if (llama_decode(ctx, llama_batch_get_one(tokens.data() + i, n)) != 0) break;
    }

    std::string output;
    char piece[512];
    for (int t = 0; t < max_tokens; t++) {
        llama_token id = llama_sampler_sample(smpl, ctx, -1);
        if (llama_vocab_is_eog(vocab, id)) break;
        int np = llama_token_to_piece(vocab, id, piece, (int32_t) sizeof(piece), 0, true);
        if (np > 0) output.append(piece, np);
        // Advance the KV cache so we can sample the next token. Skip it on the last
        // planned iteration -- that forward pass would never be sampled from.
        if (t + 1 < max_tokens && llama_decode(ctx, llama_batch_get_one(&id, 1)) != 0) break;
    }

    llama_sampler_free(smpl);
    llama_free(ctx);

    // The parse rules live in a PEG arena that templates_apply serialized into
    // cparams.parser; common_chat_parse forwards params.parser to the PEG engine,
    // so it must be deserialized here or every format returns content-only (no
    // tool calls). The converting ctor carries format + generation_prompt.
    common_chat_parser_params pp(cparams);
    if (!cparams.parser.empty()) pp.parser.load(cparams.parser);
    common_chat_msg parsed = common_chat_parse(output, false, pp);

    // Lenient fallback: small/quantized models often emit a bare {"name","arguments"}
    // tool-call JSON (frequently fenced) instead of the template's exact tag syntax,
    // so the strict parser misses it. If tools were offered and nothing parsed, pull
    // out the first JSON object that names a real tool. result["fallback"] records
    // whether this fired -- a diagnostic for judging if a model needs the crutch.
    bool fallback_used = false;
    if (parsed.tool_calls.empty() && !inputs.tools.empty()) {
        size_t a = output.find('{'), b = output.rfind('}');
        if (a != std::string::npos && b != std::string::npos && b > a) {
            try {
                json j = json::parse(output.substr(a, b - a + 1));
                std::string name = j.value("name", "");
                bool known = false;
                for (const auto &tool : inputs.tools) known |= (tool.name == name);
                if (known && j.contains("arguments")) {
                    common_chat_tool_call tc;
                    tc.name = name;
                    tc.arguments = as_json_string(j["arguments"]);
                    parsed.tool_calls.push_back(tc);
                    parsed.content.clear();
                    fallback_used = true;
                }
            } catch (...) { /* not a tool call; leave content as-is */ }
        }
    }

    result["fallback"] = fallback_used;
    result["content"] = parsed.content;
    int i = 0;
    for (const auto &tc : parsed.tool_calls) {
        result["tool_calls"].push_back({
            {"id", tc.id.empty() ? "call_" + std::to_string(i) : tc.id},
            {"name", tc.name},
            {"arguments", tc.arguments}
        });
        i++;
    }

    buf = result.dump();
    return buf.c_str();
}

void el_close(void *handle) {
    auto *h = (el_handle *) handle;
    if (!h) return;
    llama_model_free(h->model);
    delete h;
}

}  // extern "C"
