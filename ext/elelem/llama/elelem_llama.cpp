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
#include <cctype>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

using json = nlohmann::ordered_json;

extern "C" {

struct el_handle {
    llama_model *model;
    common_chat_templates_ptr tmpls;
    llama_context *ctx; // persistent across el_generate calls; memory is
                         // cleared at the start of each call so behavior
                         // stays identical to a fresh context per call
    int n_ctx;
    int n_threads;
    float temp;       // <= 0 => greedy (deterministic); used by evals
    uint32_t seed;
    bool dead = false; // set once the GPU device is lost; see el_close
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

static void el_log_callback(enum ggml_log_level level, const char *text, void * /*user_data*/) {
    if (level < el_log_threshold()) return;
    fputs(text, stderr);
}

// Smallest context worth running with; below this a model is unusable for
// agentic tool-calling (system prompt + tool schemas alone can exceed a few
// thousand tokens), so give up rather than silently hand back a useless handle.
static const int EL_MIN_CTX = 4096;

// n_ctx <= 0 means "auto": ask llama.cpp for the model's trained max (n_ctx=0
// is documented as "from model" in llama.h) rather than guessing from a memory
// formula -- a formula sized for plain transformer KV would badly overestimate
// footprint for hybrid/recurrent-state architectures (e.g. Qwen3.5's Gated
// DeltaNet layers hold fixed-size state, not per-token KV). If the resulting
// context still doesn't fit (llama_init_from_model returns null -- OOM or
// unsupported size), retry at half the size down to EL_MIN_CTX; this is
// correct for every architecture because it measures the actual allocation
// instead of predicting it.
static llama_context *el_init_context(llama_model *model, int n_ctx, int n_threads, uint32_t *out_n_ctx) {
    uint32_t requested = n_ctx > 0 ? (uint32_t) n_ctx : 0;
    llama_context_params cp = llama_context_default_params();
    cp.n_ctx = requested;
    cp.n_threads = n_threads;
    cp.n_threads_batch = n_threads;
    llama_context *ctx = llama_init_from_model(model, cp);
    if (ctx) {
        *out_n_ctx = requested > 0 ? requested : llama_model_n_ctx_train(model);
        return ctx;
    }
    if (requested == 0) return nullptr; // "from model" failed; no size to halve

    for (uint32_t size = requested / 2; size >= EL_MIN_CTX; size /= 2) {
        cp.n_ctx = size;
        ctx = llama_init_from_model(model, cp);
        if (ctx) { *out_n_ctx = size; return ctx; }
    }
    return nullptr;
}

// temp <= 0 selects greedy/deterministic sampling; seed is the RNG seed for the
// sampled path (both surfaced so callers -- notably the eval harness -- can pin them).
// n_threads <= 0 means "auto": use llama.cpp's own physical-core/SMT-aware
// detection (common_cpu_get_num_math) instead of a thread count picked in Ruby.
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

    int threads = n_threads > 0 ? n_threads : common_cpu_get_num_math();

    // common_chat_templates_init parses the model's embedded Jinja chat
    // template; a malformed/unsupported template throws instead of
    // returning null, which would otherwise abort the whole process (see
    // el_generate's comment on the FFI boundary).
    try {
        auto tmpls = common_chat_templates_init(model, "");

        uint32_t actual_n_ctx = 0;
        llama_context *ctx = el_init_context(model, n_ctx, threads, &actual_n_ctx);
        if (!ctx) { llama_model_free(model); return nullptr; }

        return new el_handle{model, std::move(tmpls), ctx, (int) actual_n_ctx, threads, temp, (uint32_t) seed};
    } catch (const std::exception &) {
        llama_model_free(model);
        return nullptr;
    }
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
// Every throwing call in here (nlohmann::json parsing, jinja template
// application, the chat PEG parser) is wrapped by the caller, el_generate,
// in one top-level try/catch: any C++ exception that unwinds past this
// function crosses the Fiddle FFI boundary into Ruby and aborts the whole
// process (std::terminate, not a catchable Ruby exception). A malformed
// request or an unfamiliar template family (new model = new chat_template)
// must degrade to an error result, never crash the caller.
static const char *el_generate_impl(void *handle, const char *messages_json, const char *tools_json, int max_tokens) {
    auto *h = (el_handle *) handle;
    const llama_vocab *vocab = llama_model_get_vocab(h->model);
    static thread_local std::string buf; // result JSON; valid until the next call on this thread

    common_chat_templates_inputs inputs;
    inputs.messages = build_msgs(json::parse(messages_json));
    if (tools_json && *tools_json) inputs.tools = build_tools(json::parse(tools_json));
    inputs.add_generation_prompt = true;
    inputs.use_jinja = true;
    // Must be set before templates_apply -- it bakes extract_reasoning into the
    // parser grammar templates_apply builds (see common_chat_parser_params below,
    // which only controls parse-time behavior for formats that already support
    // it). Without this, thinking-tag models (e.g. GLM's <think>...</think>)
    // leave reasoning text inline in content instead of reasoning_content.
    inputs.reasoning_format = COMMON_REASONING_FORMAT_AUTO;
    common_chat_params cparams = common_chat_templates_apply(h->tmpls.get(), inputs);

    json result;
    result["content"] = "";
    result["tool_calls"] = json::array();

    // ctx is persistent on the handle (see el_open); clear its KV cache so
    // each call is still stateless from the model's point of view, just
    // without paying context-alloc/graph-reserve cost every time.
    llama_context *ctx = h->ctx;
    auto t0 = std::chrono::steady_clock::now();
    llama_memory_clear(llama_get_memory(ctx), true);
    auto t1 = std::chrono::steady_clock::now();

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
    auto t2 = std::chrono::steady_clock::now();

    // The caller's max_tokens is a policy choice (how long one reply may
    // run), not a hardware quantity -- but decoding past what's left in the
    // context window is never correct regardless of what was requested, so
    // that part of the ceiling is calculated, not configured.
    int budget = std::min(max_tokens, h->n_ctx - n_prompt);

    std::string output;
    char piece[512];
    int n_decoded = 0;
    for (int t = 0; t < budget; t++) {
        llama_token id = llama_sampler_sample(smpl, ctx, -1);
        if (llama_vocab_is_eog(vocab, id)) break;
        int np = llama_token_to_piece(vocab, id, piece, (int32_t) sizeof(piece), 0, true);
        if (np > 0) output.append(piece, np);
        n_decoded++;
        // Advance the KV cache so we can sample the next token. Skip it on the last
        // planned iteration -- that forward pass would never be sampled from.
        if (t + 1 < budget && llama_decode(ctx, llama_batch_get_one(&id, 1)) != 0) break;
    }
    auto t3 = std::chrono::steady_clock::now();

    llama_sampler_free(smpl);

    auto ms = [](auto a, auto b) { return std::chrono::duration<double, std::milli>(b - a).count(); };
    result["ms_reset"] = ms(t0, t1);
    result["ms_prefill"] = ms(t1, t2);
    result["ms_decode"] = ms(t2, t3);
    result["n_prompt"] = n_prompt;
    result["n_decoded"] = n_decoded;
    // n_decoded hits the budget without an EOG token when generation was cut
    // off mid-thought (e.g. the whole budget spent on reasoning) rather than
    // finishing naturally -- surface it so callers don't mistake silence for
    // "the model had nothing to say".
    result["truncated"] = n_decoded >= budget;

    // The parse rules live in a PEG arena that templates_apply serialized into
    // cparams.parser; common_chat_parse forwards params.parser to the PEG engine,
    // so it must be deserialized here or every format returns content-only (no
    // tool calls). The converting ctor carries format + generation_prompt.
    //
    // common_chat_parse throws std::runtime_error when the model's raw output
    // doesn't match its own template's expected grammar (seen with gpt-oss's
    // harmony format on malformed/truncated generations). That exception can't
    // cross the Fiddle FFI boundary -- it aborts the whole Ruby process -- so
    // treat a parse failure the same as "nothing parsed": fall through to the
    // lenient fallback below with the raw text kept as content.
    common_chat_msg parsed;
    parsed.content = output;
    try {
        common_chat_parser_params pp(cparams);
        // AUTO routes <think>/harmony-analysis text into reasoning_content
        // instead of leaving it inline in content (the default, NONE, does not
        // split it out at all -- see gpt-oss's <|channel|>analysis<|message|>).
        pp.reasoning_format = COMMON_REASONING_FORMAT_AUTO;
        if (!cparams.parser.empty()) pp.parser.load(cparams.parser);
        parsed = common_chat_parse(output, false, pp);
    } catch (const std::exception &) {
        parsed.content = output;
    }

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

    // Harmony fallback: gpt-oss sometimes emits a malformed header --
    // e.g. a doubled "<|channel|>commentary" before "<|constrain|>json", or a
    // missing space -- that the PEG grammar in common_chat_parse rejects
    // outright (logged upstream as "unparsed peg-native output"), leaving
    // tool_calls empty and the whole raw header+JSON sitting in content. The
    // JSON-sniffing fallback above can't help: the tool name lives in the
    // "to=functions.NAME" tag, not in the trailing JSON object, which here is
    // bare arguments. Scrape the tag instead of trying to normalize every way
    // the header tags can be malformed. Search for the *last* "to=functions."
    // that is actually followed by "<|message|>" on the same call, since
    // model prose can hallucinate an earlier, unrelated "to=functions." (e.g.
    // mid-sentence speculation) before the real one.
    bool harmony_tag_fallback_used = false;
    if (parsed.tool_calls.empty() && !inputs.tools.empty()) {
        static const std::string tag = "to=functions.";
        static const std::string msg_tag = "<|message|>";
        size_t search_from = output.size();
        for (;;) {
            size_t t = output.rfind(tag, search_from);
            if (t == std::string::npos) break;
            size_t name_start = t + tag.length();
            size_t name_end = name_start;
            while (name_end < output.size() && (isalnum((unsigned char) output[name_end]) || output[name_end] == '_')) name_end++;
            std::string name = output.substr(name_start, name_end - name_start);

            size_t m = output.find(msg_tag, name_end);
            bool known = false;
            for (const auto &tool : inputs.tools) known |= (tool.name == name);

            if (known && m != std::string::npos) {
                size_t arg_start = m + msg_tag.length();
                // Assumes a single trailing JSON object (b is the last '}' in
                // the whole output, not scoped to this call) -- fine while a
                // generation carries at most one malformed tool call, but two
                // such calls or trailing prose containing '}' would make this
                // span both and fail to parse, silently giving up rather than
                // producing a wrong call.
                size_t a = output.find('{', arg_start), b = output.rfind('}');
                if (a != std::string::npos && b != std::string::npos && b > a) {
                    try {
                        json args = json::parse(output.substr(a, b - a + 1));
                        common_chat_tool_call tc;
                        tc.name = name;
                        tc.arguments = as_json_string(args);
                        parsed.tool_calls.push_back(tc);
                        parsed.content.clear();
                        harmony_tag_fallback_used = true;
                    } catch (...) { /* bare JSON didn't parse; give up on this tag */ }
                }
            }
            if (harmony_tag_fallback_used || t == 0) break;
            search_from = t - 1;
        }
    }

    result["fallback"] = fallback_used;
    result["harmony_tag_fallback"] = harmony_tag_fallback_used;
    result["content"] = parsed.content;
    result["reasoning"] = parsed.reasoning_content;
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

const char *el_generate(void *handle, const char *messages_json, const char *tools_json, int max_tokens) {
    static thread_local std::string errbuf;
    try {
        return el_generate_impl(handle, messages_json, tools_json, max_tokens);
    } catch (const std::exception &e) {
        // A GPU-level failure here (e.g. Vulkan ErrorDeviceLost from a
        // driver-side OOM) leaves the backend unusable: llama_context's
        // destructor synchronizes the backend and will throw again from
        // inside a noexcept destructor, which calls std::terminate before
        // a caller-side try/catch in el_close ever runs. Mark the handle
        // dead so el_close knows to skip teardown instead of trying (and
        // failing) to catch the uncatchable.
        if (handle) ((el_handle *) handle)->dead = true;
        json result = {{"content", ""}, {"tool_calls", json::array()}, {"reasoning", ""}, {"fallback", false}, {"error", e.what()}};
        errbuf = result.dump();
        return errbuf.c_str();
    } catch (...) {
        if (handle) ((el_handle *) handle)->dead = true;
        json result = {{"content", ""}, {"tool_calls", json::array()}, {"reasoning", ""}, {"fallback", false}, {"error", "unknown exception in el_generate"}};
        errbuf = result.dump();
        return errbuf.c_str();
    }
}

void el_close(void *handle) {
    auto *h = (el_handle *) handle;
    if (!h) return;
    // If the GPU device was already lost mid-generate, freeing the context
    // re-triggers the same failure from inside llama_context's destructor,
    // which is implicitly noexcept -- throwing there calls std::terminate
    // immediately, before it could ever reach a try/catch here. Leak
    // deliberately instead: the process is exiting and the OS reclaims
    // everything anyway.
    if (h->dead) {
        delete h;
        return;
    }
    llama_free(h->ctx);
    llama_model_free(h->model);
    delete h;
    llama_backend_free();
}

}  // extern "C"
