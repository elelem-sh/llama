// Thin, stable C façade over llama.cpp + its common_chat layer, for in-process
// inference from Ruby (Fiddle). All volatile C++/by-value types are handled here,
// compiled against the vendored headers, so Ruby only sees simple signatures:
// JSON in (OpenAI-style messages + tools), JSON out (content + tool_calls). The
// model, chat templates and context are opened once and kept resident on an
// opaque handle; clearing the KV cache at the start of each generate keeps every
// request statelessly isolated without paying to rebuild the context.
#include "llama.h"
#include "chat.h"
#include "common.h" // common_cpu_get_num_math (also reached via chat.h; explicit so it survives header churn)
#include <nlohmann/json.hpp>
#include <algorithm>
#include <cctype>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <memory>
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
// Read once: LOG_LEVEL can't change mid-process, and this ran per log line.
static enum ggml_log_level el_log_threshold() {
    static const enum ggml_log_level cached = [] {
        const char *level = std::getenv("LOG_LEVEL");
        if (!level) return GGML_LOG_LEVEL_WARN;
        std::string v(level);
        std::transform(v.begin(), v.end(), v.begin(),
                       [](unsigned char c) { return (char) std::tolower(c); });
        if (v == "debug") return GGML_LOG_LEVEL_DEBUG;
        if (v == "info") return GGML_LOG_LEVEL_INFO;
        if (v == "error") return GGML_LOG_LEVEL_ERROR;
        return GGML_LOG_LEVEL_WARN;
    }();
    return cached;
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
static llama_context *el_init_context(llama_model *model, int n_ctx, int n_threads) {
    uint32_t requested = n_ctx > 0 ? (uint32_t) n_ctx : 0;
    llama_context_params cp = llama_context_default_params();
    cp.n_ctx = requested;
    cp.n_threads = n_threads;
    cp.n_threads_batch = n_threads;
    llama_context *ctx = llama_init_from_model(model, cp);
    if (ctx) return ctx;
    if (requested == 0) return nullptr; // "from model" failed; no size to halve

    for (uint32_t size = requested / 2; size >= EL_MIN_CTX; size /= 2) {
        cp.n_ctx = size;
        ctx = llama_init_from_model(model, cp);
        if (ctx) return ctx;
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

        llama_context *ctx = el_init_context(model, n_ctx, threads);
        if (!ctx) { llama_model_free(model); return nullptr; }

        // Ask the context what it actually allocated rather than assuming the
        // request was honoured -- llama.cpp may clamp or round n_ctx down, and
        // this value is the sole gate on the decode budget in el_generate_impl.
        // Per-sequence, not total: we decode one sequence (llama_batch_get_one),
        // and llama.cpp splits the pool as n_ctx_seq = n_ctx / n_seq_max. The
        // two are equal at the default n_seq_max=1, so this only matters if
        // that ever changes -- at which point n_ctx would overstate our room.
        return new el_handle{model, std::move(tmpls), ctx, (int) llama_n_ctx_seq(ctx), threads, temp, (uint32_t) seed};
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

// Shape every early return like a normal result so the Ruby side (gguf.rb)
// can read "error" without special-casing missing keys.
static const char *el_error(std::string &buf, const std::string &message) {
    json result = {
        {"content", ""},
        {"reasoning", ""},
        {"tool_calls", json::array()},
        {"fallback", false},
        {"harmony_tag_fallback", false},
        {"truncated", false},
        {"error", message}
    };
    buf = result.dump();
    return buf.c_str();
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
    inputs.add_generation_prompt = true;
    inputs.use_jinja = true;
    // Must be set before templates_apply -- it bakes extract_reasoning into the
    // parser grammar templates_apply builds (see common_chat_parser_params below,
    // which only controls parse-time behavior for formats that already support
    // it). Without this, thinking-tag models (e.g. GLM's <think>...</think>)
    // leave reasoning text inline in content instead of reasoning_content.
    inputs.reasoning_format = COMMON_REASONING_FORMAT_AUTO;

    // Malformed request JSON and an unsupported chat template are *caller*
    // errors, not backend failures: handle them here so they return an error
    // result without marking the handle dead (which would strand a perfectly
    // healthy model+context for the rest of the process). Only errors that
    // escape to el_generate are treated as fatal to the device.
    common_chat_params cparams;
    try {
        inputs.messages = build_msgs(json::parse(messages_json));
        if (tools_json && *tools_json) inputs.tools = build_tools(json::parse(tools_json));
        cparams = common_chat_templates_apply(h->tmpls.get(), inputs);
    } catch (const std::exception &e) {
        return el_error(buf, std::string("invalid request or chat template: ") + e.what());
    }

    json result;

    // ctx is persistent on the handle (see el_open); clear its KV cache so
    // each call is still stateless from the model's point of view, just
    // without paying context-alloc/graph-reserve cost every time.
    llama_context *ctx = h->ctx;
    auto t0 = std::chrono::steady_clock::now();
    llama_memory_clear(llama_get_memory(ctx), true);
    auto t1 = std::chrono::steady_clock::now();

    // unique_ptr because everything between here and the decode loop can throw
    // (vector allocation, json assignment); a raw pointer freed at the bottom
    // would leak the whole sampler chain on that path. Deliberately per-call:
    // llama_sampler_init_dist carries RNG state, so hoisting this onto the
    // handle would make seeded runs non-reproducible across calls.
    std::unique_ptr<llama_sampler, decltype(&llama_sampler_free)> smpl(
        llama_sampler_chain_init(llama_sampler_chain_default_params()), llama_sampler_free);
    if (h->temp <= 0.0f) {
        llama_sampler_chain_add(smpl.get(), llama_sampler_init_greedy());
    } else {
        llama_sampler_chain_add(smpl.get(), llama_sampler_init_top_k(40));
        llama_sampler_chain_add(smpl.get(), llama_sampler_init_top_p(0.95f, 1));
        llama_sampler_chain_add(smpl.get(), llama_sampler_init_temp(h->temp));
        llama_sampler_chain_add(smpl.get(), llama_sampler_init_dist(h->seed));
    }

    const std::string &prompt = cparams.prompt;
    int n_prompt = -llama_tokenize(vocab, prompt.c_str(), (int32_t) prompt.size(), nullptr, 0, true, true);
    // A non-positive count means the template produced an empty prompt (or
    // tokenization failed). Sampling with an empty context asserts inside
    // llama.cpp and takes the process down, so stop here instead.
    if (n_prompt <= 0) return el_error(buf, "empty prompt after applying chat template");

    // The prompt must leave room for at least one generated token. Without
    // this, budget below goes <= 0 and the caller sees an empty reply
    // misreported as "hit max_tokens" rather than "context overflow".
    if (n_prompt >= h->n_ctx) {
        return el_error(buf, "prompt of " + std::to_string(n_prompt) +
                                 " tokens exceeds context window of " + std::to_string(h->n_ctx));
    }

    std::vector<llama_token> tokens(n_prompt);
    llama_tokenize(vocab, prompt.c_str(), (int32_t) prompt.size(), tokens.data(), (int32_t) tokens.size(), true, true);

    // Ingest the prompt in n_batch-sized chunks (a full prompt commonly exceeds one
    // batch, which llama_decode asserts against). A failure here leaves the KV
    // cache holding only part of the prompt -- sampling from that yields a reply
    // conditioned on a truncated conversation, so report it rather than continuing.
    const int n_batch = (int) llama_n_batch(ctx);
    for (int i = 0; i < n_prompt; i += n_batch) {
        int n = std::min(n_batch, n_prompt - i);
        if (llama_decode(ctx, llama_batch_get_one(tokens.data() + i, n)) != 0) {
            return el_error(buf, "failed to evaluate prompt (llama_decode failed at token " +
                                     std::to_string(i) + " of " + std::to_string(n_prompt) + ")");
        }
    }
    auto t2 = std::chrono::steady_clock::now();

    // Decoding past what's left in the context window is never correct, so the
    // hardware half of the ceiling is always calculated, never configured.
    // max_tokens <= 0 means "auto", the same convention n_ctx and n_threads use
    // in el_open.
    //
    // Auto spends at most half the remaining window on a single reply rather
    // than all of it. Conversations here are append-only (see Conversation),
    // so a reply that consumed the entire window would leave the next turn's
    // prompt with no room and hard-fail the agent loop at the n_prompt guard
    // above. Half keeps replies long enough to be useful while letting the
    // conversation continue -- the optimal value for a multi-turn agent is not
    // the largest single reply. An explicit max_tokens is still honoured (and
    // still clamped to the real headroom, which is never negotiable).
    int headroom = h->n_ctx - n_prompt;
    int budget = max_tokens > 0 ? std::min(max_tokens, headroom) : std::max(1, headroom / 2);

    std::string output;
    char piece[512];
    int n_decoded = 0;
    bool eog = false;
    bool decode_failed = false;
    for (int t = 0; t < budget; t++) {
        llama_token id = llama_sampler_sample(smpl.get(), ctx, -1);
        if (llama_vocab_is_eog(vocab, id)) { eog = true; break; }
        int np = llama_token_to_piece(vocab, id, piece, (int32_t) sizeof(piece), 0, true);
        if (np > 0) output.append(piece, np);
        n_decoded++;
        // Advance the KV cache so we can sample the next token. Skip it on the last
        // planned iteration -- that forward pass would never be sampled from.
        if (t + 1 < budget && llama_decode(ctx, llama_batch_get_one(&id, 1)) != 0) {
            decode_failed = true;
            break;
        }
    }
    auto t3 = std::chrono::steady_clock::now();

    auto ms = [](auto a, auto b) { return std::chrono::duration<double, std::milli>(b - a).count(); };
    result["ms_reset"] = ms(t0, t1);
    result["ms_prefill"] = ms(t1, t2);
    result["ms_decode"] = ms(t2, t3);
    result["n_prompt"] = n_prompt;
    result["n_decoded"] = n_decoded;
    // Truncated means "stopped without reaching an end-of-generation token" --
    // the budget ran out mid-thought (e.g. spent entirely on reasoning) rather
    // than the model finishing naturally. Keyed off the EOG flag rather than
    // n_decoded >= budget so a decode failure isn't reported as a clean stop.
    result["truncated"] = !eog;
    result["budget"] = budget;
    // A mid-generation decode failure still returns whatever was produced, but
    // the caller needs to know the reply was cut short by an error rather than
    // by policy.
    if (decode_failed) result["error"] = "llama_decode failed during generation";

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
    auto known_tool = [&inputs](const std::string &name) {
        if (name.empty()) return false;
        for (const auto &tool : inputs.tools) {
            if (tool.name == name) return true;
        }
        return false;
    };

    bool fallback_used = false;
    if (parsed.tool_calls.empty() && !inputs.tools.empty()) {
        size_t a = output.find('{'), b = output.rfind('}');
        if (a != std::string::npos && b != std::string::npos && b > a) {
            try {
                json j = json::parse(output.substr(a, b - a + 1));
                std::string name = j.value("name", "");
                if (known_tool(name) && j.contains("arguments")) {
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
            if (known_tool(name) && m != std::string::npos) {
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
    result["tool_calls"] = json::array(); // always present, even when empty
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
    if (!handle) return el_error(errbuf, "null handle");
    if (!messages_json) return el_error(errbuf, "null messages");
    // Once the device is lost every subsequent call fails the same way, and
    // some of those failures abort the process rather than throwing. Refuse up
    // front so one lost GPU degrades to clean errors instead of a crash.
    if (((el_handle *) handle)->dead) return el_error(errbuf, "handle is unusable after a previous fatal error");

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
        ((el_handle *) handle)->dead = true;
        return el_error(errbuf, e.what());
    } catch (...) {
        ((el_handle *) handle)->dead = true;
        return el_error(errbuf, "unknown exception in el_generate");
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
    // Deliberately no llama_backend_free(): the backend is process-global and
    // shared by every handle, so freeing it here would pull it out from under
    // any other live handle, and g_backend would still read true -- so a
    // subsequent el_open would skip re-init and run against freed state. It is
    // reclaimed by the OS at exit.
}

}  // extern "C"
