#include "llama.h"
#include "chat.h"
#include "common.h"
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
    llama_context *ctx;
    int n_ctx;
    int n_threads;
    float temp;
    uint32_t seed;
    bool dead = false;
    std::vector<llama_token> cached;
  };

  static bool g_backend = false;

  static enum ggml_log_level el_log_threshold() {
    static const enum ggml_log_level cached = [] {
      const char *level = std::getenv("ELELEM_LOG_LEVEL");
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

  static const int EL_MIN_CTX = 4096;
  static const int EL_FALLBACK_CTX = 65536;

  static llama_context *el_init_context(llama_model *model, int n_ctx, int n_threads) {
    int32_t trained_max = llama_model_n_ctx_train(model);
    uint32_t requested = n_ctx > 0 ? (uint32_t) n_ctx
      : trained_max > 0 ? (uint32_t) trained_max
      : (uint32_t) EL_FALLBACK_CTX;
    llama_context_params cp = llama_context_default_params();
    cp.n_ctx = requested;
    cp.n_threads = n_threads;
    cp.n_threads_batch = n_threads;
    llama_context *ctx = llama_init_from_model(model, cp);
    if (ctx) return ctx;

    for (uint32_t size = requested / 2; size >= EL_MIN_CTX; size /= 2) {
      cp.n_ctx = size;
      ctx = llama_init_from_model(model, cp);
      if (ctx) return ctx;
    }
    return nullptr;
  }

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

    try {
      auto tmpls = common_chat_templates_init(model, "");

      llama_context *ctx = el_init_context(model, n_ctx, threads);
      if (!ctx) { llama_model_free(model); return nullptr; }

      return new el_handle{model, std::move(tmpls), ctx, (int) llama_n_ctx_seq(ctx), threads, temp, (uint32_t) seed};
    } catch (const std::exception &) {
      llama_model_free(model);
      return nullptr;
    }
  }

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

  static const char *el_generate_impl(void *handle, const char *messages_json, const char *tools_json, int max_tokens) {
    auto *h = (el_handle *) handle;
    const llama_vocab *vocab = llama_model_get_vocab(h->model);
    static thread_local std::string buf;

    common_chat_templates_inputs inputs;
    inputs.add_generation_prompt = true;
    inputs.use_jinja = true;
    inputs.reasoning_format = COMMON_REASONING_FORMAT_AUTO;
    common_chat_params cparams;
    try {
      inputs.messages = build_msgs(json::parse(messages_json));
      if (tools_json && *tools_json) inputs.tools = build_tools(json::parse(tools_json));
      cparams = common_chat_templates_apply(h->tmpls.get(), inputs);
    } catch (const std::exception &e) {
      return el_error(buf, std::string("invalid request or chat template: ") + e.what());
    }

    json result;

    llama_context *ctx = h->ctx;
    llama_memory_t mem = llama_get_memory(ctx);

    struct cache_guard {
      llama_memory_t mem;
      std::vector<llama_token> *cached;
      bool committed = false;
      ~cache_guard() {
        if (committed) return;
        llama_memory_clear(mem, true);
        cached->clear();
      }
    } guard{mem, &h->cached};

    std::unique_ptr<llama_sampler, decltype(&llama_sampler_free)> smpl(llama_sampler_chain_init(llama_sampler_chain_default_params()), llama_sampler_free);
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
    if (n_prompt <= 0) return el_error(buf, "empty prompt after applying chat template");
    if (n_prompt >= h->n_ctx) {
      return el_error(buf, "prompt of " + std::to_string(n_prompt) + " tokens exceeds context window of " + std::to_string(h->n_ctx));
    }

    std::vector<llama_token> tokens(n_prompt);
    llama_tokenize(vocab, prompt.c_str(), (int32_t) prompt.size(), tokens.data(), (int32_t) tokens.size(), true, true);

    auto t0 = std::chrono::steady_clock::now();

    int n_match = 0;
    while (n_match < (int) h->cached.size() && n_match < n_prompt && h->cached[n_match] == tokens[n_match]) {
      n_match++;
    }
    if (n_match == n_prompt) n_match = n_prompt - 1;

    if (n_match > 0 && !llama_memory_seq_rm(mem, 0, n_match, -1)) {
      n_match = 0;
    }
    if (n_match == 0) {
      llama_memory_clear(mem, true);
    }
    h->cached.resize(n_match);
    auto t1 = std::chrono::steady_clock::now();

    const int n_batch = (int) llama_n_batch(ctx);
    for (int i = n_match; i < n_prompt; i += n_batch) {
      int n = std::min(n_batch, n_prompt - i);
      if (llama_decode(ctx, llama_batch_get_one(tokens.data() + i, n)) != 0) {
        return el_error(buf, "failed to evaluate prompt (llama_decode failed at token " + std::to_string(i) + " of " + std::to_string(n_prompt) + ")");
      }
      h->cached.insert(h->cached.end(), tokens.begin() + i, tokens.begin() + i + n);
    }
    auto t2 = std::chrono::steady_clock::now();

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
      if (t + 1 < budget && llama_decode(ctx, llama_batch_get_one(&id, 1)) != 0) {
        decode_failed = true;
        break;
      }
      if (t + 1 < budget) h->cached.push_back(id);
    }
    auto t3 = std::chrono::steady_clock::now();

    if (!decode_failed) guard.committed = true;

    auto ms = [](auto a, auto b) { return std::chrono::duration<double, std::milli>(b - a).count(); };
    result["ms_reset"] = ms(t0, t1);
    result["ms_prefill"] = ms(t1, t2);
    result["ms_decode"] = ms(t2, t3);
    result["n_prompt"] = n_prompt;
    result["n_reused"] = n_match;
    result["n_decoded"] = n_decoded;
    result["truncated"] = !eog;
    result["budget"] = budget;

    if (decode_failed) result["error"] = "llama_decode failed during generation";

    common_chat_msg parsed;
    parsed.content = output;
    try {
      common_chat_parser_params pp(cparams);
      pp.reasoning_format = COMMON_REASONING_FORMAT_AUTO;
      if (!cparams.parser.empty()) pp.parser.load(cparams.parser);
      parsed = common_chat_parse(output, false, pp);
    } catch (const std::exception &) {
      parsed.content = output;
    }

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
    result["tool_calls"] = json::array();
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
    if (((el_handle *) handle)->dead) return el_error(errbuf, "handle is unusable after a previous fatal error");

    try {
      return el_generate_impl(handle, messages_json, tools_json, max_tokens);
    } catch (const std::exception &e) {
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
    if (h->dead) {
      delete h;
      return;
    }
    llama_free(h->ctx);
    llama_model_free(h->model);
    delete h;
  }
}  // extern "C"
