// Thin, stable C façade over llama.cpp for in-process inference from Ruby (Fiddle).
// All volatile by-value structs (llama_model_params, llama_context_params,
// llama_batch, ...) are handled here in C, compiled against the vendored llama.h,
// so Ruby only ever sees simple, stable signatures that don't change when
// llama.cpp bumps. Model stays resident; a fresh context per generate call keeps
// each request stateless (no KV bleed between turns).
#include "llama.h"
#include <string>
#include <vector>

extern "C" {

typedef void (*el_token_cb)(const char *piece);

static bool g_backend = false;

// Load a model and keep it resident. n_gpu_layers 0 = CPU. Returns model* or NULL.
void *el_open(const char *path, int n_gpu_layers) {
    if (!g_backend) { llama_backend_init(); g_backend = true; }
    llama_model_params mp = llama_model_default_params();
    mp.n_gpu_layers = n_gpu_layers;
    return (void *) llama_model_load_from_file(path, mp);
}

// Render the message array through the model's chat template, then stream the
// generated pieces to cb. Returns tokens generated, or -1 on error.
int el_generate(void *model_, const char **roles, const char **contents, int n_msgs,
                int n_ctx, int n_threads, int max_tokens, el_token_cb cb) {
    llama_model *model = (llama_model *) model_;
    const llama_vocab *vocab = llama_model_get_vocab(model);

    std::vector<llama_chat_message> msgs(n_msgs);
    for (int i = 0; i < n_msgs; i++) { msgs[i].role = roles[i]; msgs[i].content = contents[i]; }

    std::string text;
    const char *tmpl = llama_model_chat_template(model, nullptr);
    if (tmpl && n_msgs > 0) {
        int need = llama_chat_apply_template(tmpl, msgs.data(), n_msgs, true, nullptr, 0);
        if (need > 0) {
            std::vector<char> buf(need);
            int n = llama_chat_apply_template(tmpl, msgs.data(), n_msgs, true, buf.data(), (int32_t) buf.size());
            if (n > 0) text.assign(buf.data(), n);
        }
    }
    if (text.empty()) {  // no template: crude fallback
        for (int i = 0; i < n_msgs; i++) { text += roles[i]; text += ": "; text += contents[i]; text += "\n"; }
        text += "assistant:";
    }

    llama_context_params cp = llama_context_default_params();
    cp.n_ctx = (uint32_t) n_ctx;
    cp.n_threads = n_threads;
    cp.n_threads_batch = n_threads;
    llama_context *ctx = llama_init_from_model(model, cp);
    if (!ctx) return -1;

    int n_prompt = -llama_tokenize(vocab, text.c_str(), (int32_t) text.size(), nullptr, 0, true, true);
    std::vector<llama_token> tokens(n_prompt);
    if (llama_tokenize(vocab, text.c_str(), (int32_t) text.size(), tokens.data(),
                       (int32_t) tokens.size(), true, true) < 0) {
        llama_free(ctx);
        return -1;
    }

    llama_sampler *smpl = llama_sampler_chain_init(llama_sampler_chain_default_params());
    llama_sampler_chain_add(smpl, llama_sampler_init_top_k(40));
    llama_sampler_chain_add(smpl, llama_sampler_init_top_p(0.95f, 1));
    llama_sampler_chain_add(smpl, llama_sampler_init_temp(0.7f));
    llama_sampler_chain_add(smpl, llama_sampler_init_dist(LLAMA_DEFAULT_SEED));

    llama_batch batch = llama_batch_get_one(tokens.data(), (int32_t) tokens.size());
    llama_token cur = 0;
    char piece[512];
    int generated = 0;

    for (int i = 0; i < max_tokens; i++) {
        if (llama_decode(ctx, batch) != 0) break;
        llama_token id = llama_sampler_sample(smpl, ctx, -1);
        if (llama_vocab_is_eog(vocab, id)) break;
        int np = llama_token_to_piece(vocab, id, piece, (int32_t) sizeof(piece), 0, true);
        if (np > 0 && cb) { std::string s(piece, np); cb(s.c_str()); }
        generated++;
        cur = id;                                   // must outlive the decode below
        batch = llama_batch_get_one(&cur, 1);
    }

    llama_sampler_free(smpl);
    llama_free(ctx);
    return generated;
}

void el_close(void *model) { if (model) llama_model_free((llama_model *) model); }

}  // extern "C"
