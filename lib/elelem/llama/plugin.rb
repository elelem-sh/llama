# frozen_string_literal: true

# In-process GGUF provider (see Elelem::Net::GGUF in lib/elelem/net/gguf.rb).
Elelem::Providers.register(:gguf) do
  # Offload all layers by default when a GPU backend was compiled (extconf.rb stamps
  # it); CPU builds stay at 0. GGUF_N_GPU_LAYERS overrides either way.
  gpu = %w[vulkan cuda metal].include?(Elelem::Net::GGUF.backend)

  Elelem::Net::GGUF.new(
    model_path: ENV.fetch("GGUF_MODEL", File.expand_path("~/models/Qwen2.5-Coder-7B-Instruct-Q4_K_M.gguf")),
    n_ctx: Integer(ENV.fetch("GGUF_N_CTX", "8192")), # room for system prompt + tool results
    n_threads: Integer(ENV.fetch("GGUF_THREADS", "16")),
    max_tokens: Integer(ENV.fetch("GGUF_MAX_TOKENS", "512")),
    n_gpu_layers: Integer(ENV.fetch("GGUF_N_GPU_LAYERS", gpu ? "999" : "0")), # 999 = offload all
    temp: Float(ENV.fetch("GGUF_TEMP", "0.7")),                               # 0 = greedy
    seed: Integer(ENV.fetch("GGUF_SEED", "-1"))                              # -1 = llama default
  )
end
