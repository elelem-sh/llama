# frozen_string_literal: true

Elelem::Providers.register(:gguf) do
  gpu = %w[vulkan cuda metal].include?(Elelem::Net::GGUF.backend)

  Elelem::Net::GGUF.new(
    model: ENV.fetch("GGUF_MODEL", File.expand_path("~/models/Qwen2.5-Coder-7B-Instruct-Q4_K_M.gguf")),
    n_ctx: Integer(ENV.fetch("GGUF_N_CTX", "8192")),
    n_threads: Integer(ENV.fetch("GGUF_THREADS", "16")),
    max_tokens: Integer(ENV.fetch("GGUF_MAX_TOKENS", "512")),
    n_gpu_layers: Integer(ENV.fetch("GGUF_N_GPU_LAYERS", gpu ? "999" : "0")),
    temp: Float(ENV.fetch("GGUF_TEMP", "0.7")),
    seed: Integer(ENV.fetch("GGUF_SEED", "-1"))
  )
end
