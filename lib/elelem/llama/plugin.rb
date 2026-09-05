# frozen_string_literal: true

Elelem::Providers.register(:gguf) do
  gpu = %w[vulkan cuda metal].include?(Elelem::Net::GGUF.backend)

  Elelem::Net::GGUF.new(
    model: ENV.fetch("GGUF_MODEL", File.expand_path("~/.elelem/models/Qwen3.8-27B-UD-Q4_K_XL.gguf")),
    # 0 = auto: the native shim asks llama.cpp for the model's trained max
    # context (falling back to smaller sizes if it doesn't fit), and picks
    # thread count from the host's physical cores. Set GGUF_N_CTX/GGUF_THREADS
    # to override.
    n_ctx: Integer(ENV.fetch("GGUF_N_CTX", "0")),
    n_threads: Integer(ENV.fetch("GGUF_THREADS", "0")),
    max_tokens: Integer(ENV.fetch("GGUF_MAX_TOKENS", "4096")),
    n_gpu_layers: Integer(ENV.fetch("GGUF_N_GPU_LAYERS", gpu ? "999" : "0")),
    temp: Float(ENV.fetch("GGUF_TEMP", "0.7")),
    seed: Integer(ENV.fetch("GGUF_SEED", "-1"))
  )
end
