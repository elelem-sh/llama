# frozen_string_literal: true

Elelem::Providers.register(:gguf) do
  gpu = %w[vulkan cuda metal].include?(Elelem::Net::GGUF.backend)

  Elelem::Net::GGUF.new(
    model: ENV.fetch("GGUF_MODEL", File.expand_path("~/.elelem/models/Qwen3.8-27B-UD-Q4_K_XL.gguf")),
    # n_ctx/n_threads/n_gpu_layers are calculated, not configured: the native
    # shim asks llama.cpp for the model's trained max context (falling back to
    # smaller sizes if it doesn't fit) and picks thread count from the host's
    # physical cores; every layer is offloaded whenever a GPU backend is
    # present. Convention over configuration -- there's no better answer a
    # user could supply than what the hardware and model already determine.
    n_ctx: 0,
    n_threads: 0,
    n_gpu_layers: gpu ? 999 : 0,
    max_tokens: Integer(ENV.fetch("GGUF_MAX_TOKENS", "4096")),
    temp: Float(ENV.fetch("GGUF_TEMP", "0.7")),
    seed: Integer(ENV.fetch("GGUF_SEED", "-1"))
  )
end
