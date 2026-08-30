# frozen_string_literal: true

Elelem::Providers.register(:gguf) do
  gpu = %w[vulkan cuda metal].include?(Elelem::Net::GGUF.backend)

  Elelem::Net::GGUF.new(
    model: ENV.fetch("GGUF_MODEL", File.expand_path("~/.elelem/models/gpt-oss-20b-MXFP4.gguf")),
    n_ctx: Integer(ENV.fetch("GGUF_N_CTX", "16384")),
    n_threads: Integer(ENV.fetch("GGUF_THREADS", "16")),
    max_tokens: Integer(ENV.fetch("GGUF_MAX_TOKENS", "4096")),
    n_gpu_layers: Integer(ENV.fetch("GGUF_N_GPU_LAYERS", gpu ? "999" : "0")),
    temp: Float(ENV.fetch("GGUF_TEMP", "0.7")),
    seed: Integer(ENV.fetch("GGUF_SEED", "-1"))
  )
end
