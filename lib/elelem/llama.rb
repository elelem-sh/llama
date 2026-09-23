# frozen_string_literal: true

require "fiddle"
require "json"
require "elelem"

require_relative "llama/version"
require_relative "llama/model_resolver"
require_relative "llama/provider"

Elelem.configure do |config|
  config.provider(:llama) do |model: nil, **|
    model_path = Elelem::Llama::ModelResolver.resolve(model:)

    default_gpu_layers = lambda {
      backend = Elelem::Llama::Provider.backend
      next "0" unless %w[vulkan cuda metal].include?(backend)
      next "0" if backend == "metal" && !RUBY_PLATFORM.include?("arm64")

      free = Elelem::Llama::Provider.gpu_free_memory
      free && File.size(model_path) < free * 0.8 ? "999" : "0"
    }

    Elelem::Llama::Provider.new(
      model: model_path,
      n_ctx: Integer(ENV.fetch("ELELEM_GGUF_N_CTX", "0")),
      n_threads: 0,
      n_gpu_layers: Integer(ENV.fetch("ELELEM_GGUF_N_GPU_LAYERS") { default_gpu_layers.call }),
      max_tokens: Integer(ENV.fetch("ELELEM_GGUF_MAX_TOKENS", "0")),
      temp: Float(ENV.fetch("ELELEM_GGUF_TEMP", "0.7")),
      seed: Integer(ENV.fetch("ELELEM_GGUF_SEED", "-1"))
    )
  end
end
