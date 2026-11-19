# frozen_string_literal: true

require "fiddle"
require "json"
require "elelem"

require_relative "llama/version"
require_relative "llama/context_choice"
require_relative "llama/host"
require_relative "llama/host_memory"
require_relative "llama/load_library"
require_relative "llama/model_resolver"
require_relative "llama/settings"
require_relative "llama/shim"
require_relative "llama/provider"

Elelem.configure do |config|
  config.provider(:llama) do |model: nil, **|
    model_path = File.expand_path(Elelem::Llama::ModelResolver.resolve(model:))

    Elelem::Llama::Provider.shared(
      model: model_path,
      n_ctx: Integer(ENV.fetch("ELELEM_GGUF_N_CTX", "0")),
      n_threads: Integer(ENV.fetch("ELELEM_GGUF_THREADS", "0")),
      n_gpu_layers: Integer(ENV.fetch("ELELEM_GGUF_N_GPU_LAYERS", "-1")),
      max_tokens: Integer(ENV.fetch("ELELEM_GGUF_MAX_TOKENS", "0")),
      temp: Float(ENV.fetch("ELELEM_GGUF_TEMP", "0.7")),
      seed: Integer(ENV.fetch("ELELEM_GGUF_SEED", "-1"))
    )
  end
end
