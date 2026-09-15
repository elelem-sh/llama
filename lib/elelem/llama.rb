# frozen_string_literal: true

require "fiddle"
require "json"
require "elelem"

require_relative "llama/version"
require_relative "llama/provider"

Elelem.configure do |config|
  config.provider(:llama) do |model: nil, **|
    default_model = lambda {
      dir = File.expand_path("~/.agents/models")
      link = File.join(dir, "default.gguf")
      return link if File.symlink?(link) || File.exist?(link)

      models = Dir[File.join(dir, "*.gguf")]
      case models.length
      when 1 then models.first
      when 0 then raise "no .gguf model found in #{dir}. Set ELELEM_GGUF_MODEL=/path/to/model.gguf"
      else raise "multiple .gguf models found in #{dir}. Set ELELEM_GGUF_MODEL=/path/to/model.gguf or symlink " \
                 "#{link} to the one to use by default"
      end
    }

    model_path = model || ENV.fetch("ELELEM_GGUF_MODEL") { default_model.call }

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
