# frozen_string_literal: true

require "fiddle"
require "json"
require "elelem"

require_relative "llama/version"
require_relative "llama/provider"

Elelem.configure do |config|
  config.provider(:llama) do |model: nil, **|
    gpu = %w[vulkan cuda metal].include?(Elelem::Llama::Provider.backend)

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

    Elelem::Llama::Provider.new(
      model: model || ENV.fetch("ELELEM_GGUF_MODEL") { default_model.call },
      n_ctx: 0,
      n_threads: 0,
      n_gpu_layers: gpu ? 999 : 0,
      max_tokens: Integer(ENV.fetch("ELELEM_GGUF_MAX_TOKENS", "0")),
      temp: Float(ENV.fetch("ELELEM_GGUF_TEMP", "0.7")),
      seed: Integer(ENV.fetch("ELELEM_GGUF_SEED", "-1"))
    )
  end
end
