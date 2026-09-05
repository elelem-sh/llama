# frozen_string_literal: true

Elelem::Providers.register(:gguf) do
  gpu = %w[vulkan cuda metal].include?(Elelem::Net::GGUF.backend)

  # No baked-in model path: a fresh user has no reason to have the same file
  # this project's author does. With exactly one .gguf in ~/.agents/models,
  # that's unambiguously "the model"; with zero or several, only the user
  # can say which -- so ask, rather than silently guessing.
  default_model = -> {
    dir = File.expand_path("~/.agents/models")
    link = File.join(dir, "default.gguf")
    return link if File.symlink?(link) || File.exist?(link)

    models = Dir[File.join(dir, "*.gguf")].sort
    case models.length
    when 1 then models.first
    when 0 then raise "no .gguf model found in #{dir}. Set ELELEM_GGUF_MODEL=/path/to/model.gguf"
    else raise "multiple .gguf models found in #{dir}. Set ELELEM_GGUF_MODEL=/path/to/model.gguf " \
               "or symlink #{link} to the one to use by default"
    end
  }

  Elelem::Net::GGUF.new(
    model: ENV.fetch("ELELEM_GGUF_MODEL") { default_model.call },
    # n_ctx/n_threads/n_gpu_layers/max_tokens are calculated, not configured:
    # the native shim asks llama.cpp for the model's trained max context, capped
    # at a size that leaves GPU headroom for a large prompt's compute buffer
    # (see EL_MAX_AUTO_CTX in elelem_llama.cpp), falling back to smaller sizes
    # still if that doesn't fit; thread count comes from the host's physical
    # cores; every layer is offloaded whenever a GPU backend is present; a
    # reply may use whatever context the prompt leaves free. Convention over
    # configuration -- there's no better *number* a user could supply than
    # what the hardware and model already determine.
    n_ctx: 0,
    n_threads: 0,
    n_gpu_layers: gpu ? 999 : 0,
    max_tokens: Integer(ENV.fetch("ELELEM_GGUF_MAX_TOKENS", "0")),
    temp: Float(ENV.fetch("ELELEM_GGUF_TEMP", "0.7")),
    seed: Integer(ENV.fetch("ELELEM_GGUF_SEED", "-1"))
  )
end
