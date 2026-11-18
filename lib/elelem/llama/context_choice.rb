# frozen_string_literal: true

module Elelem
  module Llama
    ROOMY_CONTEXT = 32_768

    ContextChoice = Data.define(:n_ctx_train, :on_gpu, :affordable) do
      def n_ctx = fast? ? on_gpu : affordable

      def fast? = on_gpu >= [ROOMY_CONTEXT, n_ctx_train].min

      def to_s
        return "the whole model fits on the GPU" if fast? && on_gpu >= affordable
        return "the whole model stays on the GPU for speed, though #{affordable} tokens would fit with RAM" if fast?
        return "part of the model is in RAM, because it does not fit on the GPU" if on_gpu.zero?

        "part of the model is in RAM, because only #{on_gpu} tokens would fit with all of it on the GPU"
      end
    end
  end
end
