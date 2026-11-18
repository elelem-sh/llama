# frozen_string_literal: true

module Elelem
  module Llama
    Settings = Data.define(:n_ctx, :n_ctx_train, :kv_cache, :threads, :layers, :gpu_layers, :overflowing_layers,
                           :devices, :host_bytes, :host_budget) do
      def to_s = [context, placement, *experts, memory].join("; ")

      private

      def context
        "context of #{n_ctx} tokens (the model allows #{n_ctx_train}), #{kv_cache} KV cache, #{threads} threads"
      end

      def placement
        return "all weights on the CPU" if devices.empty?
        return "weights on the CPU, #{gpus} speeds up reading long prompts" if gpu_layers.zero?

        "#{gpu_layers} of #{layers} layers on #{gpus}"
      end

      def experts
        overflowing_layers.positive? ? ["#{overflowing_layers} layers keep their expert weights in RAM"] : []
      end

      def memory
        ["#{gib(host_bytes)} of RAM used from a budget of #{gib(host_budget)}",
         *devices.map { "#{gib(it[:bytes])} on #{it[:name]}" }].join(", ")
      end

      def gpus = devices.map { it[:description].to_s.empty? ? it[:name] : it[:description] }.join(" + ")

      def gib(bytes) = format("%.1f GiB", bytes.fdiv(1024**3))
    end
  end
end
