# frozen_string_literal: true

require "fiddle"
require "json"

module Elelem
  module Net
    class GGUF
      NATIVE = File.expand_path("../native", __dir__)
      SHIM = File.join(NATIVE, "libelelem_llama.so")
      V = Fiddle::TYPE_VOIDP
      I = Fiddle::TYPE_INT
      F = Fiddle::TYPE_FLOAT

      def self.backend
        File.read(File.join(NATIVE, "backend")).strip
      rescue SystemCallError
        "cpu"
      end

      def self.functions
        @functions ||= begin
          lib = Fiddle.dlopen(SHIM)
          {
            open: Fiddle::Function.new(lib["el_open"], [V, I, I, I, F, I], V),
            generate: Fiddle::Function.new(lib["el_generate"], [V, V, V, I], V)
          }
        end
      end

      def initialize(model:, n_ctx: 8192, n_threads: 16, max_tokens: 512, n_gpu_layers: 0, temp: 0.7, seed: -1)
        @max_tokens = max_tokens
        @handle = self.class.functions[:open].call(model, n_gpu_layers, n_ctx, n_threads, temp, seed)
        raise "gguf: failed to load model at #{model}" if @handle.null?
      end

      def fetch(messages, tools = [], &block)
        ptr = self.class.functions[:generate].call(@handle, JSON.generate(messages), JSON.generate(tools), @max_tokens)
        result = JSON.parse(Fiddle::Pointer.new(ptr).to_s)

        Elelem.logger.debug("gguf: tool-call fallback used") if result["fallback"]
        if result["tool_calls"].to_a.empty? && !tools.empty? && result["content"].to_s.include?("\"name\"")
          Elelem.logger.debug("gguf: no tool calls parsed, tools offered")
        end
        Elelem.logger.warn("gguf: native generate error: #{result["error"]}") if result["error"]
        Elelem.logger.debug("gguf: reasoning: #{result["reasoning"]}") if result["reasoning"].to_s != ""

        content = result["content"].to_s
        block&.call(type: "saying", text: content) unless content.empty?

        result.fetch("tool_calls", []).map do |call|
          tool_call = {
            id: call["id"],
            name: call["name"],
            arguments: parse(call["arguments"])
          }
          block&.call(tool_call.merge(type: "doing"))
          tool_call
        end
      end

      private

      def parse(raw)
        JSON.parse(raw.to_s)
      rescue JSON::ParserError
        {}
      end
    end
  end
end
