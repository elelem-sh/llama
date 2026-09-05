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

      def initialize(model:, n_ctx: 0, n_threads: 0, max_tokens: 512, n_gpu_layers: 0, temp: 0.7, seed: -1)
        @max_tokens = max_tokens
        @handle = self.class.functions[:open].call(model, n_gpu_layers, n_ctx, n_threads, temp, seed)
        raise "gguf: failed to load model at #{model}" if @handle.null?
      end

      def fetch(messages, tools = [], &block)
        ptr = self.class.functions[:generate].call(@handle, JSON.generate(messages), JSON.generate(tools), @max_tokens)
        result = JSON.parse(Fiddle::Pointer.new(ptr).to_s)

        Elelem.logger.debug("gguf: tool-call fallback used") if result["fallback"]
        Elelem.logger.debug("gguf: harmony tag fallback used") if result["harmony_tag_fallback"]
        Elelem.logger.warn("gguf: native generate error: #{result["error"]}") if result["error"]
        if result["ms_decode"]
          Elelem.logger.debug(format(
            "gguf: reset=%.0fms prefill=%.0fms (n=%d) decode=%.0fms (n=%d)",
            result["ms_reset"], result["ms_prefill"], result["n_prompt"], result["ms_decode"], result["n_decoded"]
          ))
        end
        if result["tool_calls"].to_a.empty? && !tools.empty? && result["content"].to_s.include?("\"name\"")
          Elelem.logger.debug("gguf: no tool calls parsed, tools offered")
        end
        reasoning = result["reasoning"].to_s
        Elelem.logger.debug("gguf: reasoning: #{reasoning}") unless reasoning.empty?
        block&.call(type: "thinking", text: reasoning) unless reasoning.empty?

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
