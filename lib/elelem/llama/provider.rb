# frozen_string_literal: true

module Elelem
  module Llama
    class Provider
      NATIVE = File.expand_path("../../../ext/elelem/llama/native", __dir__)
      SHIM = File.join(NATIVE, "libelelem_llama.#{RbConfig::CONFIG["host_os"] =~ /darwin/ ? "dylib" : "so"}")
      V = Fiddle::TYPE_VOIDP
      I = Fiddle::TYPE_INT
      F = Fiddle::TYPE_FLOAT

      def self.backend
        File.read(File.join(NATIVE, "backend")).strip
      rescue SystemCallError
        "cpu"
      end

      def self.gpu_free_memory
        bytes = functions[:gpu_free_memory].call
        bytes.negative? ? nil : bytes
      end

      def self.functions
        @functions ||= begin
          lib = Fiddle.dlopen(SHIM)
          {
            open: Fiddle::Function.new(lib["el_open"], [V, I, I, I, F, I], V),
            generate: Fiddle::Function.new(lib["el_generate"], [V, V, V, I, V, V], V, need_gvl: true),
            close: Fiddle::Function.new(lib["el_close"], [V], Fiddle::TYPE_VOID),
            gpu_free_memory: Fiddle::Function.new(lib["el_gpu_free_memory"], [], Fiddle::TYPE_LONG)
          }
        end
      end

      # One loaded model per set of options. /provider llama builds the provider
      # again, and loading the weights a second time would double RAM and VRAM.
      # The instance is shared, not the handle: each instance closes its own handle.
      def self.shared(**options)
        (@shared ||= {})[options] ||= new(**options)
      end

      def self.close_at_exit(handle)
        at_exit { functions[:close].call(handle) }
      end

      def initialize(model:, n_ctx: 0, n_threads: 0, max_tokens: 0, n_gpu_layers: 0, temp: 0.7, seed: -1)
        @max_tokens = max_tokens
        @handle = self.class.functions[:open].call(model, n_gpu_layers, n_ctx, n_threads, temp, seed)
        raise "llama: failed to load model at #{model}" if @handle.null?

        self.class.close_at_exit(@handle)

        @streamed = false
        @block = nil
        # Built once: each Fiddle::Closure::BlockCaller allocates a native FFI
        # trampoline, and el_generate's reentrancy guard already prevents
        # concurrent native calls on this handle, so one instance is safe to reuse.
        @on_token = Fiddle::Closure::BlockCaller.new(Fiddle::TYPE_VOID, [V, V]) do |_userdata, piece|
          @streamed = true
          text = Fiddle::Pointer.new(piece).to_s.force_encoding(Encoding::UTF_8).scrub
          @block&.call(type: "thinking", text: text)
        rescue Exception => e # rubocop:disable Lint/RescueException
          Elelem.logger.warn("llama: streaming callback failed: #{e.message}")
        end
      end

      def fetch(messages, tools = [], &block)
        @streamed = false
        @block = block

        ptr = self.class.functions[:generate].call(
          @handle, JSON.generate(messages), JSON.generate(tools), @max_tokens, @on_token, nil
        )
        result = JSON.parse(Fiddle::Pointer.new(ptr).to_s)

        Elelem.logger.debug("llama: tool-call fallback used") if result["fallback"]
        Elelem.logger.debug("llama: harmony tag fallback used") if result["harmony_tag_fallback"]
        Elelem.logger.warn("llama: generate error: #{result["error"]}") if result["error"]
        if result["ms_decode"]
          Elelem.logger.debug(format(
                                "llama: reset=%.0fms prefill=%.0fms (n=%d, %d reused) decode=%.0fms (n=%d)",
                                result["ms_reset"], result["ms_prefill"], result["n_prompt"], result["n_reused"].to_i,
                                result["ms_decode"], result["n_decoded"]
                              ))
        end
        if result["tool_calls"].to_a.empty? && !tools.empty? && result["content"].to_s.include?("\"name\"")
          Elelem.logger.debug("llama: no tool calls parsed, tools offered")
        end
        reasoning = result["reasoning"].to_s
        Elelem.logger.debug("llama: reasoning: #{reasoning}") unless reasoning.empty?
        block&.call(type: "thinking", text: reasoning) unless reasoning.empty? || @streamed

        content = result["content"].to_s
        if result["error"] && content.empty?
          content = "[error: #{result["error"]}]"
        elsif result["truncated"] && content.empty? && result.fetch("tool_calls", []).empty?
          limit = @max_tokens.positive? ? "max_tokens" : "available context"
          Elelem.logger.warn("llama: hit #{limit} (#{result["n_decoded"]}/#{result["budget"]}) before producing " \
                             "a reply")
          content = "[no reply: ran out of output tokens before finishing]"
        end
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
