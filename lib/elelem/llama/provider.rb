# frozen_string_literal: true

module Elelem
  module Llama
    class Provider
      NATIVE = File.expand_path("../../../ext/elelem/llama/native", __dir__)
      SHIM = Shim.new(dir: NATIVE)
      V = Fiddle::TYPE_VOIDP
      I = Fiddle::TYPE_INT
      F = Fiddle::TYPE_FLOAT

      BACKENDS = { "MTL" => "metal", "ROCm" => "hip" }.freeze

      def self.backend = functions[:backend].call.to_s.then { BACKENDS.fetch(it, &:downcase) }

      def self.devices
        JSON.parse(functions[:devices].call.to_s, symbolize_names: true).map { Device.new(**it) }
      end

      class << self
        attr_writer :functions
      end

      def self.start(lib, modules: NATIVE)
        failure = Fiddle::Function.new(lib["el_start"], [V], V).call(modules.encode(Encoding::UTF_8))
        raise "llama: #{failure}" unless failure.null?
      end

      def self.functions
        @functions ||= begin
          lib = SHIM.open
          start(lib)
          {
            backend: Fiddle::Function.new(lib["el_backend"], [], V),
            devices: Fiddle::Function.new(lib["el_devices"], [], V),
            open: Fiddle::Function.new(lib["el_open"], [V, I, I, I, F, I, Fiddle::TYPE_LONG_LONG], V),
            settings: Fiddle::Function.new(lib["el_settings"], [V], V),
            generate: Fiddle::Function.new(lib["el_generate"], [V, V, V, I, V, V], V, need_gvl: false),
            close: Fiddle::Function.new(lib["el_close"], [V], Fiddle::TYPE_VOID)
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
        close = functions[:close]
        at_exit { close.call(handle) }
      end

      def initialize(model:, n_ctx: 0, n_threads: 0, max_tokens: 0, n_gpu_layers: -1, temp: 0.7, seed: -1,
                     memory: HostMemory.detect, err: $stderr)
        @max_tokens = max_tokens
        @model = File.basename(File.exist?(model) ? File.realpath(model) : model)
        budget = memory.budget
        announce(model, err)
        @handle = self.class.functions[:open].call(model, n_gpu_layers, n_ctx, n_threads, temp, seed, budget)
        raise "llama: failed to load model at #{model}" if @handle.null?

        self.class.close_at_exit(@handle)
        Elelem.logger.debug { "llama: #{settings}" }

        @streamed = false
        @block = nil
        # Built once: each Fiddle::Closure::BlockCaller allocates a native FFI
        # trampoline, and el_generate's reentrancy guard already prevents
        # concurrent native calls on this handle, so one instance is safe to reuse.
        #
        # The shim also calls it with an empty piece while it has no text, which
        # is when Ruby gets to raise an Interrupt. Nothing may unwind through the
        # shim, so whatever is raised is kept for fetch and the shim told to stop.
        @on_token = Fiddle::Closure::BlockCaller.new(I, [V, V]) do |_userdata, piece|
          @failure ? 1 : stream(piece)
        rescue Exception => e # rubocop:disable Lint/RescueException
          @failure = e
          1
        end
      end

      def fetch(messages, tools = [], &block)
        @streamed = false
        @block = block
        @failure = nil

        ptr = self.class.functions[:generate].call(
          @handle, JSON.generate(messages), JSON.generate(tools), @max_tokens, @on_token, nil
        )
        raise @failure if @failure

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

      attr_reader :model

      def settings
        Settings.new(**JSON.parse(self.class.functions[:settings].call(@handle).to_s, symbolize_names: true))
      end

      private

      def announce(model, err)
        size = File.size?(model)
        err.puts "loading #{@model}#{format(" (%.1f GB)", size / 1e9) if size}…"
      end

      def stream(piece)
        text = Fiddle::Pointer.new(piece).to_s.force_encoding(Encoding::UTF_8).scrub
        return 0 if text.empty?

        @streamed = true
        @block&.call(type: "thinking", text: text)
        0
      end

      def parse(raw)
        JSON.parse(raw.to_s)
      rescue JSON::ParserError
        {}
      end
    end
  end
end
