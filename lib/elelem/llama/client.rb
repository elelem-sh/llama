# frozen_string_literal: true

require "fiddle"
require "json"

module Elelem
  module Net
    # In-process GGUF client: loads a local model inside the elelem process via a
    # thin C shim over llama.cpp + its common_chat layer (see
    # ext/elelem_llama/elelem_llama.cpp), bound with stdlib Fiddle. The shim takes
    # OpenAI-style messages + tools as JSON and returns {content, tool_calls} as
    # JSON, so the model participates in the normal agent tool loop. No subprocess,
    # no HTTP server, no third-party gem.
    class GGUF
      SHIM = File.expand_path("../native/libelelem_llama.so", __dir__)
      V = Fiddle::TYPE_VOIDP
      I = Fiddle::TYPE_INT

      # dlopen + bindings are process-wide resources -- memoize once, like Net.http.
      def self.functions
        @functions ||= begin
          unless File.exist?(SHIM)
            raise "gguf: native shim missing at #{SHIM}\n       run: bundle exec rake compile"
          end
          lib = Fiddle.dlopen(SHIM)
          {
            open: Fiddle::Function.new(lib["el_open"], [V, I, I, I], V),
            generate: Fiddle::Function.new(lib["el_generate"], [V, V, V, I], V)
          }
        end
      end

      def initialize(model_path:, n_ctx: 4096, n_threads: 16, max_tokens: 512)
        @max_tokens = max_tokens
        @handle = self.class.functions[:open].call(model_path, 0, n_ctx, n_threads) # 0 = CPU
        raise "gguf: failed to load model at #{model_path}" if @handle.null?
      end

      # elelem provider contract: fetch(messages, tools=[]) { |event| } -> tool_calls.
      # Streams the reply as a "saying" event and each parsed tool call as a
      # "doing" event, which the agent loop executes and feeds back.
      def fetch(messages, tools = [], &block)
        ptr = self.class.functions[:generate].call(
          @handle, JSON.generate(messages), JSON.generate(tools), @max_tokens
        )
        result = JSON.parse(Fiddle::Pointer.new(ptr).to_s)

        content = result["content"].to_s
        block&.call(type: "saying", text: content) unless content.empty?

        result.fetch("tool_calls", []).map do |call|
          tool_call = { id: call["id"], name: call["name"], arguments: parse_args(call["arguments"]) }
          block&.call(tool_call.merge(type: "doing"))
          tool_call
        end
      end

      private

      def parse_args(raw)
        JSON.parse(raw.to_s)
      rescue JSON::ParserError
        {}
      end
    end
  end
end
