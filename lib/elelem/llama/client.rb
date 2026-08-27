# frozen_string_literal: true

require "fiddle"

module Elelem
  module Net
    # In-process GGUF client: loads a local model inside the elelem process via a
    # thin C shim over llama.cpp (see ext/elelem_llama/elelem_llama.cpp), bound
    # with stdlib Fiddle. No subprocess, no HTTP server, no third-party gem.
    class GGUF
      SHIM = File.expand_path("../native/libelelem_llama.so", __dir__)
      V = Fiddle::TYPE_VOIDP
      I = Fiddle::TYPE_INT

      # dlopen + bindings are process-wide resources -- memoize once, like Net.http.
      def self.functions
        @functions ||= begin
          lib = Fiddle.dlopen(SHIM)
          {
            open: Fiddle::Function.new(lib["el_open"], [V, I], V),
            generate: Fiddle::Function.new(lib["el_generate"], [V, V, V, I, I, I, I, V], I)
          }
        end
      end

      def initialize(model_path:, n_ctx: 4096, n_threads: 16, max_tokens: 512)
        @n_ctx = n_ctx
        @n_threads = n_threads
        @max_tokens = max_tokens
        @model = self.class.functions[:open].call(model_path, 0) # 0 gpu layers = CPU (GPU later)
        raise "gguf: failed to load model at #{model_path}" if @model.null?
      end

      # elelem provider contract: fetch(messages, tools=[]) { |event| } -> tool_calls
      def fetch(messages, _tools = [], &block)
        usable = messages.select { |m| m[:content] && !m[:content].to_s.empty? }
        roles, keep_roles = str_array(usable.map { |m| role_of(m[:role]) })
        contents, keep_contents = str_array(usable.map { |m| m[:content].to_s })

        callback = Fiddle::Closure::BlockCaller.new(Fiddle::TYPE_VOID, [V]) do |cstr|
          block&.call(type: "saying", text: Fiddle::Pointer.new(cstr).to_s)
        end

        fns = self.class.functions
        fns[:generate].call(@model, roles, contents, usable.length,
                            @n_ctx, @n_threads, @max_tokens, callback)
        # keep_roles/keep_contents stay referenced through the native call above so
        # GC can't free the malloc'd buffers mid-generation; they fall out of scope
        # (and RUBY_FREE reclaims them) only now that the call has returned.
        [] # tool-calling from the local model is deferred (phase 2)
      end

      private

      def role_of(role)
        r = role.to_s
        %w[system user assistant].include?(r) ? r : "user"
      end

      # Build a C `const char**` from Ruby strings; returns [pointer_array, keep]
      # where `keep` holds the per-string buffers so GC can't free them mid-call.
      def str_array(strings)
        cstrs = strings.map { |s| cstr(s) }
        arr = Fiddle::Pointer.malloc([strings.size, 1].max * Fiddle::SIZEOF_VOIDP, Fiddle::RUBY_FREE)
        cstrs.each_with_index do |ptr, i|
          arr[i * Fiddle::SIZEOF_VOIDP, Fiddle::SIZEOF_VOIDP] = [ptr.to_i].pack("J")
        end
        [arr, cstrs]
      end

      def cstr(str)
        bytes = "#{str}\0".b
        ptr = Fiddle::Pointer.malloc(bytes.bytesize, Fiddle::RUBY_FREE)
        ptr[0, bytes.bytesize] = bytes
        ptr
      end
    end
  end
end
