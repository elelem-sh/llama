# frozen_string_literal: true

require_relative "host"
require_relative "load_library"

module Elelem
  module Llama
    class Shim
      LOADED_FIRST_ON_WINDOWS = %w[ggml-base.dll ggml.dll libllama.dll libllama-common.dll].freeze

      def initialize(dir:, host: Host.new)
        @dir = dir
        @host = host
      end

      def path = File.join(@dir, name)

      def open(dlopen: Fiddle.method(:dlopen), load_library: LoadLibrary.new)
        return dlopen.call(path) unless @host.windows?

        [*LOADED_FIRST_ON_WINDOWS, name].each { load_library.call(File.join(@dir, it)) }
        dlopen.call(name)
      end

      private

      def name
        return "elelem_llama.dll" if @host.windows?
        return "libelelem_llama.dylib" if @host.darwin?

        "libelelem_llama.so"
      end
    end
  end
end
