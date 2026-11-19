# frozen_string_literal: true

require_relative "load_library"

module Elelem
  module Llama
    class Shim
      WINDOWS = /mingw|mswin/
      DARWIN = /darwin/
      LOADED_FIRST_ON_WINDOWS = %w[ggml-base.dll ggml.dll libllama.dll libllama-common.dll].freeze

      def initialize(dir:, host_os: RbConfig::CONFIG["host_os"])
        @dir = dir
        @host_os = host_os
      end

      def path = File.join(@dir, name)

      def open(dlopen: Fiddle.method(:dlopen), load_library: LoadLibrary.new)
        return dlopen.call(path) unless windows?

        [*LOADED_FIRST_ON_WINDOWS, name].each { load_library.call(File.join(@dir, it)) }
        dlopen.call(name)
      end

      private

      def name
        case @host_os
        when WINDOWS then "elelem_llama.dll"
        when DARWIN then "libelelem_llama.dylib"
        else "libelelem_llama.so"
        end
      end

      def windows? = @host_os.match?(WINDOWS)
    end
  end
end
