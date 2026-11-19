# frozen_string_literal: true

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

      def path = beside(name)

      def open(dlopen: Fiddle.method(:dlopen))
        loaded_first.each { dlopen.call(beside(it)) }
        dlopen.call(path)
      end

      private

      def name
        case @host_os
        when WINDOWS then "elelem_llama.dll"
        when DARWIN then "libelelem_llama.dylib"
        else "libelelem_llama.so"
        end
      end

      def loaded_first = windows? ? LOADED_FIRST_ON_WINDOWS : []

      def windows? = @host_os.match?(WINDOWS)

      def beside(file) = File.join(@dir, file)
    end
  end
end
