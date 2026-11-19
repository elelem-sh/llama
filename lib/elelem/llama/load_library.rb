# frozen_string_literal: true

module Elelem
  module Llama
    class LoadLibrary
      def initialize(load_library_w: nil, last_error: -> { Fiddle.win32_last_error })
        @load_library_w = load_library_w
        @last_error = last_error
      end

      def call(path)
        return unless load_library_w.call(wide(path)).null?

        raise Fiddle::DLError, "Windows cannot load #{path}: error #{@last_error.call}"
      end

      private

      def load_library_w
        @load_library_w ||= Fiddle::Function.new(
          Fiddle.dlopen("kernel32")["LoadLibraryW"], [Fiddle::TYPE_VOIDP], Fiddle::TYPE_VOIDP
        )
      end

      def wide(path) = "#{path.tr("/", "\\")}\0".encode(Encoding::UTF_16LE)
    end
  end
end
