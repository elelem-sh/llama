# frozen_string_literal: true

module Elelem
  module Llama
    class LoadLibrary
      KERNEL32 = lambda do |wide_path|
        pointer = Fiddle::TYPE_VOIDP
        Fiddle::Function.new(Fiddle.dlopen("kernel32")["LoadLibraryW"], [pointer], pointer).call(wide_path)
      end

      def initialize(load_library_w: KERNEL32)
        @load_library_w = load_library_w
      end

      def call(path)
        raise Fiddle::DLError, "Windows cannot load #{path}" if @load_library_w.call(wide(path)).null?
      end

      private

      def wide(path) = "#{path.tr("/", "\\")}\0".encode(Encoding::UTF_16LE)
    end
  end
end
