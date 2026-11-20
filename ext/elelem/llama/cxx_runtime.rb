# frozen_string_literal: true

require "fileutils"
require_relative "patchelf"

module Elelem
  module Llama
    class CxxRuntime
      OWN_NAMES = { "libstdc++.so.6" => "libstdc++-elelem.so.6", "libgcc_s.so.1" => "libgcc_s-elelem.so.1" }.freeze

      def initialize(compiler:, patchelf: Patchelf.new)
        @compiler = compiler
        @patchelf = patchelf
      end

      def install(into)
        FileUtils.mkdir_p(into)
        OWN_NAMES.each { |library, own| carry(library, File.join(into, own)) }
        Dir.children(into).sort.each { @patchelf.need(File.join(into, it), OWN_NAMES) }
      end

      private

      def carry(library, as)
        FileUtils.cp(found(library), as)
        @patchelf.name(as, File.basename(as))
      end

      def found(library)
        path = @compiler.file_name(library)
        return path if known?(path)

        raise "#{@compiler} does not know where #{library} is"
      end

      def known?(path) = File.absolute_path?(path) && File.file?(path)
    end
  end
end
