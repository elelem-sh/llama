# frozen_string_literal: true

require "fileutils"

module Elelem
  module Llama
    class SharedLibraries
      LIBRARY = /\.(so|dylib|dll)(\.\d+)*\z/
      LINUX_VERSION = /(\.so\.\d+)(?:\.\d+)+\z/
      MACOS_VERSION = /(\.\d+)(?:\.\d+)+(\.dylib)\z/

      def initialize(dir)
        @dir = dir
      end

      def install(into)
        FileUtils.mkdir_p(into)
        files.each { FileUtils.cp(File.join(@dir, it), File.join(into, loaded_as(it))) }
      end

      def files = Dir.children(@dir).select { library?(it) }.sort

      def loaded_as(file) = file.sub(LINUX_VERSION, '\1').sub(MACOS_VERSION, '\1\2')

      private

      def library?(file) = file.match?(LIBRARY) && !File.symlink?(File.join(@dir, file))
    end
  end
end
