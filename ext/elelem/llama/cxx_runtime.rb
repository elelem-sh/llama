# frozen_string_literal: true

require "fileutils"

module Elelem
  module Llama
    class CxxRuntime
      LIBRARIES = %w[libstdc++.so.6 libgcc_s.so.1].freeze

      def initialize(compiler:, locate: ->(library) { `#{compiler} -print-file-name=#{library}`.strip })
        @compiler = compiler
        @locate = locate
      end

      def install(into)
        FileUtils.mkdir_p(into)
        LIBRARIES.each { FileUtils.cp(found(it), File.join(into, it)) }
      end

      private

      def found(library)
        path = @locate.call(library)
        return path if File.file?(path)

        raise "#{@compiler} does not know where #{library} is"
      end
    end
  end
end
