# frozen_string_literal: true

module Elelem
  module Llama
    class Compiler
      def initialize(command)
        @command = command
      end

      def file_name(library) = IO.popen([@command, "-print-file-name=#{library}"], &:read).strip

      def to_s = @command
    end
  end
end
