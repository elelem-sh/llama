# frozen_string_literal: true

module Elelem
  module Llama
    class SearchPath
      PROGRAM = ->(path) { File.executable?(path) && !File.directory?(path) }

      def initialize(env: ENV, program: PROGRAM)
        @env = env
        @program = program
      end

      def include?(command) = candidates(command).any?(&@program)

      private

      def candidates(command) = directories.product(names(command)).map { File.join(*it) }

      def names(command) = suffixes.map { "#{command}#{it}" }

      def directories = @env["PATH"].to_s.split(File::PATH_SEPARATOR)

      def suffixes = ["", *@env["PATHEXT"].to_s.split(";")]
    end
  end
end
