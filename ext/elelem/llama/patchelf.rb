# frozen_string_literal: true

module Elelem
  module Llama
    class Patchelf
      def initialize(command: "patchelf")
        @command = command
      end

      def name(library, soname) = edit(library, "--set-soname", soname)

      def need(library, renamed) = edit(library, *renamed.flat_map { |from, to| ["--replace-needed", from, to] })

      private

      def edit(library, *changes) = system(@command, *changes, library, exception: true)
    end
  end
end
