# frozen_string_literal: true

module Elelem
  module Llama
    class Progress
      def initialize(output)
        @output = output
        @shown = nil
      end

      def call(done, total = nil)
        return @output.puts("elelem: #{done}") unless total

        percent = done * 100 / total
        percent -= percent % 10 unless @output.tty?
        show(percent) unless percent == @shown
      end

      private

      def show(percent)
        @shown = percent
        return @output.puts("elelem: #{percent}%") unless @output.tty?

        @output.print("\relelem: #{percent}%#{"\n" if percent == 100}")
      end
    end
  end
end
