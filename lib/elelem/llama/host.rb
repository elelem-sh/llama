# frozen_string_literal: true

module Elelem
  module Llama
    class Host
      def initialize(system: RbConfig::CONFIG["host_os"], processor: RbConfig::CONFIG["host_cpu"])
        @system = system
        @processor = processor
      end

      def windows? = @system.match?(/mingw|mswin/)

      def darwin? = @system.match?(/darwin/)

      def linux? = @system.match?(/linux/)

      def musl? = @system.match?(/musl/)

      def apple_silicon? = darwin? && @processor.match?(/arm64|aarch64/)

      def builds_from_source? = !windows? || mingw_on_x64?

      private

      def mingw_on_x64? = @system.match?(/mingw/) && @processor.match?(/x86_64|x64/)
    end
  end
end
