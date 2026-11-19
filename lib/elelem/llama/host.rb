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
    end
  end
end
