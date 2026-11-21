# frozen_string_literal: true

require_relative "host_memory"
require_relative "models"

module Elelem
  module Llama
    class ModelChoice
      def self.for_this_machine(memory: HostMemory.detect, devices: Provider.devices, **)
        new(budget: memory.budget, devices:, **)
      end

      def initialize(budget:, devices:, models: Models::ALL, at_most: Float::INFINITY)
        @budget = budget
        @devices = devices
        @models = models
        @at_most = at_most
      end

      def model = downloadable.select { it.fits?(memory) }.max_by(&:bytes) || smallest

      def memory = host + @devices.select(&:dedicated?).sum(&:room)

      def to_s = model.to_s

      private

      def downloadable = @models.select { it.bytes <= @at_most }

      def smallest = @models.min_by(&:bytes)

      def host = host_known? ? @budget : half_the_installed_memory

      def host_known? = @budget != HostMemory::UNKNOWN

      def half_the_installed_memory = @devices.select(&:cpu?).sum(&:total) / 2
    end
  end
end
