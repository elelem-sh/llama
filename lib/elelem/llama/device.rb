# frozen_string_literal: true

module Elelem
  module Llama
    DEVICE_MARGIN = 1024**3

    Device = Data.define(:name, :description, :type, :dedicated, :free, :total) do
      def cpu? = type == "cpu"

      def dedicated? = dedicated

      def room = [free - DEVICE_MARGIN, 0].max
    end
  end
end
