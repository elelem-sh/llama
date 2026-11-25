# frozen_string_literal: true

module Elelem
  module Llama
    ModelReference = Data.define(:repository, :quant) do
      def self.parse(text)
        name = /[A-Za-z0-9][\w.-]*/
        match = %r{\A(?<repository>#{name}/#{name})(:(?<quant>[A-Za-z0-9][A-Za-z0-9_-]*))?\z}.match(text)
        raise ArgumentError, "#{text.inspect} is not owner/repo or owner/repo:quant" unless match

        new(repository: match[:repository], quant: match[:quant] || "Q4_K_M")
      end

      def to_s = "#{repository}:#{quant}"
    end
  end
end
