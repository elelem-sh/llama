# frozen_string_literal: true

module Elelem
  module Llama
    Model = Data.define(:name, :repository, :commit, :file, :bytes, :sha256, :reserve) do
      def url = "https://huggingface.co/#{repository}/resolve/#{commit}/#{file}"

      def budget = bytes + reserve

      def fits?(memory) = budget <= memory

      def to_s = { name:, url:, file:, bytes:, sha256: }.map { |key, value| "#{key}=#{value}" }.join("\n")
    end
  end
end
