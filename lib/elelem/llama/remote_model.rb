# frozen_string_literal: true

require_relative "gguf_name"

module Elelem
  module Llama
    RemoteModel = Data.define(:repository, :commit, :file, :bytes, :sha256) do
      def verified
        raise "#{name} is one part of a split model, which is not supported" if name.split?
        raise "refusing the file name #{name}" unless name.safe?
        raise "#{repository} does not say the size and SHA-256 of #{file}" unless described?

        self
      end

      def path = "/#{repository}/resolve/#{commit}/#{file}"

      def to_s = format("%<file>s (%<size>.1f GB)", file:, size: bytes.fdiv(10**9))

      private

      def name = GgufName.new(file)

      def described? = sha?(commit, 40) && sha?(sha256, 64) && bytes.is_a?(Integer) && bytes.positive?

      def sha?(text, length) = text.to_s.match?(/\A\h{#{length}}\z/)
    end
  end
end
