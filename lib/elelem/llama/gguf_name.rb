# frozen_string_literal: true

module Elelem
  module Llama
    class GgufName
      PLAIN = /\A[A-Za-z0-9][\w.+-]*\.gguf\z/i
      RESERVED = /\A(default\.gguf\z|(con|prn|aux|nul|com\d|lpt\d)\.)/i
      SHARD = /-\d{5}-of-\d{5}(?=\.gguf\z)/i

      def initialize(path)
        @path = path.to_s
      end

      def model? = @path.match?(/\.gguf\z/i) && !File.basename(@path).match?(/mmproj/i)

      def split? = @path.match?(SHARD)

      def safe? = @path.match?(PLAIN) && !@path.match?(RESERVED)

      def quant?(quant) = @path.sub(SHARD, "").match?(/(\A|[-._])#{Regexp.escape(quant)}\.gguf\z/i)

      def to_s = @path
    end
  end
end
