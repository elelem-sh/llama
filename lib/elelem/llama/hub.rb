# frozen_string_literal: true

require "json"
require_relative "gguf_name"
require_relative "http"
require_relative "remote_model"

module Elelem
  module Llama
    class Hub
      ORIGIN = URI("https://huggingface.co")

      def initialize(http: Http.new, origin: ORIGIN)
        @http = http
        @origin = origin
      end

      def find(reference)
        listing = list(reference.repository)
        sibling = only(reference, models(listing))

        RemoteModel.new(repository: reference.repository, commit: listing["sha"], file: sibling["rfilename"],
                        bytes: sibling["size"], sha256: sibling.dig("lfs", "sha256")).verified
      end

      def open(model, headers, &) = @http.get(@origin + model.path, headers, &)

      private

      def list(repository)
        body = nil
        @http.get(@origin + "/api/models/#{repository}/revision/main?blobs=true") do |response|
          raise "#{repository}: #{explain(response)}" unless response.is_a?(Net::HTTPOK)

          body = JSON.parse(response.read_body)
        end
        body
      end

      def explain(response)
        return "not found, private, or it needs a login" if %w[401 403 404].include?(response.code)

        "#{@origin.host} answered HTTP #{response.code}"
      end

      def models(listing) = listing.fetch("siblings").select { name(it).model? }

      def only(reference, siblings)
        found = siblings.select { name(it).quant?(reference.quant) }
        raise "#{reference.repository} has no #{reference.quant} model; it has #{list_of(siblings)}" if found.empty?

        whole = found.reject { name(it).split? }
        raise "#{reference} matches #{list_of(whole)}; name one of them more exactly" if whole.size > 1

        whole.first || found.first
      end

      def name(sibling) = GgufName.new(sibling["rfilename"])

      def list_of(siblings) = siblings.empty? ? "no .gguf file" : siblings.map { name(it) }.join(", ")
    end
  end
end
