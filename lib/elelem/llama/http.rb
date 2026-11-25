# frozen_string_literal: true

require "net/http"
require "uri"

module Elelem
  module Llama
    class Http
      HEADERS = { "User-Agent" => "elelem-llama", "Accept-Encoding" => "identity" }.freeze
      REDIRECTS = 5

      def initialize(schemes: ["https"])
        @schemes = schemes
      end

      def get(uri, headers = {}, redirects: REDIRECTS, &)
        raise "refusing to fetch over #{uri.scheme} from #{uri.host}" unless @schemes.include?(uri.scheme)

        location = request(uri, HEADERS.merge(headers), &) or return
        raise "too many redirects from #{uri.host}" if redirects.zero?

        get(uri + location, headers, redirects: redirects - 1, &)
      end

      private

      def request(uri, headers)
        location = nil
        Net::HTTP.start(uri.hostname, uri.port, *proxy(uri), use_ssl: uri.scheme == "https", open_timeout: 30) do |http|
          http.request(Net::HTTP::Get.new(uri, headers)) do |response|
            location = response["location"] if response.is_a?(Net::HTTPRedirection)
            yield response unless location
          end
        end
        location
      end

      def proxy(uri)
        proxy = uri.find_proxy or return []
        raise "the proxy for #{uri.host} is not an http:// proxy" unless proxy.scheme == "http"

        [proxy.hostname, proxy.port, *[proxy.user, proxy.password].map { it && URI.decode_uri_component(it) }]
      rescue URI::InvalidURIError
        raise "the proxy for #{uri.host} is not a URL"
      end
    end
  end
end
