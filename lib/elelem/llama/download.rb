# frozen_string_literal: true

require "digest"
require "fileutils"
require "net/http"
require "openssl"
require_relative "free_space"

module Elelem
  module Llama
    class Download
      ATTEMPTS = 4
      INTERRUPTIONS = [SocketError, Timeout::Error, EOFError, OpenSSL::SSL::SSLError, Errno::ECONNRESET,
                       Errno::ECONNREFUSED, Errno::EPIPE, Errno::ETIMEDOUT, Errno::EHOSTUNREACH,
                       Errno::ENETUNREACH].freeze

      def initialize(hub:, dir:, free_space: FreeSpace.new, progress: ->(*) {})
        @hub = hub
        @dir = dir
        @free_space = free_space
        @progress = progress
      end

      def call(model)
        FileUtils.mkdir_p(@dir)
        path = File.join(@dir, model.file)
        return existing(path, model) if taken?(path)

        part = "#{path}.#{model.sha256[0, 12]}.part"
        File.open(part, File::RDWR | File::CREAT | File::BINARY) { fill(it, model) }
        verify(part, model)
        publish(part, path, model)
      end

      private

      def taken?(path) = File.exist?(path) || File.symlink?(path)

      def publish(part, path, model)
        return File.rename(part, path) && path unless taken?(path)

        existing(path, model).tap { File.delete(part) }
      end

      def existing(path, model)
        return path if File.file?(path) && File.size(path) == model.bytes && sha256(path) == model.sha256

        raise "#{path} already exists and is a different file; move it, then pull again"
      end

      def fill(part, model)
        raise "#{part.path} is being downloaded by another process" unless part.flock(File::LOCK_EX | File::LOCK_NB)

        part.truncate(0) if part.size > model.bytes
        ensure_room(model.bytes - part.size)
        ATTEMPTS.times { part.size < model.bytes && resume(part, model) }
        raise "the download of #{model.file} stopped early; pull again to continue" if part.size < model.bytes
      end

      def ensure_room(missing)
        free = @free_space.in(@dir) or return
        return if free >= missing * 11 / 10

        raise "not enough disk space in #{@dir}: #{missing * 11 / 10 / (10**6)} MB needed, #{free / (10**6)} MB free"
      end

      def resume(part, model)
        offset = part.size
        headers = offset.zero? ? {} : { "Range" => "bytes=#{offset}-" }
        @hub.open(model, headers) { receive(it, part, offset, model) }
      rescue *INTERRUPTIONS => e
        @progress.call("interrupted: #{e.message}")
      end

      def receive(response, part, offset, model)
        start_at(part, resumed_at(response, offset))
        response.read_body do |chunk|
          raise "#{model.file} is larger than expected" if part.pos + chunk.bytesize > model.bytes

          part.write(chunk)
          @progress.call(part.pos, model.bytes)
        end
        part.flush
      end

      def resumed_at(response, offset)
        return 0 if response.is_a?(Net::HTTPOK)
        return offset if response.is_a?(Net::HTTPPartialContent) && response.content_range&.first == offset

        raise "the server answered HTTP #{response.code}"
      end

      def start_at(part, offset)
        part.truncate(offset)
        part.seek(offset)
      end

      def verify(part, model)
        return if sha256(part) == model.sha256

        File.delete(part)
        raise "checksum mismatch for #{model.file}; the download was deleted, pull again"
      end

      def sha256(path) = Digest::SHA256.file(path).hexdigest
    end
  end
end
