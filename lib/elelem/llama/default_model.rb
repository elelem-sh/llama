# frozen_string_literal: true

require "fileutils"

module Elelem
  module Llama
    class DefaultModel
      SYMLINK = File.method(:symlink)

      def initialize(dir:, symlink: SYMLINK)
        @dir = dir
        @link = File.join(dir, "default.gguf")
        @symlink = symlink
      end

      def is?(model) = File.exist?(@link) && File.identical?(@link, model)

      def missing? = !File.exist?(@link) && !File.symlink?(@link)

      def sole?(model) = missing? && others == [model]

      def point_at(model)
        raise "#{@link} is a model and not a link; move it, then pull again" unless missing? || link?

        fresh = "#{@link}.#{Process.pid}"
        link(model, fresh)
        File.rename(fresh, @link)
      ensure
        FileUtils.rm_f(fresh) if fresh
      end

      private

      def link? = File.symlink?(@link) || others.any? { File.identical?(it, @link) }

      def others = Dir.children(@dir).grep(/\.gguf\z/i).map { File.join(@dir, it) } - [@link]

      def link(model, fresh)
        @symlink.call(File.basename(model), fresh)
      rescue Errno::EACCES, Errno::EPERM, NotImplementedError
        File.link(model, fresh)
      end
    end
  end
end
