# frozen_string_literal: true

require "fiddle"
require "open3"
require_relative "host"

module Elelem
  module Llama
    class FreeSpace
      WINDOWS = lambda do |dir|
        free = Fiddle::Pointer.malloc(8, Fiddle::RUBY_FREE)
        path = "#{dir.tr("/", "\\")}\0".encode(Encoding::UTF_16LE)
        filled = Fiddle::Function.new(
          Fiddle.dlopen("kernel32")["GetDiskFreeSpaceExW"], [Fiddle::TYPE_VOIDP] * 4, Fiddle::TYPE_INT
        ).call(path, free, nil, nil)

        free[0, 8].unpack1("Q<") unless filled.zero?
      end

      DF = lambda do |dir|
        report, status = Open3.capture2("df", "-Pk", dir)
        available(report) if status.success?
      rescue SystemCallError
        nil
      end

      def self.available(report) = report.b.lines[1].to_s[/(\d+)\s+\d+%/, 1]&.then { Integer(it) * 1024 }

      def initialize(host: Host.new, windows: WINDOWS, unix: DF)
        @measure = host.windows? ? windows : unix
      end

      def in(dir) = @measure.call(dir)
    end
  end
end
