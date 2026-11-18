# frozen_string_literal: true

module Elelem
  module Llama
    class HostMemory
      GIB = 1024**3
      UNKNOWN = -1

      def self.detect(host_os = RbConfig::CONFIG["host_os"])
        case host_os
        when /linux/ then Linux.new
        when /darwin/ then Darwin.new
        else new
        end
      end

      def available = nil

      def budget
        free = available or return UNKNOWN

        [free - reserve(free), 0].max
      end

      private

      def reserve(free) = [free / 8, GIB].max

      class Linux < HostMemory
        CGROUPS = "/sys/fs/cgroup"

        def initialize(read: File.method(:read))
          super()
          @read = read
        end

        def available = [system_wide, *cgroups.filter_map { headroom(it) }].compact.min

        private

        def system_wide = kilobytes(read("/proc/meminfo"), "MemAvailable")

        def cgroups
          path = read("/proc/self/cgroup").to_s[/^0::(.*)$/, 1] or return []
          names = path.split("/").reject(&:empty?)

          (0..names.size).map { File.join(CGROUPS, *names.first(it)) }
        end

        def headroom(cgroup)
          limit = Integer(read(File.join(cgroup, "memory.max")), exception: false) or return
          used = Integer(read(File.join(cgroup, "memory.current")), exception: false) or return

          limit - used + reclaimable(read(File.join(cgroup, "memory.stat")))
        end

        def reclaimable(stat) = %w[inactive_file active_file].sum { stat.to_s[/^#{it} (\d+)/, 1].to_i }

        def kilobytes(text, field) = text.to_s[/^#{field}:\s+(\d+) kB/, 1]&.then { Integer(it) * 1024 }

        def read(path)
          @read.call(path)
        rescue SystemCallError
          nil
        end
      end

      class Darwin < HostMemory
        RECLAIMABLE = ["Pages free", "Pages inactive", "Pages speculative", "Pages purgeable"].freeze

        def initialize(run: ->(command) { `#{command}` })
          super()
          @run = run
        end

        def available
          report = @run.call("vm_stat")
          page = report[/page size of (\d+) bytes/, 1] or return

          Integer(page) * RECLAIMABLE.sum { report[/^#{it}:\s+(\d+)/, 1].to_i }
        rescue SystemCallError
          nil
        end
      end
    end
  end
end
