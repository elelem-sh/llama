# frozen_string_literal: true

require_relative "host"

module Elelem
  module Llama
    class HostMemory
      GIB = 1024**3
      UNKNOWN = -1

      def self.detect(host = Host.new)
        return Linux.new if host.linux?
        return Darwin.new if host.darwin?
        return Windows.new if host.windows?

        new
      end

      def available = nil

      def total = nil

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

        def total = kilobytes(read("/proc/meminfo"), "MemTotal")

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

        def total
          Integer(@run.call("sysctl -n hw.memsize"), exception: false)
        rescue SystemCallError
          nil
        end
      end

      class Windows < HostMemory
        STATUS_SIZE = 64
        TOTAL_PHYSICAL = 8
        AVAILABLE_PHYSICAL = 16

        KERNEL32 = lambda do
          status = Fiddle::Pointer.malloc(STATUS_SIZE, Fiddle::RUBY_FREE)
          status[0, 4] = [STATUS_SIZE].pack("L")
          filled = Fiddle::Function.new(
            Fiddle.dlopen("kernel32")["GlobalMemoryStatusEx"], [Fiddle::TYPE_VOIDP], Fiddle::TYPE_INT
          ).call(status)

          status[0, STATUS_SIZE] unless filled.zero?
        end

        def initialize(global_memory_status: KERNEL32)
          super()
          @global_memory_status = global_memory_status
        end

        def available = physical(AVAILABLE_PHYSICAL)

        def total = physical(TOTAL_PHYSICAL)

        private

        def physical(offset) = @global_memory_status.call&.unpack1("Q<", offset:)
      end
    end
  end
end
