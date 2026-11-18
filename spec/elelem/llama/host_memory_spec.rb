# frozen_string_literal: true

require_relative "../../../lib/elelem/llama/host_memory"

RSpec.describe Elelem::Llama::HostMemory do
  gib = 1024**3

  def reading(files)
    ->(path) { files.fetch(path) { raise Errno::ENOENT, path } }
  end

  it "picks the reader for the operating system" do
    expect([described_class.detect("linux-gnu"), described_class.detect("darwin24"), described_class.detect("mingw32")])
      .to match([described_class::Linux, described_class::Darwin, be_instance_of(described_class)])
  end

  it "leaves the budget to the shim where available memory is unknown" do
    expect(described_class.new.budget).to eq(described_class::UNKNOWN)
  end

  describe described_class::Linux do
    let(:meminfo) { "MemTotal:       32512992 kB\nMemFree:  1024 kB\nMemAvailable:   16777216 kB\n" }

    it "keeps an eighth of the available memory for other programs" do
      memory = described_class.new(read: reading("/proc/meminfo" => meminfo))

      expect(memory.budget).to eq(14 * gib)
    end

    it "keeps at least one gibibyte for other programs" do
      memory = described_class.new(read: reading("/proc/meminfo" => "MemAvailable: 4194304 kB\n"))

      expect(memory.budget).to eq(3 * gib)
    end

    it "has nothing to give when less than the reserve is available" do
      memory = described_class.new(read: reading("/proc/meminfo" => "MemAvailable: 524288 kB\n"))

      expect(memory.budget).to eq(0)
    end

    it "respects the memory limit of its control group" do
      memory = described_class.new(read: reading(
        "/proc/meminfo" => meminfo,
        "/proc/self/cgroup" => "0::/user.slice/run-1.scope\n",
        "/sys/fs/cgroup/user.slice/memory.max" => "max\n",
        "/sys/fs/cgroup/user.slice/run-1.scope/memory.max" => "#{6 * gib}\n",
        "/sys/fs/cgroup/user.slice/run-1.scope/memory.current" => "#{gib}\n"
      ))

      expect(memory.available).to eq(5 * gib)
    end

    it "respects a limit set on a parent control group" do
      memory = described_class.new(read: reading(
        "/proc/meminfo" => meminfo,
        "/proc/self/cgroup" => "0::/pod/app\n",
        "/sys/fs/cgroup/pod/memory.max" => "#{4 * gib}\n",
        "/sys/fs/cgroup/pod/memory.current" => "#{gib}\n",
        "/sys/fs/cgroup/pod/app/memory.max" => "max\n"
      ))

      expect(memory.available).to eq(3 * gib)
    end

    it "sees the limit of a container, whose control group is the root" do
      memory = described_class.new(read: reading(
        "/proc/meminfo" => meminfo,
        "/proc/self/cgroup" => "0::/\n",
        "/sys/fs/cgroup/memory.max" => "#{6 * gib}\n",
        "/sys/fs/cgroup/memory.current" => "#{2 * gib}\n"
      ))

      expect(memory.available).to eq(4 * gib)
    end

    it "counts cached files in the control group as available, like MemAvailable does" do
      memory = described_class.new(read: reading(
        "/proc/meminfo" => meminfo,
        "/proc/self/cgroup" => "0::/\n",
        "/sys/fs/cgroup/memory.max" => "#{6 * gib}\n",
        "/sys/fs/cgroup/memory.current" => "#{5 * gib}\n",
        "/sys/fs/cgroup/memory.stat" => "anon 1\ninactive_file #{2 * gib}\nactive_file #{gib}\nshmem 9\n"
      ))

      expect(memory.available).to eq(4 * gib)
    end

    it "uses the system figure when it is the smaller one" do
      memory = described_class.new(read: reading(
        "/proc/meminfo" => "MemAvailable: 2097152 kB\n",
        "/proc/self/cgroup" => "0::/\n",
        "/sys/fs/cgroup/memory.max" => "#{6 * gib}\n",
        "/sys/fs/cgroup/memory.current" => "0\n"
      ))

      expect(memory.available).to eq(2 * gib)
    end

    it "does not know when /proc is missing" do
      expect(described_class.new(read: reading({})).available).to be_nil
    end
  end

  describe described_class::Darwin do
    let(:vm_stat) do
      <<~REPORT
        Mach Virtual Memory Statistics: (page size of 16384 bytes)
        Pages free:                               10000.
        Pages active:                            500000.
        Pages inactive:                           20000.
        Pages speculative:                         3000.
        Pages wired down:                        100000.
        Pages purgeable:                            500.
      REPORT
    end

    it "counts free, inactive, speculative and purgeable pages as available" do
      memory = described_class.new(run: ->(command) { command == "vm_stat" ? vm_stat : "" })

      expect(memory.available).to eq(33_500 * 16_384)
    end

    it "does not know when vm_stat is missing" do
      expect(described_class.new(run: ->(_) { raise Errno::ENOENT }).available).to be_nil
    end

    it "does not know when vm_stat prints something else" do
      expect(described_class.new(run: ->(_) { "" }).available).to be_nil
    end
  end
end
