# frozen_string_literal: true

require_relative "../../../lib/elelem/llama/device"
require_relative "../../../lib/elelem/llama/model_choice"

RSpec.describe Elelem::Llama::ModelChoice do
  gib = 1024**3

  def model(name, gibibytes)
    Elelem::Llama::Model.new(name:, repository: "someone/#{name}", commit: "c0ffee", file: "#{name}.gguf",
                             bytes: gibibytes * (1024**3), sha256: "abc123", reserve: 3 * (1024**3))
  end

  def device(type:, dedicated:, free:, total: free)
    Elelem::Llama::Device.new(name: type, description: type, type:, dedicated:, free:, total:)
  end

  def chosen(budget:, devices: [], **)
    models = [model("medium", 6), model("small", 3), model("large", 21)]

    described_class.new(budget:, devices:, models:, **).model.name
  end

  it "picks the largest model whose weights and conversation fit the host budget" do
    expect([chosen(budget: 9 * gib), chosen(budget: 24 * gib)]).to eq(%w[medium large])
  end

  it "passes over a model that misses the budget by a byte" do
    expect(chosen(budget: (9 * gib) - 1)).to eq("small")
  end

  it "falls back on the smallest model when nothing fits" do
    expect(chosen(budget: 2 * gib)).to eq("small")
  end

  it "adds the room on a dedicated graphics card to the host budget" do
    card = device(type: "gpu", dedicated: true, free: 17 * gib)

    expect(chosen(budget: 8 * gib, devices: [card])).to eq("large")
  end

  it "does not count memory a graphics card shares with the host twice" do
    integrated = device(type: "igpu", dedicated: false, free: 17 * gib)
    unified = device(type: "gpu", dedicated: false, free: 17 * gib)

    expect(chosen(budget: 8 * gib, devices: [integrated, unified])).to eq("small")
  end

  it "takes half of the installed memory where the host cannot say what is available" do
    processor = device(type: "cpu", dedicated: false, free: 18 * gib, total: 18 * gib)

    expect(chosen(budget: Elelem::Llama::HostMemory::UNKNOWN, devices: [processor])).to eq("medium")
  end

  it "picks nothing larger than the download it was limited to" do
    expect(chosen(budget: 64 * gib, at_most: 6 * gib)).to eq("medium")
  end

  it "prints its model for a shell to read" do
    choice = described_class.new(budget: 64 * gib, devices: [], models: [model("small", 3)])

    expect(choice.to_s).to eq(model("small", 3).to_s)
  end

  it "reads the budget and the devices of this machine from what it is given" do
    memory = Elelem::Llama::HostMemory::Linux.new(read: lambda { |path|
      path == "/proc/meminfo" ? "MemAvailable:   16777216 kB\n" : raise(Errno::ENOENT)
    })

    expect(described_class.for_this_machine(memory:, devices: []).memory).to eq(14 * gib)
  end
end
