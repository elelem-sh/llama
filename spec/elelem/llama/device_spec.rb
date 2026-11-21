# frozen_string_literal: true

require_relative "../../../lib/elelem/llama/device"

RSpec.describe Elelem::Llama::Device do
  gib = 1024**3

  def device(type: "gpu", dedicated: true, free: 0, total: 0)
    described_class.new(name: "Vulkan0", description: "a graphics card", type:, dedicated:, free:, total:)
  end

  it "keeps a gibibyte of its free memory for everything else on the device" do
    expect(device(free: 8 * gib).room).to eq(7 * gib)
  end

  it "has no room when less than the margin is free" do
    expect(device(free: gib / 2).room).to eq(0)
  end

  it "knows the processor from the graphics cards" do
    expect([device(type: "cpu", dedicated: false), device].map(&:cpu?)).to eq([true, false])
  end

  it "is dedicated when the shim says its memory is its own" do
    expect([device, device(type: "igpu", dedicated: false)].map(&:dedicated?)).to eq([true, false])
  end
end
