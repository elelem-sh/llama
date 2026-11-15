# frozen_string_literal: true

require "elelem"
require "fiddle"
require "json"
require_relative "../../../lib/elelem/llama/provider"

RSpec.describe Elelem::Llama::Provider do
  let(:opened) { [] }

  before do
    opens = opened
    open = ->(*args) { opens << args and Fiddle::Pointer.new(opens.size) }
    allow(described_class).to receive_messages(functions: { open: open }, close_at_exit: nil)
    described_class.instance_variable_set(:@shared, nil)
  end

  after { described_class.instance_variable_set(:@shared, nil) }

  it "loads a model once for the same options" do
    first = described_class.shared(model: "a.gguf", n_gpu_layers: 0)
    second = described_class.shared(model: "a.gguf", n_gpu_layers: 0)

    expect(second).to be(first)
    expect(opened.size).to eq(1)
  end

  it "loads again for different options" do
    described_class.shared(model: "a.gguf")
    described_class.shared(model: "b.gguf")

    expect(opened.map(&:first)).to eq(["a.gguf", "b.gguf"])
  end
end
