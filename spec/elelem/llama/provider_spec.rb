# frozen_string_literal: true

require "elelem"
require "fiddle"
require "json"
require_relative "../../../lib/elelem/llama"

RSpec.describe Elelem::Llama::Provider do
  let(:opened) { [] }

  before do
    opens = opened
    open = ->(*args) { opens << args and Fiddle::Pointer.new(opens.size) }
    allow(described_class).to receive_messages(functions: { open: open }, close_at_exit: nil)
    described_class.instance_variable_set(:@shared, nil)
    described_class.instance_variable_set(:@gpu_layers, nil)
  end

  after do
    described_class.instance_variable_set(:@shared, nil)
    described_class.instance_variable_set(:@gpu_layers, nil)
  end

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

  it "decides the GPU layers once per model, so less free memory later does not load a copy" do
    answers = %w[999 0]
    first = described_class.gpu_layers_for("/m/a.gguf") { answers.shift }
    second = described_class.gpu_layers_for("/m/a.gguf") { answers.shift }

    expect([first, second]).to eq(%w[999 999])
  end

  it "builds -p llama once on a GPU machine whose free memory drops after the first load" do
    Dir.mktmpdir do |dir|
      model = File.join(dir, "m.gguf")
      File.write(model, "x" * 100)
      free = [1000, 100]
      allow(described_class).to receive_messages(backend: "cuda")
      allow(described_class).to receive(:gpu_free_memory) { free.shift }

      first = Elelem::Config.build_provider("llama", model: model)
      second = Elelem::Config.build_provider("llama", model: model)

      expect(second).to be(first)
      expect(opened.map { |args| args[1] }).to eq([999])
    end
  end
end
