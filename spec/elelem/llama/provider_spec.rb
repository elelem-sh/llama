# frozen_string_literal: true

require "elelem"
require "fiddle"
require "json"
require_relative "../../../lib/elelem/llama"

RSpec.describe Elelem::Llama::Provider do
  let(:opened) { [] }

  let(:surveys) { ["{}"] }

  before do
    opens = opened
    found = surveys
    described_class.functions = {
      open: ->(*args) { opens << args and Fiddle::Pointer.new(opens.size) },
      survey: ->(*) { found.first },
      close: ->(_handle) {}
    }
    described_class.instance_variable_set(:@shared, nil)
  end

  after do
    described_class.functions = nil
    described_class.instance_variable_set(:@shared, nil)
  end

  def generating(result, pieces: [])
    keep = []
    lambda do |_handle, _messages, _tools, _max_tokens, on_token, _userdata|
      callback = Fiddle::Function.new(on_token, [Fiddle::TYPE_VOIDP, Fiddle::TYPE_VOIDP], Fiddle::TYPE_VOID)
      pieces.each { |piece| callback.call(nil, piece) }
      keep << Fiddle::Pointer[JSON.generate(result)]
      keep.last
    end
  end

  def events_for(generate)
    described_class.functions[:generate] = generate
    events = []
    described_class.new(model: "a.gguf").fetch([{ role: "user", content: "say hi" }]) { |event| events << event }
    events
  end

  it "does not send the reasoning again after it was streamed" do
    result = { "reasoning" => "Greet the user.", "content" => "Hi!", "tool_calls" => [] }

    expect(events_for(generating(result, pieces: ["Greet ", "the user."]))).to eq([
                                                                                    { type: "thinking",
                                                                                      text: "Greet " },
                                                                                    { type: "thinking",
                                                                                      text: "the user." },
                                                                                    { type: "saying", text: "Hi!" }
                                                                                  ])
  end

  it "sends the reasoning from the result when none was streamed" do
    result = { "reasoning" => "Greet the user.", "content" => "Hi!", "tool_calls" => [] }

    expect(events_for(generating(result, pieces: [""]))).to eq([
                                                                 { type: "thinking", text: "Greet the user." },
                                                                 { type: "saying", text: "Hi!" }
                                                               ])
  end

  it "sends only the answer when the model does not reason" do
    result = { "reasoning" => "", "content" => "Hi!", "tool_calls" => [] }

    expect(events_for(generating(result))).to eq([{ type: "saying", text: "Hi!" }])
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

  it "lets the shim choose the GPU layers and the context unless told otherwise" do
    described_class.new(model: "a.gguf")

    expect(opened.first[1..2]).to eq([-1, 0])
  end

  it "opens with the context a dedicated GPU can hold alone when that is roomy" do
    surveys.replace([JSON.generate(n_ctx_train: 262_144, on_gpu: 40_960, affordable: 131_072)])
    described_class.new(model: "a.gguf")

    expect(opened.first[1..2]).to eq([-1, 40_960])
  end

  it "does not survey when the context or the layers were chosen by hand" do
    surveys.replace([JSON.generate(n_ctx_train: 262_144, on_gpu: 40_960, affordable: 131_072)])
    described_class.new(model: "a.gguf", n_ctx: 8192)
    described_class.new(model: "a.gguf", n_gpu_layers: 20)

    expect(opened.map { it[1..2] }).to eq([[-1, 8192], [20, 0]])
  end

  it "opens the model with the memory budget of the host" do
    described_class.new(model: "a.gguf", memory: Elelem::Llama::HostMemory::Linux.new(read: lambda { |path|
      path == "/proc/meminfo" ? "MemAvailable:   16777216 kB\n" : raise(Errno::ENOENT)
    }))

    expect(opened.first.last).to eq(14 * (1024**3))
  end

  it "builds -p llama once for the same model" do
    Dir.mktmpdir do |dir|
      model = File.join(dir, "m.gguf")
      File.write(model, "x" * 100)

      first = Elelem::Config.build_provider("llama", model: model)
      second = Elelem::Config.build_provider("llama", model: model)

      expect(second).to be(first)
      expect(opened.size).to eq(1)
    end
  end
end
