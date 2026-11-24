# frozen_string_literal: true

require "elelem"
require "fiddle"
require "json"
require_relative "../../../lib/elelem/llama"

RSpec.describe Elelem::Llama::Provider do
  let(:opened) { [] }

  before do
    opens = opened
    described_class.functions = {
      open: ->(*args) { opens << args and Fiddle::Pointer.new(opens.size) },
      close: ->(_handle) {}
    }
    described_class.instance_variable_set(:@shared, nil)
    @stderr = $stderr
    $stderr = StringIO.new
  end

  after do
    $stderr = @stderr
    described_class.functions = nil
    described_class.instance_variable_set(:@shared, nil)
  end

  def generating(result, pieces: [], stops: [])
    keep = []
    lambda do |_handle, _messages, _tools, _max_tokens, on_token, _userdata|
      callback = Fiddle::Function.new(on_token, [Fiddle::TYPE_VOIDP, Fiddle::TYPE_VOIDP], Fiddle::TYPE_INT)
      pieces.each { |piece| stops << callback.call(nil, piece) }
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

  describe ".start" do
    def shim_answering(answer)
      started = Fiddle::Closure::BlockCaller.new(Fiddle::TYPE_VOIDP, [Fiddle::TYPE_VOIDP]) { |_modules| answer }
      { "el_start" => started }
    end

    it "starts when the shim loaded a backend from the gem" do
      expect { described_class.start(shim_answering(Fiddle::NULL), modules: "/gem/native") }.not_to raise_error
    end

    it "fails with what the shim reports instead of looking for a backend elsewhere" do
      refusal = Fiddle::Pointer["no CPU backend loaded from /gem/native"]

      expect { described_class.start(shim_answering(refusal), modules: "/gem/native") }
        .to raise_error(RuntimeError, "llama: no CPU backend loaded from /gem/native")
    end
  end

  describe ".backend" do
    def backend_named(name)
      described_class.functions[:backend] = -> { Fiddle::Pointer[name] }
      described_class.backend
    end

    it "names the backend of the best device in lower case" do
      expect(backend_named("Vulkan")).to eq("vulkan")
      expect(backend_named("CPU")).to eq("cpu")
    end

    it "names Metal and ROCm the way ELELEM_LLAMA_BACKEND does" do
      expect(backend_named("MTL")).to eq("metal")
      expect(backend_named("ROCm")).to eq("hip")
    end
  end

  describe ".devices" do
    it "lists what the shim found on this machine" do
      found = [{ name: "Vulkan0", description: "Radeon RX 7600", type: "gpu", dedicated: true,
                 free: 8_000_000_000, total: 8_500_000_000 }]
      described_class.functions[:devices] = -> { Fiddle::Pointer[JSON.generate(found)] }

      expect(described_class.devices).to eq([Elelem::Llama::Device.new(**found.first)])
    end
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

  it "does not send the heartbeats between pieces as text" do
    result = { "reasoning" => "Greet the user.", "content" => "Hi!", "tool_calls" => [] }
    stops = []

    events = events_for(generating(result, pieces: ["", "Greet ", "", "the user."], stops: stops))

    expect(events.map { it[:text] }).to eq(["Greet ", "the user.", "Hi!"])
    expect(stops).to eq([0, 0, 0, 0])
  end

  it "tells the shim to stop once the block raises, and raises that after generating" do
    stops = []
    described_class.functions[:generate] = generating({ "cancelled" => true }, pieces: ["Greet ", "", "the user."],
                                                                               stops: stops)
    seen = []

    expect do
      described_class.new(model: "a.gguf").fetch([]) { |event| seen << event and raise Interrupt }
    end.to raise_error(Interrupt)
    expect(stops).to eq([1, 1, 1])
    expect(seen).to eq([{ type: "thinking", text: "Greet " }])
  end

  it "generates again after a reply that was stopped" do
    provider = described_class.new(model: "a.gguf")
    described_class.functions[:generate] = generating({ "cancelled" => true }, pieces: ["Greet "])
    expect { provider.fetch([]) { raise "the terminal is gone" } }.to raise_error("the terminal is gone")

    described_class.functions[:generate] = generating({ "content" => "Hi!", "tool_calls" => [] }, pieces: [""])
    events = []
    provider.fetch([]) { |event| events << event }

    expect(events).to eq([{ type: "saying", text: "Hi!" }])
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

  it "passes a context or layers chosen by hand to the shim" do
    described_class.new(model: "a.gguf", n_ctx: 8192)
    described_class.new(model: "a.gguf", n_gpu_layers: 20)

    expect(opened.map { it[1..2] }).to eq([[-1, 8192], [20, 0]])
  end

  it "says which model it loads and how big it is before loading it" do
    Dir.mktmpdir do |dir|
      model = File.join(dir, "m.gguf")
      File.open(model, "w") { it.truncate(2_700_000_000) }
      err = StringIO.new

      described_class.new(model:, err:)

      expect(err.string).to eq("loading m.gguf (2.7 GB)…\n")
    end
  end

  it "names the model file it loaded" do
    expect(described_class.new(model: "/models/qwen.gguf").model).to eq("qwen.gguf")
  end

  it "names a model whose path came back as bytes, as file names do under LANG=C" do
    err = StringIO.new
    provider = described_class.new(model: "/models/\u6a21.gguf".b, err:)

    expect(provider.model).to eq("\u6a21.gguf")
    expect(err.string).to eq("loading \u6a21.gguf…\n")
  end

  it "names the file a default.gguf link points to" do
    Dir.mktmpdir do |dir|
      File.write(File.join(dir, "qwen.gguf"), "")
      File.symlink("qwen.gguf", File.join(dir, "default.gguf"))

      expect(described_class.new(model: File.join(dir, "default.gguf")).model).to eq("qwen.gguf")
    end
  end

  it "says which model it loads even when it cannot see the file" do
    err = StringIO.new
    described_class.new(model: "a.gguf", err:)

    expect(err.string).to eq("loading a.gguf…\n")
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
