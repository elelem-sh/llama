# frozen_string_literal: true

require_relative "../../../lib/elelem/llama/shim"

RSpec.describe Elelem::Llama::Shim do
  let(:opened) { [] }
  let(:loaded) { [] }
  let(:dlopen) { ->(library) { opened << library and "handle of #{library}" } }
  let(:load_library) { ->(path) { loaded << path } }

  def shim_on(system) = described_class.new(dir: "/gem/native", host: Elelem::Llama::Host.new(system:))

  it "is a shared object on Linux" do
    expect(shim_on("linux").path).to eq("/gem/native/libelelem_llama.so")
  end

  it "is a dynamic library on macOS" do
    expect(shim_on("darwin25").path).to eq("/gem/native/libelelem_llama.dylib")
  end

  it "is a DLL without the lib prefix on Windows" do
    expect(shim_on("mingw32").path).to eq("/gem/native/elelem_llama.dll")
  end

  it "opens only itself where the loader finds its libraries beside it" do
    expect(shim_on("linux").open(dlopen:, load_library:)).to eq("handle of /gem/native/libelelem_llama.so")
    expect(loaded).to be_empty
  end

  it "loads every library by full path, each after the ones it needs, on Windows" do
    shim_on("mingw32").open(dlopen:, load_library:)

    expect(loaded).to eq(%w[
                           /gem/native/ggml-base.dll
                           /gem/native/ggml.dll
                           /gem/native/libllama.dll
                           /gem/native/libllama-common.dll
                           /gem/native/elelem_llama.dll
                         ])
  end

  it "opens the loaded shim by name on Windows, where a path outside the ANSI code page cannot be opened" do
    expect(shim_on("mingw32").open(dlopen:, load_library:)).to eq("handle of elelem_llama.dll")
    expect(opened).to eq(["elelem_llama.dll"])
  end
end
