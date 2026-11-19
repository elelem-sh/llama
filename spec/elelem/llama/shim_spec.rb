# frozen_string_literal: true

require_relative "../../../lib/elelem/llama/shim"

RSpec.describe Elelem::Llama::Shim do
  let(:opened) { [] }
  let(:dlopen) { ->(path) { opened << path and path } }

  def shim_on(host_os) = described_class.new(dir: "/gem/native", host_os:)

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
    shim_on("linux").open(dlopen:)

    expect(opened).to eq(["/gem/native/libelelem_llama.so"])
  end

  it "opens every library it needs by full path, each after the ones it needs, on Windows" do
    shim_on("mingw32").open(dlopen:)

    expect(opened).to eq(%w[
                           /gem/native/ggml-base.dll
                           /gem/native/ggml.dll
                           /gem/native/libllama.dll
                           /gem/native/libllama-common.dll
                           /gem/native/elelem_llama.dll
                         ])
  end

  it "answers with the opened shim" do
    expect(shim_on("mingw32").open(dlopen:)).to eq("/gem/native/elelem_llama.dll")
  end
end
