# frozen_string_literal: true

require "tmpdir"
require_relative "../../../ext/elelem/llama/shared_libraries"

RSpec.describe Elelem::Llama::SharedLibraries do
  around do |example|
    Dir.mktmpdir do |root|
      @built = File.join(root, "built")
      @installed = File.join(root, "installed")
      FileUtils.mkdir_p(@built)
      example.run
    end
  end

  def build(file, links: [])
    File.write(File.join(@built, file), file)
    links.each { File.symlink(file, File.join(@built, it)) }
  end

  def installed
    described_class.new(@built).install(@installed)
    Dir.children(@installed).sort
  end

  it "installs a versioned library once, under the name the loader asks for" do
    build("libllama.so.1.0.0", links: ["libllama.so.1", "libllama.so"])

    expect(installed).to eq(["libllama.so.1"])
    expect(File.symlink?(File.join(@installed, "libllama.so.1"))).to be(false)
    expect(File.read(File.join(@installed, "libllama.so.1"))).to eq("libllama.so.1.0.0")
  end

  it "keeps the name of a backend module" do
    build("libggml-cpu-haswell.so")
    build("libggml-cpu-armv8.2_1.so")

    expect(installed).to eq(["libggml-cpu-armv8.2_1.so", "libggml-cpu-haswell.so"])
  end

  it "installs a versioned macOS library under its install name" do
    build("libggml-base.1.0.0.dylib", links: ["libggml-base.1.dylib", "libggml-base.dylib"])

    expect(installed).to eq(["libggml-base.1.dylib"])
  end

  it "keeps the name of a Windows library" do
    build("ggml-base.dll")

    expect(installed).to eq(["ggml-base.dll"])
  end

  it "leaves out everything that is not a shared library" do
    build("libllama.so.1.0.0")
    build("llama-bench")
    build("libcommon.a")

    expect(installed).to eq(["libllama.so.1"])
  end
end
