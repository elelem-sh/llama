# frozen_string_literal: true

require "tmpdir"
require_relative "../../../ext/elelem/llama/cxx_runtime"

RSpec.describe Elelem::Llama::CxxRuntime do
  around do |example|
    Dir.mktmpdir do |root|
      @toolchain = File.join(root, "toolchain")
      @installed = File.join(root, "installed")
      FileUtils.mkdir_p(@toolchain)
      example.run
    end
  end

  def runtime = described_class.new(compiler: "c++", locate: ->(library) { File.join(@toolchain, library) })

  def ship(library, as:)
    File.write(File.join(@toolchain, as), "the bytes of #{library}")
    File.symlink(as, File.join(@toolchain, library))
  end

  it "installs each library the compiler points to under the name the loader asks for" do
    ship("libstdc++.so.6", as: "libstdc++.so.6.0.34")
    ship("libgcc_s.so.1", as: "libgcc_s.so.1.0.0")

    runtime.install(@installed)

    expect(Dir.children(@installed).sort).to eq(["libgcc_s.so.1", "libstdc++.so.6"])
    expect(File.symlink?(File.join(@installed, "libstdc++.so.6"))).to be(false)
    expect(File.read(File.join(@installed, "libstdc++.so.6"))).to eq("the bytes of libstdc++.so.6")
  end

  it "refuses a compiler that does not know where its runtime is" do
    expect { runtime.install(@installed) }.to raise_error(RuntimeError, "c++ does not know where libstdc++.so.6 is")
  end
end
