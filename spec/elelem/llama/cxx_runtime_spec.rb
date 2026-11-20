# frozen_string_literal: true

require "tmpdir"
require_relative "../../../ext/elelem/llama/cxx_runtime"

RSpec.describe Elelem::Llama::CxxRuntime do
  toolchain = Struct.new(:dir) do
    def file_name(library) = File.join(dir, library)

    def to_s = "c++"
  end

  lost_toolchain = Struct.new(:dir) do
    def file_name(library) = library

    def to_s = "c++"
  end

  renames = Struct.new(:named, :needing) do
    def name(library, soname) = named[File.basename(library)] = soname

    def need(library, renamed) = needing[File.basename(library)] = renamed
  end

  around do |example|
    Dir.mktmpdir do |root|
      @toolchain = File.join(root, "toolchain")
      @installed = File.join(root, "installed")
      @patchelf = renames.new({}, {})
      FileUtils.mkdir_p([@toolchain, @installed])
      example.run
    end
  end

  def runtime(compiler = @compiler) = described_class.new(compiler:, patchelf: @patchelf)

  def ship(library, as:)
    File.write(File.join(@toolchain, as), "the bytes of #{library}")
    File.symlink(as, File.join(@toolchain, library))
  end

  def ship_the_runtime
    ship("libstdc++.so.6", as: "libstdc++.so.6.0.33")
    ship("libgcc_s.so.1", as: "libgcc_s.so.1.0.0")
  end

  before { @compiler = toolchain.new(@toolchain) }

  it "installs each library the compiler points to under a name of its own" do
    ship_the_runtime

    runtime.install(@installed)

    expect(Dir.children(@installed).sort).to eq(["libgcc_s-elelem.so.1", "libstdc++-elelem.so.6"])
    expect(File.symlink?(File.join(@installed, "libstdc++-elelem.so.6"))).to be(false)
    expect(File.read(File.join(@installed, "libstdc++-elelem.so.6"))).to eq("the bytes of libstdc++.so.6")
  end

  it "names each library after its file, so that it never answers for the one of the system" do
    ship_the_runtime

    runtime.install(@installed)

    expect(@patchelf.named).to eq(
      "libstdc++-elelem.so.6" => "libstdc++-elelem.so.6", "libgcc_s-elelem.so.1" => "libgcc_s-elelem.so.1"
    )
  end

  it "points every library beside it, and the runtime itself, at the new names" do
    ship_the_runtime
    File.write(File.join(@installed, "libllama.so.1"), "")

    runtime.install(@installed)

    expect(@patchelf.needing.keys).to eq(["libgcc_s-elelem.so.1", "libllama.so.1", "libstdc++-elelem.so.6"])
    expect(@patchelf.needing.values.uniq).to eq([described_class::OWN_NAMES])
  end

  it "refuses a compiler that does not know where its runtime is" do
    expect { runtime.install(@installed) }.to raise_error(RuntimeError, "c++ does not know where libstdc++.so.6 is")
  end

  it "refuses a compiler that answers with the bare name of a library it could not find" do
    Dir.chdir(@toolchain) do
      ship_the_runtime

      expect { runtime(lost_toolchain.new(@toolchain)).install(@installed) }
        .to raise_error(RuntimeError, "c++ does not know where libstdc++.so.6 is")
    end
  end
end
