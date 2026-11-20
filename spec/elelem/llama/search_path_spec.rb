# frozen_string_literal: true

require "tmpdir"
require_relative "../../../ext/elelem/llama/search_path"

RSpec.describe Elelem::Llama::SearchPath do
  around do |example|
    Dir.mktmpdir do |dir|
      @dir = dir
      example.run
    end
  end

  def program(name) = File.join(@dir, name).tap { File.write(it, "") }.then { File.chmod(0o755, it) }

  def search_path(**env) = described_class.new(env: { "PATH" => ["/nowhere", @dir].join(File::PATH_SEPARATOR), **env })

  it "finds a program in any of the directories" do
    program("glslc")
    expect(search_path.include?("glslc")).to be(true)
  end

  it "does not find what is not there" do
    expect(search_path.include?("glslc")).to be(false)
  end

  it "does not take a directory for a program" do
    Dir.mkdir(File.join(@dir, "glslc"))
    expect(search_path.include?("glslc")).to be(false)
  end

  it "finds a Windows program by any suffix PATHEXT names" do
    program("nvcc.EXE")
    expect(search_path("PATHEXT" => ".COM;.EXE;.BAT").include?("nvcc")).to be(true)
  end

  it "does not guess a suffix PATHEXT does not name" do
    program("nvcc.EXE")
    expect([search_path.include?("nvcc"), search_path("PATHEXT" => ".COM;.BAT").include?("nvcc")]).to eq([false, false])
  end

  it "finds nothing where there is no PATH" do
    expect(described_class.new(env: {}).include?("glslc")).to be(false)
  end

  it "asks the reader it is given whether a path is a program" do
    asked = []
    described_class.new(env: { "PATH" => "bin", "PATHEXT" => ".EXE" }, program: ->(path) { asked.push(path) && false })
                   .include?("nvcc")
    expect(asked).to eq(["bin/nvcc", "bin/nvcc.EXE"])
  end
end
