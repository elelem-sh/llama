# frozen_string_literal: true

require "tmpdir"
require_relative "../../../ext/elelem/llama/compiler"

RSpec.describe Elelem::Llama::Compiler do
  around do |example|
    Dir.mktmpdir do |dir|
      @command = File.join(dir, "c++")
      File.write(@command, "#!/bin/sh\necho \"/toolchain/${1#-print-file-name=}\"\n")
      File.chmod(0o755, @command)
      example.run
    end
  end

  it "asks the compiler where a library is" do
    expect(described_class.new(@command).file_name("libstdc++.so.6")).to eq("/toolchain/libstdc++.so.6")
  end

  it "hands the name over as it is, with no shell in between" do
    expect(described_class.new(@command).file_name("lib; echo no")).to eq("/toolchain/lib; echo no")
  end

  it "reads as the command it runs" do
    expect(described_class.new("c++").to_s).to eq("c++")
  end
end
