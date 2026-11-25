# frozen_string_literal: true

require "open3"

RSpec.describe "elelem/llama" do
  def names_after(setup)
    script = "#{setup}; require 'elelem/llama'; print Elelem::Config.names.join(' ')"
    Open3.capture2(RbConfig.ruby, "-I", File.expand_path("../../../lib", __dir__), "-r", "elelem", "-e", script).first
  end

  it "registers itself as llama and as the default provider" do
    expect(names_after("")).to eq("llama default")
  end

  it "replaces an earlier default" do
    script = "Elelem.configure { |config| config.provider(:default) { abort 'replaced nothing' } }; " \
             "ENV['ELELEM_GGUF_N_CTX'] = 'not a number'; ENV['ELELEM_GGUF_MODEL'] = '/x'; " \
             "require 'elelem/llama'; Elelem::Config.build_provider(:default)"
    lib = File.expand_path("../../../lib", __dir__)
    command = [RbConfig.ruby, "-I", lib, "-r", "elelem", "-e", script]
    _, error, = Dir.mktmpdir { |home| Open3.capture3({ "HOME" => home }, *command, chdir: home) }

    expect(error).to include("not a number")
  end
end
