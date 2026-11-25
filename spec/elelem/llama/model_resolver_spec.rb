# frozen_string_literal: true

RSpec.describe Elelem::Llama::ModelResolver do
  around do |example|
    Dir.mktmpdir { |dir| @models_dir = dir and example.run }
  end

  def resolve(model: nil, env: {})
    described_class.resolve(model:, env:, models_dir: @models_dir)
  end

  it "prefers an explicit model" do
    expect(resolve(model: "/explicit.gguf", env: { "ELELEM_GGUF_MODEL" => "/env.gguf" })).to eq("/explicit.gguf")
  end

  it "falls back to ELELEM_GGUF_MODEL when no explicit model is given" do
    expect(resolve(env: { "ELELEM_GGUF_MODEL" => "/env.gguf" })).to eq("/env.gguf")
  end

  it "ignores a blank ELELEM_GGUF_MODEL" do
    FileUtils.touch(File.join(@models_dir, "only.gguf"))

    expect(resolve(env: { "ELELEM_GGUF_MODEL" => " " })).to eq(File.join(@models_dir, "only.gguf"))
  end

  context "with no explicit model or ENV" do
    it "picks the only .gguf file in the models dir" do
      FileUtils.touch(File.join(@models_dir, "only.gguf"))

      expect(resolve).to eq(File.join(@models_dir, "only.gguf"))
    end

    it "prefers a default.gguf symlink when multiple models exist" do
      FileUtils.touch(File.join(@models_dir, "a.gguf"))
      FileUtils.touch(File.join(@models_dir, "b.gguf"))
      File.symlink("a.gguf", File.join(@models_dir, "default.gguf"))

      expect(resolve).to eq(File.join(@models_dir, "default.gguf"))
    end

    it "raises when no .gguf model is found" do
      expect { resolve }.to raise_error(/no \.gguf model found/)
    end

    it "raises when multiple .gguf models exist with no default" do
      FileUtils.touch(File.join(@models_dir, "a.gguf"))
      FileUtils.touch(File.join(@models_dir, "b.gguf"))

      expect { resolve }.to raise_error(/multiple \.gguf models found/)
    end
  end
end
