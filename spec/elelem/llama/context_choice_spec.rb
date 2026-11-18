# frozen_string_literal: true

require_relative "../../../lib/elelem/llama/context_choice"

RSpec.describe Elelem::Llama::ContextChoice do
  def choice(on_gpu:, affordable:, n_ctx_train: 262_144) = described_class.new(n_ctx_train:, on_gpu:, affordable:)

  it "takes the trained context when the whole model fits on the GPU with it" do
    fits = choice(on_gpu: 262_144, affordable: 262_144)

    expect([fits.n_ctx, fits.to_s]).to eq([262_144, "the whole model fits on the GPU"])
  end

  it "keeps the whole model on the GPU when a roomy context still fits there" do
    fast = choice(on_gpu: 40_960, affordable: 131_072)

    expect([fast.n_ctx, fast.to_s]).to eq(
      [40_960, "the whole model stays on the GPU for speed, though 131072 tokens would fit with RAM"]
    )
  end

  it "counts exactly 32768 tokens as roomy" do
    expect(choice(on_gpu: 32_768, affordable: 131_072).n_ctx).to eq(32_768)
  end

  it "shares the model with RAM when the GPU alone leaves a cramped context" do
    shared = choice(on_gpu: 32_512, affordable: 131_072)

    expect([shared.n_ctx, shared.to_s]).to eq(
      [131_072, "part of the model is in RAM, because only 32512 tokens would fit with all of it on the GPU"]
    )
  end

  it "shares the model with RAM when it does not fit on the GPU at all" do
    shared = choice(on_gpu: 0, affordable: 65_536)

    expect([shared.n_ctx, shared.to_s]).to eq(
      [65_536, "part of the model is in RAM, because it does not fit on the GPU"]
    )
  end

  it "treats the trained context as roomy for a model trained on less than 32768 tokens" do
    expect(choice(n_ctx_train: 8192, on_gpu: 8192, affordable: 8192).n_ctx).to eq(8192)
  end

  it "shares a small-context model with RAM when the GPU cannot hold its trained context" do
    expect(choice(n_ctx_train: 8192, on_gpu: 4096, affordable: 8192).n_ctx).to eq(8192)
  end
end
