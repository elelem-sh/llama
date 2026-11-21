# frozen_string_literal: true

require_relative "../../../lib/elelem/llama/models"

RSpec.describe Elelem::Llama::Models do
  gib = 1024**3

  it "names every model once" do
    expect(described_class.names).to eq(described_class.names.uniq)
  end

  it "pins every download to a commit and a checksum" do
    expect(described_class::ALL).to all(
      have_attributes(
        url: %r{\Ahttps://huggingface\.co/[\w.-]+/[\w.-]+/resolve/\h{40}/[\w.-]+\.gguf\z},
        sha256: /\A\h{64}\z/,
        bytes: be_positive
      )
    )
  end

  it "asks for the weights and room for a roomy conversation" do
    expect(described_class::ALL).to all(satisfy { it.budget == it.bytes + (3 * gib) })
  end

  it "still offers the model the installer has always downloaded" do
    expect(described_class.named("qwen3-4b")).to have_attributes(
      url: "https://huggingface.co/unsloth/Qwen3-4B-GGUF/resolve/22c9fc8a8c7700b76a1789366280a6a5a1ad1120/" \
           "Qwen3-4B-UD-Q4_K_XL.gguf",
      bytes: 2_546_341_152,
      sha256: "f6e3fb6c2cdc869d16e66c719e94f2c02095d195967230e759a2d77fe814c71f"
    )
  end

  it "refuses a name it does not know, listing the ones it does" do
    expect { described_class.named("gpt-2") }
      .to raise_error(ArgumentError, /\Ano model named gpt-2; pick from qwen3-4b, .*qwen3-coder-next\z/)
  end

  describe Elelem::Llama::Model do
    let(:model) do
      described_class.new(name: "tiny", repository: "someone/Tiny-GGUF", commit: "c0ffee", file: "tiny.gguf",
                          bytes: 4 * gib, sha256: "abc123", reserve: 2 * gib)
    end

    it "fits in exactly its budget and in nothing less" do
      expect([model.fits?(6 * gib), model.fits?((6 * gib) - 1)]).to eq([true, false])
    end

    it "prints one key=value per line for a shell to read" do
      expect(model.to_s.lines(chomp: true)).to eq(
        [
          "name=tiny",
          "url=https://huggingface.co/someone/Tiny-GGUF/resolve/c0ffee/tiny.gguf",
          "file=tiny.gguf",
          "bytes=#{4 * gib}",
          "sha256=abc123"
        ]
      )
    end
  end
end
