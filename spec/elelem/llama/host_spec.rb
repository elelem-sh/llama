# frozen_string_literal: true

require_relative "../../../lib/elelem/llama/host"

RSpec.describe Elelem::Llama::Host do
  def host(system, processor = "x86_64") = described_class.new(system:, processor:)

  it "knows Windows by either of the names Ruby gives it" do
    expect([host("mingw32"), host("mswin64"), host("linux-gnu")].map(&:windows?)).to eq([true, true, false])
  end

  it "knows macOS and Linux" do
    expect([host("darwin25").darwin?, host("linux-gnu").linux?, host("linux-gnu").darwin?]).to eq([true, true, false])
  end

  it "knows musl apart from the GNU C library" do
    expect([host("linux-musl"), host("linux-gnu"), host("linux")].map(&:musl?)).to eq([true, false, false])
  end

  it "knows Apple silicon as macOS on an ARM processor" do
    expect([host("darwin25", "arm64"), host("darwin25"), host("linux-gnu", "aarch64")].map(&:apple_silicon?))
      .to eq([true, false, false])
  end

  it "builds from source everywhere but on Windows" do
    expect([host("linux-gnu", "aarch64"), host("linux-musl"), host("darwin25", "arm64")].map(&:builds_from_source?))
      .to eq([true, true, true])
  end

  it "builds from source on Windows only with MinGW on x64" do
    expect([host("mingw32"), host("mingw32", "aarch64"), host("mswin64_140", "x64")].map(&:builds_from_source?))
      .to eq([true, false, false])
  end
end
