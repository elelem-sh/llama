# frozen_string_literal: true

require "fiddle"
require_relative "../../../lib/elelem/llama/load_library"

RSpec.describe Elelem::Llama::LoadLibrary do
  let(:asked) { [] }

  def answering(handle) = described_class.new(load_library_w: ->(wide_path) { asked << wide_path and handle })

  it "asks Windows for the path in UTF-16 with backslashes and a terminator" do
    answering(Fiddle::Pointer.new(1)).call("C:/gé/native/ggml.dll")

    expect(asked).to eq(["C:\\gé\\native\\ggml.dll\0".encode(Encoding::UTF_16LE)])
  end

  it "fails when Windows cannot load the library" do
    expect { answering(Fiddle::NULL).call("C:/gem/native/ggml.dll") }
      .to raise_error(Fiddle::DLError, "Windows cannot load C:/gem/native/ggml.dll")
  end
end
