# frozen_string_literal: true

require_relative "../../../lib/elelem/llama/settings"

RSpec.describe Elelem::Llama::Settings do
  gib = 1024**3
  igpu = { name: "Vulkan0", description: "Intel(R) Iris(R) Xe Graphics", bytes: gib / 2 }

  def settings(**changes)
    described_class.new(n_ctx: 40_960, n_ctx_train: 262_144, kv_cache: "q8_0", threads: 6, layers: 37, gpu_layers: 0,
                        overflowing_layers: 0, devices: [], host_bytes: 5 * (1024**3), host_budget: 18 * (1024**3),
                        **changes)
  end

  it "explains a CPU-only load" do
    expect(settings.to_s).to eq(
      "context of 40960 tokens (the model allows 262144), q8_0 KV cache, 6 threads; " \
      "all weights on the CPU; 5.0 GiB of RAM used from a budget of 18.0 GiB"
    )
  end

  it "says that a GPU without layers still helps with long prompts" do
    expect(settings(devices: [igpu]).to_s).to include(
      "weights on the CPU, Intel(R) Iris(R) Xe Graphics speeds up reading long prompts",
      "budget of 18.0 GiB, 0.5 GiB on Vulkan0"
    )
  end

  it "counts the layers on the GPU" do
    gpu = { name: "CUDA0", description: "RTX 4090", bytes: 20 * gib }

    expect(settings(devices: [gpu], gpu_layers: 30).to_s).to include("30 of 37 layers on RTX 4090")
  end

  it "mentions expert weights left in RAM" do
    gpu = { name: "CUDA0", description: "", bytes: 10 * gib }

    expect(settings(devices: [gpu], gpu_layers: 37, overflowing_layers: 12).to_s)
      .to include("37 of 37 layers on CUDA0; 12 layers keep their expert weights in RAM")
  end
end
