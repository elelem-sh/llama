# frozen_string_literal: true

require_relative "remote_model"

module Elelem
  module Llama
    module PinnedModels
      ROOMY = 12 * (1024**3)

      SMALL = RemoteModel.new(
        repository: "unsloth/Qwen3.5-2B-GGUF",
        commit: "f6d5376be1edb4d416d56da11e5397a961aca8ae",
        file: "Qwen3.5-2B-Q4_K_M.gguf",
        bytes: 1_280_835_840,
        sha256: "aaf42c8b7c3cab2bf3d69c355048d4a0ee9973d48f16c731c0520ee914699223"
      )

      LARGE = RemoteModel.new(
        repository: "unsloth/Qwen3.5-4B-GGUF",
        commit: "e87f176479d0855a907a41277aca2f8ee7a09523",
        file: "Qwen3.5-4B-Q4_K_M.gguf",
        bytes: 2_740_937_888,
        sha256: "00fe7986ff5f6b463e62455821146049db6f9313603938a70800d1fb69ef11a4"
      )

      def self.for(memory) = memory.total.to_i < ROOMY ? SMALL : LARGE
    end
  end
end
