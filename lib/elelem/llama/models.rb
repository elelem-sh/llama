# frozen_string_literal: true

require_relative "model"

module Elelem
  module Llama
    module Models
      ROOMY_CONVERSATION = 3 * (1024**3)

      ALL = [
        {
          name: "qwen3-4b",
          repository: "unsloth/Qwen3-4B-GGUF",
          commit: "22c9fc8a8c7700b76a1789366280a6a5a1ad1120",
          file: "Qwen3-4B-UD-Q4_K_XL.gguf",
          bytes: 2_546_341_152,
          sha256: "f6e3fb6c2cdc869d16e66c719e94f2c02095d195967230e759a2d77fe814c71f"
        },
        {
          name: "qwen3.5-4b",
          repository: "unsloth/Qwen3.5-4B-GGUF",
          commit: "e87f176479d0855a907a41277aca2f8ee7a09523",
          file: "Qwen3.5-4B-UD-Q4_K_XL.gguf",
          bytes: 2_912_109_728,
          sha256: "b252c5610a42ca82d20fe2a12813e9d069eed89292907e26c783eeb0bc961bc7"
        },
        {
          name: "ornith-1.5-9b",
          repository: "ornith-ai/Ornith-1.5-9B-GGUF",
          commit: "abdd624b12ebf020b767fff532ff44fe552b28c3",
          file: "Ornith-1.5-9B-Q4_K_M.gguf",
          bytes: 5_780_090_816,
          sha256: "70c112196e0b7023803c9762752e46d29e612a92c83f995bc3ba1ceb07e8fab6"
        },
        {
          name: "qwen3.5-9b",
          repository: "unsloth/Qwen3.5-9B-GGUF",
          commit: "3885219b6810b007914f3a7950a8d1b469d598a5",
          file: "Qwen3.5-9B-UD-Q4_K_XL.gguf",
          bytes: 5_966_095_584,
          sha256: "6f5d30666c2d8ae16a306e616d95341dcf3cc46810df84d7e6f5a7d1e4c1b293"
        },
        {
          name: "ornith-1.5-35b-a3b",
          repository: "ornith-ai/Ornith-1.5-35B-A3B-GGUF",
          commit: "12393612fd4f730ff5aadc23e9b8f9648aa49ceb",
          file: "Ornith-1.5-35B-Q4_K_M.gguf",
          bytes: 21_713_463_040,
          sha256: "42739874cc2ccfdb8523b23fbe52e29b2a7555c8176737ca9ca0b5d59859d41f"
        },
        {
          name: "qwen3.6-35b-a3b",
          repository: "unsloth/Qwen3.6-35B-A3B-GGUF",
          commit: "a483e9e6cbd595906af30beda3187c2663a1118c",
          file: "Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf",
          bytes: 22_360_456_160,
          sha256: "707a55a8a4397ecde44de0c499d3e68c1ad1d240d1da65826b4949d1043f4450"
        },
        {
          name: "qwen3-coder-next",
          repository: "unsloth/Qwen3-Coder-Next-GGUF",
          commit: "ce09c67b53bc8739eef83fe67b2f5d293c270632",
          file: "Qwen3-Coder-Next-UD-Q4_K_XL.gguf",
          bytes: 49_608_478_720,
          sha256: "4bb93f0a0221ef4ff963ca9094df629c8dfdfabc3b4fdd85c1a2e4c0624fce36"
        }
      ].map { Model.new(reserve: ROOMY_CONVERSATION, **it) }.freeze

      def self.named(name)
        ALL.find { it.name == name } or raise ArgumentError, "no model named #{name}; pick from #{names.join(", ")}"
      end

      def self.names = ALL.map(&:name)
    end
  end
end
