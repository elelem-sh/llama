# frozen_string_literal: true

module Elelem
  module Llama
    module ModelResolver
      def self.resolve(model:, env: ENV, models_dir: "~/.agents/models")
        model || named(env) || default_model(models_dir)
      end

      def self.named(env)
        env["ELELEM_GGUF_MODEL"].then { it unless it.to_s.b.strip.empty? }
      end
      private_class_method :named

      def self.default_model(models_dir)
        dir = File.expand_path(models_dir)
        link = File.join(dir, "default.gguf")
        return link if File.symlink?(link) || File.exist?(link)

        models = Dir[File.join(dir, "*.gguf")]
        case models.length
        when 1 then models.first
        when 0 then raise "no .gguf model found in #{dir}. Set ELELEM_GGUF_MODEL=/path/to/model.gguf"
        else raise "multiple .gguf models found in #{dir}. Set ELELEM_GGUF_MODEL=/path/to/model.gguf or symlink " \
                   "#{link} to the one to use by default"
        end
      end
    end
  end
end
