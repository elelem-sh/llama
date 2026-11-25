# frozen_string_literal: true

require "optparse"
require_relative "default_model"
require_relative "download"
require_relative "host_memory"
require_relative "hub"
require_relative "model_reference"
require_relative "pinned_models"
require_relative "progress"

module Elelem
  module Llama
    class Pull
      USAGE = "Usage: elelem-llama --pull [owner/repo[:quant]] [--default]"

      def initialize(args, dir: File.expand_path("~/.agents/models"), hub: Hub.new, memory: HostMemory.detect,
                     output: $stderr)
        @args = OptionParser.new(USAGE) { |o| describe_options(o) }.parse(args)
        @dir = dir
        @hub = hub
        @memory = memory
        @output = output
        @default = DefaultModel.new(dir:)
      end

      def run
        raise ArgumentError, USAGE if @args.size > 1

        @output.puts "elelem: pulling #{model}"
        path = Download.new(hub: @hub, dir: @dir, progress: Progress.new(@output)).call(model)
        choose(path)
      end

      private

      def describe_options(parser)
        parser.on("--default", "Use this model by default") { @chosen = true }
      end

      def model = @model ||= @args.empty? ? PinnedModels.for(@memory) : @hub.find(ModelReference.parse(@args.first))

      def choose(path)
        @default.point_at(path) if (@chosen || @default.missing?) && !@default.is?(path)
        return @output.puts("elelem: #{path} is the default model") if @default.is?(path)

        @output.puts "elelem: saved #{path}; use it with --model #{path}, or pull again with --default"
      end
    end
  end
end
