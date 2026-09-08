# frozen_string_literal: true

require_relative "lib/elelem/llama/version"

Gem::Specification.new do |spec|
  spec.name = "elelem-llama"
  spec.version = Elelem::Llama::VERSION
  spec.authors = ["mo khan"]
  spec.email = ["mo@mokhan.ca"]

  spec.summary = "A llama.cpp plugin for elelem."
  spec.description = "A llama.cpp plugin for elelem."
  spec.homepage = "https://src.mokhan.ca/elelem/llama"
  spec.license = "MIT"
  spec.required_ruby_version = ">= 4.0.0"
  spec.metadata["allowed_push_host"] = "https://rubygems.org"
  spec.metadata["homepage_uri"] = spec.homepage
  spec.metadata["source_code_uri"] = spec.homepage

  spec.files = [
    "LICENSE.txt",
    "lib/elelem/llama.rb",
    "lib/elelem/llama/version.rb",
    "lib/elelem/llama/provider.rb",
    "exe/elelem-llama",
    "ext/elelem/llama/elelem.cpp",
    "ext/elelem/llama/extconf.rb",
  ]
  spec.files += Dir["vendor/llama.cpp/**/*"].select { |path| File.file?(path) }
  spec.extensions = ["ext/elelem/llama/extconf.rb"]
  spec.bindir = "exe"
  spec.executables = ["elelem-llama"]
  spec.require_paths = ["lib"]

  spec.add_dependency "elelem", "~> 0.11"
  spec.add_dependency "fiddle", "~> 1.1"
  spec.add_dependency "json", "~> 3.0"
end
