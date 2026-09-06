# frozen_string_literal: true

require_relative "lib/elelem/llama/version"

Gem::Specification.new do |spec|
  spec.name = "elelem-llama"
  spec.version = Elelem::Llama::VERSION
  spec.authors = ["mo khan"]
  spec.email = ["mo@mokhan.ca"]

  spec.summary = "A llama.cpp plugin for elelem."
  spec.description = "A llama.cpp plugin for elelem."
  spec.homepage = "https://src.mokhan.ca/xlgmokha/elelem"
  spec.license = "MIT"
  spec.required_ruby_version = ">= 4.0.0"
  spec.required_rubygems_version = ">= 4.0.0"
  spec.metadata["allowed_push_host"] = "https://rubygems.org"
  spec.metadata["homepage_uri"] = spec.homepage
  spec.metadata["source_code_uri"] = "https://git.mokhan.ca/xlgmokha/elelem.git"

  spec.files = [
    "LICENSE.txt",
    "lib/elelem/llama.rb",
    "lib/elelem/llama/version.rb",
    "lib/elelem/llama/client.rb",
    "lib/elelem/llama/plugin.rb",
    "ext/elelem/llama/elelem.cpp",
    "ext/elelem/llama/extconf.rb",
  ]
  # Vendored llama.cpp source (git submodule) is compiled at install time by the
  # extension below, so it must ship in the gem.
  spec.files += Dir["vendor/llama.cpp/**/*"].select { |path| File.file?(path) }
  spec.extensions = ["ext/elelem/llama/extconf.rb"]
  spec.require_paths = ["lib"]

  spec.add_dependency "elelem", "~> 0.1"
  spec.add_dependency "fiddle", "~> 1.1"
  spec.add_dependency "json", "~> 2.21"
end
