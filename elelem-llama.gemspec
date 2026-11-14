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
  spec.metadata["rubygems_mfa_required"] = "true"

  spec.files = [
    "LICENSE.txt",
    "lib/elelem/llama.rb",
    "lib/elelem/llama/version.rb",
    "lib/elelem/llama/model_resolver.rb",
    "lib/elelem/llama/provider.rb",
    "exe/elelem-llama",
    "ext/elelem/llama/elelem.cpp",
    "ext/elelem/llama/extconf.rb"
  ]
  # Only what extconf.rb's cmake build (LLAMA_BUILD_COMMON=ON, tests/docs/
  # examples/tools/server all OFF) actually needs to configure and compile.
  # Anchored to the vendor root so this doesn't also strip required nested
  # dirs with the same name, e.g. src/models/ (per-architecture model code).
  vendor_exclude = %r{\Avendor/llama\.cpp/(tests|docs|models|examples|tools|\.github|gguf-py|scripts)/}
  spec.files += Dir["vendor/llama.cpp/**/*"].select do |path|
    File.file?(path) && path !~ vendor_exclude
  end
  spec.extensions = ["ext/elelem/llama/extconf.rb"]
  spec.bindir = "exe"
  spec.executables = ["elelem-llama"]
  spec.require_paths = ["lib"]

  spec.add_dependency "elelem", ">= 0.13", "< 1"
  spec.add_dependency "fiddle", "~> 1.1"
  spec.add_dependency "json", "~> 3.0"
end
