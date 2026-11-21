# frozen_string_literal: true

require_relative "lib/elelem/llama/version"

Gem::Specification.new do |spec|
  spec.name = "elelem-llama"
  spec.version = Elelem::Llama::VERSION
  spec.authors = ["mo khan"]
  spec.email = ["mo@mokhan.ca"]

  spec.summary = "A llama.cpp plugin for elelem."
  spec.description = "A llama.cpp plugin for elelem."
  spec.homepage = "https://elelem.sh"
  spec.licenses = ["MIT"]
  spec.required_ruby_version = ">= 4.0.0"
  spec.metadata["allowed_push_host"] = "https://rubygems.org"
  spec.metadata["homepage_uri"] = spec.homepage
  spec.metadata["source_code_uri"] = "https://github.com/elelem-sh/llama"
  spec.metadata["rubygems_mfa_required"] = "true"

  spec.files = [
    "LICENSE.txt",
    "lib/elelem/llama.rb",
    "lib/elelem/llama/version.rb",
    "lib/elelem/llama/context_choice.rb",
    "lib/elelem/llama/device.rb",
    "lib/elelem/llama/host.rb",
    "lib/elelem/llama/host_memory.rb",
    "lib/elelem/llama/load_library.rb",
    "lib/elelem/llama/model.rb",
    "lib/elelem/llama/model_choice.rb",
    "lib/elelem/llama/model_resolver.rb",
    "lib/elelem/llama/models.rb",
    "lib/elelem/llama/provider.rb",
    "lib/elelem/llama/settings.rb",
    "lib/elelem/llama/shim.rb",
    "exe/elelem-llama"
  ]
  linked_into_every_build = [
    "vendor/llama.cpp/LICENSE",
    "vendor/llama.cpp/licenses/LICENSE-jsonhpp",
    "vendor/llama.cpp/vendor/cpp-httplib/LICENSE",
    "vendor/notices/LICENSE-llamafile-sgemm",
    "vendor/notices/LICENSE-yarn"
  ]
  vulkan_headers = [
    "vendor/vulkan-headers/LICENSE.md",
    "vendor/vulkan-headers/LICENSES/Apache-2.0.txt",
    "vendor/vulkan-headers/LICENSES/MIT.txt"
  ]
  gcc_runtime = ["vendor/gcc-runtime/COPYING3", "vendor/gcc-runtime/COPYING.RUNTIME", "vendor/gcc-runtime/NOTICE"]
  winpthreads = ["vendor/winpthreads/COPYING"]
  linked_only_into = {
    "x86_64-linux-gnu" => vulkan_headers,
    "aarch64-linux-gnu" => vulkan_headers,
    "x86_64-linux-musl" => gcc_runtime,
    "x64-mingw-ucrt" => winpthreads
  }
  licensed_only_there = {
    "x86_64-linux-musl" => ["GPL-3.0-or-later WITH GCC-exception-3.1"],
    "x64-mingw-ucrt" => ["BSD-3-Clause"]
  }
  prebuilt_for = ENV.fetch("ELELEM_LLAMA_PLATFORM", "")
  if prebuilt_for.empty?
    spec.files += Dir["ext/elelem/llama/*.{cpp,rb}"]
    # Only what extconf.rb's cmake build (LLAMA_BUILD_COMMON=ON, tests/docs/
    # examples/tools/server all OFF) actually needs to configure and compile.
    # Anchored to the vendor root so this doesn't also strip required nested
    # dirs with the same name, e.g. src/models/ (per-architecture model code).
    vendor_exclude = %r{\Avendor/llama\.cpp/(tests|docs|models|examples|tools|\.github|gguf-py|scripts)/}
    spec.files += Dir["vendor/llama.cpp/**/*"].select do |path|
      File.file?(path) && path !~ vendor_exclude
    end
    spec.extensions = ["ext/elelem/llama/extconf.rb"]
  else
    spec.platform = prebuilt_for
    spec.files += Dir["ext/elelem/llama/native/*"]
    spec.files += linked_into_every_build + linked_only_into.fetch(prebuilt_for, [])
    spec.licenses += licensed_only_there.fetch(prebuilt_for, [])
  end
  spec.bindir = "exe"
  spec.executables = ["elelem-llama"]
  spec.require_paths = ["lib"]

  spec.add_dependency "elelem", ">= 0.13", "< 1"
  spec.add_dependency "fiddle", "~> 1.1"
  spec.add_dependency "json", ">= 2.18"
end
