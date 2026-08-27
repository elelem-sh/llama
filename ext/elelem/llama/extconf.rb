# frozen_string_literal: true

# Builds the vendored llama.cpp (CPU shared libs) and our thin C shim at gem
# install time, installing the shared objects into lib/elelem/native/ so the
# :gguf provider can Fiddle.dlopen them and run a model in-process. llama.cpp is
# a CMake project (not an mkmf extension), so we drive cmake ourselves and then
# emit a stub Makefile to satisfy RubyGems' `make && make install` step.
require "fileutils"

EXT_DIR   = __dir__
GEM_ROOT  = File.expand_path("../..", EXT_DIR)
VENDOR    = File.join(GEM_ROOT, "vendor", "llama.cpp")
BUILD_DIR = File.join(EXT_DIR, "build")
NATIVE    = File.join(GEM_ROOT, "lib", "elelem", "native")

# GPU offload is opt-in: ELELEM_GGML_VULKAN=ON builds the Vulkan backend (needs
# vulkan-headers, vulkan-loader-devel, and glslc). CPU-only stays the default so a
# bare `gem install` never requires GPU toolchain. Toggling this flips a cached
# CMake var, so switching modes wants a clean build dir (rm -rf ext/*/build).
VULKAN = %w[1 on true yes].include?(ENV["ELELEM_GGML_VULKAN"].to_s.strip.downcase)

def run(*cmd)
  warn "elelem: + #{cmd.join(' ')}"
  system(*cmd) || abort("elelem: build step failed: #{cmd.join(' ')}")
end

unless File.exist?(File.join(VENDOR, "CMakeLists.txt"))
  abort "elelem: vendored llama.cpp missing at #{VENDOR}\n" \
        "        run: git submodule update --init --recursive"
end

# 1. Configure + build only libllama (CPU). Examples/tests/tools/server off keeps
#    the build fast; GPU backends come later.
run("cmake", "-S", VENDOR, "-B", BUILD_DIR,
    "-DCMAKE_BUILD_TYPE=Release",
    "-DBUILD_SHARED_LIBS=ON",
    # Bake $ORIGIN rpath into the libs so, once co-located in lib/elelem/native,
    # libllama finds its ggml siblings there -- not in this throwaway build dir.
    "-DCMAKE_BUILD_WITH_INSTALL_RPATH=ON",
    "-DCMAKE_INSTALL_RPATH=$ORIGIN",
    "-DGGML_VULKAN=#{VULKAN ? 'ON' : 'OFF'}",
    "-DLLAMA_CURL=OFF",
    "-DLLAMA_BUILD_COMMON=ON", # common_chat: chat templates + tool-call parsing
    "-DLLAMA_BUILD_TESTS=OFF",
    "-DLLAMA_BUILD_EXAMPLES=OFF",
    "-DLLAMA_BUILD_TOOLS=OFF",
    "-DLLAMA_BUILD_SERVER=OFF")
run("cmake", "--build", BUILD_DIR, "--target", "llama", "llama-common", "--parallel")

# 2. Copy the shared objects (preserving SONAME symlinks) next to where the shim
#    will live, so the shim's $ORIGIN rpath resolves libllama/libggml at runtime.
FileUtils.mkdir_p(NATIVE)
libdir = File.join(BUILD_DIR, "bin")
run("sh", "-c", "cp -a #{libdir}/*.so* #{NATIVE}/")

# 3. Compile the shim against the vendored headers, link libllama, rpath $ORIGIN.
cxx = ENV["CXX"] || "c++"
run(cxx, "-std=c++17", "-O2", "-shared", "-fPIC",
    "-I", File.join(VENDOR, "include"),
    "-I", File.join(VENDOR, "ggml", "include"),
    "-I", File.join(VENDOR, "common"),   # chat.h (common_chat)
    "-I", File.join(VENDOR, "vendor"),   # nlohmann/json.hpp
    File.join(EXT_DIR, "elelem_llama.cpp"),
    "-o", File.join(NATIVE, "libelelem_llama.so"),
    "-L", libdir, "-lllama-common", "-lllama",
    "-Wl,-rpath,$ORIGIN")

# 4. Stub Makefile so `make` / `make install` succeed (work is already done).
File.write(File.join(EXT_DIR, "Makefile"), <<~MAKE)
  all:
  clean:
  install:
  .PHONY: all clean install
MAKE

warn "elelem: built llama.cpp + shim into #{NATIVE} (#{VULKAN ? 'Vulkan/GPU' : 'CPU'})"
