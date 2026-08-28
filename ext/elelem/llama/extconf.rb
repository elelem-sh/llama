# frozen_string_literal: true

# Builds the vendored llama.cpp (shared libs) and our thin C shim at gem install
# time, installing the shared objects into lib/elelem/native/ so the :gguf provider
# can Fiddle.dlopen them and run a model in-process. llama.cpp is a CMake project
# (not an mkmf extension), so we drive cmake ourselves and then emit a stub Makefile
# to satisfy RubyGems' `make && make install` step.
#
# The GPU backend is chosen automatically from the host's toolchain (see
# detect_backend): NVIDIA/CUDA, then Vulkan (covers AMD/Intel/NVIDIA), else CPU. A
# GPU build that fails to configure/compile falls back to CPU so a bare `gem install`
# never hard-fails. Override with ELELEM_LLAMA_BACKEND=auto|cpu|vulkan|cuda.
require "fileutils"
# NB: we deliberately do NOT `require "mkmf"` -- its at_exit hook aborts unless
# create_makefile ran, which conflicts with the stub Makefile we emit below. A
# small PATH probe (which) covers the toolchain detection we need.

EXT_DIR   = __dir__
GEM_ROOT  = File.expand_path("../..", EXT_DIR)
VENDOR    = File.join(GEM_ROOT, "vendor", "llama.cpp")
BUILD_DIR = File.join(EXT_DIR, "build")
NATIVE    = File.join(GEM_ROOT, "lib", "elelem", "native")
STAMP     = File.join(BUILD_DIR, ".elelem_backend") # last backend built here

# macOS/Metal is deferred: it needs .dylib naming, @loader_path rpath, an embedded
# .metallib, and extra cmake targets -- and a Mac to test on. Bail clearly rather
# than emit a broken build. (Detection + build path will slot in here later.)
if RbConfig::CONFIG["host_os"] =~ /darwin/
  abort "elelem: macOS/Metal is not supported yet -- Linux (CPU/CUDA/Vulkan) only."
end

class BuildError < StandardError; end

def run(*cmd)
  warn "elelem: + #{cmd.join(' ')}"
  system(*cmd) || raise(BuildError, "build step failed: #{cmd.join(' ')}")
end

# Is `cmd` an executable on PATH? (stdlib stand-in for mkmf's find_executable.)
def which(cmd)
  ENV["PATH"].to_s.split(File::PATH_SEPARATOR).any? do |dir|
    path = File.join(dir, cmd)
    File.executable?(path) && !File.directory?(path)
  end
end

# A Vulkan build links libvulkan at runtime, so glslc (shader compiler) alone is not
# enough -- probe for the loader too, or we'd build a lib that fails at dlopen.
def vulkan_loader?
  return true if which("vulkaninfo")
  %w[/usr/lib64 /usr/lib /usr/local/lib /lib64 /lib].any? do |dir|
    !Dir.glob(File.join(dir, "libvulkan.so*")).empty?
  end
end

VALID_BACKENDS = %w[cpu vulkan cuda].freeze

# Pick the backend from an explicit override, then the host toolchain, else CPU.
def detect_backend
  forced = ENV["ELELEM_LLAMA_BACKEND"].to_s.strip.downcase
  unless forced.empty? || forced == "auto" || VALID_BACKENDS.include?(forced)
    abort "elelem: ELELEM_LLAMA_BACKEND=#{forced.inspect} invalid; use auto|#{VALID_BACKENDS.join('|')}"
  end
  return forced unless forced.empty? || forced == "auto"
  # Legacy opt-in still honored.
  return "vulkan" if %w[1 on true yes].include?(ENV["ELELEM_GGML_VULKAN"].to_s.strip.downcase)

  return "cuda" if which("nvcc")
  return "vulkan" if which("glslc") && vulkan_loader?
  "cpu"
end

GPU_FLAGS = {
  "cpu"    => [],                     # GGML_NATIVE is ON by default -> host CPU SIMD
  "vulkan" => ["-DGGML_VULKAN=ON"],
  "cuda"   => ["-DGGML_CUDA=ON"]
}.freeze

# Configure + build only libllama + libllama-common (chat templates + tool-call
# parsing). Examples/tests/tools/server off keeps the build fast. $ORIGIN rpath lets
# the co-located libs in lib/elelem/native find their ggml siblings at runtime.
def configure_and_build(backend)
  flags = GPU_FLAGS.fetch(backend) { abort "elelem: unknown backend #{backend.inspect}" }
  run("cmake", "-S", VENDOR, "-B", BUILD_DIR,
      "-DCMAKE_BUILD_TYPE=Release",
      "-DBUILD_SHARED_LIBS=ON",
      "-DCMAKE_BUILD_WITH_INSTALL_RPATH=ON",
      "-DCMAKE_INSTALL_RPATH=$ORIGIN",
      "-DLLAMA_CURL=OFF",
      "-DLLAMA_BUILD_COMMON=ON",
      "-DLLAMA_BUILD_TESTS=OFF",
      "-DLLAMA_BUILD_EXAMPLES=OFF",
      "-DLLAMA_BUILD_TOOLS=OFF",
      "-DLLAMA_BUILD_SERVER=OFF",
      *flags)
  run("cmake", "--build", BUILD_DIR, "--target", "llama", "llama-common", "--parallel")
  File.write(STAMP, backend)
end

unless File.exist?(File.join(VENDOR, "CMakeLists.txt"))
  abort "elelem: vendored llama.cpp missing at #{VENDOR}\n" \
        "        run: git submodule update --init --recursive"
end

backend = detect_backend

# Toggling backends flips cached CMake vars, so a re-run targeting a different
# backend needs a clean build dir.
if File.exist?(STAMP) && File.read(STAMP).strip != backend
  warn "elelem: backend changed -> wiping #{BUILD_DIR}"
  FileUtils.rm_rf(BUILD_DIR)
end

# 1. Build llama.cpp for the chosen backend; on GPU failure, fall back to CPU so a
#    bare install never hard-fails on a half-present GPU toolchain.
begin
  warn "elelem: building llama.cpp backend=#{backend}"
  configure_and_build(backend)
rescue BuildError => e
  raise if backend == "cpu"
  warn "elelem: #{backend} build failed (#{e.message}); falling back to CPU"
  FileUtils.rm_rf(BUILD_DIR)
  backend = "cpu"
  configure_and_build(backend)
end

# 2. Copy the shared objects (preserving SONAME symlinks) next to where the shim
#    will live, so the shim's $ORIGIN rpath resolves libllama/libggml at runtime.
#    Clear stale libs first: switching backends (e.g. vulkan -> cpu) must not leave a
#    previous backend's libggml-*.so behind for ggml to pick up at runtime.
FileUtils.mkdir_p(NATIVE)
FileUtils.rm_f(Dir.glob(File.join(NATIVE, "*.so*")))
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

# 4. Record the backend the runtime actually got, so the :gguf provider can default
#    n_gpu_layers appropriately (see lib/elelem/plugins/gguf.rb).
File.write(File.join(NATIVE, "backend"), backend)

# 5. Stub Makefile so `make` / `make install` succeed (work is already done).
File.write(File.join(EXT_DIR, "Makefile"), <<~MAKE)
  all:
  clean:
  install:
  .PHONY: all clean install
MAKE

warn "elelem: built llama.cpp + shim into #{NATIVE} (backend=#{backend})"
