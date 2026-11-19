# frozen_string_literal: true

require "etc"
require "fileutils"
require_relative "cxx_runtime"
require_relative "shared_libraries"
require_relative "../../../lib/elelem/llama/shim"

EXT_DIR   = __dir__
GEM_ROOT  = File.expand_path("../../..", EXT_DIR)
VENDOR    = File.join(GEM_ROOT, "vendor", "llama.cpp")
BUILD_DIR = File.join(EXT_DIR, "build")
NATIVE    = File.join(GEM_ROOT, "ext", "elelem", "llama", "native")
STAMP     = File.join(BUILD_DIR, ".elelem_backend") # last backend built here
VALID_BACKENDS = %w[cpu vulkan cuda hip metal portable].freeze
FALLBACK = { "vulkan" => "cpu", "cuda" => "cpu", "hip" => "cpu", "metal" => "cpu" }.freeze
HOST_CPU = ["-DGGML_NATIVE=ON"].freeze
EVERY_CPU = ["-DGGML_NATIVE=OFF", "-DGGML_BACKEND_DL=ON", "-DGGML_CPU_ALL_VARIANTS=ON", "-DGGML_OPENMP=OFF"].freeze
HOST = Elelem::Llama::Host.new
DARWIN = HOST.darwin?
WINDOWS = HOST.windows?
MUSL = HOST.musl?
APPLE_SILICON = HOST.apple_silicon?
METAL = ["-DGGML_METAL=ON", "-DGGML_METAL_EMBED_LIBRARY=ON"].freeze
NO_METAL = ["-DGGML_METAL=OFF"].freeze
BESIDE_THE_LOADER = DARWIN ? "@loader_path" : "$ORIGIN"
OLDEST_MACOS = "13.3"
LLAMA_LIBRARIES = %w[llama-common llama ggml ggml-base].freeze
LLAMA_DLLS = Elelem::Llama::Shim::LOADED_FIRST_ON_WINDOWS.reverse.freeze
APPLE_FRAMEWORKS = %w[Metal Foundation Accelerate].flat_map { ["-framework", it] }.freeze
STATIC_RUNTIME = ["-static", "-s", "-Wl,--exclude-libs,libgcc_eh.a:libwinpthread.a:libpthread.a"].freeze
STATIC_RUNTIME_IN_EVERY_DLL = %w[SHARED MODULE].map { "-DCMAKE_#{it}_LINKER_FLAGS=#{STATIC_RUNTIME.join(" ")}" }.freeze

class BuildError < StandardError; end

def run(*cmd)
  warn "elelem: + #{cmd.join(" ")}"
  system(*cmd) || raise(BuildError, "build step failed: #{cmd.join(" ")}")
end

def which(cmd)
  ENV["PATH"].to_s.split(File::PATH_SEPARATOR).any? do |dir|
    path = File.join(dir, cmd)
    File.executable?(path) && !File.directory?(path)
  end
end

def vulkan_loader?
  return true if which("vulkaninfo")

  %w[/usr/lib64 /usr/lib /usr/local/lib /lib64 /lib].any? do |dir|
    !Dir.glob(File.join(dir, "libvulkan.so*")).empty?
  end
end

# Most specific gfxNNNN target reported by rocminfo (e.g. "gfx1151"),
# skipping the generic "gfxNN" family aliases it also prints.
def hip_gfx_target
  out = `rocminfo 2>/dev/null`
  targets = out.scan(/gfx[0-9a-f]+/).uniq
  targets.max_by(&:length)
rescue Errno::ENOENT
  nil
end

def hip_available?
  which("hipcc") && !hip_gfx_target.nil?
end

def hip_architectures = ENV.fetch("ELELEM_HIP_ARCH") { hip_gfx_target || "gfx1100" }

def chosen_cuda_architectures = ENV.values_at("ELELEM_CUDA_ARCH").compact.map { "-DCMAKE_CUDA_ARCHITECTURES=#{it}" }

def portable_gpu
  return [] if WINDOWS || MUSL
  return ["-DGGML_VULKAN=ON"] unless DARWIN

  APPLE_SILICON ? METAL : NO_METAL
end

def backend_flags(backend)
  {
    "cpu" => [*HOST_CPU, *(DARWIN ? NO_METAL : [])],
    "vulkan" => [*HOST_CPU, "-DGGML_VULKAN=ON"],
    "cuda" => [*HOST_CPU, "-DGGML_CUDA=ON", *chosen_cuda_architectures],
    "hip" => [*HOST_CPU, "-DGGML_HIP=ON", "-DCMAKE_HIP_ARCHITECTURES=#{hip_architectures}"],
    "metal" => [*HOST_CPU, *METAL],
    "portable" => [*EVERY_CPU, *portable_gpu]
  }.fetch(backend) { abort "elelem: unknown backend #{backend.inspect}" }
end

def detect_backend
  forced = ENV["ELELEM_LLAMA_BACKEND"].to_s.strip.downcase
  unless forced.empty? || forced == "auto" || VALID_BACKENDS.include?(forced)
    abort "elelem: ELELEM_LLAMA_BACKEND=#{forced.inspect} invalid; use auto|#{VALID_BACKENDS.join("|")}"
  end
  return forced unless forced.empty? || forced == "auto"

  return "metal" if APPLE_SILICON
  return "cpu" if DARWIN
  return "cuda" if which("nvcc")
  return "vulkan" if which("glslc") && vulkan_loader?
  return "hip" if hip_available?

  "cpu"
end

def configure_and_build(backend)
  run(
    "cmake", "-S", VENDOR, "-B", BUILD_DIR,
    "-DCMAKE_BUILD_TYPE=Release",
    "-DBUILD_SHARED_LIBS=ON",
    "-DCMAKE_BUILD_WITH_INSTALL_RPATH=ON",
    "-DCMAKE_INSTALL_RPATH=#{BESIDE_THE_LOADER}",
    "-DLLAMA_OPENSSL=OFF",
    "-DLLAMA_BUILD_COMMON=ON",
    "-DLLAMA_BUILD_APP=OFF",
    "-DLLAMA_BUILD_TESTS=OFF",
    "-DLLAMA_BUILD_EXAMPLES=OFF",
    "-DLLAMA_BUILD_TOOLS=OFF",
    "-DLLAMA_BUILD_SERVER=OFF",
    *(WINDOWS ? STATIC_RUNTIME_IN_EVERY_DLL : []),
    *backend_flags(backend)
  )
  run("cmake", "--build", BUILD_DIR, "--target", "llama", "llama-common", "--parallel",
      ENV.fetch("ELELEM_BUILD_JOBS") { Etc.nprocessors.to_s })
  File.write(STAMP, backend)
end

unless File.exist?(File.join(VENDOR, "CMakeLists.txt"))
  abort "elelem: vendored llama.cpp missing at #{VENDOR}\n        run: git submodule update --init --recursive"
end

backend = detect_backend
ENV["MACOSX_DEPLOYMENT_TARGET"] ||= OLDEST_MACOS if backend == "portable"

if File.exist?(STAMP) && File.read(STAMP).strip != backend
  warn "elelem: backend changed -> wiping #{BUILD_DIR}"
  FileUtils.rm_rf(BUILD_DIR)
end

def on_unix(libdir)
  ["-fPIC", "-L", libdir, *LLAMA_LIBRARIES.map { "-l#{it}" }, "-Wl,-rpath,#{BESIDE_THE_LOADER}"]
end

def on_macos(libdir)
  [*on_unix(libdir), "-Wl,-install_name,@rpath/libelelem_llama.dylib", *APPLE_FRAMEWORKS]
end

def on_windows(libdir) = [*LLAMA_DLLS.map { File.join(libdir, it) }, *STATIC_RUNTIME]

def link_flags(libdir)
  return on_windows(libdir) if WINDOWS
  return on_macos(libdir) if DARWIN

  on_unix(libdir)
end

def compiler = ENV["CXX"] || "c++"

def carries_its_cxx_runtime?(backend) = MUSL && backend == "portable"

def link_shim(libdir)
  FileUtils.rm_rf(NATIVE)
  Elelem::Llama::SharedLibraries.new(libdir).install(NATIVE)

  run(compiler, "-std=c++17", "-O2", "-Wall", "-Wextra", "-shared",
      "-I", File.join(VENDOR, "include"),
      "-I", File.join(VENDOR, "ggml", "include"),
      "-I", File.join(VENDOR, "common"),
      "-I", File.join(VENDOR, "vendor"),
      File.join(EXT_DIR, "elelem.cpp"),
      "-o", Elelem::Llama::Shim.new(dir: NATIVE).path,
      *link_flags(libdir))
end

begin
  warn "elelem: building llama.cpp backend=#{backend}"
  configure_and_build(backend)
  link_shim(File.join(BUILD_DIR, "bin"))
rescue BuildError => e
  backend = FALLBACK.fetch(backend) { raise }
  warn "elelem: build failed (#{e.message}); falling back to #{backend}"
  FileUtils.rm_rf(BUILD_DIR)
  configure_and_build(backend)
  link_shim(File.join(BUILD_DIR, "bin"))
end

Elelem::Llama::CxxRuntime.new(compiler:).install(NATIVE) if carries_its_cxx_runtime?(backend)

File.write(File.join(EXT_DIR, "Makefile"), <<~MAKE)
  all:
  clean:
  install:
  .PHONY: all clean install
MAKE

warn "elelem: built llama.cpp + shim into #{NATIVE} (backend=#{backend})"
