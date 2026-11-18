# frozen_string_literal: true

require "etc"
require "fileutils"

EXT_DIR   = __dir__
GEM_ROOT  = File.expand_path("../../..", EXT_DIR)
VENDOR    = File.join(GEM_ROOT, "vendor", "llama.cpp")
BUILD_DIR = File.join(EXT_DIR, "build")
NATIVE    = File.join(GEM_ROOT, "ext", "elelem", "llama", "native")
STAMP     = File.join(BUILD_DIR, ".elelem_backend") # last backend built here
VALID_BACKENDS = %w[cpu vulkan cuda hip metal].freeze
DARWIN = RbConfig::CONFIG["host_os"] =~ /darwin/

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

def gpu_flags(backend)
  {
    "cpu" => [],
    "vulkan" => ["-DGGML_VULKAN=ON"],
    "cuda" => ["-DGGML_CUDA=ON", *chosen_cuda_architectures],
    "hip" => ["-DGGML_HIP=ON", "-DCMAKE_HIP_ARCHITECTURES=#{hip_architectures}"],
    "metal" => ["-DGGML_METAL=ON", "-DGGML_METAL_EMBED_LIBRARY=ON"]
  }.fetch(backend) { abort "elelem: unknown backend #{backend.inspect}" }
end

def detect_backend
  forced = ENV["ELELEM_LLAMA_BACKEND"].to_s.strip.downcase
  unless forced.empty? || forced == "auto" || VALID_BACKENDS.include?(forced)
    abort "elelem: ELELEM_LLAMA_BACKEND=#{forced.inspect} invalid; use auto|#{VALID_BACKENDS.join("|")}"
  end
  return forced unless forced.empty? || forced == "auto"

  return "metal" if DARWIN
  return "cuda" if which("nvcc")
  return "vulkan" if which("glslc") && vulkan_loader?
  return "hip" if hip_available?

  "cpu"
end

def configure_and_build(backend)
  flags = gpu_flags(backend)
  run(
    "cmake", "-S", VENDOR, "-B", BUILD_DIR,
    "-DCMAKE_BUILD_TYPE=Release",
    "-DBUILD_SHARED_LIBS=ON",
    "-DCMAKE_BUILD_WITH_INSTALL_RPATH=ON",
    "-DCMAKE_INSTALL_RPATH=$ORIGIN",
    "-DGGML_NATIVE=ON",
    "-DLLAMA_OPENSSL=OFF",
    "-DLLAMA_BUILD_COMMON=ON",
    "-DLLAMA_BUILD_APP=OFF",
    "-DLLAMA_BUILD_TESTS=OFF",
    "-DLLAMA_BUILD_EXAMPLES=OFF",
    "-DLLAMA_BUILD_TOOLS=OFF",
    "-DLLAMA_BUILD_SERVER=OFF",
    *flags
  )
  run("cmake", "--build", BUILD_DIR, "--target", "llama", "llama-common", "--parallel",
      ENV.fetch("ELELEM_BUILD_JOBS") { Etc.nprocessors.to_s })
  File.write(STAMP, backend)
end

unless File.exist?(File.join(VENDOR, "CMakeLists.txt"))
  abort "elelem: vendored llama.cpp missing at #{VENDOR}\n        run: git submodule update --init --recursive"
end

backend = detect_backend

if File.exist?(STAMP) && File.read(STAMP).strip != backend
  warn "elelem: backend changed -> wiping #{BUILD_DIR}"
  FileUtils.rm_rf(BUILD_DIR)
end

def link_shim(libdir)
  ext = DARWIN ? "dylib" : "so"
  shim = File.join(NATIVE, "libelelem_llama.#{ext}")

  FileUtils.mkdir_p(NATIVE)
  FileUtils.rm_f(Dir.glob(File.join(NATIVE, "*.#{ext}*")))
  FileUtils.cp(Dir.glob(File.join(libdir, "*.#{ext}*")), NATIVE)

  cxx = ENV["CXX"] || "c++"
  link_flags = if DARWIN
                 ["-Wl,-rpath,@loader_path", "-Wl,-install_name,@rpath/libelelem_llama.dylib",
                  "-framework", "Metal", "-framework", "Foundation", "-framework", "Accelerate"]
               else
                 ["-Wl,-rpath,$ORIGIN"]
               end
  run(cxx, "-std=c++17", "-O2", "-Wall", "-Wextra", "-shared", "-fPIC",
      "-I", File.join(VENDOR, "include"),
      "-I", File.join(VENDOR, "ggml", "include"),
      "-I", File.join(VENDOR, "common"),   # chat.h (common_chat)
      "-I", File.join(VENDOR, "vendor"),   # nlohmann/json.hpp
      File.join(EXT_DIR, "elelem.cpp"),
      "-o", shim,
      "-L", libdir, "-lllama-common", "-lllama", "-lggml", "-lggml-base",
      *link_flags)
end

begin
  warn "elelem: building llama.cpp backend=#{backend}"
  configure_and_build(backend)
  link_shim(File.join(BUILD_DIR, "bin"))
rescue BuildError => e
  raise if backend == "cpu"

  warn "elelem: #{backend} build failed (#{e.message}); falling back to CPU"
  FileUtils.rm_rf(BUILD_DIR)
  backend = "cpu"
  configure_and_build(backend)
  link_shim(File.join(BUILD_DIR, "bin"))
end

File.write(File.join(NATIVE, "backend"), backend)
File.write(File.join(EXT_DIR, "Makefile"), <<~MAKE)
  all:
  clean:
  install:
  .PHONY: all clean install
MAKE

warn "elelem: built llama.cpp + shim into #{NATIVE} (backend=#{backend})"
