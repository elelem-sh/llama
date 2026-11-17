As a `user installing elelem-llama on any machine`, I `want CI to
prove that gem install elelem-llama builds and runs on my platform`, so
that `installs do not fail on an OS or CPU nobody tried`.

# SYNOPSIS

Add a CI matrix that builds, installs and runs the gem on each
supported OS and CPU architecture before every release.

# DESCRIPTION

elelem-llama compiles the vendored llama.cpp at install time
(extconf.rb). Today CI only builds it on one Linux x86_64 runner, so a
break on arm64, macOS, musl or a GPU backend is found by users, not by
CI. Example: the unbounded `--parallel` build (backlog 003) failed for
a user on a 4-core Linux container.

Proposed matrix, Ruby 4.0 (the gemspec floor), CPU backend unless
noted:

| Runner / image       | Arch    | Backend      |
|----------------------|---------|--------------|
| `ubuntu-latest`      | x86_64  | cpu          |
| `ubuntu-24.04-arm`   | aarch64 | cpu          |
| `macos-latest`       | arm64   | metal        |
| `macos-15-intel`     | x86_64  | cpu          |
| Fedora container     | x86_64  | cpu          |
| Alpine container     | x86_64  | cpu (musl)   |
| `ubuntu-latest`      | x86_64  | vulkan, cuda, hip: build only |

Each cell:

1. `gem build elelem-llama.gemspec` (checkout with submodules, so
   `vendor/llama.cpp` is in the gem)
2. `CMAKE_GENERATOR=Ninja gem install ./elelem-llama-*.gem`
3. `elelem-llama --help`
4. Load `tinyllamas/stories260K.gguf` from `ggml-org/models` (about
   1 MB, the model llama.cpp's own CI uses; cache it) and generate a
   few tokens

GPU backends have no runners on the current plan, so those cells
install the SDK (Vulkan SDK, CUDA toolkit, ROCm) and only check that
the build succeeds with `ELELEM_LLAMA_BACKEND=<name>`.

Windows is out of scope for now: `link_shim` only copies `.so` or
`.dylib` (extconf.rb:109). Revisit in design.

# SEE ALSO

* [ ] ext/elelem/llama/extconf.rb:11 - VALID_BACKENDS
* [ ] ext/elelem/llama/extconf.rb:62 - detect_backend
* [ ] ext/elelem/llama/extconf.rb:109 - shared library extension
* [ ] .github/workflows/release.yml
* [ ] .agents/backlog/002-gpu-offload-and-threads.md
* [ ] .agents/backlog/003-bounded-build-parallelism.md

# Tasks

* [ ] TBD (filled in design mode)

# Acceptance Criteria

* [ ] Every cell in the matrix builds, installs the built `.gem`, runs
      `elelem-llama --help`, and generates tokens from the tiny model
      (GPU cells: build only)
* [ ] The matrix runs on pull requests and before release
* [ ] A failing cell blocks the release job
* [ ] The README lists the supported OS and CPU combinations
