As a `user installing elelem-llama`, I `want the native build to use a
bounded number of jobs`, so that `gem install does not run out of
memory on an ordinary machine`.

# SYNOPSIS

Pass an explicit job count to `cmake --build --parallel`.

# DESCRIPTION

extconf.rb builds llama.cpp with:

```ruby
run("cmake", "--build", BUILD_DIR, "--target", "llama", "llama-common", "--parallel")
```

(extconf.rb:93). `--parallel` with no number, under CMake's default
Unix Makefiles generator, runs `make -j` with no limit. Installing
elelem-llama 0.3.0 on a 4-core, 16 GB Linux container ran about 180
compiler processes at once; the kernel killed processes for lack of
memory and the install failed. `CMAKE_GENERATOR=Ninja gem install
elelem-llama` works around it, and the elelem.sh local models guide
documents that.

Proposed:

```ruby
jobs = ENV.fetch("ELELEM_BUILD_JOBS") { Etc.nprocessors.to_s }
run("cmake", "--build", BUILD_DIR, "--target", "llama", "llama-common", "--parallel", jobs)
```

`CMAKE_BUILD_PARALLEL_LEVEL` is only read when `--parallel` has no
number, so passing the number is the reliable fix. `require "etc"` in
extconf.rb (it requires only fileutils today).

# SEE ALSO

* [x] ext/elelem/llama/extconf.rb:93 - unbounded --parallel
* [x] ext/elelem/llama/extconf.rb:3 - requires
* [x] bin/compile
* [x] Formerly elelem-sh/elelem.sh#43

# Tasks

* [x] Pass `--parallel` a job count from `ELELEM_BUILD_JOBS`, else `Etc.nprocessors`
* [x] Build, install and run the gem in CI on Linux x86_64, arm64 and Fedora (`bin/platform-check`)

# Acceptance Criteria

* [x] The cmake build command passes an explicit job count
* [x] The default job count is the number of processors
* [x] `ELELEM_BUILD_JOBS` overrides it
* [x] `gem install` of the built gem succeeds on a 4-core, 16 GB Linux
      machine with the default generator (CI `linux` job: private-repo
      runners have 2 cores and 8 GB, the same memory per build job)
