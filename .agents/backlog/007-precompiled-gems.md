As a `user installing elelem with curl -fsSL https://elelem.sh/install | sh`,
I `want a prebuilt elelem-llama for my platform`, so that `install needs
no compiler and works on any machine, Windows included`.

# SYNOPSIS

Publish platform gems of elelem-llama with the native library already
built, next to the source gem.

# DESCRIPTION

Every install of elelem-llama compiles the vendored llama.cpp
(extconf.rb). That needs a C/C++ compiler, CMake, pkg-config and the
libffi headers, takes minutes on a laptop, and fails on platforms the
build has not been tried on. It is the one step that keeps the elelem.sh
install script from being a single command everywhere: the script stops
and prints a package-manager command when build tools are missing,
refuses musl Linux, and has no Windows version.

Proposed: build `libelelem_llama` in the 006 install matrix and publish a
platform gem per cell (`x86_64-linux-gnu`, `aarch64-linux-gnu`,
`x86_64-linux-musl`, `aarch64-linux-musl`, `arm64-darwin`,
`x86_64-darwin`), with the CPU backend plus Metal on arm64-darwin. The
source gem stays for GPU backends (`ELELEM_LLAMA_BACKEND`) and other
platforms. Windows needs `link_shim` to handle `.dll` first
(extconf.rb:109).

Open questions for design mode:

* rake-compiler-dock versus native runners per platform
* whether rv (rv.dev) installs platform gems from gem.coop
* how a user picks a GPU backend when a prebuilt CPU gem exists

# SEE ALSO

* [ ] .agents/backlog/006-install-matrix.md
* [ ] ext/elelem/llama/extconf.rb:109 - shared library extension
* [ ] elelem-llama.gemspec
* [ ] https://github.com/elelem-sh/elelem.sh/blob/main/static/install

# Tasks

* [ ] TBD (filled in design mode)

# Acceptance Criteria

* [ ] `gem install elelem-llama` on each listed platform installs a
      prebuilt gem without a compiler
* [ ] The elelem.sh install script no longer checks for build tools on
      those platforms
* [ ] The source gem still builds GPU backends
