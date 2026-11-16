As a `person who wants to run a GGUF model in process`, I `want a
README in the elelem-llama repository`, so that `I can install it, pick
a model and a GPU backend, and avoid known build problems`.

# SYNOPSIS

Add a short README.

# DESCRIPTION

The repository has no README. It should cover:

* Install, including the build requirements (a C++ compiler, CMake)
  and that the gem compiles llama.cpp, which takes a while.
* Backends: auto detection and `ELELEM_LLAMA_BACKEND`
  (cpu, vulkan, cuda, hip, metal), and `ELELEM_HIP_ARCH`.
* Models: how a model is found (`--model`, `~/.agents/models`, see
  model_resolver.rb) and what bin/setup downloads.
* Runtime settings: `ELELEM_GGUF_N_CTX`, `ELELEM_GGUF_N_GPU_LAYERS`,
  `ELELEM_GGUF_MAX_TOKENS`, `ELELEM_GGUF_TEMP`, `ELELEM_GGUF_SEED`, and
  any added by other stories (threads, build jobs).
* The out of memory build problem and its workaround, until the
  bounded build story ships.
* A link to https://elelem.sh/plugins/llama/ for the full guide.

Show commands without a leading `$`.

# SEE ALSO

* [ ] lib/elelem/llama.rb - provider block and environment
* [ ] lib/elelem/llama/model_resolver.rb - model lookup
* [ ] ext/elelem/llama/extconf.rb:62 - backend detection
* [ ] bin/setup - model download
* [ ] https://elelem.sh/plugins/llama/
* [ ] Formerly elelem-sh/elelem.sh#23

# Tasks

* [ ] TBD (filled in design mode)

# Acceptance Criteria

* [ ] README.md exists and covers install, backends, model lookup and
      every environment variable the gem reads
* [ ] Defaults in the README match the code
* [ ] Commands have no leading `$`
