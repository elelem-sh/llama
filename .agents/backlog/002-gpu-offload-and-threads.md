As a `user with a GPU or a many-core machine`, I `want elelem-llama to
offload to HIP GPUs, leave room for the KV cache, and let me set the
thread count`, so that `the default settings use my hardware without
running out of VRAM`.

# SYNOPSIS

Fix the GPU layer default for HIP and the KV cache, and add a threads
setting.

# DESCRIPTION

`default_gpu_layers` in lib/elelem/llama.rb decides whether to offload
all layers (`999`) or none (`0`) when `ELELEM_GGUF_N_GPU_LAYERS` is not
set:

* Only `vulkan`, `cuda` and `metal` builds are considered
  (llama.rb:17). extconf.rb can build a `hip` backend
  (extconf.rb:11, :55), but a HIP build always defaults to 0 GPU
  layers, so ROCm users get CPU inference unless they set the variable.
* The check compares only the model file size with 80% of free VRAM
  (llama.rb:21). The KV cache for the chosen context is not counted, so
  a model that just fits can fail to allocate, or run with a much
  smaller context, once the cache is added.

Threads: `n_threads: 0` is always passed (llama.rb:27). The shim then
uses `common_cpu_get_num_math()` (elelem.cpp:152), which is a sensible
default, but there is no way to override it. Add
`ELELEM_GGUF_THREADS`.

Already fixed: the model in bin/setup,
`unsloth/Qwen3.8-27B-GGUF/Qwen3.8-27B-UD-Q4_K_XL.gguf`, resolves on
Hugging Face (HTTP 200 on 2026-11-16).

# SEE ALSO

* [x] lib/elelem/llama.rb - `default_gpu_layers` is gone; the shim decides
* [x] lib/elelem/llama/host_memory.rb - the RAM the host can spare
* [x] ext/elelem/llama/elelem.cpp - `el_fit_to_devices`, `el_largest_context`
* [x] vendor/llama.cpp/common/fit.h - `common_fit_params`
* [x] Formerly elelem-sh/elelem.sh#28

# Tasks

* [x] Let `common_fit_params` place layers, tensor split and MoE experts
      on every GPU backend, HIP included
* [x] Pick the largest context that fits the RAM the host can spare
      (Linux `MemAvailable` and cgroup limits, macOS `vm_stat`) and the
      free memory of each GPU, measured with the KV cache type in use
* [x] An integrated GPU keeps the weights on the CPU and only speeds up
      reading long prompts
* [x] `ELELEM_GGUF_THREADS`

# Acceptance Criteria

* [ ] On a HIP build with enough free VRAM, layers are offloaded by
      default (by design: fit treats every GPU backend alike; not run
      on HIP hardware)
* [x] The offload estimate includes the KV cache for the context that
      will be used
* [x] `ELELEM_GGUF_THREADS=N` sets the thread count; unset keeps
      today's default
* [x] `ELELEM_GGUF_N_GPU_LAYERS` still overrides the estimate
      (`0` also keeps every GPU out, as the escape from driver bugs)
* [x] Specs cover the memory budget and the explanation of the chosen
      settings; placement itself is llama.cpp's and has no Ruby side
