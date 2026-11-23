# Changelog

## [0.10.0](https://github.com/elelem-sh/llama/compare/v0.9.1...v0.10.0) (2026-11-23)


### Features

* choose the largest pinned model that fits the memory of this machine ([db9c25e](https://github.com/elelem-sh/llama/commit/db9c25ef95232ef6e58b183de23c9dcc72ffb16f))
* turn thinking off with ELELEM_THINKING=off so replies start at once ([63d940b](https://github.com/elelem-sh/llama/commit/63d940b156f8021baf27c564a6131daf62870610))

## [0.9.1](https://github.com/elelem-sh/llama/compare/v0.9.0...v0.9.1) (2026-11-20)


### Bug Fixes

* find nvcc, glslc and vulkaninfo on Windows by the suffixes PATHEXT names ([04fe1a6](https://github.com/elelem-sh/llama/commit/04fe1a67c91ecf272c537f11a0fbee56273c6749))
* refuse in one sentence a Windows Ruby the source gem cannot build on ([80535a1](https://github.com/elelem-sh/llama/commit/80535a13914463b766786dd618aa435b2326e261))

## [0.9.0](https://github.com/elelem-sh/llama/compare/v0.8.0...v0.9.0) (2026-11-20)


### Features

* build a CPU-only portable gem on musl that carries its own C++ runtime ([2b763d3](https://github.com/elelem-sh/llama/commit/2b763d3be94f0903726cb04826aa614a35c92289))


### Bug Fixes

* carry the GCC runtime under names of its own, so it never stands in for the one of the system ([85204b7](https://github.com/elelem-sh/llama/commit/85204b71aa78b30e93ad5ec520c53922ec6a71e0))

## [0.8.0](https://github.com/elelem-sh/llama/compare/v0.7.0...v0.8.0) (2026-11-19)


### Features

* budget the context from the physical memory Windows has available ([d0bf824](https://github.com/elelem-sh/llama/commit/d0bf824c787e033717b41d62bd7103939e98d351))
* link the shim and every DLL on Windows with the C and C++ runtime inside and nothing to find at load time ([26e0a77](https://github.com/elelem-sh/llama/commit/26e0a770ef5dd52902dc43557b682681aa4cbf40))


### Bug Fixes

* refuse to start when no backend loads from the gem instead of taking one from the working directory ([aa33ad4](https://github.com/elelem-sh/llama/commit/aa33ad400f8a74c92c358065826d57240ac4a0ee))
* say which Windows error stopped a DLL from loading and look LoadLibraryW up once ([e27d7b9](https://github.com/elelem-sh/llama/commit/e27d7b927718e7195658f3096512ebe7200a7782))

## [0.7.0](https://github.com/elelem-sh/llama/compare/v0.6.0...v0.7.0) (2026-11-18)


### Features

* build the prebuilt gem on macOS, with Metal on Apple silicon, for macOS 13.3 and libraries found beside their loader ([d7ba392](https://github.com/elelem-sh/llama/commit/d7ba392ce0ca90b4134ce38b1e230c800175eddc))

## [0.6.0](https://github.com/elelem-sh/llama/compare/v0.5.0...v0.6.0) (2026-11-18)


### Features

* build a prebuilt gem that picks the best code for the CPU when it runs, with ELELEM_LLAMA_BACKEND=portable ([223a5af](https://github.com/elelem-sh/llama/commit/223a5af08dd80bbc49acee30a31603c2072712a8))
* name the backend in use from the device registry and load backend modules from the gem ([f7e7e1b](https://github.com/elelem-sh/llama/commit/f7e7e1b728f46f4d854dfb50f580a2cd86f66979))


### Bug Fixes

* accept the json that ships with Ruby so installing it compiles nothing ([ec0a2dc](https://github.com/elelem-sh/llama/commit/ec0a2dce93e7e712027963cf57a188eca7a70dad))

## [0.5.0](https://github.com/elelem-sh/llama/compare/v0.4.3...v0.5.0) (2026-11-18)


### Features

* fit the GPU layers and the largest context to the memory of the machine ([f10cd42](https://github.com/elelem-sh/llama/commit/f10cd42ee018d3d24f87d1c4196da773e89cc243))
* prefer the whole model on a dedicated GPU while a context of 32768 tokens still fits there ([7f9a4e8](https://github.com/elelem-sh/llama/commit/7f9a4e8f6998e3848c6f7549d4a4bccd623ea21b))
* read the memory the host can spare on Linux and macOS, control group limits included ([02bccaa](https://github.com/elelem-sh/llama/commit/02bccaa371ee9da491858582a06874ce45592a94))
* set the thread count with ELELEM_GGUF_THREADS ([435acf7](https://github.com/elelem-sh/llama/commit/435acf7769535b99f0f4649fd6f821a902ddf2f3))
* survey the dedicated GPU before loading and inject the native functions in specs instead of stubbing them ([fb7ea21](https://github.com/elelem-sh/llama/commit/fb7ea21980d753d7f085b523acc70b738c8bb506))


### Bug Fixes

* build the cpu backend on Intel Macs, where Metal generates garbage ([c132ab5](https://github.com/elelem-sh/llama/commit/c132ab587234b5332416dfa6951ae7ab7b77f14d))

## [0.4.3](https://github.com/elelem-sh/llama/compare/v0.4.2...v0.4.3) (2026-11-18)


### Bug Fixes

* build llama.cpp for the host CPU under gem install ([f32ba7e](https://github.com/elelem-sh/llama/commit/f32ba7e3a5536c4c5a04914854c96aa5f4dd2bbc))
* **deps:** bump llama.cpp to v1.0.0 ([c1c677c](https://github.com/elelem-sh/llama/commit/c1c677c14106ed4c2824b578f68a92837d770cc1))
* let ELELEM_CUDA_ARCH choose the cuda architectures ([5f2b9ff](https://github.com/elelem-sh/llama/commit/5f2b9ff3a749d8936ee6137f197c9dfc37f7b1ab))

## [0.4.2](https://github.com/elelem-sh/llama/compare/v0.4.1...v0.4.2) (2026-11-17)


### Bug Fixes

* stream only the parsed reasoning so think tags and the answer are not shown as thinking ([c3d62cb](https://github.com/elelem-sh/llama/commit/c3d62cbd30341952b9a0723be348434e48a8369a))

## [0.4.1](https://github.com/elelem-sh/llama/compare/v0.4.0...v0.4.1) (2026-11-17)


### Bug Fixes

* build llama.cpp with one job per processor, overridable with ELELEM_BUILD_JOBS, so gem install does not run out of memory ([f8b4f82](https://github.com/elelem-sh/llama/commit/f8b4f8245fa4d88464dae1b9f5b24abdae10bf44))

## [0.4.0](https://github.com/elelem-sh/llama/compare/v0.3.0...v0.4.0) (2026-11-17)


### Features

* accept model: from provider_options, fall back to ENV ([fa7c5e9](https://github.com/elelem-sh/llama/commit/fa7c5e982b3845a98dd7b1bb14428d1442e50624))
* add macOS Metal backend support ([67b4292](https://github.com/elelem-sh/llama/commit/67b4292a0b6f0120e745e184aa22f4fa8e6d6707))


### Bug Fixes

* /provider llama reuses the loaded model instead of loading a second copy ([16d8aae](https://github.com/elelem-sh/llama/commit/16d8aae20ecd58d4e085e45fe049cedf74b1b7b8))
* add rspec to Gemfile so bundle exec rspec runs at all ([70061ba](https://github.com/elelem-sh/llama/commit/70061ba111dc7a82a132c30bc119d5d7a51692be))
* auto-detect safe GPU offload, default off on non-Apple-Silicon Metal ([30795df](https://github.com/elelem-sh/llama/commit/30795df7f5d6c8763433e8a2697e8fe0f8b764f0))
* detect integrated GPU devices when querying free VRAM ([4a4516c](https://github.com/elelem-sh/llama/commit/4a4516cb1e323159bd9a201aa553d892d1be5d62))
* llama decides GPU layers once per model, so /provider llama never loads a second copy ([38deaaa](https://github.com/elelem-sh/llama/commit/38deaaa042d2c53e85c168a63384a830bde2173c))
* require elelem ~&gt; 0.12, the floor builtins/skills actually need ([bef01fa](https://github.com/elelem-sh/llama/commit/bef01fa70d04aa3219be5f64743cf53a13b4edd8))
