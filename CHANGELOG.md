# Changelog

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
