As a `user chatting with a local GGUF model`, I `want the reply to
stream as reply text while it is generated`, so that `I read the answer
as it arrives instead of raw model output shown as thinking`.

# SYNOPSIS

Stream parsed reply text as `saying`, and only reasoning as `thinking`.

# DESCRIPTION

The native shim calls back once per generated piece. The callback
emits every piece as `thinking` (provider.rb:62-65). That text is the
raw model output: chat template markers, reasoning, tool call JSON and
the reply all mixed together. The parsed result, with `content`,
`reasoning` and `tool_calls` split apart, only arrives when generation
ends; then `content` is emitted as `saying` in one piece
(provider.rb:97-106). When anything streamed, the parsed reasoning is
not emitted again (provider.rb:95).

So during a turn the user sees unparsed output dimmed as thinking, and
then the reply appears all at once.

Proposed: parse incrementally as pieces arrive, and emit reasoning as
`thinking` and reply text as `saying` as soon as each is known. Tool
call text should not be streamed to the user as either. llama.cpp's
common chat parser supports partial parsing, which the shim can use;
choose the approach in design.

# SEE ALSO

* [ ] lib/elelem/llama/provider.rb:62-68 - on_token callback
* [ ] lib/elelem/llama/provider.rb:93-106 - final reasoning and content
* [ ] ext/elelem/llama/elelem.cpp - el_generate and the token callback
* [ ] spec/elelem/llama/provider_spec.rb
* [ ] Formerly elelem-sh/elelem.sh#28

# Tasks

* [ ] TBD (filled in design mode)

# Acceptance Criteria

* [ ] Reply text reaches the output as `saying` events during
      generation, not only at the end
* [ ] Reasoning reaches the output as `thinking` events
* [ ] Chat template markers and tool call JSON are not shown as
      thinking or saying
* [ ] The final reply recorded in the conversation is the same as today
* [ ] Specs cover a reply, a reply with reasoning, and a tool call
