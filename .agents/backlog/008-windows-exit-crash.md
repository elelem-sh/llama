As a `user on Windows`, I `want elelem-shell to exit cleanly after it answers`, so that `scripts and CI see exit 0 instead of a crash`.

# SYNOPSIS

Fix the intermittent heap corruption (`0xC0000374`) when elelem-shell
exits on Windows after a llama chat.

# DESCRIPTION

The `docs` job in `.github/workflows/windows-e2e.yml` runs the Windows
steps from https://elelem.sh/docs/start/ verbatim in PowerShell, then a
one-shot chat. In run 45521694260 (0.11.0, x64-mingw-ucrt), the model
read `nonce.txt` and repeated the nonce correctly, and then the process
ended with:

    Program "elelem-shell.bat" ended with non-zero exit code: -1073740940 (0xC0000374).

`0xC0000374` is STATUS_HEAP_CORRUPTION. It happens after the reply, so
the likely place is teardown: freeing the llama context, model or
sampler at exit, the order of `el_close` against Ruby's finalizers, or
memory freed by a different CRT than the one that allocated it.

It is intermittent:

- The `chat` job, which runs a one-shot chat from bash with
  `</dev/null`, passed in the same run and in run 45521335689.
- The `docs` job passed once (run 45521335689) and failed once.

# Tasks

* [ ] Reproduce: loop the `docs` one-shot 10 times in one job and count
      the failures. Compare `$null |` from pwsh with `</dev/null` from bash.
* [ ] Narrow it: close the provider explicitly before exit, and run once
      with `ELELEM_GGUF_N_GPU_LAYERS=0` to rule out the backend.
* [ ] Fix it in the extension's teardown, not in the shell.

# Acceptance Criteria

* [ ] The `docs` job passes 10 one-shot runs in a row
* [ ] `elelem-shell` exits 0 after a successful one-shot chat on Windows

# SEE ALSO

* `.github/workflows/windows-e2e.yml`
* 007-precompiled-gems.md
