# Sandbox

The `bash` tool runs inside a bubblewrap mount namespace when
`[tools].sandbox_bash = true`.

## Contract

Inside the sandbox:

- the host filesystem is mounted read-only
- the current working directory is mounted writable at its real path
- `/tmp` is a fresh tmpfs per call
- `/proc` and `/dev` are available
- the docker socket is bound in when present

Result:

- reads outside `cwd` may succeed
- writes outside `cwd` must fail
- writes inside `cwd` must succeed
- `/tmp` state does not persist across bash calls

## Why Real-Path Binding Matters

The writable bind uses the real host path for `cwd`, not a remapped path like
`/work`. That keeps host-side tools such as `docker run -v $PWD:...` working,
because the daemon resolves host paths, not namespace-only aliases.

## What Is Not Sandboxed

The harness's own bookkeeping is not wrapped:

- `.trace.jsonl`
- `.solver/state.json`
- `checkpoint.json`
- `metrics.json`

Those files are written by trusted host code, not by the model's shell.

## Verification

The minimum escape checks are:

- write inside `cwd`: succeeds
- write to `$HOME` or another absolute host path: fails read-only
- write via `../../..`: fails read-only

The automated check for this contract lives in:

- `tests/test_sandbox_escape.py`
