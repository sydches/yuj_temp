# Harness Public Staging

Clean extraction staging repo for the public version of the harness.

This repo is being built clinically from the lab repo by whitelist, not by
history-preserving migration. The goal is to keep the runtime identity intact:

- model and tool-call quirk absorption
- output admission control before context bloat
- deterministic artifacts and replay
- clean code/config separation
- local-first assistant mode without product sprawl

Start with these docs:

- `docs/harness_spec.md`
- `docs/assistant_spec.md`
- `docs/coupling_spec.md`
- `docs/public_repo_spec.md`

Current status:

- internal paths preserved intentionally
- benchmark shell intentionally left behind
- assistant shell is usable now under `scripts/llm_assist/`
- session inspection, approval/suspend, and smoke bootstrap are implemented
- primary operator entrypoint is `yuj`

This repo is not the old experiment repo with names changed. It is the start of
the public runtime.

## Current entrypoints

```bash
python3 -m scripts.llm_solver --help
yuj --help
yuj code "Fix the failing test."
yuj smoke
yuj sessions
yuj show
yuj approve
yuj resume
```

`yuj` is the operator-facing command. The repo also includes `./yuj` and
`python3 -m scripts.yuj` as equivalent entrypoints if you are running from the
checkout directly.

`code` is an alias of `run` for the common coding-agent entry path.

## Practical CLI Flow

```bash
yuj inspect presets
yuj inspect knobs runtime

yuj code "Fix the failing test, run the relevant test, then finish."
yuj code --cwd /path/to/other/repo "Fix the failing test, run the relevant test, then finish."
yuj run --prompt-file task.txt

yuj sessions
yuj show
yuj resume
```

Assistant artifacts default to `<project>/.llm_assist/`. To keep them elsewhere
for smoke runs or temporary sessions, set `HARNESS_ASSIST_HOME=/tmp/...`.

Ease-of-use defaults:

- `yuj code "..."` runs against the current directory by default
- positional prompt text is accepted for the common path; `--prompt-text` and
  `--prompt-file` still work
- `yuj show`, `yuj resume`, and `yuj approve` target the latest relevant
  session by default, preferring the current repo cwd before falling back to
  the global latest session

`run` and `smoke` reconcile the requested model against `/v1/models` at session
creation: alias-resolved ids that are not served verbatim fall back to the
first served id. The exact served id is what gets persisted in session
metadata.

While a session is running, `run` and `resume` print incremental progress to
stdout — session start, tool calls, approval requests, and session end — by
tailing `.trace.jsonl`. No engine changes are required for this.

Each session now prints an immediate startup banner with the session id, cwd,
model, artifact path, and served model list before the live trace begins, so
the operator does not need to wait for completion just to learn what started.

If a risky assistant-mode bash command is blocked, the session pauses with an
approval request. Assistant-mode classifier coverage: `rm`, `git reset --hard`,
`git clean`, `git checkout --`, `chmod`, `chown`, plus `mv`/`cp` when any
positional path resolves outside the repo root. Measurement mode is unaffected.

```bash
yuj show
yuj approve
yuj resume
yuj show <session_id>
yuj approve <session_id>
yuj resume <session_id>
```

## Smoke Test

Use the built-in smoke command to bootstrap a throwaway repo, resolve the exact
served model id from `/v1/models`, and run one assistant session. A green smoke
run means: the assistant session exited successfully, `calc.py` contains the
fixed `return a + b` body, `tests/test_calc.py` passes, and no pending approval
is outstanding. Any of those failing produces a non-zero exit with the smoke
repo path, session id, artifact dir, final status, and finish reason.

```bash
yuj smoke
HARNESS_ASSIST_HOME=/tmp/harness-assist-smoke yuj smoke
```
