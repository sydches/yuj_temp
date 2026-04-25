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

This repo is not the old experiment repo with names changed. It is the start of
the public runtime.

## Current entrypoints

```bash
python3 -m scripts.llm_solver --help
python3 -m scripts.llm_assist run --cwd /path/to/repo --prompt-text "Fix the failing test."
python3 -m scripts.llm_assist code --cwd /path/to/repo --prompt-text "Fix the failing test."
python3 -m scripts.llm_assist smoke
python3 -m scripts.llm_assist sessions
python3 -m scripts.llm_assist show <session_id>
python3 -m scripts.llm_assist approve <session_id>
python3 -m scripts.llm_assist resume <session_id>
```

`code` is an alias of `run` for the common coding-agent entry path.

## Practical CLI Flow

```bash
python3 -m scripts.llm_assist inspect presets
python3 -m scripts.llm_assist inspect knobs runtime

python3 -m scripts.llm_assist run \
  --cwd /path/to/repo \
  --prompt-text "Fix the failing test, run the relevant test, then finish."

python3 -m scripts.llm_assist sessions
python3 -m scripts.llm_assist show <session_id>
python3 -m scripts.llm_assist resume <session_id>
```

Assistant artifacts default to `<project>/.llm_assist/`. To keep them elsewhere
for smoke runs or temporary sessions, set `HARNESS_ASSIST_HOME=/tmp/...`.

`run` now reconciles the requested model against `/v1/models` at session
creation: alias-resolved ids that are not served verbatim fall back to the
first served id, matching `smoke`. The exact served id is what gets persisted
in session metadata.

While a session is running, `run` and `resume` print incremental progress to
stdout — session start, tool calls, approval requests, and session end — by
tailing `.trace.jsonl`. No engine changes are required for this.

If a risky assistant-mode bash command is blocked, the session pauses with an
approval request. Assistant-mode classifier coverage: `rm`, `git reset --hard`,
`git clean`, `git checkout --`, `chmod`, `chown`, plus `mv`/`cp` when any
positional path resolves outside the repo root. Measurement mode is unaffected.

```bash
python3 -m scripts.llm_assist show <session_id>
python3 -m scripts.llm_assist approve <session_id>
python3 -m scripts.llm_assist resume <session_id>
```

## Smoke Test

Use the built-in smoke command to bootstrap a throwaway repo, resolve the exact
served model id from `/v1/models`, and run one assistant session. A green smoke
run means: the assistant session exited successfully, `calc.py` contains the
fixed `return a + b` body, `tests/test_calc.py` passes, and no pending approval
is outstanding. Any of those failing produces a non-zero exit with the smoke
repo path, session id, artifact dir, final status, and finish reason.

```bash
python3 -m scripts.llm_assist smoke
HARNESS_ASSIST_HOME=/tmp/harness-assist-smoke python3 -m scripts.llm_assist smoke
```
