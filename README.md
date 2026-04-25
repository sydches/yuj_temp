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
yuj status
yuj current
yuj show
yuj approve
yuj resume
```

`yuj` is the operator-facing command. The repo also includes `./yuj` and
`python3 -m scripts.yuj` as equivalent entrypoints if you are running from the
checkout directly.

`code` is an alias of `run` for the common coding-agent entry path.

Install as a real CLI entrypoint:

```bash
python3 -m pip install -e . --no-build-isolation
yuj --help
yuj setup
```

## Practical CLI Flow

```bash
yuj setup

yuj inspect presets
yuj inspect knobs runtime

yuj code "Fix the failing test, run the relevant test, then finish."
yuj code --cwd /path/to/other/repo "Fix the failing test, run the relevant test, then finish."
yuj code --provider openai --model gpt-5.4 "Fix the failing test, run the relevant test, then finish."
yuj code --provider anthropic --model claude-sonnet-4-5 "Fix the failing test, run the relevant test, then finish."
yuj code --provider openrouter --model anthropic/claude-sonnet-4.5 "Fix the failing test, run the relevant test, then finish."
yuj run --prompt-file task.txt

yuj sessions
yuj status
yuj current
yuj show
yuj resume
```

Assistant artifacts default to `<project>/.llm_assist/`. To keep them elsewhere
for smoke runs or temporary sessions, set `HARNESS_ASSIST_HOME=/tmp/...`.

First-run setup:

- `yuj setup` prompts for local LLM settings or hosted provider credentials and
  writes `config.local.toml`
- interactive `yuj` with no arguments launches setup first if `config.local.toml`
  is missing
- the first interactive `yuj code`, `yuj run`, or `yuj smoke` offers setup when
  no local config exists
- `config.local.toml` is gitignored and machine-local
- scriptable setup works with `yuj setup --provider openai --model <id>
  --api-key-env OPENAI_API_KEY`

Ease-of-use defaults:

- `yuj code "..."` runs against the current directory by default
- positional prompt text is accepted for the common path; `--prompt-text` and
  `--prompt-file` still work
- each repo cwd keeps an active session pointer
- `yuj show`, `yuj resume`, and `yuj approve` resolve the active session for
  the current repo first, then fall back to the latest relevant session
- `yuj sessions` prints a compact table with `ref` and `flags` columns
- active/current/locked are surfaced as `active`, `cwd`, and `locked` flags
- explicit session references accept the full id, the 8-char `session_ref`,
  or any unique prefix
- `yuj status [session_ref]` gives a concise state/next-action view
- `yuj current` is a fast alias for `yuj status latest`
- `--provider` accepts `local`, `openai`, `anthropic`, `zai`, `openrouter`,
  and `custom`; provider settings are persisted with the session so `resume`
  uses the same supplier
- secrets are referenced by env var (`--api-key-env NAME`) rather than stored
  in the session database

`run` and `smoke` reconcile the model against `/v1/models` at session creation.
For local/default runs, alias-resolved ids that are not served verbatim fall
back to the first served id. For hosted provider runs with explicit `--model`,
the requested model id is honored even if the provider's model-list endpoint is
incomplete. The resolved model id is what gets persisted in session metadata.

Provider defaults:

| Provider | Base URL | Key env |
|---|---|---|
| `local` | configured `[server].base_url` | configured `[server].api_key` |
| `openai` | `https://api.openai.com/v1` | `OPENAI_API_KEY` |
| `anthropic` | `https://api.anthropic.com/v1` | `ANTHROPIC_API_KEY` |
| `zai` | `https://api.z.ai/api/paas/v4` | `ZAI_API_KEY` |
| `openrouter` | `https://openrouter.ai/api/v1` | `OPENROUTER_API_KEY` |
| `custom` | pass `--base-url` | pass `--api-key-env` or use config |

While a session is running, `run` and `resume` print incremental progress to
stdout — session start, tool calls, approval requests, and session end — by
tailing `.trace.jsonl`. No engine changes are required for this.

Each session now prints an immediate startup banner with the session id, cwd,
model, artifact path, and served model list before the live trace begins, so
the operator does not need to wait for completion just to learn what started.

Completed runs print a compact summary (changed files + last observed test
command/outcome) derived from trace events.

Only one terminal may own a session at a time. If you try to resume a locked
session from another terminal, `yuj` refuses cleanly and shows the owning pid,
host, and acquisition time. `yuj show` also prints the current lock state.

If you interrupt `yuj` with `Ctrl-C`, the session is marked `interrupted`,
left resumable, and surfaced by `yuj show` as a paused session rather than a
ghost running session.

If the local model server cannot be reached during model resolution, `yuj`
fails with a short operator message naming the base URL and `/v1/models`
instead of dumping a Python traceback.

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
