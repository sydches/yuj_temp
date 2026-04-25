# Handoff

Last updated: 2026-04-24

## Purpose

This repo is the clean public staging extract of the harness. It is meant to be
built forward from here, not recontaminated from the lab repo.

Primary constraints:

- keep one Python codebase
- keep config/code separation clean
- keep output admission control upstream
- keep assistant artifacts outside the repo being edited
- do not reintroduce benchmark shell, TS wrapper, web-first UI, memory, subagents, or dashboard sprawl

## Read First

1. `README.md`
2. `AGENTS.md`
3. `docs/harness_spec.md`
4. `docs/assistant_spec.md`
5. `docs/coupling_spec.md`
6. `docs/public_repo_spec.md`

## What Exists

- core runtime under `scripts/llm_solver/`
- minimum assistant shell under `scripts/llm_assist/`
- knob query CLI at `scripts/knob.py`
- public-facing docs under `docs/`
- selected profiles under `profiles/`
- copied test suite under `tests/`

Assistant shell commands currently implemented:

- `python3 -m scripts.llm_assist run --cwd ... --prompt-text ...`
- `python3 -m scripts.llm_assist code --cwd ... --prompt-text ...` (alias of `run`)
- `python3 -m scripts.llm_assist smoke`
- `python3 -m scripts.llm_assist resume <session_id>`
- `python3 -m scripts.llm_assist approve <session_id>`
- `python3 -m scripts.llm_assist sessions`
- `python3 -m scripts.llm_assist show <session_id>`
- `python3 -m scripts.llm_assist inspect knobs [query]`
- `python3 -m scripts.llm_assist inspect presets`

## Verified Working

- `python3 -m scripts.llm_solver --help`
- `python3 -m scripts.llm_assist --help`
- `python3 -m scripts.llm_assist inspect presets`
- `python3 -m scripts.llm_assist inspect knobs runtime`
- `python3 -m scripts.llm_assist show --help`
- `python3 -m scripts.llm_assist approve --help`
- `python3 -m scripts.llm_assist smoke --help`
- assistant session store + resume metadata
- assistant artifact isolation under `.llm_assist/`-style storage
- assistant session detail + trace-tail inspection
- compact turn renderer in `llm_assist show`
- assistant-only approval/suspend for risky bash actions (now includes
  `mv`/`cp` when any positional path resolves outside the repo root)
- `run` and `smoke` both resolve the exact served model id via
  `resolve_served_model` and persist it in session metadata
- incremental progress rendering during `run` and `resume` via a
  background trace follower (`scripts/llm_assist/progress.py`)
- `show` derives live status from `.trace.jsonl` + `approval_request.json`
  (approval_pending / running / completed / paused / error) instead of
  trusting the SQLite row blindly
- `smoke` acceptance checks: `calc.py` contains the fix, `tests/test_calc.py`
  passes, and no pending approval is outstanding; failure prints the repo
  path, session id, artifact dir, final status, and finish reason
- live approval/pause path verified against a running local model server
- standalone invocation of `tests/test_harness_and_pipeline.py` works again

Recent important fixes already present here:

- `runtime.mode` gate in `scripts/llm_solver/config.py`
- assistant `work_dir` vs `artifact_dir` split in `scripts/llm_solver/harness/loop.py`
- adaptive policy test-signal fix in `scripts/llm_solver/harness/loop.py`
  successful test runs without `[exit code: ...]` now count as test signal

## Current Known Gaps

1. The smoke command is wired and tested with stubs, but live localhost access
   still depends on the actual local server being up and reachable from the
   current environment. Live runs are also noticeably slower than the test
   suite, so expect multi-second pauses between turns on the local model.

2. Approval coverage is intentionally narrow.
   It currently gates obviously risky bash actions (`rm`, destructive git
   cleanup/reset, `chmod`, `chown`, `mv`/`cp` with paths outside the repo root)
   rather than a full shell-risk classifier.

3. The shell is still CLI-first.
   There is no richer TUI or dashboard, by design.

## Known Traps

- Do not put assistant traces/state/checkpoints inside the repo being edited.
  They belong under `.llm_assist/`.

- Do not copy more files from the lab repo unless they are explicitly whitelisted.
  If something is missing, update `docs/public_repo_spec.md` first.

- Do not infer model reachability from aliases.
  Example from live verification: the server served `Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf`, not `qwen3.5-9b`.

- For local server checks from this Codex environment, sandboxed commands may fail to reach localhost.
  If a localhost probe fails, retry with escalation before concluding the server is down.

## Ready-Now Workflow

1. Verify the CLI surface:
   `python3 -m scripts.llm_assist --help`

2. Inspect available knobs/presets:
   `python3 -m scripts.llm_assist inspect presets`
   `python3 -m scripts.llm_assist inspect knobs runtime`

3. Run the built-in smoke task if the local server is up:
   `python3 -m scripts.llm_assist smoke`

4. Start a real repo session:
   `python3 -m scripts.llm_assist run --cwd <repo> --prompt-text "..."`
   If alias resolution is ambiguous, pass the exact served model id.

5. Inspect or resume sessions:
   `python3 -m scripts.llm_assist sessions`
   `python3 -m scripts.llm_assist show <session_id>`
   `python3 -m scripts.llm_assist resume <session_id>`

6. If a risky action is paused for approval:
   `python3 -m scripts.llm_assist approve <session_id>`
   then resume the session.

## Useful Commands

```bash
cd /home/syd/projects/harness-public-staging

python3 -m scripts.llm_solver --help
python3 -m scripts.llm_assist --help
python3 -m scripts.llm_assist inspect presets
python3 -m scripts.llm_assist inspect knobs runtime
python3 -m scripts.llm_assist smoke

python3 -m pytest tests/ -q -p no:cacheprovider
python3 -m pytest tests/test_harness_and_pipeline.py -q -p no:cacheprovider
```

Live server checks:

```bash
curl -sS http://localhost:8080/v1/models
curl -sS http://localhost:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"<exact-served-id>","messages":[{"role":"user","content":"Reply with ok"}],"max_tokens":16}'
```

Built-in assistant smoke:

```bash
python3 -m scripts.llm_assist smoke
HARNESS_ASSIST_HOME=/tmp/harness-assist-smoke python3 -m scripts.llm_assist smoke
```

Direct exact-model run:

```bash
python3 -m scripts.llm_assist run \
  --cwd /path/to/repo \
  --model Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf \
  --prompt-text "Fix the failing test, run the relevant test, then finish."
```

## If You Need The Lab Repo

The lab/source repo is:

`/home/syd/projects/yuj-public`

Do not copy from it casually. Treat this staging repo as the product candidate and the lab repo as reference material only.
