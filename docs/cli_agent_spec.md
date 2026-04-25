# CLI Agent Spec

## Purpose

Define the remaining work required to make `yuj` / `scripts.llm_assist` functionally
competitive with a terminal coding agent such as Claude Code or Codex CLI,
using the existing harness engine and only minimal additions.

This is an implementation spec, not a brainstorming note.

The goal is not feature breadth or product polish.
The goal is a **usable coding-agent CLI** that a human can invoke on a repo,
watch work happen, approve risky actions, inspect progress, resume sessions,
and trust the artifact trail.

## Read First

1. `docs/assistant_spec.md`
2. `docs/harness_spec.md`
3. `docs/separation_of_concerns.md`
4. `HANDOFF.md`

## Non-Negotiable Constraints

1. Keep one Python codebase.
2. Do not fork the harness loop.
3. Do not add a TypeScript shell, web app, dashboard, memory, history search,
   subagents, or gateway integrations.
4. Do not let assistant-shell features leak into `runtime.mode = "measurement"`.
5. Do not create a second config system. Reuse the existing config/knob surface.
6. Assistant artifacts stay outside the repo being edited.

## Current State

Already implemented:

- `run`
- `resume`
- `approve`
- `sessions`
- `show`
- `yuj` operator-facing wrapper
- `inspect knobs`
- `inspect presets`
- `smoke`
- session store
- turn renderer from `.trace.jsonl`
- assistant-only approval/suspend
- smoke bootstrap repo
- exact served model id resolution for `run` and `smoke`
- positional task text with current-cwd default for `code` / `run`
- latest-session defaults for `show` / `resume` / `approve`
- per-cwd active session pointers used before latest-session fallback
- single-owner session locking for active runs and resumes
- clean interrupted-session handling with resumable paused state
- concise model-resolution failure messages for unreachable local servers
- short deterministic session references via `session_ref` and unique prefixes
- startup banner before live trace output

Already good enough:

- artifact isolation
- session metadata persistence
- approval pause before execution for selected risky bash actions
- trace/state/checkpoint/metrics output
- test suite baseline

## Definition Of Done

The CLI is at the target level when all of the following are true:

1. A user can invoke one command against a repo and a prompt, with no manual
   model-id lookup, and the session starts reliably against the local server.
2. The user can see progress while the session is running, not only after it exits.
3. Risky actions pause cleanly with a clear approval message and can be resumed
   cleanly after approval.
4. `show` accurately reflects live state during running and resumed sessions.
5. A fresh smoke run completes end to end against a running local server.
6. A direct live repo task completes end to end against a running local server.
7. The default CLI ergonomics are good enough that a normal user does not need
   to read code or inspect `.trace.jsonl` manually.

## Acceptance Record

The items below were the targeted gaps for this shell and now serve as the
acceptance record for what has been implemented.

### 1. Exact Served Model Resolution For `run`

Problem:

- `smoke` resolves the exact served model id from `/v1/models`.
- `run` still stores the alias-resolved model string directly.
- In live use, the server may expose only a concrete model id such as
  `Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf`.

Required behavior:

- `yuj run` must resolve the exact served model id before creating the session.
- If `--model` is omitted:
  - start from the configured default model
  - query `/v1/models`
  - if the configured/default model exists exactly in the served list, use it
  - otherwise use the first served model id
- If `--model` is provided:
  - resolve friendly aliases first
  - then apply the same exact-served-id reconciliation
- The resolved exact model id must be what gets persisted in session metadata.
- This behavior must stay assistant-shell-only. Do not change measurement mode.

Implementation:

- Reuse the logic pattern already present in `resolve_smoke_model`.
- Refactor that logic into a shared helper used by both `run` and `smoke`.

Acceptance:

- A live `run` invocation succeeds when given an alias that is not literally
  present in `/v1/models`, as long as the server is up and serving at least one model.

### 2. Live Progress Rendering During `run` And `resume`

Problem:

- `run` and `resume` currently block until the session exits.
- Progress is available only by inspecting artifacts separately.
- For a coding-agent CLI, this is functionally weak even if artifacts are correct.

Required behavior:

- While a session is running, the CLI must print incremental progress to stdout.
- Minimum required live events:
  - session started
  - each tool call as it lands in `.trace.jsonl`
  - session end
  - approval request
- The rendering may be plain text. Rich styling is optional.
- Do not require engine token streaming.
- Do not change the harness loop protocol to push live messages.
- The shell must tail artifacts it already owns.

Implementation:

- Add a trace-follow helper in `scripts/llm_assist/runner.py` or a new
  assistant-shell module under `scripts/llm_assist/`.
- `cmd_run` and `cmd_resume` must:
  - start the session run
  - poll `.trace.jsonl`
  - print newly observed events in order
  - stop when the session reaches a terminal or paused state
- Use the existing trace formatting helpers as the canonical event renderer.

Acceptance:

- During a real live run, the user sees tool-call progress before the command exits.

### 3. Accurate Live Session State In `show`

Problem:

- `show` uses persisted session metadata plus artifact inspection.
- Live sessions can have stale `updated_at`, stale finish reason, or incomplete
  state if the process is still running or has resumed.

Required behavior:

- `show` must derive live status from artifacts when that is more accurate than
  the SQLite row.
- Status precedence:
  1. If an approval request exists with `status = pending`, show `approval_pending`.
  2. Else if the latest trace has a `session_end`, use its finish reason to map
     to `completed`, `paused`, or `error`.
  3. Else if a `session_start` exists without a later `session_end`, show `running`.
  4. Else fall back to the SQLite row.
- `show` must not display a stale old finish reason during an active resumed session.
- `show` must display the current session number inferred from trace events.

Implementation:

- Add a helper that infers live shell status from `.trace.jsonl` and
  `approval_request.json`.
- `cmd_show` must render that derived status instead of trusting the row blindly.

Acceptance:

- In a resumed approved session, `show` reports `running`, not the stale prior
  `approval_required` finish reason.

### 4. Approval Classifier Expansion

Problem:

- Current approval gating covers only a narrow set of destructive bash commands.
- Live coding use needs slightly broader coverage so the shell is trustworthy.

Required behavior:

- Expand the assistant-only risky-action detector to include:
  - `rm`
  - `git reset --hard`
  - `git clean`
  - `git checkout --`
  - `chmod`
  - `chown`
  - `mv` when the source or target is outside the current repo root
  - `cp` when the source or target is outside the current repo root
- Keep the classifier content-blind and syntax-based.
- Do not attempt a full shell security policy.
- Do not block safe in-repo file moves/copies by default.

Implementation:

- Extend the existing bash segment parser and approval detector in
  `scripts/llm_solver/harness/loop.py`.
- Reuse `_path_within_cwd` where possible.

Acceptance:

- Risky external or destructive file operations in assistant mode pause before execution.
- Measurement mode remains unaffected.

### 5. Single-Command Practical Entry Path

Problem:

- `run --cwd ... --prompt-text ...` is functional but verbose.
- A coding-agent CLI needs one obvious way to start work.

Required behavior:

- Add one shorthand entry path:
  - `yuj code "..."` from the current repo
  - or `yuj run --cwd <path> --prompt-text ...`
- This may be an alias to `run`.
- Do not remove `run`.
- Do not invent shell-like slash commands.

Implementation choice:

- Preferred name: `code`
- `code` must behave identically to `run`.

Acceptance:

- `code` and `run` produce identical session artifacts and behavior.

### 6. Stable Live Smoke Acceptance Command

Problem:

- `smoke` exists, but the repo still lacks a crisp acceptance contract for
  what “live smoke succeeded” means.

Required behavior:

- `smoke` success means all of the following:
  - assistant session exits with success
  - smoke repo file is fixed
  - relevant test passes
  - session ends cleanly without pending approval
- `smoke` failure output must print:
  - smoke repo path
  - session id
  - artifact dir
  - final status
  - final finish reason

Implementation:

- After `run_session`, `cmd_smoke` must assert the smoke repo outcome explicitly:
  - read `calc.py`
  - confirm it contains `return a + b`
  - optionally run `python3 -m pytest tests/test_calc.py -q`
- If those checks fail, return non-zero even if the harness session exited.

Acceptance:

- A green smoke command means the coding-agent path actually worked, not merely
  that a session artifact bundle was created.

### 7. Docs For Immediate Use

Problem:

- Operational docs exist, but the coding-agent parity target is split across files.

Required behavior:

- Keep this doc as the canonical “remaining functionality” spec.
- Update README usage examples only if command names or semantics change.
- Do not create product-marketing copy.

## Explicit Non-Goals

Do not add any of the following in response to this spec:

- web UI
- Textual TUI
- chat history search
- memory across sessions
- skills marketplace
- subagents
- remote execution
- daemon/server split
- websocket streaming
- benchmark integration

## File-Level Ownership

Expected write targets:

- `scripts/llm_assist/__main__.py`
- `scripts/llm_assist/runner.py`
- optionally one new helper module under `scripts/llm_assist/`
- `scripts/llm_solver/harness/loop.py`
- tests in already-whitelisted files only:
  - `tests/test_assist_store.py`
  - `tests/test_assist_resume.py`
  - `tests/test_runtime_mode.py`
  - `tests/test_harness_and_pipeline.py`

Do not add new top-level packages or new shell frameworks.

## Implementation Order

1. Exact served model resolution for `run`
2. Live progress rendering for `run` and `resume`
3. Accurate live-state derivation for `show`
4. Approval classifier expansion
5. `code` alias command
6. Strong smoke acceptance checks
7. Tests and docs refresh

## Acceptance Test Matrix

The implementation is not done until all rows below pass.

### Static / Unit

- `python3 -m pytest tests/ -q -p no:cacheprovider`
- `python3 -m pytest tests/test_harness_and_pipeline.py -q -p no:cacheprovider`
- `python3 -m pytest tests/test_assist_store.py -q -p no:cacheprovider`

### Live CLI

Assumes localhost server is up.

1. Normal fix task
   - Run against a fresh temp repo with a one-line bug.
   - Expected:
     - relevant source file read
     - relevant test file read
     - minimal edit
     - relevant test run
     - clean finish

2. Approval task
   - Run against a fresh temp repo with a `build/` directory.
   - Prompt asks to delete it.
   - Expected:
     - assistant pauses before `rm`
     - `show` exposes the pending request
     - `approve` updates the request
     - `resume` executes the delete

3. Smoke task
   - `yuj smoke`
   - Expected:
     - non-zero on real failure
     - zero only when repo fix and test success are both true

## Worker Execution Checklist

This section is written for a low-context coding agent.

Follow it in order.
Do not skip ahead.
Do not redesign the system.

### Global Rules

1. Read this file fully before editing anything.
2. Do not edit architectural docs unless a command name or behavior in them
   becomes false after implementation.
3. Do not add new dependencies.
4. Do not add new top-level packages.
5. Do not add a web UI, TUI, memory, history search, subagents, or daemon.
6. Do not change measurement-mode behavior.
7. Only edit the files explicitly named in this spec unless blocked.
8. If blocked by a missing file or forbidden import, stop and report it.

### Allowed Write Set

Primary:

- `scripts/llm_assist/__main__.py`
- `scripts/llm_assist/runner.py`
- optionally one new helper module under `scripts/llm_assist/`
- `scripts/llm_solver/harness/loop.py`

Tests:

- `tests/test_assist_store.py`
- `tests/test_assist_resume.py`
- `tests/test_runtime_mode.py`
- `tests/test_harness_and_pipeline.py`

Docs only after code/tests are green:

- `README.md`
- `HANDOFF.md`
- `docs/assistant_spec.md`
- this file

### Stop Conditions

Stop immediately and report if any of the following happens:

1. You think a solution requires changing `runtime.mode = "measurement"` behavior.
2. You think a solution requires a TypeScript shell, service split, or web UI.
3. You think a solution requires new top-level artifact locations inside the repo
   being edited instead of assistant-home artifacts.
4. You need to copy new files from the lab repo not already present here.
5. You need a second config system instead of existing config/knob plumbing.

### Task 1: Exact Served Model Resolution For `run`

Goal:

- Make `run` use the exact served model id, not just the alias/default string.

Files:

- `scripts/llm_assist/__main__.py`
- `scripts/llm_assist/runner.py`
- tests in `tests/test_assist_store.py`

Required edits:

1. Refactor the model resolution logic already used by `smoke` into a shared helper.
2. Make `run` call that helper before session creation.
3. Persist the exact resolved model id in the created session.
4. Preserve current `smoke` behavior.

Required tests:

- alias/default model not literally present in `/v1/models` falls back to first served model
- exact served model id is what gets persisted in session metadata

Do not proceed to Task 2 until:

- the new tests pass
- existing assistant-store tests still pass

### Task 2: Live Progress Rendering For `run` And `resume`

Goal:

- Print incremental progress while the session is running.

Files:

- `scripts/llm_assist/__main__.py`
- `scripts/llm_assist/runner.py`
- optionally one new helper under `scripts/llm_assist/`
- tests in `tests/test_assist_store.py`

Required edits:

1. Add a trace-follow helper that tails `.trace.jsonl`.
2. Make `run` print new events while the session is active.
3. Make `resume` print new events while the session is active.
4. Render at least:
   - session start
   - tool call
   - approval request
   - session end
5. Use existing event formatting helpers where possible.

Required tests:

- a stubbed running session emits printed progress before final result
- progress renderer does not duplicate old events already seen

Do not proceed to Task 3 until:

- the new tests pass
- existing full suite still passes

### Task 3: Accurate Live State In `show`

Goal:

- Make `show` derive live state from artifacts instead of stale SQLite fields.

Files:

- `scripts/llm_assist/__main__.py`
- `scripts/llm_assist/runner.py`
- tests in `tests/test_assist_store.py`

Required edits:

1. Add a helper that derives live state from:
   - `.trace.jsonl`
   - `approval_request.json`
   - session metadata row as fallback
2. Implement the exact precedence defined earlier in this spec.
3. Ensure resumed sessions do not show stale previous finish reasons.
4. Surface the current inferred session number in `show`.

Required tests:

- pending approval shows `approval_pending`
- resumed running session shows `running`
- completed session still shows final finish reason

Do not proceed to Task 4 until:

- the new tests pass

### Task 4: Expand Approval Classifier

Goal:

- Broaden risky-action coverage slightly without becoming a shell security project.

Files:

- `scripts/llm_solver/harness/loop.py`
- tests in `tests/test_harness_and_pipeline.py`

Required edits:

1. Keep current risky-action detection.
2. Add gating for:
   - `mv` outside repo root
   - `cp` outside repo root
3. Keep safe in-repo `mv` and `cp` allowed.
4. Keep all approval logic assistant-only.

Required tests:

- assistant-mode risky external `cp` pauses
- assistant-mode risky external `mv` pauses
- safe in-repo `cp` or `mv` does not pause
- measurement mode still does not use approval pause

Do not proceed to Task 5 until:

- the new tests pass

### Task 5: Add `code` Alias Command

Goal:

- Provide one obvious coding-agent entry command besides `run`.

Files:

- `scripts/llm_assist/__main__.py`
- tests in `tests/test_assist_store.py`

Required edits:

1. Add `code` as a subcommand alias to `run`.
2. It must accept the same flags and produce identical behavior.
3. Do not remove or rename `run`.

Required tests:

- `code --help` works
- `code` routes to the same logic as `run`

Do not proceed to Task 6 until:

- the new tests pass

### Task 6: Strengthen `smoke` Acceptance

Goal:

- Make `smoke` fail unless the coding task actually succeeded.

Files:

- `scripts/llm_assist/__main__.py`
- `scripts/llm_assist/runner.py`
- tests in `tests/test_assist_store.py`

Required edits:

1. After the session exits, check the smoke repo result explicitly.
2. Minimum checks:
   - source file contains the fixed implementation
   - relevant test passes
   - no pending approval exists
3. On failure, print:
   - smoke repo path
   - session id
   - artifact dir
   - final status
   - final finish reason
4. Return non-zero when any acceptance check fails.

Required tests:

- stubbed successful session but unfixed repo returns non-zero
- fixed repo and passing test returns zero
- pending approval state returns non-zero

Do not proceed to Task 7 until:

- the new tests pass

### Task 7: Final Test Pass

Run all of these:

```bash
python3 -m pytest tests/test_assist_store.py -q -p no:cacheprovider
python3 -m pytest tests/test_harness_and_pipeline.py -q -p no:cacheprovider
python3 -m pytest tests/ -q -p no:cacheprovider
```

If any fail:

- fix code or tests
- rerun all three

Do not touch docs before this step is green.

### Task 8: Minimal Docs Refresh

Only after Task 7 is green:

1. Update `README.md` command examples if needed.
2. Update `HANDOFF.md` current-state and workflow sections if needed.
3. Update `docs/assistant_spec.md` only if command surface or behavior changed.
4. Do not add narrative, product copy, or design discussion.

### Final Delivery Format

The worker must report exactly:

1. Which tasks were completed.
2. Which files were changed.
3. Which tests were run.
4. Whether live localhost validation was attempted.
5. Any remaining limitation that still blocks the full Definition Of Done.

## Practical Rule

If a proposed addition makes the shell easier to drive as a coding agent without
forking the loop or adding product sprawl, it belongs here.

If it adds product mass more than coding-agent capability, it is out of scope.
