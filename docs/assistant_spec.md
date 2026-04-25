# Assistant Spec

## Purpose
Make the harness into a usable agent without corrupting measurement mode.

Companion docs:

- `docs/harness_spec.md`
- `docs/coupling_spec.md`
- `docs/public_repo_spec.md`

This doc answers Q2 directly: yes, we can build the parts required to make this a car,
but the minimum car is much smaller than Hermes.

Design rule: **same engine, separate cabin**.

Hermes is not the blueprint. It is a comparison point for consumer shell features only.
This harness remains identity-first: practical, composable, generalized, and hostile to
junk context.

## Practical Stack

The assistant shell should use the same core stack as the harness unless there is a
clear failure forcing a change:

- Python
- TOML + dataclass config
- SQLite
- Rich

Recommended libraries:

- `Typer` for the assistant CLI entrypoint
- `Rich` for turn rendering, status, approvals, and session inspection

Not recommended for V1:

- TypeScript shell over a Python engine
- Textual-first UI
- FastAPI/web dashboard as the primary interface
- service boundaries between shell and engine

Why:

- one language keeps the extension seams clean
- one process keeps the runtime simple and debuggable
- SQLite is enough for sessions, approvals, and later local FTS
- Rich gives enough terminal UX without turning the shell into a framework project

## Required V1 Parts

| Part | Why it is required | Placement |
|---|---|---|
| Separate assistant entrypoint | A real agent needs a human entrypoint that is not the batch/benchmark runner. | `scripts/llm_assist/` |
| Session store and resume | Users need `new`, `resume`, and `list`, not run-dir archaeology. | assistant shell |
| Direct task substrate | Assistant sessions must start from `cwd + prompt`, not from a benchmark-shaped run directory. | assistant shell using `TaskSpec`-style inputs |
| Turn UI | CLI first. Show model turns, tool calls, and final results. Trace tailing is enough; token streaming is optional cosmetics. | assistant shell |
| Approval channel | Destructive or sensitive actions need a user-facing suspend/approve path in assistant mode. | assistant shell + one new harness suspend action |
| Queryable control plane | The user needs one place to inspect knobs, presets, profile choice, and consequences. Reuse `scripts/knob.py`; do not invent a second config surface. | assistant shell |
| Mode gate | `measurement` and `assistant` must be enforced at session start so memory, approvals, and UI features cannot leak into experiments. | harness runtime check |

## Not Required For V1

These are good-looking-car features, not minimum-car features:

- Persistent cross-session memory
- History search
- Personalities and skills
- Multi-backend execution
- Subagents
- Messaging gateways
- Cron / scheduler
- Web dashboard
- RL trajectory export

Ship them only after the minimum shell is stable.

Several of them should be treated as suspect by default, not as missing prestige items:

- persistent memory
- subagents
- streaming-first interaction
- gateway/dashboard sprawl

They make a product look larger. They do not automatically make the engine better.

## Required Commands

Human-facing command name:

- `yuj`

Internal module path:

- `scripts.llm_assist/`
- repo-local wrappers: `./yuj` and `python3 -m scripts.yuj`

Minimum queryable surface:

| Command | Effect |
|---|---|
| `yuj code "..."` | Start a session in the current directory from positional task text |
| `yuj run --cwd <path> --prompt-file <file>` | Start from a prompt file |
| `yuj run --cwd <path> --prompt-text ...` | Explicit flag-based start path |
| `yuj resume [session_id]` | Resume an existing session; default is latest resumable session |
| `yuj sessions` | List sessions with cwd, model, and latest status |
| `yuj inspect knobs [query]` | Search/describe knobs via `scripts/knob.py` |
| `yuj inspect presets` | Show curated presets |

Everything else is optional.

Implemented and useful in the current staging shell:

| Command | Effect |
|---|---|
| `yuj show [session_id]` | Show status, paths, recent turns, and trace tail; default is latest session |
| `yuj approve [session_id]` | Approve a pending risky action; default is latest pending approval |
| `yuj smoke` | Bootstrap a throwaway repo, resolve the exact served model id, run one end-to-end assistant session, and assert repo fix + tests pass + no pending approval |

Practical notes:

- `run` and `smoke` both reconcile the requested model against `/v1/models`
  via a shared `resolve_served_model` helper. Alias-resolved ids that are not
  served verbatim fall back to the first served id. The exact served id is
  persisted in session metadata.
- `code` and `run` accept positional task text and default `--cwd` to the
  current working directory for the common path.
- `run` and `resume` print live tool-call progress to stdout while the session
  runs by tailing `.trace.jsonl`. No engine changes are required.
- `run`, `resume`, and `smoke` print a startup banner with session id, cwd,
  model, artifact path, and served-model information before the live trace.
- `show` derives live status from `.trace.jsonl` + `approval_request.json`
  (approval_pending / running / completed / paused / error / fallback) instead
  of trusting the SQLite row blindly, so resumed running sessions do not show
  stale prior finish reasons.
- `show`, `resume`, and `approve` accept no session id and default to the
  latest relevant session, preferring the current repo cwd before the global
  latest session list.
- Assistant-mode approval classifier covers: `rm`, `git reset --hard`,
  `git clean`, `git checkout --`, `chmod`, `chown`, plus `mv`/`cp` when any
  positional path resolves outside the repo root. Measurement mode is
  unaffected.

## Session Layout

Assistant mode keeps its own artifacts outside the repo being edited.

Default root:

`<project-root>/.llm_assist/`

Required contents:

- `sessions.sqlite3` — session index
- `sessions/<session_id>/prompt.txt`
- `sessions/<session_id>/.trace.jsonl`
- `sessions/<session_id>/.solver/state.json`
- `sessions/<session_id>/checkpoint.json`
- `sessions/<session_id>/metrics.json`
- `sessions/<session_id>/transcript.log`
- `sessions/<session_id>/savings.jsonl`

This is load-bearing. Assistant sessions must not dump replay artifacts into the
repo they are editing.

Operational note:

- `HARNESS_ASSIST_HOME=/tmp/...` can be used to relocate assistant artifacts for
  smoke runs or temporary sessions.

## Build Order

1. Harden the engine first.
   Ship the core gaps from `docs/harness_spec.md`: exit-code semantics, empty-output confirmation, secret redaction, injection scanning, multi-file patching, duplicate-read ladder.

2. Add the mode gate.
   `measurement` remains default and locked down. `assistant` explicitly enables shell-only behavior.

3. Create `scripts/llm_assist/`.
   Add `run`, `resume`, `sessions`, and `inspect` commands. Use the existing engine and artifacts; do not fork the loop.
   Separate `work_dir` from `artifact_dir` so session state lives under `.llm_assist/`.

4. Add approval support.
   Assistant mode gets suspend/approve for risky actions. Measurement mode rejects that path.

5. Add a simple CLI renderer.
   Render completed turns and tool activity from the trace. Do not make engine streaming a prerequisite.

6. Reassess after V1.
   Only then decide whether memory, history search, or personalities are worth the added surface area.

Current staging status:

- `show` exists and renders recent turns plus trace tail.
- `approve` exists and unblocks assistant-only risky actions after explicit user approval.
- `smoke` exists and is suitable for local end-to-end checks when the server is up.
- `yuj` is the intended operator-facing command name.

## Non-Negotiable Constraints

- No benchmark vocabulary in the assistant shell.
- No shell feature may change measurement-mode loop behavior.
- No second config system; the knob catalog remains the source of truth.
- No large-output slurping followed by summarization as the primary strategy.
  Output admission control stays upstream.

## Short Answer

Hermes is a reference for what a consumer shell can contain.
It is not the blueprint for this system, and cloning it would be a category error.

For this repo, the minimum car is:
assistant entrypoint, session lifecycle, approval path, and a queryable control plane,
all wrapped around the existing engine.
