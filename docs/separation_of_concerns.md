# Separation Of Concerns

This repo has four runtime-adjacent layers. Keep them separate.

This boundary is also the update contract for pulling in newer harness code.
The harness can be replaced or refreshed from upstream as long as the public
interface below stays intact; the wrapper must remain an adapter around that
interface, not a fork of the loop.

## 1. Harness Core

Owns the model loop itself:

- package path: `scripts/llm_solver/**`
- tool surface
- tool-output admission control
- guardrails
- context strategies
- profile normalize/denormalize behavior
- trace, state, savings, checkpoint artifacts

The harness core must stay task-agnostic.

It can know:

- what tool was called
- whether a subprocess exited non-zero
- how to truncate, rewrite, or sink output
- how to project its own trace into state

It must not know:

- `scripts/llm_assist`
- `scripts.yuj`
- benchmark identities
- task-suite-specific patches
- evaluator rules
- campaign bookkeeping

## 2. Assistant Shell

Owns the human-facing cabin:

- package path: `scripts/llm_assist/**`
- wrapper entrypoints: `yuj` and `scripts/yuj.py`
- `llm_assist run`
- `llm_assist resume`
- `llm_assist sessions`
- `llm_assist inspect ...`
- session metadata and resume index

The assistant shell may add UX and persistence, but it must reuse the same
engine and artifact contract.

It does not get to fork the loop.

The assistant shell may import these core interfaces:

- `scripts.llm_solver.config`
- `scripts.llm_solver.harness.TaskSpec`
- `scripts.llm_solver.harness.solve_task`
- `scripts.llm_solver.harness.context_strategies`
- `scripts.llm_solver.models`
- `scripts.llm_solver.server`
- shared artifact readers where no separate public helper exists

Any additional core import should be treated as a boundary decision. If the
wrapper needs a new capability, prefer adding a small reusable core helper
instead of reaching into shell-specific state from the harness.

The assistant shell owns all of this and it must not be required by measurement
mode:

- session database and active-session pointers
- live trace rendering
- approvals and session-local approval decisions
- provider setup UX
- `.llm_assist/` artifact root selection

## 3. Environment Setup

Owns everything required to hand the harness a ready working directory:

- workspace selection
- optional pretest script
- optional system prompt file
- container or interpreter setup, if any

The harness runs the provided pretest script. It does not know how that script
was produced.

## 4. Post-Run Analysis

Owns offline inspection of produced artifacts:

- replay
- trace inspection
- coherence/thrash analysis
- denormalization audits
- summaries and comparisons

Analysis may be task-aware because it runs **after** the session completes.
That task-awareness must not leak back into the live loop.

## Dependency Rule

Allowed:

- assistant shell -> harness core
- wrapper entrypoint -> assistant shell
- environment setup -> harness core
- post-run analysis -> harness artifacts

Forbidden:

- harness core -> assistant shell
- harness core -> wrapper entrypoint
- harness core -> benchmark/eval pipeline
- assistant shell -> benchmark/eval pipeline
- harness core -> post-run analysis logic

## Harness Refresh Rule

When bringing in newer harness code, update only the harness-owned paths first:

- `scripts/llm_solver/**`
- `profiles/**`
- `configs/knobs.toml`
- harness docs and tests that describe core behavior

Do not overwrite these wrapper-owned paths as part of a harness refresh:

- `scripts/llm_assist/**`
- `scripts/yuj.py`
- `yuj`
- assistant tests
- `.llm_assist/**`

After the refresh, run the boundary tests before fixing functional fallout:

```bash
python3 -m pytest tests/test_runtime_mode.py
```

If the refreshed harness needs a new shell-facing hook, add it to the core as a
task-agnostic helper and document the interface here. Do not let the harness
call back into `scripts.llm_assist`.

## Practical Test

Ask one question:

`Would this still make sense if the task were a local repo, not a benchmark run?`

If yes, it probably belongs in the harness core or assistant shell.
If no, it belongs in environment/setup, analysis, or a separate lab repo.
