# Separation Of Concerns

This repo has four runtime-adjacent layers. Keep them separate.

## 1. Harness Core

Owns the model loop itself:

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

- benchmark identities
- task-suite-specific patches
- evaluator rules
- campaign bookkeeping

## 2. Assistant Shell

Owns the human-facing cabin:

- `llm_assist run`
- `llm_assist resume`
- `llm_assist sessions`
- `llm_assist inspect ...`
- session metadata and resume index

The assistant shell may add UX and persistence, but it must reuse the same
engine and artifact contract.

It does not get to fork the loop.

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
- environment setup -> harness core
- post-run analysis -> harness artifacts

Forbidden:

- harness core -> assistant shell
- harness core -> benchmark/eval pipeline
- assistant shell -> benchmark/eval pipeline
- harness core -> post-run analysis logic

## Practical Test

Ask one question:

`Would this still make sense if the task were a local repo, not a benchmark run?`

If yes, it probably belongs in the harness core or assistant shell.
If no, it belongs in environment/setup, analysis, or a separate lab repo.
