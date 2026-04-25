# Harness Spec

## Purpose
Canonical statement of what belongs in the harness core and what does not.

Companion docs:

- `docs/assistant_spec.md`
- `docs/coupling_spec.md`
- `docs/public_repo_spec.md`

This doc answers Q1 directly: what else belongs in the harness column.
It is not a wishlist. The goal is to compare the right layers and defend
the harness identity:

- **Harness core**: reusable engine behavior we would keep without FeatureBench.
- **Assistant shell**: user-facing control plane that can wrap the engine later.
- **Explicit non-goals**: consumer trim that should not be misfiled as missing core.

Objective conclusion from `.private/hermes_analysis.md` plus the other agent notes:
Hermes is relevant, but mostly as a **comparison foil for shell/control-plane concerns**.
It is not the target architecture. Much of its visible surface is product packaging,
distribution, and marketing-layer breadth rather than loop quality.

## Identity Correction

This harness should not become a Hermes clone.

For this repo's priorities, the harness is already more practical on the parts that matter:

- model and tool-call quirk absorption
- output admission control before context bloat happens
- clean code/config separation
- composable extension seams
- deterministic artifacts, replay, and analysis
- task-agnostic runtime behavior rather than product-specific theatrics

Hermes is useful mainly as a reminder of what a consumer shell can contain.
It is not evidence that those features improve the engine.

## Practical Stack

The practical implementation stack for the harness and its first assistant shell
should stay deliberately small:

- Python for the engine and the assistant shell
- TOML + dataclass config as the only config surface
- SQLite for session metadata and later local search
- Rich for terminal rendering
- pytest for verification

V1 should remain one Python codebase and one process. No service split, no
TypeScript wrapper, no web-first stack.

Why this stack:

- matches the current codebase and extension seams
- keeps plugin boundaries in one language
- avoids schema duplication across Python and TypeScript
- makes config, tools, guardrails, and artifacts easy to evolve together
- preserves composability and low-friction development

## Required Third-Column Rows

| Row | What the harness column should say | Status |
|---|---|---|
| Output admission control | `bash_quirks` + `language_quirks` + `harness/tools.py` decide what output is allowed into context: rewrites, filtering, search pagination, read reminders, truncation, optional structured projection and sink-and-surface. Philosophy: remove junk before it reaches the model. | Present |
| Edit reliability contract | Strict `edit`/`write` surface, fuzzy replacer cascade, candidate surfacing on miss, optional post-edit checks with revert/block. | Present |
| Guardrail architecture | Phase-indexed guardrail registry with explicit precedence and finish reasons; session behavior is code, not prompt text. | Present |
| Artifact and replay contract | `.trace.jsonl` is append-only ground truth; `.solver/state.json` is a projection; `state_replay.py`, `state_verify.py`, `run_summary.py`, and `.savings.jsonl` make runs inspectable. | Present |
| Extension seams | Profiles, normalize/denormalize rules, bash quirks, language quirks, injections, context strategies, and the tool registry are separate extension surfaces. | Present |
| Knob/control plane | `config.toml`, overlays, `configs/knobs.toml`, and `scripts/knob.py` define a queryable tuning surface with presets, tags, blast radius, and mode metadata. | Present but not exposed through a user-facing entrypoint |
| Mode boundary | The harness needs an explicit runtime split between `measurement` and `assistant`, so shell features cannot silently leak into scientific runs. | Documented, not yet the full runtime contract |
| Execution substrate | Current runtime is a single local sandboxed engine. Multi-backend execution is not part of the current core identity. | Present by design |
| Assistant shell boundary | No built-in cabin yet: no first-class chat entrypoint, approval UX, session browser, or end-user control plane. This is absence by design, not evidence of a weak engine. | Missing, intentionally outside the loop today |

## Core Gaps That Still Belong In The Harness

Only the items below are necessary additions to the core. Everything else should stay in the shell or remain out of scope.

| Item | Why it is necessary | Placement |
|---|---|---|
| Exit-code semantic annotation | Prevents wasted turns on non-errors like `grep` exit 1 and `diff` exit 1. This is output hygiene, not UX. | `bash_quirks` + `harness/tools.py` |
| Empty-output confirmation | Blank observations cause pointless re-runs. A successful no-output command should say so explicitly. | `harness/tools.py` |
| Secret redaction on tool output and artifacts | Credentials should never enter prompt context, trace, or state projection in raw form. | `harness/tools.py` + artifact writers |
| Prompt-injection scan for loaded text | Any loaded markdown or profile fragment is untrusted input once assistant-mode memory/injections exist. Reject bad content at load time. | profile/injection loaders |
| Multi-file patch tool | Cross-file fixes currently pay unnecessary turn tax. A patch DSL is a general edit primitive, not benchmark-specific logic. | `harness/tools.py` + parser module |
| Duplicate-read ladder | Re-reading the same file without mutation is a direct no-progress signal and should be handled in-loop, not only in post-hoc analysis. | `harness/guardrails.py` |

## Not Core Gaps

These should not be added to the harness column as if they were missing engine parts:

- TUI, slash commands, streaming UI, messaging gateways
- Cron, web dashboard, voice, mobile clients
- Cross-session memory, history search, personalities, skills marketplace
- Subagents, ensemble routing, RL export
- Multi-backend infrastructure beyond what the assistant shell actually needs

Those are shell or product-surface features. Useful later, but not part of the core comparison.

Some are also active risks to the harness identity when treated as defaults:

- persistent memory that changes behavior across sessions
- streaming and interrupt paths that weaken replayability
- subagents and ensemble behavior that break single-loop interpretability
- broad gateway/dashboard surface that adds product mass without improving the core loop

## Practical Rule

If a feature's main job is to keep bad bytes out of context, keep the loop reproducible,
or preserve clean code/config separation, it belongs in the harness.

If its main job is to make the system pleasant for a human to drive, it belongs in the
assistant shell.
