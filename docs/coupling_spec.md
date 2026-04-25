# Coupling Spec

## Purpose
Classify the current repo's surfaces by architectural ownership so the public
extraction does not drag the lab along with it.

This doc accompanies:

- `docs/harness_spec.md`
- `docs/assistant_spec.md`

This is the boundary document for extraction work.

## Non-Negotiable Rules

1. **Copy by whitelist, never by blacklist.**
2. **Phase 1 preserves internal paths.**
   Keep `scripts/llm_solver/`, `profiles/`, and current import paths unchanged
   until the extracted repo is green.
3. **`bwrap` is core.**
   It is a runtime primitive, not FeatureBench coupling.
4. **FeatureBench, fb-eval, DOE, and campaign bookkeeping are not core.**
5. **Artifact-centric analysis may survive.**
   Benchmark-centric analysis does not.

## Coupling Classes

| Class | Meaning | Public repo action |
|---|---|---|
| `core` | Runtime machinery we would keep without FeatureBench | copy |
| `reusable-analysis` | Post-hoc tools that reason over harness artifacts | copy selectively |
| `rewrite` | Useful surface, but current file is benchmark- or lab-shaped | rewrite, do not copy verbatim |
| `benchmark-shell` | Exists to prepare/evaluate/score benchmark runs | leave behind |
| `private-sediment` | Local caches, notes, junk, archived experiments | leave behind |

## Path Classification

| Path / glob | Class | Action | Notes |
|---|---|---|---|
| `scripts/llm_solver/harness/**` | `core` | copy | loop, tools, sandbox, context, guardrails, state writer |
| `scripts/llm_solver/server/**` | `core` | copy | transport boundary + profile loading |
| `scripts/llm_solver/_shared/**` | `core` | copy | content-blind shared helpers |
| `scripts/llm_solver/bash_quirks/**` | `core` | copy | output admission control |
| `scripts/llm_solver/language_quirks/**` | `core` | copy | runner/task-format descriptors |
| `scripts/llm_solver/models/**` | `core` | copy | local model registry/helpers |
| `scripts/llm_solver/profiles/**` | `core` | copy selectively | profile tooling and verification helpers |
| `scripts/llm_solver/__main__.py` | `core` | copy | current entrypoint |
| `scripts/llm_solver/config.py` | `core` | copy | config loader and dataclass |
| `scripts/__init__.py` | `core` | copy | package root for `python -m scripts.llm_solver` |
| `profiles/_base/**` | `core` | copy | base profile system |
| `profiles/glm-4-flash/**` | `core` | copy | current live profile |
| `profiles/qwen3-8b-q4/**` | `core` | copy | current live profile |
| `profiles/qwen3.5-9b/**` | `core` | copy | current live profile |
| `profiles/qwen3.5-9b-q6k/**` | `core` | copy | current live profile |
| `profiles/qwen3.6-35b-a3b/**` | `core` | copy | current live profile |
| `profiles/qwen3.6-35b-a3b-q5/**` | `core` | copy | current live profile |
| `profiles/deprecated/**` | `private-sediment` | leave behind | dead surface |
| `scripts/llm_solver/analysis/coherence.py` | `reusable-analysis` | copy | artifact-based thrash analysis |
| `scripts/llm_solver/analysis/compare.py` | `reusable-analysis` | copy | generic run comparison |
| `scripts/llm_solver/analysis/denorm_audit.py` | `reusable-analysis` | copy | profile/runtime artifact analysis |
| `scripts/llm_solver/analysis/denorm_discover.py` | `reusable-analysis` | copy | profile improvement loop |
| `scripts/llm_solver/analysis/run_summary.py` | `reusable-analysis` | copy | artifact summary |
| `scripts/llm_solver/analysis/savings_summary.py` | `reusable-analysis` | copy | token-savings artifact summary |
| `scripts/llm_solver/analysis/state_replay.py` | `reusable-analysis` | copy | replay from trace |
| `scripts/llm_solver/analysis/state_verify.py` | `reusable-analysis` | copy | projection invariant checker |
| `scripts/llm_solver/analysis/transcript.py` | `reusable-analysis` | copy | transcript parsing |
| `scripts/llm_solver/analysis/transcript_clipper.py` | `reusable-analysis` | copy | transcript clipping |
| `scripts/llm_solver/analysis/verbatim_audit.py` | `reusable-analysis` | copy | raw request/response audit |
| `scripts/llm_solver/analysis/_coherence/**` | `reusable-analysis` | copy | support package |
| `scripts/llm_solver/analysis/_denorm_discover/**` | `reusable-analysis` | copy | support package |
| `scripts/llm_solver/analysis/_task_format.py` | `reusable-analysis` | copy | analysis task-format support |
| `scripts/llm_solver/analysis/experiments/**` | `benchmark-shell` | leave behind | DOE/campaign ledgering |
| `scripts/eval/**` | `benchmark-shell` | leave behind | FeatureBench evaluator adapter |
| `scripts/evaluate.py` | `benchmark-shell` | leave behind | fb-eval entrypoint |
| `scripts/prepare.sh` | `benchmark-shell` | leave behind | benchmark prep substrate |
| `scripts/collect_patches.sh` | `benchmark-shell` | leave behind | benchmark patch collection |
| `scripts/doe/**` | `benchmark-shell` | leave behind | DOE automation |
| `tasks/**` | `benchmark-shell` | leave behind | benchmark inputs/masks |
| `experiments/**` | `benchmark-shell` | leave behind | campaign records |
| `results/**` | `private-sediment` | leave behind | run artifacts |
| `.repo_cache/**` | `private-sediment` | leave behind | local cache |
| `.junk/**` | `private-sediment` | leave behind | junk |
| `.private/**` | `private-sediment` | leave behind | private notes |
| `archive/**` | `private-sediment` | leave behind | historical sediment |
| `config.toml` | `rewrite` | rewrite | keep shape, drop benchmark-era defaults |
| `configs/knobs.toml` | `core` | copy | authoritative knob catalog |
| `configs/toggles/**` | `benchmark-shell` | leave behind | DOE/ablation surface |
| `configs/doe/**` | `benchmark-shell` | leave behind | campaign surface |
| `configs/qwen36_diag/**` | `private-sediment` | leave behind | local experiment surface |
| `README.md` | `rewrite` | rewrite | current repo narrative is experiment-shaped |
| `CLAUDE.md` | `rewrite` | rewrite | current instructions are lab-shaped |
| `AGENTS.md` | `core` | copy | extraction guidance |
| `docs/harness_spec.md` | `core` | copy | public spec |
| `docs/assistant_spec.md` | `core` | copy | public spec |
| `docs/coupling_spec.md` | `core` | copy | public spec |
| `docs/public_repo_spec.md` | `core` | copy | public spec |
| `docs/separation_of_concerns.md` | `core` | copy | architectural law |
| `docs/state_writer.md` | `core` | copy | artifact contract |
| `docs/sandbox.md` | `core` | copy | sandbox contract |
| `docs/knob_catalog.md` | `core` | copy | control surface |
| `docs/config_layering.md` | `core` | copy | config contract |
| `docs/model_profiles.md` | `core` | copy | profile system reference |
| `docs/harness_boundary.md` | `rewrite` | rewrite or drop | names FeatureBench explicitly |
| `docs/doe/**` | `benchmark-shell` | leave behind | campaign methods |
| `docs/evaluation_pipeline.md` | `benchmark-shell` | leave behind | benchmark pipeline |
| `docs/prep_artifacts.md` | `benchmark-shell` | leave behind | benchmark prep details |

## Runtime Artifacts

| Artifact | Status | Rationale |
|---|---|---|
| `.trace.jsonl` | keep | append-only ground truth |
| `.savings.jsonl` | keep | transform/accounting artifact |
| `.solver/state.json` | keep | mechanical projection |
| `transcript.log` | keep for phase 1 | current runtime emits it; harmless transitional artifact |
| `checkpoint.json` | keep for phase 1 | current runtime expectation |
| `metrics.json` | keep for phase 1 | current runtime expectation |
| `predictions.jsonl` | leave behind | benchmark patch submission artifact |
| `eval/**` | leave behind | benchmark evaluation output |

## Import Boundary Rules

Allowed:

- assistant shell → core runtime
- reusable analysis → `_shared`, `language_quirks`
- benchmark shell → core runtime

Forbidden:

- core runtime → benchmark shell
- core runtime → private sediment
- assistant shell → benchmark shell
- harness → analysis
- server → benchmark shell

If a copied file requires a forbidden import, stop and amend the spec before
copying anything else.

## Clarifications

### `bwrap`

Keep it. It is the default runtime sandbox and part of the harness identity.

### Docker

Leave behind for v1 public extraction. In this repo it mainly exists as part of
the benchmark/eval story. A future public repo can add container backends later
behind a clean sandbox interface.

### Analysis suite

Keep the parts that answer questions like:

- what happened in the trace?
- did the state projection stay valid?
- where did the model thrash?
- what did token-shaping transforms save?

Leave behind the parts that answer questions like:

- how did this arm perform on FeatureBench?
- how do campaign deltas look across DOE cells?
- how do hidden-test verdicts aggregate across runs?

## Extraction Rule

When in doubt:

1. If the file improves runtime behavior or artifact introspection independent
   of FeatureBench, keep it or keep it selectively.
2. If the file exists to feed, score, or report benchmark runs, leave it behind.
3. If the file is useful but still benchmark-shaped, rewrite it instead of copying it.
