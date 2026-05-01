# Public Repo Spec

## Purpose
Define the initial extraction from this lab repo into a clean public staging repo.

This spec is written to be executable by a low-context coding agent. It is
deliberately explicit.

## Staging Repo

Create and use:

`/home/syd/projects/harness-public-staging`

This is a staging repo name, not final branding.

## Extraction Strategy

1. **Copy only from the whitelist in this document.**
2. **Do not move files out of the lab repo.** Copy only.
3. **Do not rename internal paths in phase 1.**
   Keep `scripts/llm_solver/` and `profiles/` intact until tests pass.
4. **Do not copy any file not named here.**
5. **If a missing file is required, stop and update this spec first.**

## Phase Plan

### Phase 0

Docs only.

Deliverables:

- `docs/harness_spec.md`
- `docs/assistant_spec.md`
- `docs/coupling_spec.md`
- `docs/public_repo_spec.md`
- `AGENTS.md`

### Phase 1

Core runtime extraction with current internal paths preserved.

### Phase 2

Add the assistant shell.

### Phase 3

Add selected reusable analysis.

### Phase 4

Only after green tests: consider cosmetic renames or package cleanup.

## Phase 1 Copy Whitelist

### Root files

- `AGENTS.md`
- `HANDOFF.md` — **write fresh in staging/public repo**
- `config.toml` — **rewrite**, do not copy verbatim
- `README.md` — **rewrite**, do not copy verbatim
- `.gitignore` — **rewrite**, do not copy verbatim
- `CLAUDE.md` — **do not copy**

### Docs

- `docs/harness_spec.md`
- `docs/assistant_spec.md`
- `docs/coupling_spec.md`
- `docs/public_repo_spec.md`
- `docs/separation_of_concerns.md`
- `docs/state_writer.md`
- `docs/sandbox.md`
- `docs/knob_catalog.md`
- `docs/config_layering.md`
- `docs/model_profiles.md`

Do not copy any other doc in phase 1.

### Config

- `configs/knobs.toml`

Do not copy:

- `configs/toggles/**`
- `configs/doe/**`
- `configs/qwen36_diag/**`

### Core runtime code

- `scripts/__init__.py`
- `scripts/llm_solver/__init__.py`
- `scripts/llm_solver/__main__.py`
- `scripts/llm_solver/config.py`
- `scripts/llm_solver/_shared/**`
- `scripts/llm_solver/bash_quirks/**`
- `scripts/llm_solver/harness/**`
- `scripts/llm_solver/language_quirks/**`
- `scripts/llm_solver/models/**`
- `scripts/llm_solver/profiles/**`
- `scripts/llm_solver/server/**`

Do not copy:

- `scripts/eval/**`
- `scripts/doe/**`
- `scripts/prepare.sh`
- `scripts/collect_patches.sh`
- `scripts/evaluate.py`

### Profiles

Copy:

- `profiles/_base/**`
- `profiles/glm-4-flash/**`
- `profiles/qwen3-8b-q4/**`
- `profiles/qwen3.5-9b/**`
- `profiles/qwen3.5-9b-q6k/**`
- `profiles/qwen3.6-35b-a3b/**`
- `profiles/qwen3.6-35b-a3b-q5/**`

Do not copy:

- `profiles/deprecated/**`

### Phase 1 test whitelist

Copy only these test support files and tests:

- `tests/__init__.py`
- `tests/_config_helpers.py`
- `tests/_shell_patterns.py`
- `tests/context.py`
- `tests/context_strategies/**`
- `tests/edit_replacers.py`
- `tests/experiment.py`
- `tests/guardrails.py`
- `tests/injections.py`
- `tests/loop.py`
- `tests/post_edit.py`
- `tests/sandbox.py`
- `tests/savings.py`
- `tests/schemas.py`
- `tests/solver.py`
- `tests/state_writer.py`
- `tests/tools.py`
- `tests/test_composability.py`
- `tests/test_compound_context.py`
- `tests/test_compound_selective_context.py`
- `tests/test_concise_contexts.py`
- `tests/test_edit_candidates.py`
- `tests/test_edit_replacers.py`
- `tests/test_harness_and_pipeline.py`
- `tests/test_injections.py`
- `tests/test_loop_detect_guardrail.py`
- `tests/test_parallel_readonly.py`
- `tests/test_post_edit_check.py`
- `tests/test_profile_behavioral_toml.py`
- `tests/test_profile_verify.py`
- `tests/test_read_reminders.py`
- `tests/test_sandbox_escape.py`
- `tests/test_savings_ledger.py`
- `tests/test_search_pagination.py`
- `tests/test_sink_pointer_xml.py`
- `tests/test_state_writer.py`

Do not copy any other test in phase 1.

Rationale for explicit exclusions:

- `tests/test_profile_system.py` still expects lab-only
  `experiments/profiler-comparison/**` generated profiles.
- `tests/test_smoke.py` is hardware- and model-file-dependent.

## Phase 2 Copy Whitelist

Only after phase 1 is green:

- `scripts/knob.py`
- `scripts/llm_assist/__init__.py`
- `scripts/llm_assist/__main__.py`
- `scripts/llm_assist/runner.py`
- `scripts/llm_assist/store.py`
- `tests/test_assist_store.py`
- `tests/test_assist_resume.py`
- `tests/test_runtime_mode.py`

Do not copy any existing benchmark runner to impersonate the assistant shell.

## Phase 3 Copy Whitelist

Only after phase 2 is green:

- `scripts/llm_solver/analysis/coherence.py`
- `scripts/llm_solver/analysis/compare.py`
- `scripts/llm_solver/analysis/denorm_audit.py`
- `scripts/llm_solver/analysis/denorm_discover.py`
- `scripts/llm_solver/analysis/run_summary.py`
- `scripts/llm_solver/analysis/savings_summary.py`
- `scripts/llm_solver/analysis/state_replay.py`
- `scripts/llm_solver/analysis/state_verify.py`
- `scripts/llm_solver/analysis/transcript.py`
- `scripts/llm_solver/analysis/transcript_clipper.py`
- `scripts/llm_solver/analysis/verbatim_audit.py`
- `scripts/llm_solver/analysis/_coherence/**`
- `scripts/llm_solver/analysis/_denorm_discover/**`
- `scripts/llm_solver/analysis/_task_format.py`
- `tests/test_coherence.py`
- `tests/test_compare.py`
- `tests/test_denorm_audit.py`
- `tests/test_denorm_discover.py`

Do not copy:

- `scripts/llm_solver/analysis/experiments/**`
- `tests/test_experiments_ledger.py`

## Never Copy

- `.private/**`
- `.junk/**`
- `.repo_cache/**`
- `archive/**`
- `experiments/**`
- `results/**`
- `tasks/**`
- `.pytest_cache/**`

## Files To Rewrite Fresh

These should be written fresh in the public repo, not copied verbatim:

- `README.md`
- `config.toml`
- `AGENTS.md`
- `HANDOFF.md`
- `.gitignore`

`README.md` must describe the public repo as a harness + assistant project, not as a
FeatureBench experiment.

`config.toml` must start minimal. Do not drag the full lab knob surface into v1 unless
the copied runtime actually needs it.

## Acceptance Criteria

### Structural

- No copied file imports `scripts.eval`
- No copied file imports `tasks`
- No copied file imports `experiments`
- No copied file mentions `FeatureBench` outside historical docs
- `scripts/llm_solver/**` does not import `scripts.llm_assist` or `scripts.yuj`
- assistant-shell code stays under `scripts/llm_assist/**`, `scripts/yuj.py`, and `yuj`

### Runtime

- `python3 -m scripts.llm_solver --help` works
- core tests pass

### Documentation

- the four spec docs are present
- `AGENTS.md` is present
- rewritten `README.md` does not frame the repo as a benchmark harness

## Verification Commands

Run from the staging repo root:

```bash
python3 -m pytest tests/ -q
rg -n "FeatureBench|fb eval|mask_patches|tasks/prepared|predictions.jsonl|scripts\\.eval" .
python3 -m scripts.llm_solver --help
```

The grep command should return nothing except possibly this spec itself while the
staging repo is still documentation-only.

## Dumb Agent Protocol

If you are an automated agent executing this extraction:

1. Read `AGENTS.md`.
2. Read these docs in order:
   `docs/harness_spec.md`,
   `docs/assistant_spec.md`,
   `docs/coupling_spec.md`,
   `docs/public_repo_spec.md`.
3. Copy only the exact files and directories named here.
4. Preserve internal paths in phase 1.
5. If a missing import appears, do not improvise. Update the spec first.
6. If a file is benchmark-shaped, do not copy it just because tests mention it.
7. Do not add TypeScript, a web server, a dashboard, or subagents in v1.

## Recommendation

The first clean public repo should be boring:

- one Python codebase
- one config surface
- one default sandbox
- one assistant shell
- one set of artifact contracts

That is the point.
