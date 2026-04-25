# Config layering

`scripts/llm_solver/config.py`. Single source of truth for runtime parameters. Every CLI entry point resolves its settings through `load_config()`.

## Precedence (lowest to highest)

```
config.toml         (checked in, project-wide defaults)
    ↓ deep merge
config.local.toml   (gitignored, machine-specific overrides)
    ↓ deep merge
--config <path>     (optional extra TOML, per-invocation)
    ↓ flatten
CLI flag overrides  (highest priority)
```

Deep merge: dicts recurse, scalars and arrays replace wholesale. `[tools]` in `config.local.toml` replaces only the keys it names; it does not drop the ones it omits.

## Project root discovery

`_find_project_root()` walks up from `config.py` until it finds `config.toml`, or uses `$HARNESS_CONFIG` (with legacy `$YUJ_CONFIG` fallback) pointing directly at a `config.toml` file. Failing either raises `FileNotFoundError`. The walk caps at 10 levels, which is plenty for any sane layout.

## Required keys

`_extract_config_fields()` calls `_require(data, section, key)` for every field that has no defensible default. A missing required key raises `KeyError` with the full path (`config.toml missing key 'max_tokens' in [model]`). There are no silent defaults for core parameters. The only optional keys are `[experiment]` fields (`prompt_addendum`, `variant_name`, `tool_desc`) which default to empty or `"minimal"`.

This is a deliberate tightening. Silent `.get(..., default)` fallback masks typos and lets a `config.local.toml` that accidentally deletes a section continue running on stale defaults. If you need a new field, add it to both `config.toml` and `_REQUIRED_SECTIONS` / `_require()`.

## Config dataclass

`Config` is a frozen dataclass with every runtime parameter as a typed field. No section structure, no nesting — consumers access `cfg.max_turns` directly. `_extract_config_fields()` is the one place that knows which TOML section a field lives in.

## CLI override map

| Flag | Config field | Entry point |
|---|---|---|
| `--model <name>` | `model` (after `resolve_model` alias lookup) | `python -m scripts.llm_solver` |
| `--port <int>` | `base_url` (port component only; scheme+host come from `[server]`) | `python -m scripts.llm_solver` |
| `--max-sessions <N>` | `max_sessions` | `python -m scripts.llm_solver` |
| `--prompt-addendum <str>` | `prompt_addendum` | `python -m scripts.llm_solver` |
| `--variant-name <str>` | `variant_name` | `python -m scripts.llm_solver` |
| `--tool-desc minimal|opencode` | `tool_desc` | `python -m scripts.llm_solver` |
| `--config <path>` | extra TOML layered above `config.local.toml` | all entry points that accept it |

CLI override behavior: `overrides` dict is passed to `load_config()`. Any key in `_extract_config_fields`'s output is overridable; the override is coerced with `type(flat[k])(v)`. Unknown keys are silently ignored (the dataclass would reject them anyway).

## Complete field reference

Every required key in `config.toml`, grouped by section. All have inline comments in the TOML file itself.

### `[server]` — model server transport
| Key | Type | What it controls |
|-----|------|-----------------|
| `base_url` | str | OpenAI-compatible endpoint URL |
| `api_key` | str | Authorization header value |
| `provider` | str | Optional transport selector. Defaults to `openai-compatible`; `anthropic` uses the Anthropic Messages adapter. |

`api_key` may be an environment reference such as `$ENV:OPENAI_API_KEY`.
Assistant sessions created with `--provider` persist a session-local
`provider.toml` overlay containing provider/base URL/API-key-env references, not
literal API keys.
| `timeout_connect` | int | TCP connection timeout (seconds) |
| `timeout_read` | int | Chat completion response timeout (seconds) |
| `health_poll_interval` | int | Seconds between `/health` polls during launch/wait |
| `health_timeout` | int | Per-attempt socket timeout for `/health` probe |
| `launch_timeout` | int | Max wait for server to become healthy on launch |
| `stop_settle` | int | Seconds to wait after killing server processes |

### `[model]` — model identity and context budget
| Key | Type | What it controls |
|-----|------|-----------------|
| `name` | str | Model identifier sent to the server |
| `context_size` | int | Server's context window in tokens |
| `context_fill_ratio` | float | Fraction of context reserved for prompt |
| `max_tokens` | int | Max tokens the model can generate per turn |

### `[loop]` — session lifecycle and interventions
| Key | Type | What it controls |
|-----|------|-----------------|
| `max_turns` | int | Hard cap on tool-call turns per session |
| `max_sessions` | int | Max session restarts |
| `duplicate_abort` | int | Abort after N identical consecutive tool calls |
| `max_transient_retries` | int | Retry count for transient API errors |
| `retry_backoff` | list[int] | Seconds between retries |
| `error_nudge_threshold` | int | Consecutive tool errors before nudge |
| `rumination_nudge_threshold` | int | Non-write calls before gate arms |
| `rumination_same_target_warn_count` | int | Repeated same-target inspections before same-target nudge (0 disables) |
| `rumination_same_target_arm_count` | int | Repeated same-target inspections before arming the rumination gate (0 disables) |
| `test_read_warn_after` | int | Verification runs without reading the target test file before nudging (0 disables) |
| `context_inspect_repeat_threshold` | int | Repeated inspect actions before concise/yconcise switch to an exit-inspect obligation (0 disables) |
| `contract_commit_warn_after` | int | Off-contract actions after a concrete source file is in view before warning (0 disables) |
| `contract_commit_block_after` | int | Off-contract actions after a concrete source file is in view before blocking (0 disables) |
| `contract_recovery_same_target_threshold` | int | Same-target repeats before contract recovery mode activates (0 disables) |
| `contract_recovery_verify_repeat_threshold` | int | Verify-without-refine repeats before contract recovery mode activates (0 disables) |
| `contract_invalid_repeat_abort_after` | int | Repeated blocked contract violations with the same target/signature before ending the session (0 disables) |
| `contract_abort_min_turns_since_commit_arm` | int | Minimum contract-gate calls after commit arm before commit-abort is allowed (0 disables) |
| `contract_abort_min_turns_since_recovery_arm` | int | Minimum contract-gate calls after recovery arm before recovery-abort is allowed (0 disables) |
| `contract_abort_requires_zero_mutation` | bool | Allow contract aborts only before the first successful write/edit |
| `contract_equivalent_action_classes_enabled` | bool | Collapse semantically equivalent off-contract moves into one violation class |
| `mutation_repeat_warn_after` | int | Warn when repeating the same successful mutation N times in a row (0 disables) |
| `mutation_repeat_block_after` | int | Block when repeating the same successful mutation N times in a row (0 disables) |
| `mutation_repeat_abort_after` | int | End session after N blocked identical mutation retries (0 disables) |
| `min_turns_before_context` | int | Raw message history turns before context strategy switches |
| `adaptive_policy_enabled` | bool | Enable config-driven base→phase2 runtime policy switch |
| `adaptive_switch_min_turn` | int | Earliest turn index where adaptive switch may fire |
| `adaptive_requires_mutation` | bool | Require at least one successful write/edit before switch |
| `adaptive_requires_test_signal` | bool | Require observed test-signal command before switch |
| `adaptive_low_pressure_window` | int | Rolling turn window used for low-pressure gate (0 disables) |
| `adaptive_low_pressure_max_events` | int | Max pressure events allowed in window to permit switch |
| `adaptive_phase2_done_guard_enabled` | bool | Phase2 value for `done_guard_enabled` |
| `adaptive_phase2_bash_task_format_enabled` | bool | Phase2 value for task-format transforms |
| `adaptive_phase2_bash_structured_output_enabled` | bool | Phase2 value for structured output projection |
| `adaptive_phase2_bash_sink_threshold_chars` | int | Phase2 sink threshold |

Preset example: `configs/doe/adaptive_explore_then_converge.toml` starts with
explore toggles (`done_guard` and task-format transforms off) and switches to
converge toggles after mutation/pressure conditions.

### `[output]` — truncation, windowing, and rendering
| Key | Type | What it controls |
|-----|------|-----------------|
| `max_output_chars` | int | Per-tool-result char cap |
| `truncate_head_ratio` | float | Fraction of truncation budget allocated to head |
| `truncate_head_lines` | int | Minimum head lines preserved |
| `truncate_tail_lines` | int | Minimum tail lines preserved |
| `args_summary_chars` | int | Per-value char limit in args summaries |
| `trace_args_summary_chars` | int | Args-summary cap in .trace.jsonl events |
| `trace_reasoning_store_chars` | int | Max reasoning chars stored in .trace.jsonl |
| `solver_trace_lines` | int | Tail entries kept from state.json trace |
| `solver_evidence_lines` | int | Tail entries kept from state.json evidence |
| `solver_inference_lines` | int | Tail entries kept from state.json inference |
| `recent_tool_results_chars` | int | Rolling tool-result window budget |
| `focused_compound_trace_lines` | int | Trace budget override for `focused_compound` |
| `focused_compound_evidence_lines` | int | Evidence budget override for `focused_compound` |
| `focused_compound_recent_tool_results_chars` | int | Tool-result window override for `focused_compound` |
| `focused_compound_include_resolved_evidence` | bool | Whether `focused_compound` keeps resolved/passing evidence |
| `compound_selective_trace_lines` | int | Trace budget override for `compound_selective` |
| `compound_selective_unresolved_evidence_lines` | int | Unresolved evidence budget override for `compound_selective` |
| `compound_selective_resolved_evidence_lines` | int | Resolved evidence budget override for `compound_selective` |
| `compound_selective_resolved_evidence_stub_chars` | int | Stub length for resolved evidence payloads in `compound_selective` |
| `compound_selective_recent_tool_results_chars` | int | Tool-result window override for `compound_selective` |
| `compound_selective_trace_action_repeat_cap` | int | Max identical trace actions kept in `compound_selective` |
| `compound_selective_resolved_action_repeat_cap` | int | Max identical resolved-evidence actions kept in `compound_selective` |
| `compound_selective_trace_anchor_lines` | int | Older trace actions reserved as anchors in `compound_selective` |
| `compound_selective_resolved_anchor_lines` | int | Older resolved-evidence actions reserved as anchors in `compound_selective` |
| `compound_selective_trace_source_anchor_lines` | int | Older non-test source anchors reserved in `compound_selective` trace selection |
| `compound_selective_trace_test_anchor_lines` | int | Older test/verification anchors reserved in `compound_selective` trace selection |
| `compound_selective_resolved_source_anchor_lines` | int | Older non-test source anchors reserved in `compound_selective` resolved evidence |
| `compound_selective_resolved_test_anchor_lines` | int | Older test/verification anchors reserved in `compound_selective` resolved evidence |
| `trace_stub_chars` | int | Per-entry stub length in trace rendering |
| `trace_reasoning_chars` | int | Reasoning snippet length in trace entries |
| `pretest_head_chars` | int | Pretest output head budget |
| `pretest_tail_chars` | int | Pretest output tail budget |

### `[tools]` — tool execution and filtering
| Key | Type | What it controls |
|-----|------|-----------------|
| `bash_timeout` | int | Seconds before a bash call is killed |
| `grep_timeout` | int | Seconds before a grep call is killed |
| `pretest_timeout` | int | Seconds before the pretest script is killed |
| `strip_ansi` | bool | Strip ANSI escape sequences from bash output |
| `collapse_blank_lines` | bool | Collapse 3+ consecutive newlines to 2 |
| `collapse_duplicate_lines` | bool | Collapse runs of byte-identical consecutive lines into `<line> [×N]` |
| `llama_server_bin` | str | Path to llama-server binary |
| `bwrap_bin` | str | Path to bubblewrap binary |
| `sandbox_bash` | bool | Run bash calls in bwrap sandbox |

### `[experiment]` — per-run overrides
| Key | Type | What it controls |
|-----|------|-----------------|
| `prompt_addendum` | str | Text appended to task prompt |
| `variant_name` | str | Tag for this experiment variant |
| `tool_desc` | str | Tool description mode: `"minimal"` or `"opencode"` |

### `[analysis]` — post-hoc forensic analysis settings
Analysis tools run AFTER a task completes. They are allowed to be task-aware,
but the task-awareness is data-driven: each task family can define a
``TaskFormat`` descriptor naming verification-command patterns. Analysis
CLIs accept ``--task-format <name>``; this config key sets the default.
The harness never reads this setting.

| Key | Type | What it controls |
|-----|------|-----------------|
| `task_format` | str | Default task format name for analysis CLIs. Resolves to `task_formats/<name>.toml`. Override per-invocation with `--task-format`. Names: `pytest`, `cargo`, `go`, `jest`, `ctest`, `generic`, or a custom descriptor path. |

### `[prompts]` — all harness-injected model-steering text
Every prompt literal the harness can inject lives here. No prompt text is hardcoded in harness code. Each model-facing string the solver loop produces is owned by config.
| Key | Type | What it controls |
|-----|------|-----------------|
| `system_header` | str | Harness system prompt header (prepended with --system-prompt file if supplied) |
| `state_context_suffix` | str | Suffix injected at the end of every stateful user message |
| `intent_gate_first` | str | Intent gate first-rejection text for silent tool calls |
| `intent_gate_repeat` | str | Intent gate subsequent-rejection text (`{count}`, `{first_turn}`) |
| `resume_base` | str | Base continuation prompt at session boundaries |
| `error_nudge` | str | Nudge text after consecutive tool errors (`{count}`) |
| `rumination_nudge` | str | Nudge text when non-write threshold crossed (`{count}`) |
| `rumination_gate` | str | Gate rejection message for blocked non-write calls |
| `rumination_same_target_nudge` | str | Nudge text when the same file / normalized command target repeats (`{count}`, `{target}`) |
| `rumination_outside_cwd_nudge` | str | Nudge text when repeated inspection is anchored to an absolute target outside the repo root (`{count}`, `{target}`) |
| `test_read_nudge` | str | Nudge text when verification repeats before the target test file is read (`{count}`, `{target}`) |
| `resume_duplicate_abort` | str | Resume text after duplicate abort (`{n}`, `{call}`) |
| `resume_context_full` | str | Resume text after context full (`{pct}`) |
| `resume_max_turns` | str | Resume text after max turns (`{n}`, `{actions}`) |
| `resume_length` | str | Resume text after response truncation |
| `resume_last_n_actions` | int | Number of recent actions shown in max_turns resume |

## Accessors for non-harness tools

Some entry points need only a slice of the config (e.g. `scripts/solve_bare.py` needs `[sdk]`, `profiles/run_scenarios.py` needs only server transport settings). They use module-level accessors instead of building a full `Config`:

- `get_sdk_config()` — `[sdk]` section with model alias resolved.
- `get_cli_config()` — `[cli]` section with model alias resolved.
- `get_server_config()` — `[server]` section, raw.
- `get_server_base_url()` — `[server].base_url`, raises if missing.
- `get_model_default_max_tokens()` — `[model].max_tokens`, raises if missing.

These read from the already-merged `_LAYERED` dict and never re-run the merge. They exist so ad-hoc scripts can consume config without inheriting the full required-key contract of `Config`.

## Provenance echo

Every run writes the resolved `Config` into `metrics.json` under `provenance.config` (see `solver.collect_provenance`). A run that was started with a wrong `config.local.toml` override can be reconstructed from artifacts alone — `git log` is not enough. `dump_config(cfg)` is the helper.

At startup, `__main__.py` also logs one line with the load-bearing fields so a terminal-watcher sees exactly what resolved:

```
Config: model=qwen3.5-9b ctx=40960 max_turns=60 max_sessions=10 tool_desc=minimal variant=(none)
```

## Adding a new field

1. Add the key to `config.toml` with a sensible default and an inline comment.
2. Add the matching field to the `Config` dataclass.
3. Add a `_require(d, "<section>", "<key>")` line to `_extract_config_fields()`.
4. If the field has no defensible default, leave it without a default value in the dataclass; required fields come before defaulted fields.
5. Update test helpers (`make_config` in `tests/test_harness_and_pipeline.py`, `_make_config` in `tests/test_profile_system.py`) to include the new field.
