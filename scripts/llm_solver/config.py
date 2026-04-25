"""Configuration loader — TOML defaults + user overrides + CLI flags.

Layered config resolution:
  1. config.toml        (project root, checked into git)
  2. config.local.toml  (same directory, gitignored, optional)
  3. CLI overrides       (highest priority)

Project root is located by walking up from this file until config.toml is found,
or via ``HARNESS_CONFIG`` / legacy ``YUJ_CONFIG`` pointing directly to
config.toml.
"""
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path

from ._shared.toml_compat import tomllib


VALID_RUNTIME_MODES = ("measurement", "assistant")


def _find_project_root() -> Path:
    """Walk up from this file to find config.toml, or use config env vars."""
    env = os.environ.get("HARNESS_CONFIG") or os.environ.get("YUJ_CONFIG")
    if env:
        p = Path(env)
        if p.is_file():
            return p.parent
        raise FileNotFoundError(
            f"HARNESS_CONFIG/YUJ_CONFIG points to missing file: {env}"
        )

    d = Path(__file__).resolve().parent
    for _ in range(10):
        if (d / "config.toml").is_file():
            return d
        parent = d.parent
        if parent == d:
            break
        d = parent
    raise FileNotFoundError(
        "config.toml not found. Set HARNESS_CONFIG or run from project root."
    )


PROJECT_ROOT = _find_project_root()
_DEFAULT_CONFIG = PROJECT_ROOT / "config.toml"
_LOCAL_CONFIG = PROJECT_ROOT / "config.local.toml"


def _load_toml(path: Path) -> dict:
    with open(path, "rb") as f:
        return tomllib.load(f)


def _deep_merge(base: dict, overlay: dict) -> dict:
    """Recursively merge overlay into base (overlay wins)."""
    for k, v in overlay.items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            base[k] = _deep_merge(base[k], v)
        else:
            base[k] = v
    return base


def _load_layered() -> dict:
    """Load config.toml, merge config.local.toml if present."""
    data = _load_toml(_DEFAULT_CONFIG)
    if _LOCAL_CONFIG.is_file():
        _deep_merge(data, _load_toml(_LOCAL_CONFIG))
    return data


# ---------------------------------------------------------------------------
# Model aliases — exported for other scripts
# ---------------------------------------------------------------------------

def _build_model_map(data: dict) -> dict[str, str]:
    return dict(data.get("models", {}).get("aliases", {}))


# Eagerly loaded so importers can do: from scripts.llm_solver.config import MODEL_MAP
_LAYERED = _load_layered()
MODEL_MAP: dict[str, str] = _build_model_map(_LAYERED)


# ---------------------------------------------------------------------------
# SDK / CLI section accessors
# ---------------------------------------------------------------------------

def get_sdk_config() -> dict:
    """Return the [sdk] section with model alias resolved."""
    section = dict(_LAYERED.get("sdk", {}))
    model = section.get("default_model", "sonnet")
    section["default_model_resolved"] = MODEL_MAP.get(model, model)
    return section


def get_cli_config() -> dict:
    """Return the [cli] section with model alias resolved."""
    section = dict(_LAYERED.get("cli", {}))
    model = section.get("default_model", "haiku")
    section["default_model_resolved"] = MODEL_MAP.get(model, model)
    return section


def get_server_base_url() -> str:
    """Return ``[server] base_url`` from the layered config.

    Used by CLI code that needs the scheme+host before building a full Config.
    """
    return _require(_LAYERED, "server", "base_url")  # type: ignore[return-value]


def get_server_config() -> dict:
    """Return the ``[server]`` section for callers outside the Config dataclass.

    Read by tools (e.g. ``run_scenarios``) that only need transport settings.
    """
    return dict(_LAYERED.get("server", {}))


def get_model_default_max_tokens() -> int:
    """Return ``[model] max_tokens`` for tools that build ad-hoc OpenAI clients."""
    return int(_require(_LAYERED, "model", "max_tokens"))  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# llm_solver Config dataclass (existing interface, preserved)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Config:
    base_url: str
    api_key: str
    timeout_connect: int
    timeout_read: int
    health_poll_interval: int
    health_timeout: int
    launch_timeout: int
    stop_settle: int
    model: str
    context_size: int
    context_fill_ratio: float
    max_tokens: int
    max_turns: int
    max_sessions: int
    duplicate_abort: int
    error_nudge_threshold: int
    rumination_nudge_threshold: int
    require_intent: bool
    intent_grace_turns: int
    min_turns_before_context: int
    max_output_chars: int
    truncate_head_ratio: float
    truncate_head_lines: int
    truncate_tail_lines: int
    args_summary_chars: int
    trace_args_summary_chars: int
    trace_reasoning_store_chars: int
    solver_trace_lines: int
    solver_evidence_lines: int
    solver_inference_lines: int
    recent_tool_results_chars: int
    trace_stub_chars: int
    trace_reasoning_chars: int
    pretest_head_chars: int
    pretest_tail_chars: int
    bash_timeout: int
    grep_timeout: int
    pretest_timeout: int
    llama_server_bin: str
    sandbox_bash: bool
    strip_ansi: bool
    collapse_blank_lines: bool
    collapse_duplicate_lines: bool
    collapse_similar_lines: bool
    bwrap_bin: str
    max_transient_retries: int
    retry_backoff: tuple[int, ...]
    # Prompt fragments (session boundaries, gates, nudges)
    system_header: str
    state_context_suffix: str
    intent_gate_first: str
    intent_gate_repeat: str
    resume_base: str
    error_nudge: str
    rumination_nudge: str
    rumination_gate: str
    rumination_same_target_nudge: str
    rumination_outside_cwd_nudge: str
    test_read_nudge: str
    contract_commit_warn: str
    contract_commit_block: str
    contract_recovery_block: str
    mutation_repeat_warn: str
    mutation_repeat_block: str
    resume_duplicate_abort: str
    resume_context_full: str
    resume_max_turns: str
    resume_length: str
    resume_last_n_actions: int
    tool_desc: str = "minimal"
    prompt_addendum: str = ""
    variant_name: str = ""
    runtime_mode: str = "measurement"
    analysis_task_format: str = "pytest"
    provider: str = "openai-compatible"
    rumination_gate_max_blocks: int = 0
    resume_gate_escalation: str = "Session ended: {n} consecutive tool calls were blocked by the rumination gate. Your current code has been preserved."
    # Model-facing strings emitted by guardrails (done_guard, rumination_gate)
    # and by the sink-and-surface mechanism in loop.py. Kept in config per
    # the SoD anti-pattern "prompt text in harness code" — harness code
    # should carry no model-facing text directly.
    done_reject_no_mutation: str = "REJECTED: No code changes made yet. Write or edit code before calling done."
    done_reject_no_verify: str = "REJECTED: No successful command since last code change. Run a command to verify, then call done."
    done_reject_parity_no_run: str = "REJECTED: done_require_pretest_parity is on but no structured test run has been observed yet. Run the test suite first."
    done_reject_parity_still_failing: str = "REJECTED: pretest-failing tests not yet passing: {shown}{extra}"
    done_reject_parity_regression: str = "REJECTED: regression — previously-passing tests now failing: {shown}{extra}"
    done_reject_parity_streak: str = "REJECTED: pretest parity observed {count} time(s); need {required}. Run tests again to confirm."
    rumination_gate_grace_prefix: str = "[HARNESS: Gate armed. Next call must be write or edit — all else blocked.]"
    sink_pointer: str = '<tool_result_meta truncated="true" original_bytes="{chars}" original_lines="{lines}" full_path="{path}"/>'
    # Head/tail slice sizes around the sink marker — prompt literals
    # previously hardcoded in loop.py::_filter_bash_output live here
    # so the whole sink-surface shape is config-adjustable.
    sink_head_bytes: int = 1000
    sink_tail_bytes: int = 1000
    sink_body_marker: str = "... [body truncated — full output available via full_path attribute] ..."
    # read() tool system-reminder text (placeholders filled at call site).
    read_truncated_reminder: str = (
        "<system-reminder>Read returned the first {returned_lines} lines of "
        "{path}. The file is longer — re-read with a higher limit or a "
        "specific offset to see more.</system-reminder>"
    )
    read_empty_reminder: str = (
        "<system-reminder>File {path} exists but is empty (0 bytes)."
        "</system-reminder>"
    )
    # Guardrail escalation ladders (additions, 0 = disabled).
    # Each guardrail follows: WARN (append text) → BLOCK (reject call) → END (end session).
    # See docs/separation_of_concerns.md, "Harness sub-concern: Guardrails".
    rumination_gate_arm_threshold: int = 0  # % of max_turns; when > nudge, provides a warning window before the gate arms. 0 = arm at nudge.
    rumination_gate_arm_threshold_abs: int = 0  # Absolute non-write count; when > 0, overrides percentage. Decouples gate from max_turns.
    rumination_nudge_threshold_abs: int = 0  # Absolute non-write count for nudge; when > 0, overrides percentage. Decouples nudge from max_turns.
    rumination_nudge_threshold_abs_post_mutation: int = 0  # Absolute non-write count for nudge AFTER state.has_mutated flips True. When > 0, overrides the pre-mutation threshold for post-mutation rumination. When 0, post-mutation uses the same threshold as pre-mutation.
    rumination_nudge_only_pre_mutation: bool = False  # When True, suppress nudge entirely after state.has_mutated=True. Equivalent to post_mutation_threshold = infinity.
    rumination_same_target_warn_count: int = 0  # Repeated same-target non-write calls before same-target nudge (0 = disabled).
    rumination_same_target_arm_count: int = 0  # Repeated same-target non-write calls before arming the rumination gate (0 = disabled).
    test_read_warn_after: int = 0  # Verification runs without reading the target test file before nudging (0 = disabled).
    context_inspect_repeat_threshold: int = 0  # Repeated inspect actions before concise/yconcise switch to an exit-inspect obligation (0 = disabled).
    contract_commit_warn_after: int = 0  # After a source-file read, warn on non-commit actions after N violations (0 = disabled).
    contract_commit_block_after: int = 0  # After a source-file read, block non-commit actions after N violations (0 = disabled).
    contract_recovery_same_target_threshold: int = 0  # Arm recovery mode after N same-target non-write actions (0 = disabled).
    contract_recovery_verify_repeat_threshold: int = 0  # Arm recovery mode after N verify runs against the same target without mutation (0 = disabled).
    contract_invalid_repeat_abort_after: int = 0  # End session after N repeated blocked contract violations with the same target/signature (0 = disabled).
    contract_abort_min_turns_since_commit_arm: int = 0  # Minimum contract-gate calls since commit arm before abort can fire (0 = disabled).
    contract_abort_min_turns_since_recovery_arm: int = 0  # Minimum contract-gate calls since recovery arm before abort can fire (0 = disabled).
    contract_abort_requires_zero_mutation: bool = False  # If True, contract abort is allowed only before the first successful write/edit.
    contract_equivalent_action_classes_enabled: bool = False  # Collapse semantically-equivalent off-contract moves into one violation class.
    mutation_repeat_warn_after: int = 0  # Warn when repeating the same successful mutation N times in a row (0 = disabled).
    mutation_repeat_block_after: int = 0  # Block when repeating the same successful mutation N times in a row (0 = disabled).
    mutation_repeat_abort_after: int = 0  # End session after N blocked identical mutation retries (0 = disabled).
    duplicate_warn_count: int = 0  # append warning text at N identical consecutive calls (0 = disabled)
    duplicate_warn: str = "[harness: {count} identical tool calls in a row. Change approach — session ends at {abort} identical.]"
    error_abort_threshold: int = 0  # end session after N consecutive errors of any kind (0 = disabled)
    intent_abort_threshold: int = 0  # end session after N consecutive silent intent-gate rejections (0 = disabled)
    # ── Guardrail enabled flags (default True = preserve current behaviour).
    # Flip any to False in configs/substantive.toml or configs/ablation-*.toml to
    # isolate that guardrail's contribution in a campaign. Note: require_intent
    # predates this pattern and remains the canonical intent_gate toggle.
    duplicate_guard_enabled: bool = True
    # Post-edit validation (runs per-extension check cmd after each
    # edit/write). Map keyed on extension (dot-prefixed). Off by default.
    post_edit_check_enabled: bool = False
    post_edit_check_timeout: int = 10
    # List of declared check dicts. Each dict has: name, trigger,
    # when, cmd, on_fail. Empty list = no-op.
    post_edit_checks: list = field(default_factory=list)
    # Paginated search envelopes for grep/glob. Defaults ship on.
    search_pagination_enabled: bool = True
    grep_max_matches_per_page: int = 25
    glob_max_matches_per_page: int = 25
    # edit() match policy. Strict is the default (database-of-
    # primitives principle: no silent relaxation). Cascade restores
    # the auto-apply behavior shipped in aa81a62 as a DOE arm.
    edit_strict_match: bool = True
    edit_fuzzy_cascade_enabled: bool = False
    edit_candidate_count: int = 3
    # loop_detect guardrail (N consecutive identical tool-call signatures).
    # WARN on first reach-threshold (inject recovery text). END if the
    # pattern repeats once more after the warning. Disabled by default so
    # enabling it is an explicit DOE knob per the guardrail convention.
    loop_detect_enabled: bool = False
    loop_detect_threshold: int = 5
    # Parallel read-only tool dispatch. When enabled and the turn's
    # tool_calls are all read-only (>1 call, no write/edit/bash),
    # dispatch() runs concurrently via a ThreadPoolExecutor. Guardrail
    # state still updates sequentially per-tc after concurrent I/O.
    parallel_readonly_enabled: bool = False
    parallel_max_workers: int = 4
    # Injection subsystem (keyword-triggered markdown fragments).
    # Off by default; data-directory convention .harness/injections/.
    injections_enabled: bool = False
    injections_dir: str = ".harness/injections"
    loop_detect_recovery: str = (
        "<system-reminder>Loop detected: the last {streak} tool calls all "
        "have identical name and arguments. Stop repeating. Re-read the "
        "task, read a file you have not inspected yet, or change approach. "
        "One more repeat ends the session.</system-reminder>"
    )
    done_guard_enabled: bool = True
    rumination_enabled: bool = True
    error_ladder_enabled: bool = True
    bash_transforms_universal_enabled: bool = True
    bash_transforms_task_format_enabled: bool = True
    bash_transforms_structured_output_enabled: bool = False  # parse test output into digest; replace raw with digest in context
    bash_transforms_sink_threshold_chars: int = 0  # write raw bash output to .tool_output/ when result exceeds N chars (0 = disabled)
    # ── Guardrail internals surfaced from hardcoded values.
    rumination_gate_grace_calls: int = 1       # number of non-write calls after gate arm that pass with a warning before full blocking
    rumination_min_threshold: int = 6          # absolute floor on the derived rumination nudge threshold
    done_require_mutation: bool = True         # done_guard: accept only after at least one successful write/edit
    done_require_verify: bool = True           # done_guard: accept only after verified_since_mutation flipped
    done_verified_bash_min_chars: int = 200    # content-blind threshold for "substantial" bash run that counts as verification
    done_require_pretest_parity: bool = False  # done_guard: accept only when latest test run matches the pretest-failing set now PASSED and no pretest-passing regressed (requires [output_parser])
    done_parity_runs_required: int = 1         # number of consecutive parity-green runs required before done accepts (guards against flakiness)
    # ── Adaptive policy controller (config-driven phase switch).
    adaptive_policy_enabled: bool = False
    adaptive_switch_min_turn: int = 0
    adaptive_requires_mutation: bool = True
    adaptive_requires_test_signal: bool = True
    adaptive_low_pressure_window: int = 0
    adaptive_low_pressure_max_events: int = 0
    adaptive_phase2_done_guard_enabled: bool = True
    adaptive_phase2_bash_task_format_enabled: bool = True
    adaptive_phase2_bash_structured_output_enabled: bool = True
    adaptive_phase2_bash_sink_threshold_chars: int = 0
    context_slot_max_candidates: int = 1
    context_slot_inline_files: int = 1
    focused_compound_trace_lines: int = 0  # Trace budget override for focused_compound (0 = use solver_trace_lines).
    focused_compound_evidence_lines: int = 0  # Evidence budget override for focused_compound (0 = use solver_evidence_lines).
    focused_compound_recent_tool_results_chars: int = 0  # Rolling tool-result budget override for focused_compound (0 = use recent_tool_results_chars).
    focused_compound_include_resolved_evidence: bool = False  # Whether focused_compound renders resolved/passing evidence.
    compound_selective_trace_lines: int = 0  # Trace budget override for compound_selective (0 = use solver_trace_lines).
    compound_selective_unresolved_evidence_lines: int = 0  # Unresolved evidence budget override for compound_selective (0 = use solver_evidence_lines).
    compound_selective_resolved_evidence_lines: int = 0  # Resolved evidence budget override for compound_selective (0 = hide resolved evidence).
    compound_selective_resolved_evidence_stub_chars: int = 0  # Result stub chars for resolved evidence in compound_selective (0 = use trace_stub_chars).
    compound_selective_recent_tool_results_chars: int = 0  # Rolling tool-result budget override for compound_selective (0 = use recent_tool_results_chars).
    compound_selective_trace_action_repeat_cap: int = 0  # Max identical trace actions kept in compound_selective (0 = no cap).
    compound_selective_resolved_action_repeat_cap: int = 0  # Max identical resolved-evidence actions kept in compound_selective (0 = no cap).
    compound_selective_trace_anchor_lines: int = 0  # Older trace actions reserved as anchors in compound_selective (0 = no anchors).
    compound_selective_resolved_anchor_lines: int = 0  # Older resolved-evidence actions reserved as anchors in compound_selective (0 = no anchors).
    compound_selective_trace_source_anchor_lines: int = 0  # Older non-test source anchors reserved in compound_selective trace selection (0 = disabled).
    compound_selective_trace_test_anchor_lines: int = 0  # Older test/verification anchors reserved in compound_selective trace selection (0 = disabled).
    compound_selective_resolved_source_anchor_lines: int = 0  # Older non-test source anchors reserved in compound_selective resolved evidence (0 = disabled).
    compound_selective_resolved_test_anchor_lines: int = 0  # Older test/verification anchors reserved in compound_selective resolved evidence (0 = disabled).


# Every key must exist in config.toml — no silent defaults at read time.
# Values here are the hardcoded safety net only for keys intentionally optional.
_REQUIRED_SECTIONS = ("server", "model", "loop", "output", "tools", "experiment", "prompts")


def _require(data: dict, section: str, key: str) -> object:
    if section not in data:
        raise KeyError(f"config.toml missing section [{section}]")
    if key not in data[section]:
        raise KeyError(f"config.toml missing key '{key}' in [{section}]")
    return data[section][key]


def _extract_config_fields(d: dict) -> dict:
    """Project the nested TOML dict onto Config field names.

    Required keys raise KeyError on absence (no silent fallback). Experiment
    fields default to empty since they are meant to be per-run overrides.
    """
    missing = [s for s in _REQUIRED_SECTIONS if s not in d]
    if missing:
        raise KeyError(f"config.toml missing section(s): {missing}")

    experiment = d.get("experiment", {})
    analysis = d.get("analysis", {})
    return {
        "base_url": _require(d, "server", "base_url"),
        "api_key": _require(d, "server", "api_key"),
        "timeout_connect": _require(d, "server", "timeout_connect"),
        "timeout_read": _require(d, "server", "timeout_read"),
        "health_poll_interval": _require(d, "server", "health_poll_interval"),
        "health_timeout": _require(d, "server", "health_timeout"),
        "launch_timeout": _require(d, "server", "launch_timeout"),
        "stop_settle": _require(d, "server", "stop_settle"),
        "model": _require(d, "model", "name"),
        "context_size": _require(d, "model", "context_size"),
        "context_fill_ratio": _require(d, "model", "context_fill_ratio"),
        "max_tokens": _require(d, "model", "max_tokens"),
        "max_turns": _require(d, "loop", "max_turns"),
        "max_sessions": _require(d, "loop", "max_sessions"),
        "duplicate_abort": _require(d, "loop", "duplicate_abort"),
        "error_nudge_threshold": _require(d, "loop", "error_nudge_threshold"),
        "rumination_nudge_threshold": _require(d, "loop", "rumination_nudge_threshold"),
        "rumination_gate_max_blocks": d.get("loop", {}).get("rumination_gate_max_blocks", 0),
        "rumination_gate_arm_threshold": d.get("loop", {}).get("rumination_gate_arm_threshold", 0),
        "rumination_gate_arm_threshold_abs": d.get("loop", {}).get("rumination_gate_arm_threshold_abs", 0),
        "rumination_nudge_threshold_abs": d.get("loop", {}).get("rumination_nudge_threshold_abs", 0),
        "rumination_nudge_threshold_abs_post_mutation": d.get("loop", {}).get("rumination_nudge_threshold_abs_post_mutation", 0),
        "rumination_nudge_only_pre_mutation": d.get("loop", {}).get("rumination_nudge_only_pre_mutation", False),
        "rumination_same_target_warn_count": d.get("loop", {}).get("rumination_same_target_warn_count", 0),
        "rumination_same_target_arm_count": d.get("loop", {}).get("rumination_same_target_arm_count", 0),
        "test_read_warn_after": d.get("loop", {}).get("test_read_warn_after", 0),
        "context_inspect_repeat_threshold": d.get("loop", {}).get("context_inspect_repeat_threshold", 0),
        "contract_commit_warn_after": d.get("loop", {}).get("contract_commit_warn_after", 0),
        "contract_commit_block_after": d.get("loop", {}).get("contract_commit_block_after", 0),
        "contract_recovery_same_target_threshold": d.get("loop", {}).get("contract_recovery_same_target_threshold", 0),
        "contract_recovery_verify_repeat_threshold": d.get("loop", {}).get("contract_recovery_verify_repeat_threshold", 0),
        "contract_invalid_repeat_abort_after": d.get("loop", {}).get("contract_invalid_repeat_abort_after", 0),
        "contract_abort_min_turns_since_commit_arm": d.get("loop", {}).get("contract_abort_min_turns_since_commit_arm", 0),
        "contract_abort_min_turns_since_recovery_arm": d.get("loop", {}).get("contract_abort_min_turns_since_recovery_arm", 0),
        "contract_abort_requires_zero_mutation": d.get("loop", {}).get("contract_abort_requires_zero_mutation", False),
        "contract_equivalent_action_classes_enabled": d.get("loop", {}).get("contract_equivalent_action_classes_enabled", False),
        "mutation_repeat_warn_after": d.get("loop", {}).get("mutation_repeat_warn_after", 0),
        "mutation_repeat_block_after": d.get("loop", {}).get("mutation_repeat_block_after", 0),
        "mutation_repeat_abort_after": d.get("loop", {}).get("mutation_repeat_abort_after", 0),
        "duplicate_warn_count": d.get("loop", {}).get("duplicate_warn_count", 0),
        "error_abort_threshold": d.get("loop", {}).get("error_abort_threshold", 0),
        "intent_abort_threshold": d.get("loop", {}).get("intent_abort_threshold", 0),
        "duplicate_guard_enabled": d.get("loop", {}).get("duplicate_guard_enabled", True),
        "post_edit_check_enabled": d.get("post_edit_check", {}).get("enabled", False),
        "post_edit_check_timeout": d.get("post_edit_check", {}).get("timeout", 10),
        "post_edit_checks": list(d.get("post_edit_check", {}).get("checks", [])),
        "parallel_readonly_enabled": d.get("loop", {}).get("parallel_readonly_enabled", False),
        "parallel_max_workers": d.get("loop", {}).get("parallel_max_workers", 4),
        "injections_enabled": d.get("injections", {}).get("enabled", False),
        "injections_dir": d.get("injections", {}).get("dir", ".harness/injections"),
        "loop_detect_enabled": d.get("loop", {}).get("loop_detect_enabled", False),
        "loop_detect_threshold": d.get("loop", {}).get("loop_detect_threshold", 5),
        "loop_detect_recovery": d.get("prompts", {}).get(
            "loop_detect_recovery",
            "<system-reminder>Loop detected: the last {streak} tool calls "
            "all have identical name and arguments. Stop repeating. Re-read "
            "the task, read a file you have not inspected yet, or change "
            "approach. One more repeat ends the session.</system-reminder>",
        ),
        "done_guard_enabled": d.get("loop", {}).get("done_guard_enabled", True),
        "rumination_enabled": d.get("loop", {}).get("rumination_enabled", True),
        "error_ladder_enabled": d.get("loop", {}).get("error_ladder_enabled", True),
        "bash_transforms_universal_enabled": d.get("loop", {}).get("bash_transforms_universal_enabled", True),
        "bash_transforms_task_format_enabled": d.get("loop", {}).get("bash_transforms_task_format_enabled", True),
        "bash_transforms_structured_output_enabled": d.get("loop", {}).get("bash_transforms_structured_output_enabled", False),
        "bash_transforms_sink_threshold_chars": d.get("loop", {}).get("bash_transforms_sink_threshold_chars", 0),
        "rumination_gate_grace_calls": d.get("loop", {}).get("rumination_gate_grace_calls", 1),
        "rumination_min_threshold": d.get("loop", {}).get("rumination_min_threshold", 6),
        "done_require_mutation": d.get("loop", {}).get("done_require_mutation", True),
        "done_require_verify": d.get("loop", {}).get("done_require_verify", True),
        "done_verified_bash_min_chars": d.get("loop", {}).get("done_verified_bash_min_chars", 200),
        "done_require_pretest_parity": d.get("loop", {}).get("done_require_pretest_parity", False),
        "done_parity_runs_required": d.get("loop", {}).get("done_parity_runs_required", 1),
        "adaptive_policy_enabled": d.get("loop", {}).get("adaptive_policy_enabled", False),
        "adaptive_switch_min_turn": d.get("loop", {}).get("adaptive_switch_min_turn", 0),
        "adaptive_requires_mutation": d.get("loop", {}).get("adaptive_requires_mutation", True),
        "adaptive_requires_test_signal": d.get("loop", {}).get("adaptive_requires_test_signal", True),
        "adaptive_low_pressure_window": d.get("loop", {}).get("adaptive_low_pressure_window", 0),
        "adaptive_low_pressure_max_events": d.get("loop", {}).get("adaptive_low_pressure_max_events", 0),
        "adaptive_phase2_done_guard_enabled": d.get("loop", {}).get("adaptive_phase2_done_guard_enabled", True),
        "adaptive_phase2_bash_task_format_enabled": d.get("loop", {}).get("adaptive_phase2_bash_task_format_enabled", True),
        "adaptive_phase2_bash_structured_output_enabled": d.get("loop", {}).get("adaptive_phase2_bash_structured_output_enabled", True),
        "adaptive_phase2_bash_sink_threshold_chars": d.get("loop", {}).get("adaptive_phase2_bash_sink_threshold_chars", 0),
        "require_intent": d.get("loop", {}).get("require_intent", False),
        "intent_grace_turns": _require(d, "loop", "intent_grace_turns"),
        "min_turns_before_context": _require(d, "loop", "min_turns_before_context"),
        "max_output_chars": _require(d, "output", "max_output_chars"),
        "truncate_head_ratio": _require(d, "output", "truncate_head_ratio"),
        "truncate_head_lines": _require(d, "output", "truncate_head_lines"),
        "truncate_tail_lines": _require(d, "output", "truncate_tail_lines"),
        "args_summary_chars": _require(d, "output", "args_summary_chars"),
        "trace_args_summary_chars": _require(d, "output", "trace_args_summary_chars"),
        "trace_reasoning_store_chars": _require(d, "output", "trace_reasoning_store_chars"),
        "solver_trace_lines": _require(d, "output", "solver_trace_lines"),
        "solver_evidence_lines": _require(d, "output", "solver_evidence_lines"),
        "solver_inference_lines": _require(d, "output", "solver_inference_lines"),
        "recent_tool_results_chars": _require(d, "output", "recent_tool_results_chars"),
        "trace_stub_chars": _require(d, "output", "trace_stub_chars"),
        "trace_reasoning_chars": _require(d, "output", "trace_reasoning_chars"),
        "context_slot_max_candidates": d.get("output", {}).get("context_slot_max_candidates", 1),
        "context_slot_inline_files": d.get("output", {}).get("context_slot_inline_files", 1),
        "focused_compound_trace_lines": d.get("output", {}).get("focused_compound_trace_lines", 0),
        "focused_compound_evidence_lines": d.get("output", {}).get("focused_compound_evidence_lines", 0),
        "focused_compound_recent_tool_results_chars": d.get("output", {}).get("focused_compound_recent_tool_results_chars", 0),
        "focused_compound_include_resolved_evidence": d.get("output", {}).get("focused_compound_include_resolved_evidence", False),
        "compound_selective_trace_lines": d.get("output", {}).get("compound_selective_trace_lines", 0),
        "compound_selective_unresolved_evidence_lines": d.get("output", {}).get("compound_selective_unresolved_evidence_lines", 0),
        "compound_selective_resolved_evidence_lines": d.get("output", {}).get("compound_selective_resolved_evidence_lines", 0),
        "compound_selective_resolved_evidence_stub_chars": d.get("output", {}).get("compound_selective_resolved_evidence_stub_chars", 0),
        "compound_selective_recent_tool_results_chars": d.get("output", {}).get("compound_selective_recent_tool_results_chars", 0),
        "compound_selective_trace_action_repeat_cap": d.get("output", {}).get("compound_selective_trace_action_repeat_cap", 0),
        "compound_selective_resolved_action_repeat_cap": d.get("output", {}).get("compound_selective_resolved_action_repeat_cap", 0),
        "compound_selective_trace_anchor_lines": d.get("output", {}).get("compound_selective_trace_anchor_lines", 0),
        "compound_selective_resolved_anchor_lines": d.get("output", {}).get("compound_selective_resolved_anchor_lines", 0),
        "compound_selective_trace_source_anchor_lines": d.get("output", {}).get("compound_selective_trace_source_anchor_lines", 0),
        "compound_selective_trace_test_anchor_lines": d.get("output", {}).get("compound_selective_trace_test_anchor_lines", 0),
        "compound_selective_resolved_source_anchor_lines": d.get("output", {}).get("compound_selective_resolved_source_anchor_lines", 0),
        "compound_selective_resolved_test_anchor_lines": d.get("output", {}).get("compound_selective_resolved_test_anchor_lines", 0),
        "pretest_head_chars": _require(d, "output", "pretest_head_chars"),
        "pretest_tail_chars": _require(d, "output", "pretest_tail_chars"),
        "bash_timeout": _require(d, "tools", "bash_timeout"),
        "grep_timeout": _require(d, "tools", "grep_timeout"),
        "search_pagination_enabled": d.get("tools", {}).get("search_pagination_enabled", True),
        "grep_max_matches_per_page": d.get("tools", {}).get("grep_max_matches_per_page", 25),
        "glob_max_matches_per_page": d.get("tools", {}).get("glob_max_matches_per_page", 25),
        "edit_strict_match": d.get("tools", {}).get("edit_strict_match", True),
        "edit_fuzzy_cascade_enabled": d.get("tools", {}).get("edit_fuzzy_cascade_enabled", False),
        "edit_candidate_count": d.get("tools", {}).get("edit_candidate_count", 3),
        "pretest_timeout": _require(d, "tools", "pretest_timeout"),
        "llama_server_bin": _require(d, "tools", "llama_server_bin"),
        "sandbox_bash": _require(d, "tools", "sandbox_bash"),
        "strip_ansi": _require(d, "tools", "strip_ansi"),
        "collapse_blank_lines": _require(d, "tools", "collapse_blank_lines"),
        "collapse_duplicate_lines": _require(d, "tools", "collapse_duplicate_lines"),
        "collapse_similar_lines": _require(d, "tools", "collapse_similar_lines"),
        "bwrap_bin": _require(d, "tools", "bwrap_bin"),
        "max_transient_retries": _require(d, "loop", "max_transient_retries"),
        "retry_backoff": tuple(_require(d, "loop", "retry_backoff")),
        "system_header": _require(d, "prompts", "system_header"),
        "state_context_suffix": _require(d, "prompts", "state_context_suffix"),
        "intent_gate_first": _require(d, "prompts", "intent_gate_first"),
        "intent_gate_repeat": _require(d, "prompts", "intent_gate_repeat"),
        "resume_base": _require(d, "prompts", "resume_base"),
        "error_nudge": _require(d, "prompts", "error_nudge"),
        "rumination_nudge": _require(d, "prompts", "rumination_nudge"),
        "rumination_gate": _require(d, "prompts", "rumination_gate"),
        "rumination_same_target_nudge": d.get(
            "prompts", {}
        ).get(
            "rumination_same_target_nudge",
            "[HARNESS: same target hit {count} times without a write/edit ({target}). Stop rereading it; either edit, verify, or move to a different target.]",
        ),
        "rumination_outside_cwd_nudge": d.get(
            "prompts", {}
        ).get(
            "rumination_outside_cwd_nudge",
            "[HARNESS: repeated inspection is anchored outside the repo root ({target}). The working directory is already correct; search and read relative to it.]",
        ),
        "test_read_nudge": d.get(
            "prompts", {}
        ).get(
            "test_read_nudge",
            "[HARNESS: ran verification {count} time(s) without reading the target test file ({target}). Read the test before more checks.]",
        ),
        "contract_commit_warn": d.get(
            "prompts", {}
        ).get(
            "contract_commit_warn",
            "[HARNESS: source file {source} is already in view. Choose a concrete next move: edit/write, read a test file, or run verification. Do not continue broad inspection.]",
        ),
        "contract_commit_block": d.get(
            "prompts", {}
        ).get(
            "contract_commit_block",
            "[HARNESS: commit contract active from {source}. This tool call was not executed. Allowed next moves: edit/write, read a test file, or run verification.]",
        ),
        "contract_recovery_block": d.get(
            "prompts", {}
        ).get(
            "contract_recovery_block",
            "[HARNESS: recovery mode for {reason} ({target}). This tool call was not executed. Allowed next moves: read a concrete file, edit/write, or run verification.]",
        ),
        "mutation_repeat_warn": d.get(
            "prompts", {}
        ).get(
            "mutation_repeat_warn",
            "[HARNESS: the same mutation was already applied to {target}. Do not repeat it unchanged; read new evidence, run verification, or change the mutation.]",
        ),
        "mutation_repeat_block": d.get(
            "prompts", {}
        ).get(
            "mutation_repeat_block",
            "[HARNESS: repeated identical mutation on {target}. This tool call was not executed. Read new evidence, run verification, or change the mutation.]",
        ),
        "read_truncated_reminder": d.get("prompts", {}).get(
            "read_truncated_reminder",
            "<system-reminder>Read returned the first {returned_lines} lines of {path}. The file is longer — re-read with a higher limit or a specific offset to see more.</system-reminder>",
        ),
        "read_empty_reminder": d.get("prompts", {}).get(
            "read_empty_reminder",
            "<system-reminder>File {path} exists but is empty (0 bytes).</system-reminder>",
        ),
        "sink_pointer": d.get("prompts", {}).get(
            "sink_pointer",
            '<tool_result_meta truncated="true" original_bytes="{chars}" original_lines="{lines}" full_path="{path}"/>',
        ),
        "sink_body_marker": d.get("prompts", {}).get(
            "sink_body_marker",
            "... [body truncated — full output available via full_path attribute] ...",
        ),
        "sink_head_bytes": d.get("tools", {}).get("sink_head_bytes", 1000),
        "sink_tail_bytes": d.get("tools", {}).get("sink_tail_bytes", 1000),
        "resume_duplicate_abort": _require(d, "prompts", "resume_duplicate_abort"),
        "resume_context_full": _require(d, "prompts", "resume_context_full"),
        "resume_max_turns": _require(d, "prompts", "resume_max_turns"),
        "resume_length": _require(d, "prompts", "resume_length"),
        "resume_gate_escalation": d.get("prompts", {}).get("resume_gate_escalation", "Session ended: rumination gate blocked {n} consecutive calls. Your current code has been preserved."),
        "resume_last_n_actions": _require(d, "prompts", "resume_last_n_actions"),
        "tool_desc": experiment.get("tool_desc", "minimal"),
        "prompt_addendum": experiment.get("prompt_addendum", ""),
        "variant_name": experiment.get("variant_name", ""),
        "runtime_mode": d.get("runtime", {}).get("mode", "measurement"),
        "analysis_task_format": analysis.get("task_format", "pytest"),
        "provider": d.get("server", {}).get("provider", "openai-compatible"),
    }


def _validate_coupling(cfg: Config) -> None:
    """Reject config combinations that produce silent fallthrough.

    Bucket B toggles have coupling constraints (one feature's effective
    behaviour depends on another being enabled). Without validation the
    model sees an unhelpful fallback — e.g. structured output enabled
    but task-format control disabled produces no parser, no digest, and
    no error. Campaigns waste hours on silently-degraded runs.

    Rules:
      - bash_transforms_structured_output_enabled requires
        bash_transforms_task_format_enabled — the parser is loaded
        through the task-format path.
      - done_require_pretest_parity only meaningfully activates when
        bash_transforms_structured_output_enabled is also on. Parity
        falls back to the heuristic otherwise; a warning makes the
        silent downgrade visible.
    """
    if (cfg.bash_transforms_structured_output_enabled
            and not cfg.bash_transforms_task_format_enabled):
        raise ValueError(
            "config error: bash_transforms_structured_output_enabled = true "
            "requires bash_transforms_task_format_enabled = true — the "
            "structured parser is loaded via the task-format path."
        )
    if (cfg.done_require_pretest_parity
            and not cfg.bash_transforms_structured_output_enabled):
        log = logging.getLogger(__name__)
        log.warning(
            "done_require_pretest_parity is on but structured output is "
            "disabled; parity will fall back to the heuristic preconditions "
            "(done_require_mutation / done_require_verify). Enable "
            "bash_transforms_structured_output_enabled for ground-truth parity."
        )
    if (cfg.adaptive_policy_enabled
            and cfg.adaptive_phase2_bash_structured_output_enabled
            and not cfg.adaptive_phase2_bash_task_format_enabled):
        raise ValueError(
            "config error: adaptive phase2 structured output requires "
            "adaptive_phase2_bash_task_format_enabled = true."
        )
    if cfg.runtime_mode not in VALID_RUNTIME_MODES:
        raise ValueError(
            "config error: runtime.mode must be one of "
            f"{', '.join(VALID_RUNTIME_MODES)}; got {cfg.runtime_mode!r}"
        )


def require_runtime_mode(cfg: Config, *, expected: str, caller: str) -> None:
    """Reject a runtime invoked under the wrong mode.

    Measurement and assistant entrypoints share the same engine but must not
    silently cross-enable one another's behavior. Entry points call this early.
    """
    if expected not in VALID_RUNTIME_MODES:
        raise ValueError(f"unknown runtime mode expectation: {expected!r}")
    if cfg.runtime_mode != expected:
        raise ValueError(
            f"{caller} requires runtime.mode={expected!r}, "
            f"but resolved runtime.mode={cfg.runtime_mode!r}"
        )


def load_config(
    user_config: Path | list[Path] | None = None,
    overrides: dict | None = None,
) -> Config:
    """Load layered config, merge optional extra user TOML(s), apply CLI overrides.

    user_config: a single path OR a list of paths. When a list, overlays
                 layer in the given order — later entries win on conflict.
                 This lets a campaign compose atomic toggles (e.g.
                 configs/substantive.toml + configs/toggles/intent.on.toml)
                 without pre-baking every combination.
    overrides:   flat dict of CLI flag overrides (highest priority)
    """
    base = dict(_LAYERED)  # start from already-merged base + local

    if user_config is not None:
        paths: list[Path] = (
            [user_config] if isinstance(user_config, (str, Path))
            else list(user_config)
        )
        for p in paths:
            _deep_merge(base, _load_toml(Path(p)))

    flat = _extract_config_fields(base)

    if overrides:
        for k, v in overrides.items():
            if v is not None and k in flat:
                flat[k] = type(flat[k])(v)
    flat["api_key"] = _resolve_env_secret(str(flat["api_key"]), "server.api_key")

    cfg = Config(**flat)
    _validate_coupling(cfg)
    return cfg


def _resolve_env_secret(value: str, config_key: str) -> str:
    """Resolve persisted env references without storing API keys in session DBs."""
    for prefix in ("$ENV:", "env:"):
        if value.startswith(prefix):
            env_name = value[len(prefix):]
            resolved = os.environ.get(env_name)
            if resolved is None:
                raise KeyError(f"{config_key} references unset environment variable {env_name!r}")
            return resolved
    return value


def dump_config(cfg: Config) -> dict:
    """Return a serializable snapshot of a resolved Config for provenance logging."""
    from dataclasses import asdict
    d = asdict(cfg)
    # retry_backoff is a tuple; convert for JSON
    d["retry_backoff"] = list(cfg.retry_backoff)
    return d
