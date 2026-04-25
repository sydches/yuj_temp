"""Guardrails — L5 harness sub-concern that prevents model thrash and loops.

Each guardrail is a small state machine with a uniform interface:

    evaluate(state, cfg, ctx) -> Decision

Where ``Decision.action`` is one of PASS / WARN / BLOCK / END. The turn
loop walks guardrails in fixed precedence order and acts on each
decision:

    PASS   — no-op, proceed.
    WARN   — append ``decision.text`` to the tool result (after dispatch).
    BLOCK  — reject the tool call without dispatching; ``decision.text``
             becomes the tool result; the call is marked gate_blocked.
    END    — terminate the session with ``decision.reason`` as the
             SessionResult.finish_reason.

The ladder for every guardrail is WARN → BLOCK → END (absent tiers are
PASS). Ordering the tiers this way gives the model progressive feedback
instead of jumping straight to session termination.

See docs/separation_of_concerns.md "Harness sub-concern: Guardrails".
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from enum import Enum
import hashlib
import json
import re
import shlex
from typing import Any, Callable

from ._shell_patterns import TEST_COMMAND_RE as _TEST_COMMAND_RE


# ─── Decision type ────────────────────────────────────────────────────────

class Action(Enum):
    PASS = "pass"
    WARN = "warn"
    BLOCK = "block"
    END = "end"


@dataclass(frozen=True)
class Decision:
    action: Action
    text: str = ""
    reason: str = ""

    @classmethod
    def pass_(cls) -> "Decision":
        return cls(Action.PASS)

    @classmethod
    def warn(cls, text: str) -> "Decision":
        return cls(Action.WARN, text=text)

    @classmethod
    def block(cls, text: str) -> "Decision":
        return cls(Action.BLOCK, text=text)

    @classmethod
    def end(cls, reason: str) -> "Decision":
        return cls(Action.END, reason=reason)


PASS = Decision.pass_()


TURN_PRE_DISPATCH_ORDER = ("intent_gate", "duplicate_guard", "loop_detect")
TOOL_PRE_DISPATCH_ORDER = ("done_guard", "mutation_repeat_guard", "contract_gate", "rumination_gate")
TOOL_POST_DISPATCH_ORDER = ("error_ladder", "test_read_ladder", "rumination_ladder")
OBSERVER_ORDER = ("mark_bash_verified", "observe_test_file_read", "observe_contract_state")


@dataclass(frozen=True)
class GuardrailRegistry:
    """Composable guardrail callables grouped by run-loop phase."""

    turn_pre_dispatch: dict[str, Callable[..., Decision]]
    tool_pre_dispatch: dict[str, Callable[..., Decision]]
    tool_post_dispatch: dict[str, Callable[..., Decision]]
    observers: dict[str, Callable[..., None]]


# ─── Shared state ─────────────────────────────────────────────────────────

@dataclass
class GuardrailState:
    """All per-session state owned by guardrails.

    Session holds one of these and passes it to each guardrail's
    evaluate() call. Moving this out of Session keeps the ~15
    thrash-control fields together (they belong together) and makes the
    turn loop's responsibility clear: it orchestrates, the guardrails
    own their own state.
    """
    recent_calls: deque = field(default_factory=lambda: deque(maxlen=0))
    consecutive_errors: dict[str, int] = field(default_factory=dict)
    intent_block_count: int = 0
    intent_first_block_turn: int | None = None
    consecutive_intent_rejections: int = 0
    non_write_calls_since_write: int = 0
    same_target_key: str = ""
    same_target_display: str = ""
    same_target_count: int = 0
    same_target_nudge_emitted: bool = False
    rumination_gate: bool = False
    rumination_gate_grace: int = 0
    rumination_nudge_emitted: bool = False
    gate_block_count: int = 0
    has_mutated: bool = False
    verified_since_mutation: bool = False
    # Running count of successful write/edit operations. Used by the
    # regression observer to decide whether "PASSED→FAILED" on a test
    # had an intervening mutation (flaky-vs-caused-by-edit disambiguation).
    mutation_count: int = 0

    # Derived thresholds (computed at session init, read during the loop).
    # `rumination_nudge_threshold` is the PRE-first-mutation value.
    # `rumination_nudge_threshold_post_mutation` is used once has_mutated=True.
    # Both default to the same computed value if no asymmetric config is set.
    rumination_nudge_threshold: int = 0
    rumination_nudge_threshold_post_mutation: int = 0
    rumination_arm_threshold: int = 0

    # Pretest parity (done_guard ground truth — filled at session 1 start
    # by the harness parsing pretest output through the task format's
    # [output_parser]. Empty sets = pretest not parseable; done_guard
    # falls back to heuristic preconditions in that case).
    pretest_failing_tests: set[str] = field(default_factory=set)
    pretest_passing_tests: set[str] = field(default_factory=set)
    latest_test_parsed: dict[str, str] = field(default_factory=dict)
    green_parity_streak: int = 0
    test_file_reads: set[str] = field(default_factory=set)
    test_runs_without_test_read: int = 0
    last_test_target: str = ""
    test_read_nudge_target: str = ""

    # Regression observability (independent of pretest-parity mode).
    # Holds the prior test run's parsed verdicts and the mutation count
    # observed alongside it. Session compares incoming parsed verdicts
    # against prev_test_parsed to detect PASSED→FAILED transitions
    # with at least one intervening mutation.
    prev_test_parsed: dict[str, str] = field(default_factory=dict)
    mutation_count_at_prev_test: int = 0
    commit_pending: bool = False
    commit_source_path: str = ""
    commit_violation_count: int = 0
    commit_turns_since_arm: int = 0
    contract_block_sig: str = ""
    contract_block_count: int = 0
    recovery_mode_active: bool = False
    recovery_reason: str = ""
    recovery_target: str = ""
    recovery_turns_since_arm: int = 0
    verify_repeat_sig: str = ""
    verify_repeat_count: int = 0
    mutation_count_at_last_verify: int = 0
    mutation_repeat_sig: str = ""
    mutation_repeat_target: str = ""
    mutation_repeat_count: int = 0
    mutation_repeat_block_sig: str = ""
    mutation_repeat_block_count: int = 0
    # loop_detect guardrail (tighter than duplicate_guard, with a
    # recovery-inject step before hard abort). Tracks the current
    # consecutive-identical streak plus a one-shot flag that arms END
    # on the very next repeat after the WARN has been emitted.
    loop_detect_last_sig: tuple = ()
    loop_detect_streak: int = 0
    loop_detect_warned: bool = False


def init_guardrail_state(cfg: Any) -> GuardrailState:
    """Build a GuardrailState seeded from cfg at session start."""
    # Nudge threshold: absolute (if > 0) overrides percentage-of-max_turns.
    # Still clamped to the min-threshold floor so trivial max_turns values
    # don't collapse the nudge below usefulness.
    if cfg.rumination_nudge_threshold_abs > 0:
        nudge = max(cfg.rumination_min_threshold, cfg.rumination_nudge_threshold_abs)
    else:
        nudge = max(
            cfg.rumination_min_threshold,
            int(cfg.max_turns * cfg.rumination_nudge_threshold / 100),
        )
    # Post-mutation nudge threshold: separate knob. When > 0, overrides the
    # pre-mutation value for the rumination ladder's post-has_mutated checks.
    # When 0, post = pre (symmetric default).
    if cfg.rumination_nudge_threshold_abs_post_mutation > 0:
        nudge_post = max(cfg.rumination_min_threshold,
                         cfg.rumination_nudge_threshold_abs_post_mutation)
    else:
        nudge_post = nudge
    # Arm threshold: absolute (if > 0) overrides percentage-of-max_turns.
    # Absolute form decouples the gate from max_turns so reducing the
    # turn budget doesn't tighten the gate proportionally.
    if cfg.rumination_gate_arm_threshold_abs > 0:
        arm = max(nudge, cfg.rumination_gate_arm_threshold_abs)
    else:
        arm_pct = cfg.rumination_gate_arm_threshold or cfg.rumination_nudge_threshold
        arm = max(nudge, int(cfg.max_turns * arm_pct / 100))
    # Deque max length must tolerate duplicate_abort=0 (guardrail disabled).
    # maxlen=0 would make the deque never retain anything; treat 0 as "no
    # abort" by giving the deque a nominal length of 1 so it still appends
    # (the guardrail function will short-circuit on its enabled flag).
    deque_len = max(1, cfg.duplicate_abort)
    return GuardrailState(
        recent_calls=deque(maxlen=deque_len),
        rumination_nudge_threshold=nudge,
        rumination_nudge_threshold_post_mutation=nudge_post,
        rumination_arm_threshold=arm,
    )


# ─── Turn-level pre-dispatch guardrails ──────────────────────────────────
# These look at the whole turn (all tool_calls together) before any
# tool executes. Called once per turn.

def intent_gate(state: GuardrailState, cfg: Any, *,
                turn: int, content: str, tool_calls: list) -> Decision:
    """Reject silent tool calls.

    GRACE: first ``cfg.intent_grace_turns`` turns get a free pass.
    BLOCK: tool_calls present + no reasoning content → reject this turn.
    END: ``cfg.intent_abort_threshold`` consecutive rejections → end session.
    """
    if not cfg.require_intent or not tool_calls:
        state.consecutive_intent_rejections = 0
        return PASS
    if turn < cfg.intent_grace_turns:
        state.consecutive_intent_rejections = 0
        return PASS
    if (content or "").strip():
        state.consecutive_intent_rejections = 0
        return PASS

    state.intent_block_count += 1
    state.consecutive_intent_rejections += 1
    if state.intent_first_block_turn is None:
        state.intent_first_block_turn = turn
        text = cfg.intent_gate_first
    else:
        text = cfg.intent_gate_repeat.format(
            count=state.intent_block_count,
            first_turn=state.intent_first_block_turn,
        )
    if (cfg.intent_abort_threshold > 0
            and state.consecutive_intent_rejections >= cfg.intent_abort_threshold):
        # END overrides BLOCK: record the rejection text but end the session.
        return Decision(Action.END, text=text, reason="intent_abort")
    return Decision.block(text)


def loop_detect(state: GuardrailState, cfg: Any, *,
                tool_calls_sig: tuple) -> Decision:
    """Tight loop detector with one recovery-inject before hard abort.

    Borrowed in spirit from Gemini CLI's LoopDetectionService
    (``packages/core/src/services/loopDetectionService.ts``, PR #8231):
    on the same structural hash repeating for N turns, inject a
    synthetic steering message and allow the model one more turn to
    change approach. If the pattern persists, end the session.

    Differs from ``duplicate_guard`` by firing at a much tighter
    threshold (default 5) and by the WARN tier — duplicate_guard's WARN
    is a threshold announcement; this WARN is a recovery-inject.

    State slice: ``loop_detect_*`` on ``GuardrailState``. Registry
    phase: ``turn_pre_dispatch``.
    """
    if not cfg.loop_detect_enabled:
        state.loop_detect_streak = 0
        state.loop_detect_last_sig = ()
        state.loop_detect_warned = False
        return PASS
    if tool_calls_sig == state.loop_detect_last_sig:
        state.loop_detect_streak += 1
    else:
        state.loop_detect_last_sig = tool_calls_sig
        state.loop_detect_streak = 1
        state.loop_detect_warned = False
    if state.loop_detect_streak >= cfg.loop_detect_threshold:
        if not state.loop_detect_warned:
            state.loop_detect_warned = True
            return Decision.warn(
                cfg.loop_detect_recovery.format(streak=state.loop_detect_streak)
            )
        return Decision.end("loop_detected")
    return PASS


def duplicate_guard(state: GuardrailState, cfg: Any, *,
                    tool_calls_sig: tuple) -> Decision:
    """End session on N identical consecutive calls; WARN one turn earlier.

    Fires on every turn, including while the rumination gate is armed —
    pausing here would let the model cycle the same blocked call up to
    rumination_gate_max_blocks times before any terminal action fires,
    which is LARGER tolerance during a MORE dangerous state.
    """
    if not cfg.duplicate_guard_enabled:
        return PASS
    state.recent_calls.append(tool_calls_sig)
    # END
    if (len(state.recent_calls) == cfg.duplicate_abort
            and len(set(state.recent_calls)) == 1):
        return Decision.end("duplicate_abort")
    # WARN (optional, config-gated)
    if cfg.duplicate_warn_count > 0:
        tail = 0
        for s in reversed(state.recent_calls):
            if s == tool_calls_sig:
                tail += 1
            else:
                break
        if tail >= cfg.duplicate_warn_count:
            return Decision.warn(
                cfg.duplicate_warn.format(count=tail, abort=cfg.duplicate_abort)
            )
    return PASS


# ─── Per-tool-call pre-dispatch guardrails ───────────────────────────────

def done_guard(state: GuardrailState, cfg: Any, *, tc_name: str) -> Decision:
    """Verify `done` is premature or legitimate.

    Two modes, selected by cfg.done_require_pretest_parity:

    PARITY MODE (opt-in, ground-truth):
      Requires the structured output pipeline. At session 1 start, the
      harness parses pretest output into failing/passing test sets and
      stores them in state.pretest_failing_tests / pretest_passing_tests.
      Each subsequent test run updates state.latest_test_parsed and the
      green_parity_streak counter. `done` is accepted only when:
        1. the latest test run covers every pretest-failing test and
           every one of those now shows PASSED, AND
        2. no pretest-passing test is now FAILED/ERROR (no regression),
           AND
        3. the parity streak has reached cfg.done_parity_runs_required.
      Reject otherwise with a reason naming the specific tests that
      block acceptance.

      When pretest was not parseable (sets empty), PARITY MODE falls
      back to HEURISTIC MODE so non-test tasks / runners without an
      [output_parser] block stay functional.

    HEURISTIC MODE (default):
      Requires the mutation + verified_since_mutation preconditions
      (the 200-char bash heuristic — content-blind, task-agnostic,
      but imprecise: the tracker recorded false rejections under it).
    """
    if tc_name != "done":
        return PASS
    if not cfg.done_guard_enabled:
        return PASS

    use_parity = (
        getattr(cfg, "done_require_pretest_parity", False)
        and state.pretest_failing_tests
    )
    if use_parity:
        latest = state.latest_test_parsed
        if not latest:
            return Decision.block(cfg.done_reject_parity_no_run)
        passed_now = {t for t, v in latest.items() if v in ("PASSED", "PASS")}
        still_failing = state.pretest_failing_tests - passed_now
        if still_failing:
            shown = sorted(still_failing)[:5]
            extra_count = len(still_failing) - len(shown)
            return Decision.block(cfg.done_reject_parity_still_failing.format(
                shown=shown,
                extra=f" (+{extra_count} more)" if extra_count > 0 else "",
            ))
        regressed = {
            t for t, v in latest.items()
            if t in state.pretest_passing_tests and v not in ("PASSED", "PASS")
        }
        if regressed:
            shown = sorted(regressed)[:5]
            extra_count = len(regressed) - len(shown)
            return Decision.block(cfg.done_reject_parity_regression.format(
                shown=shown,
                extra=f" (+{extra_count} more)" if extra_count > 0 else "",
            ))
        required = getattr(cfg, "done_parity_runs_required", 1)
        if state.green_parity_streak < required:
            return Decision.block(cfg.done_reject_parity_streak.format(
                count=state.green_parity_streak,
                required=required,
            ))
        # Parity satisfied — bypass heuristic preconditions.
        return PASS

    # Fallback / heuristic mode.
    if cfg.done_require_mutation and not state.has_mutated:
        return Decision.block(cfg.done_reject_no_mutation)
    if cfg.done_require_verify and not state.verified_since_mutation:
        return Decision.block(cfg.done_reject_no_verify)
    return PASS


def _extract_read_path(
    tc_name: str,
    tc_args: dict | None = None,
    *,
    focus_key: str = "",
    focus_display: str = "",
) -> str:
    if tc_name == "read" and isinstance(tc_args, dict):
        raw = tc_args.get("path") or tc_args.get("file_path")
        if isinstance(raw, str):
            return raw
    if tc_name == "bash" and focus_key.startswith("file:"):
        return focus_display
    return ""


def _is_concrete_file_path(path: str) -> bool:
    if not path:
        return False
    candidate = path.split("::", 1)[0].rstrip(",")
    if candidate in {".", "..", "/"}:
        return False
    if candidate.endswith("/"):
        return False
    name = candidate.rsplit("/", 1)[-1]
    return bool(name) and name not in {".", ".."}


def _extract_bash_cmd(tc_name: str, tc_args: dict | None = None) -> str:
    if tc_name != "bash" or not isinstance(tc_args, dict):
        return ""
    raw = tc_args.get("cmd")
    return raw if isinstance(raw, str) else ""


def _is_test_command(tc_name: str, tc_args: dict | None = None) -> bool:
    return bool(_TEST_COMMAND_RE.search(_extract_bash_cmd(tc_name, tc_args)))


def _is_test_read(
    tc_name: str,
    tc_args: dict | None = None,
    *,
    focus_key: str = "",
    focus_display: str = "",
) -> bool:
    path = _extract_read_path(tc_name, tc_args, focus_key=focus_key, focus_display=focus_display)
    return _is_concrete_file_path(path) and _looks_like_test_path(path)


def _is_concrete_read(
    tc_name: str,
    tc_args: dict | None = None,
    *,
    focus_key: str = "",
    focus_display: str = "",
) -> bool:
    path = _extract_read_path(tc_name, tc_args, focus_key=focus_key, focus_display=focus_display)
    return _is_concrete_file_path(path)


def _clear_commit_contract(state: GuardrailState) -> None:
    state.commit_pending = False
    state.commit_source_path = ""
    state.commit_violation_count = 0
    state.commit_turns_since_arm = 0
    state.contract_block_sig = ""
    state.contract_block_count = 0


def _clear_recovery_mode(state: GuardrailState) -> None:
    state.recovery_mode_active = False
    state.recovery_reason = ""
    state.recovery_target = ""
    state.recovery_turns_since_arm = 0
    state.contract_block_sig = ""
    state.contract_block_count = 0


def _clear_mutation_repeat_state(state: GuardrailState) -> None:
    state.mutation_repeat_sig = ""
    state.mutation_repeat_target = ""
    state.mutation_repeat_count = 0
    state.mutation_repeat_block_sig = ""
    state.mutation_repeat_block_count = 0


def _arm_recovery_mode(state: GuardrailState, *, reason: str, target: str) -> None:
    state.recovery_mode_active = True
    state.recovery_reason = reason
    state.recovery_target = target
    state.recovery_turns_since_arm = 0


def _contract_violation_signature(
    cfg: Any,
    tc_name: str,
    tc_args: dict | None = None,
    *,
    focus_key: str = "",
    focus_display: str = "",
) -> str:
    if getattr(cfg, "contract_equivalent_action_classes_enabled", False):
        coarse = _equivalent_contract_violation_signature(
            tc_name, tc_args, focus_key=focus_key, focus_display=focus_display
        )
        if coarse:
            return coarse
    if focus_key:
        return focus_key
    if focus_display:
        return f"{tc_name}:{focus_display}"
    raw = json.dumps(tc_args or {}, sort_keys=True)
    return f"{tc_name}:{raw}"


def _equivalent_contract_violation_signature(
    tc_name: str,
    tc_args: dict | None = None,
    *,
    focus_key: str = "",
    focus_display: str = "",
) -> str:
    """Collapse semantically-equivalent non-progress moves into stable classes."""
    if tc_name == "read":
        path = _extract_read_path(tc_name, tc_args, focus_key=focus_key, focus_display=focus_display)
        if path:
            return f"read:{path}"
    if tc_name == "bash":
        cmd = (_extract_bash_cmd(tc_name, tc_args) or "").strip().lower()
        target = (focus_display or "").strip().rstrip("/")
        if "python -c" in cmd and "import " in cmd:
            return "bash:python-c-import-probe"
        if cmd.startswith("ls ") or cmd.startswith("ls\t"):
            return f"bash:ls:{target or '.'}"
        if cmd.startswith("find ") or cmd.startswith("cd ") and " find " in cmd:
            return f"bash:find:{target or '.'}"
        if cmd.startswith("pwd"):
            return "bash:pwd"
    if tc_name in ("glob", "grep"):
        path = (tc_args or {}).get("path", "")
        if isinstance(path, str) and path:
            return f"{tc_name}:path:{path}"
    return ""


def _mutation_signature(
    tc_name: str,
    tc_args: dict | None = None,
    *,
    focus_display: str = "",
) -> tuple[str, str]:
    if tc_name not in ("write", "edit") or not isinstance(tc_args, dict):
        return "", ""
    raw_path = tc_args.get("path") or tc_args.get("file_path") or focus_display
    path = raw_path if isinstance(raw_path, str) else ""
    if not _is_concrete_file_path(path):
        path = focus_display if _is_concrete_file_path(focus_display) else "current file"
    if tc_name == "edit":
        payload = json.dumps(
            {
                "old_str": tc_args.get("old_str", ""),
                "new_str": tc_args.get("new_str", ""),
            },
            sort_keys=True,
            ensure_ascii=True,
        )
    else:
        payload = str(tc_args.get("content", ""))
    digest = hashlib.sha1(payload.encode("utf-8", errors="ignore")).hexdigest()[:12]
    return f"{tc_name}:{path}:{digest}", path


def _record_mutation_repeat_block(state: GuardrailState, sig: str) -> int:
    if not sig:
        state.mutation_repeat_block_sig = ""
        state.mutation_repeat_block_count = 0
        return 0
    if state.mutation_repeat_block_sig == sig:
        state.mutation_repeat_block_count += 1
    else:
        state.mutation_repeat_block_sig = sig
        state.mutation_repeat_block_count = 1
    return state.mutation_repeat_block_count


def mutation_repeat_guard(
    state: GuardrailState,
    cfg: Any,
    *,
    tc_name: str,
    tc_args: dict | None = None,
    focus_display: str = "",
    **_: Any,
) -> Decision:
    """Warn/block/end repeated identical mutation attempts.

    Keyed by the mutation signature (tool + target + payload digest), so
    iterative edits to the same file still pass when the mutation changes.
    """
    warn_after = int(getattr(cfg, "mutation_repeat_warn_after", 0) or 0)
    block_after = int(getattr(cfg, "mutation_repeat_block_after", 0) or 0)
    abort_after = int(getattr(cfg, "mutation_repeat_abort_after", 0) or 0)
    if tc_name not in ("write", "edit") or (warn_after <= 0 and block_after <= 0 and abort_after <= 0):
        return PASS
    sig, target = _mutation_signature(tc_name, tc_args, focus_display=focus_display)
    if not sig or state.mutation_repeat_sig != sig or state.mutation_repeat_count <= 0:
        state.mutation_repeat_block_sig = ""
        state.mutation_repeat_block_count = 0
        return PASS

    next_count = state.mutation_repeat_count + 1
    shown_target = target or state.mutation_repeat_target or "current file"
    if block_after > 0 and next_count >= block_after:
        repeat_blocks = _record_mutation_repeat_block(state, sig)
        text = cfg.mutation_repeat_block.format(target=shown_target)
        if abort_after > 0 and repeat_blocks >= abort_after:
            return Decision(Action.END, text=text, reason="mutation_repeat_abort")
        return Decision.block(text)
    if warn_after > 0 and next_count >= warn_after:
        return Decision.warn(cfg.mutation_repeat_warn.format(target=shown_target))
    return PASS


def _contract_abort_allowed(
    state: GuardrailState,
    cfg: Any,
    *,
    lane: str,
) -> bool:
    if getattr(cfg, "contract_abort_requires_zero_mutation", False) and state.has_mutated:
        return False
    if lane == "commit":
        min_turns = int(getattr(cfg, "contract_abort_min_turns_since_commit_arm", 0) or 0)
        if min_turns > 0 and state.commit_turns_since_arm < min_turns:
            return False
        return True
    min_turns = int(getattr(cfg, "contract_abort_min_turns_since_recovery_arm", 0) or 0)
    if min_turns > 0 and state.recovery_turns_since_arm < min_turns:
        return False
    return True


def _record_contract_block(state: GuardrailState, sig: str) -> int:
    if not sig:
        state.contract_block_sig = ""
        state.contract_block_count = 0
        return 0
    if state.contract_block_sig == sig:
        state.contract_block_count += 1
    else:
        state.contract_block_sig = sig
        state.contract_block_count = 1
    return state.contract_block_count


def contract_gate(
    state: GuardrailState,
    cfg: Any,
    *,
    tc_name: str,
    tc_args: dict | None = None,
    focus_key: str = "",
    focus_display: str = "",
) -> Decision:
    """Block broad exploration once a tighter contract is active.

    Two content-blind contracts are supported:

    - Commit contract: after a non-test file read, the next useful move must be
      edit/write, read a test file, or run verification.
    - Recovery contract: once same-target / verify-repeat recovery arms, only a
      concrete read, edit/write, or verification command may execute.
    """
    if tc_name == "done":
        return PASS

    is_commit_allowed = (
        tc_name in ("write", "edit")
        or _is_test_read(tc_name, tc_args, focus_key=focus_key, focus_display=focus_display)
        or _is_test_command(tc_name, tc_args)
    )
    is_recovery_allowed = is_commit_allowed or _is_concrete_read(
        tc_name, tc_args, focus_key=focus_key, focus_display=focus_display,
    )

    if state.recovery_mode_active:
        state.recovery_turns_since_arm += 1
        if is_recovery_allowed:
            state.contract_block_sig = ""
            state.contract_block_count = 0
            return PASS
        target = state.recovery_target or focus_display or focus_key or "current focus"
        reason = state.recovery_reason or "repeated exploration"
        sig = _contract_violation_signature(
            cfg, tc_name, tc_args, focus_key=focus_key, focus_display=focus_display,
        )
        repeat_count = _record_contract_block(state, sig)
        abort_after = int(getattr(cfg, "contract_invalid_repeat_abort_after", 0) or 0)
        text = cfg.contract_recovery_block.format(reason=reason, target=target)
        if (
            abort_after > 0
            and repeat_count >= abort_after
            and _contract_abort_allowed(state, cfg, lane="recovery")
        ):
            return Decision(Action.END, text=text, reason="contract_recovery_abort")
        return Decision.block(text)

    warn_after = int(getattr(cfg, "contract_commit_warn_after", 0) or 0)
    block_after = int(getattr(cfg, "contract_commit_block_after", 0) or 0)
    if not state.commit_pending or (warn_after <= 0 and block_after <= 0):
        state.contract_block_sig = ""
        state.contract_block_count = 0
        return PASS
    state.commit_turns_since_arm += 1
    if is_commit_allowed:
        state.commit_violation_count = 0
        state.contract_block_sig = ""
        state.contract_block_count = 0
        return PASS

    state.commit_violation_count += 1
    source = state.commit_source_path
    if not _is_concrete_file_path(source):
        source = focus_display if _is_concrete_file_path(focus_display) else "current source"
    if block_after > 0 and state.commit_violation_count >= block_after:
        sig = _contract_violation_signature(
            cfg, tc_name, tc_args, focus_key=focus_key, focus_display=focus_display,
        )
        repeat_count = _record_contract_block(state, sig)
        text = cfg.contract_commit_block.format(source=source)
        abort_after = int(getattr(cfg, "contract_invalid_repeat_abort_after", 0) or 0)
        if (
            abort_after > 0
            and repeat_count >= abort_after
            and _contract_abort_allowed(state, cfg, lane="commit")
        ):
            return Decision(Action.END, text=text, reason="contract_commit_abort")
        return Decision.block(text)
    if warn_after > 0 and state.commit_violation_count >= warn_after:
        return Decision.warn(cfg.contract_commit_warn.format(source=source))
    return PASS


def rumination_gate(state: GuardrailState, cfg: Any, *,
                    tc_name: str) -> Decision:
    """Hard gate armed by the rumination ladder: block non-writes.

    GRACE (1 call): execute with a warning prefix (returned as WARN so
      the caller knows to dispatch but append the prefix).
    BLOCK: reject non-writes; count toward gate_max_blocks.
    END: after ``cfg.rumination_gate_max_blocks`` blocks → end session.

    Write/edit passes through (PASS); a successful write clears the
    gate via reset_on_successful_write().
    """
    if not cfg.rumination_enabled:
        return PASS
    if not state.rumination_gate:
        return PASS
    if tc_name in ("write", "edit"):
        return PASS
    if state.rumination_gate_grace > 0:
        state.rumination_gate_grace -= 1
        # Not a BLOCK — dispatch still runs — but carry a WARN so the
        # caller appends the warning prefix to the real tool result.
        return Decision.warn(cfg.rumination_gate_grace_prefix)
    state.gate_block_count += 1
    if (cfg.rumination_gate_max_blocks > 0
            and state.gate_block_count >= cfg.rumination_gate_max_blocks):
        return Decision(Action.END, text=cfg.rumination_gate, reason="gate_escalation")
    return Decision.block(cfg.rumination_gate)


# ─── Per-tool-call post-dispatch guardrails ──────────────────────────────

def _looks_like_test_path(path: str) -> bool:
    return bool(path) and "test" in path.lower()


def _canon_test_path(path: str) -> str:
    return path.split("::", 1)[0].lstrip("./").rstrip("/") or path


def _extract_test_target(cmd: str) -> str:
    if not cmd or "test" not in cmd.lower():
        return ""
    try:
        tokens = shlex.split(cmd, posix=True)
    except ValueError:
        return ""
    for token in tokens:
        candidate = token.split("::", 1)[0].rstrip(",")
        if _looks_like_test_path(candidate) and (
            "/" in candidate
            or candidate.startswith("/")
            or candidate.lower().startswith("test")
        ):
            return candidate
    return ""


def _test_target_is_covered(state: GuardrailState, target: str) -> bool:
    if not target:
        return False
    canon = _canon_test_path(target)
    if canon in state.test_file_reads:
        return True
    prefix = canon.rstrip("/")
    return any(read == prefix or read.startswith(prefix + "/") for read in state.test_file_reads)


def error_ladder(state: GuardrailState, cfg: Any, *,
                 tc_name: str, result: str) -> Decision:
    """Error cascade: WARN at nudge threshold, END at abort threshold.

    Consecutive error counter is per-tool-name. Reset on any non-error
    result for that tool.
    """
    if not cfg.error_ladder_enabled:
        return PASS
    if not result.startswith("ERROR:"):
        state.consecutive_errors[tc_name] = 0
        return PASS
    state.consecutive_errors[tc_name] = state.consecutive_errors.get(tc_name, 0) + 1
    count = state.consecutive_errors[tc_name]
    if (cfg.error_abort_threshold > 0
            and count >= cfg.error_abort_threshold):
        return Decision.end("error_abort")
    if count == cfg.error_nudge_threshold:
        return Decision.warn(cfg.error_nudge.format(count=count))
    return PASS


def test_read_ladder(
    state: GuardrailState,
    cfg: Any,
    *,
    tc_name: str,
    result: str,
    gate_blocked: bool,
    tc_args: dict | None = None,
) -> Decision:
    """Warn when verification repeats before the relevant test file is read."""
    warn_after = int(getattr(cfg, "test_read_warn_after", 0) or 0)
    if warn_after <= 0 or tc_name != "bash" or gate_blocked or result.startswith("ERROR:"):
        return PASS
    cmd = ""
    if isinstance(tc_args, dict):
        raw = tc_args.get("cmd")
        if isinstance(raw, str):
            cmd = raw
    target = _extract_test_target(cmd)
    if not target:
        return PASS
    state.last_test_target = target
    if _test_target_is_covered(state, target):
        state.test_runs_without_test_read = 0
        state.test_read_nudge_target = ""
        return PASS
    state.test_runs_without_test_read += 1
    if state.test_runs_without_test_read < warn_after:
        return PASS
    if state.test_read_nudge_target == target:
        return PASS
    state.test_read_nudge_target = target
    return Decision.warn(cfg.test_read_nudge.format(
        count=state.test_runs_without_test_read,
        target=target,
    ))


def rumination_ladder(state: GuardrailState, cfg: Any, *,
                      tc_name: str, result: str, gate_blocked: bool,
                      already_blocked_this_turn: bool,
                      focus_key: str = "", focus_display: str = "") -> Decision:
    """Post-dispatch tier of the rumination ladder: WARN + ARM.

    On successful write/edit: reset counter + gate + nudge flag.
    On any other tc: increment counter, WARN once at nudge threshold,
      ARM the gate at arm threshold.

    ``already_blocked_this_turn`` signals that rumination_gate blocked
    or graced this call — in that case skip the counter bump to avoid
    double counting.
    """
    if tc_name in ("write", "edit"):
        if not result.startswith("ERROR:"):
            state.non_write_calls_since_write = 0
            state.same_target_key = ""
            state.same_target_display = ""
            state.same_target_count = 0
            state.same_target_nudge_emitted = False
            state.rumination_gate = False
            state.rumination_gate_grace = 0
            state.rumination_nudge_emitted = False
            state.gate_block_count = 0
            state.has_mutated = True
            state.verified_since_mutation = False
            state.mutation_count += 1
            _clear_commit_contract(state)
            _clear_recovery_mode(state)
            state.verify_repeat_sig = ""
            state.verify_repeat_count = 0
            state.mutation_count_at_last_verify = state.mutation_count
        return PASS

    if not cfg.rumination_enabled:
        return PASS
    if already_blocked_this_turn:
        return PASS

    state.non_write_calls_since_write += 1
    _update_same_target_streak(state, focus_key=focus_key, focus_display=focus_display)

    recovery_same_target = int(
        getattr(cfg, "contract_recovery_same_target_threshold", 0) or 0
    )
    if (recovery_same_target > 0
            and state.same_target_key
            and state.same_target_count >= recovery_same_target):
        target = state.same_target_display or state.same_target_key
        if state.same_target_key.startswith("outside:"):
            reason = "repeated inspection outside repo root"
        else:
            reason = "repeated same-target inspection"
        _arm_recovery_mode(state, reason=reason, target=target)

    # WARN: one-shot nudge text.
    # Threshold depends on state.has_mutated: the pre-mutation threshold is a
    # "start editing" push for stuck tasks; the post-mutation threshold
    # preserves baseline-like behavior for tasks already editing productively.
    # Defaults make post == pre; set rumination_nudge_threshold_abs_post_mutation
    # to decouple them.
    # rumination_nudge_only_pre_mutation acts as a hard suppress post-mutation
    # (equivalent to post_mutation_threshold = infinity).
    threshold = (state.rumination_nudge_threshold_post_mutation
                 if state.has_mutated
                 else state.rumination_nudge_threshold)
    warn_parts: list[str] = []
    if (not state.rumination_nudge_emitted
            and state.non_write_calls_since_write >= threshold
            and not (cfg.rumination_nudge_only_pre_mutation and state.has_mutated)):
        warn_parts.append(cfg.rumination_nudge.format(
            count=state.non_write_calls_since_write,
        ))
        state.rumination_nudge_emitted = True

    same_target_warn = int(getattr(cfg, "rumination_same_target_warn_count", 0) or 0)
    if (same_target_warn > 0
            and state.same_target_key
            and not state.same_target_nudge_emitted
            and state.same_target_count >= same_target_warn):
        target = state.same_target_display or state.same_target_key
        if state.same_target_key.startswith("outside:"):
            warn_parts.append(cfg.rumination_outside_cwd_nudge.format(
                count=state.same_target_count,
                target=target,
            ))
        else:
            warn_parts.append(cfg.rumination_same_target_nudge.format(
                count=state.same_target_count,
                target=target,
            ))
        state.same_target_nudge_emitted = True

    # ARM: flip the gate flag. The block tier runs next turn.
    same_target_arm = int(getattr(cfg, "rumination_same_target_arm_count", 0) or 0)
    if (not state.rumination_gate
            and (
                state.non_write_calls_since_write >= state.rumination_arm_threshold
                or (
                    same_target_arm > 0
                    and state.same_target_key
                    and state.same_target_count >= same_target_arm
                )
            )):
        state.rumination_gate = True
        state.rumination_gate_grace = cfg.rumination_gate_grace_calls

    warn_text = "\n".join(part for part in warn_parts if part)
    return Decision.warn(warn_text) if warn_text else PASS


def _update_same_target_streak(
    state: GuardrailState,
    *,
    focus_key: str,
    focus_display: str,
) -> None:
    """Track repeated inspection of the same target between mutations.

    ``focus_key`` is a content-blind signature supplied by the loop: a file
    path for file tools / bash file reads, or a normalized bash command for
    generic bash inspections. Changing targets resets the streak.
    """
    if not focus_key:
        state.same_target_key = ""
        state.same_target_display = ""
        state.same_target_count = 0
        state.same_target_nudge_emitted = False
        return
    if state.same_target_key == focus_key:
        state.same_target_count += 1
        if focus_display:
            state.same_target_display = focus_display
        return
    state.same_target_key = focus_key
    state.same_target_display = focus_display or focus_key
    state.same_target_count = 1
    state.same_target_nudge_emitted = False


def mark_bash_verified(state: GuardrailState, cfg: Any, *,
                       tc_name: str, result: str, gate_blocked: bool, **_: Any) -> None:
    """Update verified_since_mutation on a content-blind signal.

    A successful substantial bash run (exit code 0, output length above
    cfg.done_verified_bash_min_chars) after a mutation counts as
    verification for the done-tool guard.
    """
    if tc_name != "bash":
        return
    if not state.has_mutated or gate_blocked:
        return
    if result.startswith("ERROR:"):
        return
    has_success_exit = "[exit code: 0]" in result
    has_failure_exit = "[exit code:" in result and not has_success_exit
    if has_failure_exit:
        return
    if len(result) > cfg.done_verified_bash_min_chars:
        state.verified_since_mutation = True


def observe_test_file_read(
    state: GuardrailState,
    cfg: Any,
    *,
    tc_name: str,
    result: str,
    gate_blocked: bool,
    tc_args: dict | None = None,
    focus_key: str = "",
    focus_display: str = "",
) -> None:
    """Track which test files have been read so test runs can demand them."""
    del cfg
    if gate_blocked or result.startswith("ERROR:"):
        return
    path = ""
    if tc_name == "read" and isinstance(tc_args, dict):
        raw = tc_args.get("path") or tc_args.get("file_path")
        if isinstance(raw, str):
            path = raw
    elif tc_name == "bash" and focus_key.startswith("file:") and _looks_like_test_path(focus_display):
        path = focus_display
    if not _looks_like_test_path(path):
        return
    state.test_file_reads.add(_canon_test_path(path))
    if state.last_test_target and _test_target_is_covered(state, state.last_test_target):
        state.test_runs_without_test_read = 0
        state.test_read_nudge_target = ""


def observe_contract_state(
    state: GuardrailState,
    cfg: Any,
    *,
    tc_name: str,
    result: str,
    gate_blocked: bool,
    tc_args: dict | None = None,
    focus_key: str = "",
    focus_display: str = "",
) -> None:
    """Track contract state from successful, content-blind tool outcomes."""
    if gate_blocked or result.startswith("ERROR:"):
        return

    if tc_name in ("write", "edit"):
        sig, target = _mutation_signature(tc_name, tc_args, focus_display=focus_display)
        if sig and state.mutation_repeat_sig == sig:
            state.mutation_repeat_count += 1
        elif sig:
            state.mutation_repeat_sig = sig
            state.mutation_repeat_target = target
            state.mutation_repeat_count = 1
        else:
            _clear_mutation_repeat_state(state)
        state.mutation_repeat_block_sig = ""
        state.mutation_repeat_block_count = 0
        _clear_commit_contract(state)
        _clear_recovery_mode(state)
        state.verify_repeat_sig = ""
        state.verify_repeat_count = 0
        state.mutation_count_at_last_verify = state.mutation_count
        return

    if _is_test_read(tc_name, tc_args, focus_key=focus_key, focus_display=focus_display):
        _clear_mutation_repeat_state(state)
        _clear_commit_contract(state)
        _clear_recovery_mode(state)
        return

    if _is_test_command(tc_name, tc_args):
        _clear_mutation_repeat_state(state)
        cmd = _extract_bash_cmd(tc_name, tc_args)
        target = _extract_test_target(cmd) or cmd
        if (state.verify_repeat_sig == target
                and state.mutation_count_at_last_verify == state.mutation_count):
            state.verify_repeat_count += 1
        else:
            state.verify_repeat_sig = target
            state.verify_repeat_count = 1
            state.mutation_count_at_last_verify = state.mutation_count
        _clear_commit_contract(state)
        _clear_recovery_mode(state)
        threshold = int(getattr(cfg, "contract_recovery_verify_repeat_threshold", 0) or 0)
        if threshold > 0 and state.verify_repeat_count >= threshold:
            _arm_recovery_mode(
                state,
                reason="repeated verification without refinement",
                target=target or "verification target",
            )
        return

    read_path = _extract_read_path(
        tc_name, tc_args, focus_key=focus_key, focus_display=focus_display,
    )
    if (
        read_path
        and _is_concrete_file_path(read_path)
        and not _looks_like_test_path(read_path)
        and not state.has_mutated
        and not focus_key.startswith("outside:")
    ):
        _clear_mutation_repeat_state(state)
        state.commit_pending = True
        state.commit_source_path = read_path
        state.commit_violation_count = 0
        state.commit_turns_since_arm = 0
        if state.recovery_mode_active and state.recovery_target == read_path:
            _clear_recovery_mode(state)
        return

    _clear_mutation_repeat_state(state)


def build_guardrail_registry(
    *,
    turn_pre_overrides: dict[str, Callable[..., Decision]] | None = None,
    tool_pre_overrides: dict[str, Callable[..., Decision]] | None = None,
    tool_post_overrides: dict[str, Callable[..., Decision]] | None = None,
    observer_overrides: dict[str, Callable[..., None]] | None = None,
) -> GuardrailRegistry:
    """Build the effective guardrail registry with optional overrides."""
    turn_pre_dispatch = {
        "intent_gate": intent_gate,
        "duplicate_guard": duplicate_guard,
        "loop_detect": loop_detect,
    }
    tool_pre_dispatch = {
        "done_guard": done_guard,
        "mutation_repeat_guard": mutation_repeat_guard,
        "contract_gate": contract_gate,
        "rumination_gate": rumination_gate,
    }
    tool_post_dispatch = {
        "error_ladder": error_ladder,
        "test_read_ladder": test_read_ladder,
        "rumination_ladder": rumination_ladder,
    }
    observers = {
        "mark_bash_verified": mark_bash_verified,
        "observe_test_file_read": observe_test_file_read,
        "observe_contract_state": observe_contract_state,
    }
    if turn_pre_overrides:
        turn_pre_dispatch.update(turn_pre_overrides)
    if tool_pre_overrides:
        tool_pre_dispatch.update(tool_pre_overrides)
    if tool_post_overrides:
        tool_post_dispatch.update(tool_post_overrides)
    if observer_overrides:
        observers.update(observer_overrides)
    return GuardrailRegistry(
        turn_pre_dispatch=turn_pre_dispatch,
        tool_pre_dispatch=tool_pre_dispatch,
        tool_post_dispatch=tool_post_dispatch,
        observers=observers,
    )


def validate_guardrail_registry(registry: GuardrailRegistry) -> None:
    """Fail fast when the registry is missing required call-site names."""

    def _missing(required: tuple[str, ...], registered: dict[str, Callable]) -> list[str]:
        return [name for name in required if name not in registered]

    missing = (
        _missing(TURN_PRE_DISPATCH_ORDER, registry.turn_pre_dispatch)
        + _missing(TOOL_PRE_DISPATCH_ORDER, registry.tool_pre_dispatch)
        + _missing(TOOL_POST_DISPATCH_ORDER, registry.tool_post_dispatch)
        + _missing(OBSERVER_ORDER, registry.observers)
    )
    if missing:
        raise ValueError(f"Guardrail registry missing required handlers: {', '.join(sorted(missing))}")
