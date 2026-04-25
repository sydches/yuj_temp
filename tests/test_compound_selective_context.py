"""Tests for the compound_selective context mode."""
from __future__ import annotations

import json
from pathlib import Path

from scripts.llm_solver.harness.context_strategies import CompoundSelectiveContext


def _make_context(
    tmp_path: Path,
    *,
    tool_budget: int = 60,
    trace_repeat_cap: int = 1,
    resolved_repeat_cap: int = 1,
    trace_lines: int = 2,
    resolved_lines: int = 2,
    trace_anchor_lines: int = 0,
    resolved_anchor_lines: int = 0,
    trace_source_anchor_lines: int = 0,
    trace_test_anchor_lines: int = 0,
    resolved_source_anchor_lines: int = 0,
    resolved_test_anchor_lines: int = 0,
) -> CompoundSelectiveContext:
    ctx = CompoundSelectiveContext(
        cwd=str(tmp_path),
        original_prompt="Solve the task.",
        trace_lines=50,
        evidence_lines=30,
        inference_lines=20,
        recent_tool_results_chars=30000,
        trace_stub_chars=200,
        min_turns=2,
        suffix="Continue working. Your progress is tracked in .solver/state.json.",
        selective_trace_lines=trace_lines,
        selective_unresolved_evidence_lines=2,
        selective_resolved_evidence_lines=resolved_lines,
        selective_resolved_evidence_stub_chars=40,
        selective_recent_tool_results_chars=tool_budget,
        selective_trace_action_repeat_cap=trace_repeat_cap,
        selective_resolved_action_repeat_cap=resolved_repeat_cap,
        selective_trace_anchor_lines=trace_anchor_lines,
        selective_resolved_anchor_lines=resolved_anchor_lines,
        selective_trace_source_anchor_lines=trace_source_anchor_lines,
        selective_trace_test_anchor_lines=trace_test_anchor_lines,
        selective_resolved_source_anchor_lines=resolved_source_anchor_lines,
        selective_resolved_test_anchor_lines=resolved_test_anchor_lines,
    )
    ctx.add_system("You are a solver.")
    ctx._turn_count = 5
    return ctx


def _write_state(tmp_path: Path, state: dict) -> None:
    solver_dir = tmp_path / ".solver"
    solver_dir.mkdir(parents=True, exist_ok=True)
    (solver_dir / "state.json").write_text(json.dumps(state))


def test_selective_trace_budget_is_smaller(tmp_path: Path):
    _write_state(tmp_path, {
        "state": {"current_attempt": "", "last_verify": "", "next_action": ""},
        "trace": [
            {"step": 1, "session": 1, "turn": 0, "reasoning": "first", "action": "read(path='a.py')", "result": "", "next": ""},
            {"step": 2, "session": 1, "turn": 1, "reasoning": "second", "action": "read(path='b.py')", "result": "", "next": ""},
            {"step": 3, "session": 1, "turn": 2, "reasoning": "third", "action": "edit(path='c.py')", "result": "OK", "next": ""},
        ],
        "gates": [],
        "evidence": [],
        "inference": [],
    })
    ctx = _make_context(tmp_path)
    user_text = ctx.get_messages()[1]["content"]
    assert "read(path='a.py')" not in user_text
    assert "read(path='b.py')" in user_text
    assert "edit(path='c.py')" in user_text


def test_selective_evidence_keeps_fails_and_stubbed_resolved_tail(tmp_path: Path):
    _write_state(tmp_path, {
        "state": {"current_attempt": "", "last_verify": "", "next_action": ""},
        "trace": [],
        "gates": [],
        "evidence": [
            {"step": 1, "action": "bash(cmd='find a')", "result": "OLDEST_RESOLVED_" + ("x" * 80), "verdict": "OK", "gate_blocked": False},
            {"step": 2, "action": "bash(cmd='find b')", "result": "KEEP_RESOLVED_ONE_" + ("y" * 80), "verdict": "OK", "gate_blocked": False},
            {"step": 3, "action": "bash(cmd='find c')", "result": "KEEP_RESOLVED_TWO_" + ("z" * 80), "verdict": "OK", "gate_blocked": False},
            {"step": 4, "action": "bash(cmd='pytest a')", "result": "FAIL_ONE\ntraceback line", "verdict": "FAIL", "gate_blocked": False},
            {"step": 5, "action": "bash(cmd='pytest b')", "result": "FAIL_TWO\nassert line", "verdict": "FAIL", "gate_blocked": False},
        ],
        "inference": [],
    })
    ctx = _make_context(tmp_path)
    user_text = ctx.get_messages()[1]["content"]
    assert "FAIL_ONE\ntraceback line" in user_text
    assert "FAIL_TWO\nassert line" in user_text
    assert "OLDEST_RESOLVED_" not in user_text
    assert "KEEP_RESOLVED_ONE_" in user_text
    assert "KEEP_RESOLVED_TWO_" in user_text
    assert "KEEP_RESOLVED_ONE_" + ("y" * 50) not in user_text
    assert "-- resolved --" in user_text


def test_selective_trace_keeps_diverse_actions_under_repeat_cap(tmp_path: Path):
    _write_state(tmp_path, {
        "state": {"current_attempt": "", "last_verify": "", "next_action": ""},
        "trace": [
            {"step": 1, "session": 1, "turn": 0, "reasoning": "", "action": "bash(cmd='find tests')", "result": "", "next": ""},
            {"step": 2, "session": 1, "turn": 1, "reasoning": "", "action": "read(path='a.py')", "result": "", "next": ""},
            {"step": 3, "session": 1, "turn": 2, "reasoning": "", "action": "read(path='a.py')", "result": "", "next": ""},
            {"step": 4, "session": 1, "turn": 3, "reasoning": "", "action": "read(path='a.py')", "result": "", "next": ""},
            {"step": 5, "session": 1, "turn": 4, "reasoning": "", "action": "grep(pattern='foo')", "result": "", "next": ""},
        ],
        "gates": [],
        "evidence": [],
        "inference": [],
    })
    ctx = _make_context(tmp_path, trace_lines=3, trace_anchor_lines=1)
    user_text = ctx.get_messages()[1]["content"]
    assert "bash(cmd='find tests')" in user_text
    assert user_text.count("read(path='a.py')") == 1
    assert "grep(pattern='foo')" in user_text


def test_selective_trace_preserves_source_and_test_transition_anchors(tmp_path: Path):
    _write_state(tmp_path, {
        "state": {"current_attempt": "", "last_verify": "", "next_action": ""},
        "trace": [
            {"step": 1, "session": 1, "turn": 0, "reasoning": "", "action": "read(path='src/module.py')", "result": "", "next": ""},
            {"step": 2, "session": 1, "turn": 1, "reasoning": "", "action": "bash(cmd='grep -n \"target\" tests/test_module.py')", "result": "", "next": ""},
            {"step": 3, "session": 1, "turn": 2, "reasoning": "", "action": "read(path='src/module.py')", "result": "", "next": ""},
            {"step": 4, "session": 1, "turn": 3, "reasoning": "", "action": "read(path='src/module.py')", "result": "", "next": ""},
            {"step": 5, "session": 1, "turn": 4, "reasoning": "", "action": "grep(pattern='symbol', path='src')", "result": "", "next": ""},
        ],
        "gates": [],
        "evidence": [],
        "inference": [],
    })
    ctx = _make_context(
        tmp_path,
        trace_lines=3,
        trace_repeat_cap=1,
        trace_anchor_lines=0,
        trace_source_anchor_lines=1,
        trace_test_anchor_lines=1,
    )
    user_text = ctx.get_messages()[1]["content"]
    assert "read(path='src/module.py')" in user_text
    assert "grep -n \"target\" tests/test_module.py" in user_text
    assert "grep(pattern='symbol', path='src')" in user_text


def test_selective_resolved_evidence_keeps_diverse_actions_under_repeat_cap(tmp_path: Path):
    _write_state(tmp_path, {
        "state": {"current_attempt": "", "last_verify": "", "next_action": ""},
        "trace": [],
        "gates": [],
        "evidence": [
            {"step": 1, "action": "bash(cmd='find tests')", "result": "FOUND_TEST", "verdict": "OK", "gate_blocked": False},
            {"step": 2, "action": "read(path='build.py')", "result": "READ_ONE", "verdict": "OK", "gate_blocked": False},
            {"step": 3, "action": "read(path='build.py')", "result": "READ_TWO", "verdict": "OK", "gate_blocked": False},
            {"step": 4, "action": "grep(pattern='get_parser')", "result": "FOUND_SYMBOL", "verdict": "OK", "gate_blocked": False},
        ],
        "inference": [],
    })
    ctx = _make_context(tmp_path, resolved_lines=3, resolved_anchor_lines=1)
    user_text = ctx.get_messages()[1]["content"]
    assert "FOUND_TEST" in user_text
    assert "FOUND_SYMBOL" in user_text
    assert user_text.count("read(path='build.py')") == 1
    assert "READ_TWO" in user_text
    assert "READ_ONE" not in user_text


def test_selective_resolved_evidence_preserves_test_anchor(tmp_path: Path):
    _write_state(tmp_path, {
        "state": {"current_attempt": "", "last_verify": "", "next_action": ""},
        "trace": [],
        "gates": [],
        "evidence": [
            {"step": 1, "action": "read(path='src/module.py')", "result": "SOURCE_ONE", "verdict": "OK", "gate_blocked": False},
            {"step": 2, "action": "bash(cmd='pytest -q tests/test_module.py')", "result": "VERIFY_ONE", "verdict": "OK", "gate_blocked": False},
            {"step": 3, "action": "read(path='src/module.py')", "result": "SOURCE_TWO", "verdict": "OK", "gate_blocked": False},
            {"step": 4, "action": "grep(pattern='symbol', path='src')", "result": "RECENT_SEARCH", "verdict": "OK", "gate_blocked": False},
        ],
        "inference": [],
    })
    ctx = _make_context(
        tmp_path,
        resolved_lines=3,
        resolved_repeat_cap=1,
        resolved_anchor_lines=0,
        resolved_source_anchor_lines=1,
        resolved_test_anchor_lines=1,
    )
    user_text = ctx.get_messages()[1]["content"]
    assert "SOURCE_ONE" in user_text
    assert "VERIFY_ONE" in user_text
    assert "RECENT_SEARCH" in user_text


def test_selective_tool_result_window_is_smaller(tmp_path: Path):
    _write_state(tmp_path, {
        "state": {"current_attempt": "", "last_verify": "", "next_action": ""},
        "trace": [],
        "gates": [],
        "evidence": [],
        "inference": [],
    })
    ctx = _make_context(tmp_path, tool_budget=50)
    older = "OLDER_RESULT_" + ("x" * 30)
    newer = "NEWER_RESULT_" + ("y" * 30)
    ctx.add_tool_result("call-1", older, tool_name="bash", cmd_signature='{"cmd":"echo older"}')
    ctx.add_tool_result("call-2", newer, tool_name="bash", cmd_signature='{"cmd":"echo newer"}')
    user_text = ctx.get_messages()[1]["content"]
    assert "NEWER_RESULT_" in user_text
    assert "OLDER_RESULT_" not in user_text
