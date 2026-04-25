"""Tests for scripts.llm_solver.harness.state_writer.

Validates the pure projection function (`project`) and the file helpers
(`project_from_trace`, `write_state_from_trace`) against fixture trace event
streams. No LLM dependency, no filesystem state beyond temp dirs.

The projection is content-blind by design: the harness must not parse task
output. Trace entries carry action/result verbatim; no is_test, no verdict,
no evidence. Any task-format parsing belongs in the analysis layer.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.llm_solver.harness.state_writer import (
    project,
    project_from_trace,
    write_state_from_trace,
)

# Standard cap for test projections. Matches config.toml [output] max_output_chars
# for the test-config fixture.
_CAP = 20000


def _project(events):
    return project(events, max_result_chars=_CAP)


def _project_from_trace(path):
    return project_from_trace(path, max_result_chars=_CAP)


def _write_state_from_trace(trace_path, state_path):
    return write_state_from_trace(trace_path, state_path, max_result_chars=_CAP)


# ── project() — pure projection ──────────────────────────────────────────────

class TestProjectEmpty:
    def test_empty_events_yields_empty_schema(self):
        out = _project([])
        assert out == {
            "state": {"current_attempt": "", "last_verify": "", "next_action": ""},
            "trace": [],
            "gates": [],
            "evidence": [],
            "inference": [],
        }

    def test_only_session_start_leaves_state_blank(self):
        out = _project([{"event": "session_start", "session_number": 1}])
        assert out["state"]["current_attempt"] == ""
        assert out["state"]["last_verify"] == ""
        assert out["trace"] == []
        assert out["evidence"] == []


class TestProjectTraceAccumulation:
    def test_single_tool_call_becomes_trace_entry(self):
        events = [
            {"event": "session_start", "session_number": 1},
            {
                "event": "tool_call",
                "session_number": 1,
                "turn_number": 0,
                "tool_name": "bash",
                "args_summary": "cmd='ls'",
                "result_summary": "file1\nfile2\n",
            },
        ]
        out = _project(events)
        assert len(out["trace"]) == 1
        entry = out["trace"][0]
        assert entry["step"] == 1
        assert entry["action"] == "bash(cmd='ls')"
        assert entry["result"] == "file1\nfile2\n"
        assert entry["next"] == ""
        assert entry["session"] == 1
        assert entry["turn"] == 0
        # Content-blind schema: no is_test, no verdict.
        assert "is_test" not in entry
        assert "verdict" not in entry

    def test_step_counter_is_monotonic_across_sessions(self):
        events = [
            {"event": "session_start", "session_number": 1},
            {"event": "tool_call", "session_number": 1, "turn_number": 0,
             "tool_name": "bash", "args_summary": "a", "result_summary": "x"},
            {"event": "tool_call", "session_number": 1, "turn_number": 1,
             "tool_name": "bash", "args_summary": "b", "result_summary": "y"},
            {"event": "session_end", "session_number": 1,
             "finish_reason": "stop", "turns": 2},
            {"event": "session_start", "session_number": 2},
            {"event": "tool_call", "session_number": 2, "turn_number": 0,
             "tool_name": "read", "args_summary": "path='f'", "result_summary": "z"},
        ]
        out = _project(events)
        assert [t["step"] for t in out["trace"]] == [1, 2, 3]
        assert out["trace"][2]["session"] == 2

    def test_current_attempt_tracks_latest_tool_call(self):
        events = [
            {"event": "tool_call", "session_number": 1, "turn_number": 0,
             "tool_name": "bash", "args_summary": "first", "result_summary": "r"},
            {"event": "tool_call", "session_number": 1, "turn_number": 1,
             "tool_name": "read", "args_summary": "second", "result_summary": "r"},
        ]
        out = _project(events)
        assert out["state"]["current_attempt"] == "read(second)"


class TestProjectLastVerify:
    def test_session_end_populates_last_verify(self):
        events = [
            {"event": "session_end", "session_number": 1,
             "finish_reason": "max_turns", "turns": 60},
        ]
        out = _project(events)
        assert "max_turns" in out["state"]["last_verify"]
        assert "60" in out["state"]["last_verify"]
        assert "session 1" in out["state"]["last_verify"]

    def test_last_verify_reflects_most_recent_session_end(self):
        events = [
            {"event": "session_end", "session_number": 1,
             "finish_reason": "stop", "turns": 5},
            {"event": "session_end", "session_number": 2,
             "finish_reason": "max_turns", "turns": 60},
        ]
        out = _project(events)
        assert "session 2" in out["state"]["last_verify"]
        assert "max_turns" in out["state"]["last_verify"]


class TestProjectEvidence:
    """Evidence population is content-blind.

    The rule is purely structural: bash tool calls that actually ran (not
    gate-blocked) become evidence entries, with their verdict derived from
    harness-generated markers only ([exit code: N] appended by tools.py,
    ERROR: wrapper written on exception). The projection never reads the
    raw task output for pass/fail cues — no pytest nodeid parsing, no
    compiler error detection, no summary-line regex.
    """

    def test_bash_exit_zero_is_ok_evidence(self):
        events = [
            {"event": "tool_call", "session_number": 1, "turn_number": 0,
             "tool_name": "bash", "args_summary": "cmd='./whatever'",
             "result_summary": "done"},
        ]
        out = _project(events)
        assert len(out["evidence"]) == 1
        ev = out["evidence"][0]
        assert ev["step"] == 1
        assert ev["action"] == "bash(cmd='./whatever')"
        assert ev["verdict"] == "OK"
        assert ev["gate_blocked"] is False

    def test_bash_nonzero_exit_is_fail_evidence(self):
        events = [
            {"event": "tool_call", "session_number": 1, "turn_number": 0,
             "tool_name": "bash", "args_summary": "cmd='./anything'",
             "result_summary": "some output\n[exit code: 1]"},
        ]
        out = _project(events)
        assert len(out["evidence"]) == 1
        assert out["evidence"][0]["verdict"] == "FAIL"

    def test_error_wrapper_is_fail_evidence(self):
        events = [
            {"event": "tool_call", "session_number": 1, "turn_number": 0,
             "tool_name": "bash", "args_summary": "cmd='run'",
             "result_summary": "ERROR: command timed out after 60s"},
        ]
        out = _project(events)
        assert len(out["evidence"]) == 1
        assert out["evidence"][0]["verdict"] == "FAIL"

    def test_read_tool_does_not_generate_evidence(self):
        # Non-bash tools are harness I/O, not gate verdicts — structural filter.
        events = [
            {"event": "tool_call", "session_number": 1, "turn_number": 0,
             "tool_name": "read", "args_summary": "path='anything.txt'",
             "result_summary": "content"},
        ]
        out = _project(events)
        assert out["evidence"] == []

    def test_write_tool_does_not_generate_evidence(self):
        events = [
            {"event": "tool_call", "session_number": 1, "turn_number": 0,
             "tool_name": "write", "args_summary": "path='x.py'",
             "result_summary": "OK: wrote 10 bytes to x.py"},
        ]
        out = _project(events)
        assert out["evidence"] == []

    def test_edit_tool_does_not_generate_evidence(self):
        events = [
            {"event": "tool_call", "session_number": 1, "turn_number": 0,
             "tool_name": "edit", "args_summary": "path='x.py'",
             "result_summary": "OK"},
        ]
        out = _project(events)
        assert out["evidence"] == []

    def test_gate_blocked_bash_is_not_evidence(self):
        events = [
            {"event": "tool_call", "session_number": 1, "turn_number": 0,
             "tool_name": "bash", "args_summary": "cmd='grep x'",
             "result_summary": "[harness gate] blocked",
             "gate_blocked": True},
        ]
        out = _project(events)
        assert out["evidence"] == []

    def test_mixed_stream_preserves_order(self):
        events = [
            {"event": "tool_call", "session_number": 1, "turn_number": 0,
             "tool_name": "bash", "args_summary": "cmd='a'",
             "result_summary": "ok"},
            {"event": "tool_call", "session_number": 1, "turn_number": 1,
             "tool_name": "read", "args_summary": "path='b'", "result_summary": "data"},
            {"event": "tool_call", "session_number": 1, "turn_number": 2,
             "tool_name": "bash", "args_summary": "cmd='c'",
             "result_summary": "oops\n[exit code: 2]"},
        ]
        out = _project(events)
        # Two bash calls → two evidence entries, read is skipped.
        assert len(out["evidence"]) == 2
        assert out["evidence"][0]["verdict"] == "OK"
        assert out["evidence"][1]["verdict"] == "FAIL"

    def test_evidence_result_is_truncated_to_evidence_cap(self):
        # Evidence entries get a tighter cap than trace entries because
        # they render into the model's live context every turn.
        big = "x" * 5000
        events = [
            {"event": "tool_call", "session_number": 1, "turn_number": 0,
             "tool_name": "bash", "args_summary": "cmd='a'",
             "result_summary": big},
        ]
        out = _project(events)
        ev_result = out["evidence"][0]["result"]
        assert len(ev_result) < len(big)
        assert ev_result.endswith("...")


class TestProjectTruncation:
    def test_long_args_summary_is_truncated(self):
        long_args = "x" * 500
        events = [
            {"event": "tool_call", "session_number": 1, "turn_number": 0,
             "tool_name": "bash", "args_summary": long_args,
             "result_summary": "r"},
        ]
        out = _project(events)
        entry = out["trace"][0]
        assert len(entry["action"]) < 500
        # Args truncated → "...)" suffix (ellipsis from args, then action paren close).
        assert entry["action"].endswith("...)")

    def test_long_result_summary_is_truncated(self):
        # Result cap is supplied by the caller (wired from cfg.max_output_chars
        # in production). A result longer than the cap is head-truncated with
        # a "..." suffix.
        long_result = "y" * 25000
        events = [
            {"event": "tool_call", "session_number": 1, "turn_number": 0,
             "tool_name": "bash", "args_summary": "short",
             "result_summary": long_result},
        ]
        out = _project(events)
        assert len(out["trace"][0]["result"]) < 25000
        assert len(out["trace"][0]["result"]) == _CAP
        assert out["trace"][0]["result"].endswith("...")

    def test_mid_length_result_is_not_truncated(self):
        # A result under the cap must land in state.json verbatim.
        code_read = "def foo():\n    pass\n" * 250  # 5000 chars
        assert len(code_read) == 5000
        events = [
            {"event": "tool_call", "session_number": 1, "turn_number": 0,
             "tool_name": "read", "args_summary": "path='foo.py'",
             "result_summary": code_read},
        ]
        out = _project(events)
        assert out["trace"][0]["result"] == code_read
        assert not out["trace"][0]["result"].endswith("...")


class TestProjectDeterminism:
    def test_same_input_yields_same_output(self):
        events = [
            {"event": "tool_call", "session_number": 1, "turn_number": 0,
             "tool_name": "bash", "args_summary": "a", "result_summary": "r"},
            {"event": "session_end", "session_number": 1,
             "finish_reason": "stop", "turns": 1},
        ]
        assert _project(events) == _project(events)


# ── File helpers ─────────────────────────────────────────────────────────────

class TestProjectFromTrace:
    def test_missing_file_yields_empty_schema(self, tmp_path: Path):
        out = _project_from_trace(tmp_path / "does-not-exist.jsonl")
        assert out["state"]["current_attempt"] == ""
        assert out["trace"] == []

    def test_reads_multiple_events_from_file(self, tmp_path: Path):
        trace = tmp_path / ".trace.jsonl"
        trace.write_text(
            "\n".join([
                json.dumps({"event": "session_start", "session_number": 1}),
                json.dumps({
                    "event": "tool_call", "session_number": 1, "turn_number": 0,
                    "tool_name": "bash", "args_summary": "echo hi",
                    "result_summary": "hi",
                }),
            ]) + "\n"
        )
        out = _project_from_trace(trace)
        assert len(out["trace"]) == 1
        assert out["state"]["current_attempt"] == "bash(echo hi)"

    def test_blank_lines_are_skipped(self, tmp_path: Path):
        trace = tmp_path / ".trace.jsonl"
        trace.write_text("\n\n" + json.dumps({
            "event": "tool_call", "session_number": 1, "turn_number": 0,
            "tool_name": "bash", "args_summary": "a", "result_summary": "r",
        }) + "\n\n")
        out = _project_from_trace(trace)
        assert len(out["trace"]) == 1


class TestWriteStateFromTrace:
    def test_writes_state_to_target_path(self, tmp_path: Path):
        trace = tmp_path / ".trace.jsonl"
        trace.write_text(json.dumps({
            "event": "tool_call", "session_number": 1, "turn_number": 0,
            "tool_name": "bash", "args_summary": "ls", "result_summary": "r",
        }) + "\n")
        state_path = tmp_path / ".solver" / "state.json"
        _write_state_from_trace(trace, state_path)
        assert state_path.is_file()
        data = json.loads(state_path.read_text())
        assert len(data["trace"]) == 1

    def test_creates_parent_directories(self, tmp_path: Path):
        trace = tmp_path / ".trace.jsonl"
        trace.write_text("")
        state_path = tmp_path / "a" / "b" / "c" / "state.json"
        _write_state_from_trace(trace, state_path)
        assert state_path.is_file()

    def test_idempotent_overwrite(self, tmp_path: Path):
        trace = tmp_path / ".trace.jsonl"
        trace.write_text(json.dumps({
            "event": "tool_call", "session_number": 1, "turn_number": 0,
            "tool_name": "bash", "args_summary": "a", "result_summary": "r",
        }) + "\n")
        state_path = tmp_path / ".solver" / "state.json"
        _write_state_from_trace(trace, state_path)
        first = state_path.read_text()
        _write_state_from_trace(trace, state_path)
        second = state_path.read_text()
        assert first == second

    def test_no_tmp_file_left_behind(self, tmp_path: Path):
        trace = tmp_path / ".trace.jsonl"
        trace.write_text("")
        state_path = tmp_path / ".solver" / "state.json"
        _write_state_from_trace(trace, state_path)
        leftover = list((tmp_path / ".solver").glob("*.tmp"))
        assert leftover == []

    def test_round_trip_through_solver_state_context_schema(self, tmp_path: Path):
        """The projection output must match the keys SolverStateContext reads."""
        trace = tmp_path / ".trace.jsonl"
        trace.write_text(json.dumps({
            "event": "tool_call", "session_number": 1, "turn_number": 0,
            "tool_name": "bash", "args_summary": "echo",
            "result_summary": "out",
        }) + "\n")
        state_path = tmp_path / ".solver" / "state.json"
        _write_state_from_trace(trace, state_path)
        data = json.loads(state_path.read_text())
        # Exact key set the read path expects.
        assert set(data.keys()) == {"state", "trace", "gates", "evidence", "inference"}
        assert set(data["state"].keys()) >= {"current_attempt", "last_verify", "next_action"}
        for entry in data["trace"]:
            assert {"step", "action", "result", "next"} <= set(entry.keys())
        for ev in data["evidence"]:
            assert {"step", "action", "result", "verdict"} <= set(ev.keys())


# ── Edge cases and robustness ────────────────────────────────────────────────

class TestProjectEdgeCases:
    """Guard against malformed events and missing fields. The harness writes
    these events from trusted code, but a corrupted trace or a breaking change
    upstream should not leave state.json in a garbage state or crash the writer.
    """

    def test_tool_call_with_missing_tool_name(self):
        events = [
            {"event": "tool_call", "session_number": 1, "turn_number": 0,
             "args_summary": "a", "result_summary": "r"},
        ]
        out = _project(events)
        assert len(out["trace"]) == 1
        assert out["trace"][0]["action"].startswith("?(")

    def test_tool_call_with_missing_args_summary(self):
        events = [
            {"event": "tool_call", "session_number": 1, "turn_number": 0,
             "tool_name": "bash", "result_summary": "r"},
        ]
        out = _project(events)
        assert out["trace"][0]["action"] == "bash()"

    def test_tool_call_with_none_result_summary(self):
        events = [
            {"event": "tool_call", "session_number": 1, "turn_number": 0,
             "tool_name": "bash", "args_summary": "a", "result_summary": None},
        ]
        out = _project(events)
        assert out["trace"][0]["result"] == ""

    def test_unknown_event_type_is_ignored(self):
        events = [
            {"event": "noise", "session_number": 1},
            {"event": "tool_call", "session_number": 1, "turn_number": 0,
             "tool_name": "bash", "args_summary": "a", "result_summary": "r"},
        ]
        out = _project(events)
        # Only the tool_call contributes.
        assert len(out["trace"]) == 1

    def test_session_end_without_matching_start(self):
        """A truncated trace file could start with a session_end. Don't crash."""
        events = [
            {"event": "session_end", "session_number": 1,
             "finish_reason": "stop", "turns": 1},
        ]
        out = _project(events)
        assert "stop" in out["state"]["last_verify"]
        assert out["trace"] == []

    def test_unicode_in_args_and_results(self):
        """Tool args can contain any UTF-8 — file paths with accents, emoji
        in strings, non-ASCII identifiers. The projection must pass them
        through without corruption or re-encoding."""
        events = [
            {"event": "tool_call", "session_number": 1, "turn_number": 0,
             "tool_name": "bash", "args_summary": "cmd='echo café ñ 日本語 🔥'",
             "result_summary": "café ñ 日本語 🔥"},
        ]
        out = _project(events)
        assert "café" in out["trace"][0]["action"]
        assert "日本語" in out["trace"][0]["result"]
        assert "🔥" in out["trace"][0]["result"]

    def test_trailing_partial_json_line_raises(self, tmp_path: Path):
        """Partial last line (process killed mid-write) is malformed JSON and
        must surface explicitly rather than silently skip. Silent skip would
        let a truncated trace produce a state.json that disagrees with the
        model's actual history — exactly the failure mode mechanical state is
        meant to prevent."""
        trace = tmp_path / ".trace.jsonl"
        trace.write_text(
            json.dumps({"event": "tool_call", "session_number": 1,
                        "turn_number": 0, "tool_name": "bash",
                        "args_summary": "a", "result_summary": "r"}) + "\n"
            + '{"event": "tool_call", "session_number": 1, "turn_numb'  # truncated
        )
        with pytest.raises(json.JSONDecodeError):
            _project_from_trace(trace)

    def test_atomic_write_leaves_no_partial_file(self, tmp_path: Path, monkeypatch):
        """If the JSON serialization fails mid-write, the target state.json
        must not be touched — the tmp file is written separately and renamed
        only on success."""
        trace = tmp_path / ".trace.jsonl"
        trace.write_text("")
        state_path = tmp_path / ".solver" / "state.json"
        state_path.parent.mkdir(parents=True)
        state_path.write_text('{"state": {"current_attempt": "pre-existing"}, '
                              '"trace": [], "gates": [], "evidence": [], '
                              '"inference": []}')
        pre_existing = state_path.read_text()

        import scripts.llm_solver.harness.state_writer as sw

        def fail_replace(self, target):
            raise OSError("simulated rename failure")
        monkeypatch.setattr(Path, "replace", fail_replace)

        try:
            _write_state_from_trace(trace, state_path)
        except OSError:
            pass

        # Pre-existing state.json must be untouched.
        assert state_path.read_text() == pre_existing

    def test_projection_is_not_affected_by_event_order_within_type(self):
        """Tool call order within the same session_number matters (step counter
        is input-order-based). The projection does not sort or reorder."""
        events = [
            {"event": "tool_call", "session_number": 1, "turn_number": 5,
             "tool_name": "bash", "args_summary": "B", "result_summary": "r"},
            {"event": "tool_call", "session_number": 1, "turn_number": 3,
             "tool_name": "bash", "args_summary": "A", "result_summary": "r"},
        ]
        out = _project(events)
        # First event in input is step 1, regardless of turn_number.
        assert out["trace"][0]["action"] == "bash(B)"
        assert out["trace"][1]["action"] == "bash(A)"


# ── End-to-end via Session._write_trace ─────────────────────────────────────

def _make_session_cfg():
    from scripts.llm_solver.config import Config
    return Config(
        base_url="http://localhost:8080/v1",
        api_key="local",
        timeout_connect=10,
        timeout_read=120,
        health_poll_interval=2, health_timeout=2, launch_timeout=120, stop_settle=2,
        model="test",
        context_size=8000,
        context_fill_ratio=0.85,
        max_tokens=1024,
        max_turns=10,
        max_sessions=1,
        duplicate_abort=3,
        error_nudge_threshold=3, rumination_nudge_threshold=200, require_intent=False,
        intent_grace_turns=3,
        min_turns_before_context=2,
        max_output_chars=_CAP,
        truncate_head_ratio=0.6,
        truncate_head_lines=100,
        truncate_tail_lines=50,
        args_summary_chars=80, trace_args_summary_chars=200,
        trace_reasoning_store_chars=800,
        solver_trace_lines=50,
        solver_evidence_lines=30,
        solver_inference_lines=20,
        recent_tool_results_chars=30000,
        trace_stub_chars=200,
        trace_reasoning_chars=150,
        pretest_head_chars=2000,
        pretest_tail_chars=1500,
        bash_timeout=60,
        grep_timeout=30,
        pretest_timeout=240,
        llama_server_bin="/bin/true",
        sandbox_bash=False,
        strip_ansi=True,
        collapse_blank_lines=True,
        collapse_duplicate_lines=True,
        collapse_similar_lines=True,
        bwrap_bin="/usr/bin/bwrap",
        max_transient_retries=0,
        retry_backoff=(1, 4, 16),
        system_header="You are a solver.",
        state_context_suffix="Continue working.",
        intent_gate_first="[harness] silent rejected",
        intent_gate_repeat="[intent gate: {count} since {first_turn}]",
        resume_base="Continue.",
        error_nudge="{count} errors",
        rumination_nudge="{count} non-write",
        rumination_gate="blocked",
        rumination_same_target_nudge="same target {target}",
        rumination_outside_cwd_nudge="outside {target}",
        test_read_nudge="read test {target}",
        contract_commit_warn="warn {source}",
        contract_commit_block="block {source}",
        contract_recovery_block="recover {reason} {target}",
        mutation_repeat_warn="warn mutation {target}",
        mutation_repeat_block="block mutation {target}",
        resume_duplicate_abort="{n} identical: {call}",
        resume_context_full="{pct}% full",
        resume_max_turns="{n}: {actions}",
        resume_length="truncated",
        resume_last_n_actions=3,
        tool_desc="minimal",
        prompt_addendum="",
        variant_name="",
    )


class TestWriteTraceIntegration:
    """Exercise the actual harness loop path: Session._write_trace → write-
    through to trace.jsonl → _refresh_state → write_state_from_trace →
    state.json updated. No live LLM client; we call the private method
    directly to simulate what the tool dispatch loop does."""

    def _make_session(self, tmp_path: Path):
        from scripts.llm_solver.harness.loop import Session
        from scripts.llm_solver.harness.context import FullTranscript
        cfg = _make_session_cfg()
        trace_path = tmp_path / ".trace.jsonl"
        state_path = tmp_path / ".solver" / "state.json"
        state_path.parent.mkdir(parents=True)
        state_path.write_text(
            '{"state": {}, "trace": [], "gates": [], "evidence": [], "inference": []}'
        )
        trace_file = open(trace_path, "a")
        session = Session(
            cfg=cfg,
            client=None,
            system_prompt="sys",
            initial_message="task",
            cwd=str(tmp_path),
            context_manager=FullTranscript(),
            trace_file=trace_file,
            session_number=1,
            trace_path=trace_path,
            state_path=state_path,
        )
        return session, trace_file, trace_path, state_path

    def test_write_trace_updates_state_json(self, tmp_path: Path):
        session, trace_file, trace_path, state_path = self._make_session(tmp_path)
        try:
            session._write_trace({
                "event": "tool_call", "session_number": 1, "turn_number": 0,
                "tool_name": "bash", "args_summary": "ls",
                "result_summary": "out",
            })
            data = json.loads(state_path.read_text())
            assert len(data["trace"]) == 1
            assert data["trace"][0]["action"] == "bash(ls)"
            assert data["state"]["current_attempt"] == "bash(ls)"
        finally:
            trace_file.close()

    def test_no_state_write_when_state_path_is_none(self, tmp_path: Path):
        """Gating check: a Session instantiated without state_path must NOT
        create or touch a .solver/ dir, even if one exists as a sibling. This
        is the wo_yuj arm's guarantee."""
        from scripts.llm_solver.harness.loop import Session
        from scripts.llm_solver.harness.context import FullTranscript
        cfg = _make_session_cfg()
        trace_path = tmp_path / ".trace.jsonl"
        trace_file = open(trace_path, "a")
        try:
            session = Session(
                cfg=cfg, client=None, system_prompt="sys",
                initial_message="task", cwd=str(tmp_path),
                context_manager=FullTranscript(),
                trace_file=trace_file, session_number=1,
                trace_path=trace_path, state_path=None,
            )
            session._write_trace({
                "event": "tool_call", "session_number": 1, "turn_number": 0,
                "tool_name": "bash", "args_summary": "ls",
                "result_summary": "r",
            })
            # Nothing should exist under tmp_path/.solver
            assert not (tmp_path / ".solver").exists()
        finally:
            trace_file.close()

    def test_multiple_writes_grow_state_trace(self, tmp_path: Path):
        session, trace_file, trace_path, state_path = self._make_session(tmp_path)
        try:
            for i in range(5):
                session._write_trace({
                    "event": "tool_call", "session_number": 1, "turn_number": i,
                    "tool_name": "bash", "args_summary": f"cmd{i}",
                    "result_summary": f"r{i}",
                })
            data = json.loads(state_path.read_text())
            assert len(data["trace"]) == 5
            assert [t["action"] for t in data["trace"]] == [
                f"bash(cmd{i})" for i in range(5)
            ]
            assert data["state"]["current_attempt"] == "bash(cmd4)"
        finally:
            trace_file.close()

    def test_state_matches_offline_replay_after_live_writes(self, tmp_path: Path):
        """Live write path must be byte-equivalent to calling project_from_trace
        on the resulting trace.jsonl. This is the load-bearing guarantee that
        makes .trace.jsonl the single source of truth."""
        session, trace_file, trace_path, state_path = self._make_session(tmp_path)
        try:
            for i in range(10):
                session._write_trace({
                    "event": "tool_call", "session_number": 1, "turn_number": i,
                    "tool_name": "bash" if i % 2 == 0 else "read",
                    "args_summary": f"arg{i}",
                    "result_summary": f"result{i}",
                })
        finally:
            trace_file.close()

        live = json.loads(state_path.read_text())
        replay = _project_from_trace(trace_path)
        assert live == replay


# ── reasoning field — CompoundContext support ──────────────────────────────

class TestProjectReasoning:
    def test_reasoning_defaults_to_empty_when_field_missing(self):
        # Back-compat: old trace events that predate the reasoning field
        # must still project cleanly. The trace entry gets an empty string,
        # not a KeyError or None.
        events = [
            {"event": "tool_call", "session_number": 1, "turn_number": 0,
             "tool_name": "bash", "args_summary": "cmd='ls'",
             "result_summary": "file1\n"},
        ]
        out = _project(events)
        assert out["trace"][0]["reasoning"] == ""

    def test_reasoning_propagates_from_event_to_trace_entry(self):
        events = [
            {"event": "tool_call", "session_number": 1, "turn_number": 0,
             "tool_name": "bash", "args_summary": "cmd='ls'",
             "result_summary": "file1\n",
             "reasoning": "Let me check what files are in the repo first."},
        ]
        out = _project(events)
        assert out["trace"][0]["reasoning"] == \
            "Let me check what files are in the repo first."

    def test_multi_tool_turn_shares_reasoning_across_entries(self):
        # Same turn_number, multiple tool calls, same reasoning text.
        # state_writer makes no attempt to deduplicate — the renderer does.
        events = [
            {"event": "tool_call", "session_number": 1, "turn_number": 3,
             "tool_name": "read", "args_summary": "path='a.py'",
             "result_summary": "...", "reasoning": "I need a.py and b.py"},
            {"event": "tool_call", "session_number": 1, "turn_number": 3,
             "tool_name": "read", "args_summary": "path='b.py'",
             "result_summary": "...", "reasoning": "I need a.py and b.py"},
        ]
        out = _project(events)
        assert len(out["trace"]) == 2
        assert out["trace"][0]["reasoning"] == out["trace"][1]["reasoning"]
        assert out["trace"][0]["reasoning"] == "I need a.py and b.py"

    def test_reasoning_none_becomes_empty_string(self):
        events = [
            {"event": "tool_call", "session_number": 1, "turn_number": 0,
             "tool_name": "bash", "args_summary": "x", "result_summary": "y",
             "reasoning": None},
        ]
        out = _project(events)
        assert out["trace"][0]["reasoning"] == ""
