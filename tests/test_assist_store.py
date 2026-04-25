import json
from pathlib import Path
from unittest.mock import patch

from scripts.llm_assist.__main__ import main
from scripts.llm_assist.progress import TraceFollower
from scripts.llm_assist.runner import (
    approval_request_path,
    load_approval_request,
    prepare_smoke_repo,
    resolve_served_model,
    resolve_smoke_model,
    save_approval_request,
    session_trace_tail,
    session_turn_tail,
)
from scripts.llm_assist.store import SessionStore


def test_session_store_round_trip(tmp_path: Path):
    store = SessionStore(tmp_path)

    record = store.create_session(
        cwd=tmp_path / "work",
        model="qwen3-8b",
        prompt_text="Fix the failing test",
        prompt_source="inline",
        context_mode="full",
        system_prompt_path=None,
        config_paths=[],
    )

    fetched = store.get_session(record.session_id)
    assert fetched is not None
    assert fetched.cwd == str((tmp_path / "work").resolve())
    assert fetched.status == "created"
    assert fetched.context_mode == "full"

    store.update_session(
        record.session_id,
        status="paused",
        last_finish_reason="max_turns",
    )
    updated = store.get_session(record.session_id)
    assert updated is not None
    assert updated.status == "paused"
    assert updated.last_finish_reason == "max_turns"

    sessions = store.list_sessions(limit=5)
    assert [item.session_id for item in sessions] == [record.session_id]


def test_session_trace_tail_formats_recent_events(tmp_path: Path):
    artifact_dir = tmp_path / "session"
    artifact_dir.mkdir(parents=True)
    trace_path = artifact_dir / ".trace.jsonl"
    events = [
        {"event": "session_start", "session_number": 1},
        {
            "event": "tool_call",
            "session_number": 1,
            "turn_number": 0,
            "tool_name": "bash",
            "args_summary": "cmd='pytest -q tests/test_app.py'",
            "result_summary": "1 passed",
        },
        {
            "event": "session_end",
            "session_number": 1,
            "finish_reason": "stop",
            "turns": 2,
        },
    ]
    trace_path.write_text("".join(json.dumps(event) + "\n" for event in events))

    lines = session_trace_tail(artifact_dir, limit=2)

    assert lines == [
        "tool_call turn=0 bash(cmd='pytest -q tests/test_app.py') => 1 passed",
        "session_end session=1 finish_reason=stop turns=2",
    ]


def test_session_turn_tail_groups_reasoning_and_tools(tmp_path: Path):
    artifact_dir = tmp_path / "session"
    artifact_dir.mkdir(parents=True)
    trace_path = artifact_dir / ".trace.jsonl"
    events = [
        {"event": "session_start", "session_number": 1},
        {
            "event": "tool_call",
            "session_number": 1,
            "turn_number": 0,
            "tool_name": "read",
            "args_summary": "path='calc.py'",
            "result_summary": "def add(a, b): return a - b",
            "reasoning": "Inspect the implementation before editing.",
            "gate_blocked": False,
        },
        {
            "event": "tool_call",
            "session_number": 1,
            "turn_number": 0,
            "tool_name": "bash",
            "args_summary": "cmd='pytest -q tests/test_calc.py'",
            "result_summary": "1 failed",
            "reasoning": "Inspect the implementation before editing.",
            "gate_blocked": False,
        },
        {
            "event": "tool_call",
            "session_number": 1,
            "turn_number": 1,
            "tool_name": "done",
            "args_summary": "",
            "result_summary": "Session ended by model.",
            "reasoning": "",
            "gate_blocked": False,
        },
    ]
    trace_path.write_text("".join(json.dumps(event) + "\n" for event in events))

    lines = session_turn_tail(artifact_dir, limit=2)

    assert lines == [
        "turn 0 (session 1)",
        "  reasoning: Inspect the implementation before editing.",
        "  tool: read(path='calc.py')",
        "    result: def add(a, b): return a - b",
        "  tool: bash(cmd='pytest -q tests/test_calc.py')",
        "    result: 1 failed",
        "turn 1 (session 1)",
        "  tool: done()",
        "    result: Session ended by model.",
    ]


def test_show_command_prints_session_details_and_trace_tail(tmp_path, capsys):
    store = SessionStore(tmp_path)
    record = store.create_session(
        cwd=tmp_path / "work",
        model="qwen3-8b",
        prompt_text="Fix the failing test",
        prompt_source="inline",
        context_mode="full",
        system_prompt_path=None,
        config_paths=[],
    )
    artifact_dir = Path(record.artifact_dir)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    trace_path = artifact_dir / ".trace.jsonl"
    events = [
        {"event": "session_start", "session_number": 1},
        {
            "event": "tool_call",
            "session_number": 1,
            "turn_number": 0,
            "tool_name": "read",
            "args_summary": "path='calc.py'",
            "result_summary": "def add(a, b): return a - b",
            "reasoning": "Read the buggy implementation first.",
        },
        {
            "event": "session_end",
            "session_number": 1,
            "finish_reason": "stop",
            "turns": 1,
        },
    ]
    trace_path.write_text("".join(json.dumps(event) + "\n" for event in events))
    store.update_session(record.session_id, status="completed", last_finish_reason="stop")

    with patch("scripts.llm_assist.__main__.SessionStore", return_value=store):
        rc = main(["show", record.session_id, "--trace-lines", "2"])

    captured = capsys.readouterr()
    assert rc == 0
    assert f"session_id: {record.session_id}" in captured.out
    assert "status: completed" in captured.out
    assert "finish_reason: stop" in captured.out
    assert "approval: none" in captured.out
    assert "recent_turns:" in captured.out
    assert "turn 0 (session 1)" in captured.out
    assert "reasoning: Read the buggy implementation first." in captured.out
    assert "trace_tail:" in captured.out
    assert "tool_call turn=0 read(path='calc.py')" in captured.out
    assert "session_end session=1 finish_reason=stop turns=1" in captured.out


def test_approve_command_marks_pending_request_approved(tmp_path, capsys):
    store = SessionStore(tmp_path)
    record = store.create_session(
        cwd=tmp_path / "work",
        model="qwen3-8b",
        prompt_text="Fix the failing test",
        prompt_source="inline",
        context_mode="full",
        system_prompt_path=None,
        config_paths=[],
    )
    save_approval_request(
        Path(record.artifact_dir),
        {
            "status": "pending",
            "tool_name": "bash",
            "cmd": "rm -rf build",
            "args_summary": "cmd='rm -rf build'",
            "reason": "destructive file deletion via rm",
        },
    )

    with patch("scripts.llm_assist.__main__.SessionStore", return_value=store):
        rc = main(["approve", record.session_id])

    captured = capsys.readouterr()
    approval = load_approval_request(Path(record.artifact_dir))
    assert rc == 0
    assert approval is not None
    assert approval["status"] == "approved"
    assert f"approved: {record.session_id}" in captured.out


def test_resume_rejects_pending_approval_request(tmp_path):
    store = SessionStore(tmp_path)
    record = store.create_session(
        cwd=tmp_path / "work",
        model="qwen3-8b",
        prompt_text="Fix the failing test",
        prompt_source="inline",
        context_mode="full",
        system_prompt_path=None,
        config_paths=[],
    )
    save_approval_request(
        Path(record.artifact_dir),
        {
            "status": "pending",
            "tool_name": "bash",
            "cmd": "rm -rf build",
            "args_summary": "cmd='rm -rf build'",
            "reason": "destructive file deletion via rm",
        },
    )
    store.update_session(record.session_id, status="paused", last_finish_reason="approval_required")

    with patch("scripts.llm_assist.__main__.SessionStore", return_value=store):
        try:
            main(["resume", record.session_id])
        except SystemExit as exc:
            assert "pending approval request" in str(exc)
        else:
            raise AssertionError("expected SystemExit")


def test_show_command_prints_pending_approval_request(tmp_path, capsys):
    store = SessionStore(tmp_path)
    record = store.create_session(
        cwd=tmp_path / "work",
        model="qwen3-8b",
        prompt_text="Fix the failing test",
        prompt_source="inline",
        context_mode="full",
        system_prompt_path=None,
        config_paths=[],
    )
    save_approval_request(
        Path(record.artifact_dir),
        {
            "status": "pending",
            "tool_name": "bash",
            "cmd": "rm -rf build",
            "args_summary": "cmd='rm -rf build'",
            "reason": "destructive file deletion via rm",
        },
    )

    with patch("scripts.llm_assist.__main__.SessionStore", return_value=store):
        rc = main(["show", record.session_id, "--trace-lines", "0", "--turns", "0"])

    captured = capsys.readouterr()
    assert rc == 0
    assert "status: approval_pending" in captured.out
    assert "approval: pending" in captured.out
    assert "approval_reason: destructive file deletion via rm" in captured.out
    assert "approval_action: bash(cmd='rm -rf build')" in captured.out


def test_show_command_reports_running_for_resumed_active_session(tmp_path, capsys):
    store = SessionStore(tmp_path)
    record = store.create_session(
        cwd=tmp_path / "work",
        model="qwen3-8b",
        prompt_text="Fix the failing test",
        prompt_source="inline",
        context_mode="full",
        system_prompt_path=None,
        config_paths=[],
    )
    # SQLite row still reflects the prior paused state after approval.
    store.update_session(record.session_id, status="paused", last_finish_reason="approval_required")

    artifact_dir = Path(record.artifact_dir)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    trace_path = artifact_dir / ".trace.jsonl"
    events = [
        {"event": "session_start", "session_number": 1},
        {
            "event": "session_end",
            "session_number": 1,
            "finish_reason": "approval_required",
            "turns": 2,
        },
        {"event": "session_start", "session_number": 2},
    ]
    trace_path.write_text("".join(json.dumps(e) + "\n" for e in events))

    with patch("scripts.llm_assist.__main__.SessionStore", return_value=store):
        rc = main(["show", record.session_id, "--trace-lines", "0", "--turns", "0"])

    captured = capsys.readouterr()
    assert rc == 0
    assert "status: running" in captured.out
    assert "current_session: 2" in captured.out
    assert "finish_reason: approval_required" not in captured.out


def test_show_command_preserves_completed_finish_reason(tmp_path, capsys):
    store = SessionStore(tmp_path)
    record = store.create_session(
        cwd=tmp_path / "work",
        model="qwen3-8b",
        prompt_text="Fix the failing test",
        prompt_source="inline",
        context_mode="full",
        system_prompt_path=None,
        config_paths=[],
    )
    store.update_session(record.session_id, status="completed", last_finish_reason="stop")
    artifact_dir = Path(record.artifact_dir)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    trace_path = artifact_dir / ".trace.jsonl"
    events = [
        {"event": "session_start", "session_number": 1},
        {
            "event": "session_end",
            "session_number": 1,
            "finish_reason": "stop",
            "turns": 3,
        },
    ]
    trace_path.write_text("".join(json.dumps(e) + "\n" for e in events))

    with patch("scripts.llm_assist.__main__.SessionStore", return_value=store):
        rc = main(["show", record.session_id, "--trace-lines", "0", "--turns", "0"])

    captured = capsys.readouterr()
    assert rc == 0
    assert "status: completed" in captured.out
    assert "finish_reason: stop" in captured.out
    assert "current_session: 1" in captured.out


def test_prepare_smoke_repo_creates_bugged_repo(tmp_path: Path):
    repo = prepare_smoke_repo(tmp_path / "assist-smoke")

    assert repo == (tmp_path / "assist-smoke").resolve()
    assert (repo / "calc.py").read_text() == "def add(a, b):\n    return a - b\n"
    assert "assert add(2, 3) == 5" in (repo / "tests" / "test_calc.py").read_text()


def test_resolve_smoke_model_prefers_exact_served_id_when_alias_missing():
    with patch("scripts.llm_assist.runner._default_model", return_value="qwen3-8b"), \
            patch("scripts.llm_assist.runner.load_config", return_value=object()), \
            patch("scripts.llm_assist.runner._load_profile", return_value=None), \
            patch("scripts.llm_assist.runner.LlamaClient") as client_cls:
        client_cls.return_value.health_check.return_value = [
            "Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf"
        ]

        model, served = resolve_smoke_model([])

    assert model == "Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf"
    assert served == ["Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf"]


def test_resolve_served_model_returns_exact_id_when_alias_present():
    with patch("scripts.llm_assist.runner._default_model", return_value="qwen3-8b"), \
            patch("scripts.llm_assist.runner.load_config", return_value=object()), \
            patch("scripts.llm_assist.runner._load_profile", return_value=None), \
            patch("scripts.llm_assist.runner.LlamaClient") as client_cls:
        client_cls.return_value.health_check.return_value = [
            "qwen3-8b",
            "Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf",
        ]

        model, served = resolve_served_model([])

    assert model == "qwen3-8b"
    assert "Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf" in served


def test_resolve_served_model_falls_back_to_first_served_id():
    with patch("scripts.llm_assist.runner._default_model", return_value="qwen3-8b"), \
            patch("scripts.llm_assist.runner.load_config", return_value=object()), \
            patch("scripts.llm_assist.runner._load_profile", return_value=None), \
            patch("scripts.llm_assist.runner.LlamaClient") as client_cls:
        client_cls.return_value.health_check.return_value = [
            "Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf",
            "fallback-model.gguf",
        ]

        model, served = resolve_served_model([], requested_model="qwen3-8b")

    assert model == "Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf"
    assert served[0] == "Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf"


def test_trace_follower_prints_new_events(tmp_path: Path):
    artifact_dir = tmp_path / "session"
    artifact_dir.mkdir(parents=True)
    trace_path = artifact_dir / ".trace.jsonl"
    trace_path.write_text("")

    rendered: list[str] = []
    follower = TraceFollower(artifact_dir, print_fn=rendered.append, poll_interval=0.01)
    follower.start()

    events = [
        {"event": "session_start", "session_number": 1},
        {
            "event": "tool_call",
            "session_number": 1,
            "turn_number": 0,
            "tool_name": "read",
            "args_summary": "path='calc.py'",
            "result_summary": "def add(a, b): return a - b",
        },
        {
            "event": "approval_request",
            "session_number": 1,
            "turn_number": 1,
            "tool_name": "bash",
            "args_summary": "cmd='rm -rf build'",
            "reason": "destructive file deletion via rm",
        },
        {
            "event": "session_end",
            "session_number": 1,
            "finish_reason": "approval_required",
            "turns": 2,
        },
    ]
    with open(trace_path, "a") as f:
        for event in events:
            f.write(json.dumps(event) + "\n")
            f.flush()

    follower.stop()

    assert any(line.startswith("session_start session=1") for line in rendered)
    assert any("tool_call turn=0 read" in line for line in rendered)
    assert any("approval_request turn=1 bash" in line for line in rendered)
    assert any("session_end session=1 finish_reason=approval_required" in line for line in rendered)


def test_trace_follower_does_not_duplicate_existing_events(tmp_path: Path):
    artifact_dir = tmp_path / "session"
    artifact_dir.mkdir(parents=True)
    trace_path = artifact_dir / ".trace.jsonl"

    historical = [
        {"event": "session_start", "session_number": 1},
        {
            "event": "tool_call",
            "session_number": 1,
            "turn_number": 0,
            "tool_name": "read",
            "args_summary": "path='old.py'",
            "result_summary": "stale contents",
        },
        {
            "event": "session_end",
            "session_number": 1,
            "finish_reason": "stop",
            "turns": 1,
        },
    ]
    trace_path.write_text("".join(json.dumps(e) + "\n" for e in historical))

    rendered: list[str] = []
    follower = TraceFollower(artifact_dir, print_fn=rendered.append, poll_interval=0.01)
    follower.start()

    new_event = {
        "event": "tool_call",
        "session_number": 2,
        "turn_number": 0,
        "tool_name": "edit",
        "args_summary": "path='new.py'",
        "result_summary": "ok",
    }
    with open(trace_path, "a") as f:
        f.write(json.dumps(new_event) + "\n")
        f.flush()

    follower.stop()

    assert not any("old.py" in line for line in rendered)
    assert any("tool_call turn=0 edit" in line and "new.py" in line for line in rendered)


def test_cmd_run_prints_progress_before_final_result(tmp_path, capsys):
    store = SessionStore(tmp_path / "assist-home")
    work_dir = tmp_path / "work"
    work_dir.mkdir()

    def fake_run_session(store_obj, record, *, resume):
        artifact_dir = Path(record.artifact_dir)
        artifact_dir.mkdir(parents=True, exist_ok=True)
        trace_path = artifact_dir / ".trace.jsonl"
        events = [
            {"event": "session_start", "session_number": 1},
            {
                "event": "tool_call",
                "session_number": 1,
                "turn_number": 0,
                "tool_name": "bash",
                "args_summary": "cmd='pytest -q'",
                "result_summary": "1 passed",
            },
            {
                "event": "session_end",
                "session_number": 1,
                "finish_reason": "stop",
                "turns": 1,
            },
        ]
        # Write in a loop with small delays so the follower's poll sees them.
        import time as _t
        with open(trace_path, "a") as f:
            for event in events:
                f.write(json.dumps(event) + "\n")
                f.flush()
                _t.sleep(0.02)
        store_obj.update_session(record.session_id, status="completed", last_finish_reason="stop")
        return True, "stop"

    with patch("scripts.llm_assist.__main__.SessionStore", return_value=store), \
            patch(
                "scripts.llm_assist.__main__.resolve_served_model",
                return_value=("exact-served.gguf", ["exact-served.gguf"]),
            ), \
            patch("scripts.llm_assist.__main__.run_session", side_effect=fake_run_session):
        rc = main([
            "run",
            "--cwd", str(work_dir),
            "--prompt-text", "do it",
        ])

    captured = capsys.readouterr()
    assert rc == 0
    # Progress output from the follower precedes the closing "status: completed"
    # block printed by _print_session_result.
    idx_progress = captured.out.find("tool_call turn=0 bash")
    idx_status = captured.out.find("status: completed")
    idx_start = captured.out.find("starting:")
    assert idx_start != -1
    assert idx_progress != -1 and idx_status != -1
    assert idx_start < idx_progress
    assert idx_progress < idx_status


def test_code_alias_routes_to_run(tmp_path, capsys):
    store = SessionStore(tmp_path / "assist-home")
    work_dir = tmp_path / "work"
    work_dir.mkdir()

    def fake_run_session(store_obj, record, *, resume):
        artifact_dir = Path(record.artifact_dir)
        artifact_dir.mkdir(parents=True, exist_ok=True)
        (artifact_dir / ".trace.jsonl").write_text(
            json.dumps({
                "event": "session_end",
                "session_number": 1,
                "finish_reason": "stop",
                "turns": 1,
            }) + "\n"
        )
        store_obj.update_session(record.session_id, status="completed", last_finish_reason="stop")
        return True, "stop"

    with patch("scripts.llm_assist.__main__.SessionStore", return_value=store), \
            patch(
                "scripts.llm_assist.__main__.resolve_served_model",
                return_value=("exact-served.gguf", ["exact-served.gguf"]),
            ), \
            patch("scripts.llm_assist.__main__.run_session", side_effect=fake_run_session) as run_mock:
        rc = main([
            "code",
            "--cwd", str(work_dir),
            "--prompt-text", "do it",
        ])

    assert rc == 0
    assert run_mock.called is True
    sessions = store.list_sessions(limit=1)
    assert sessions and sessions[0].status == "completed"


def test_code_alias_help_exits_cleanly(capsys):
    try:
        main(["code", "--help"])
    except SystemExit as exc:
        assert exc.code == 0
    else:
        raise AssertionError("expected SystemExit")
    captured = capsys.readouterr()
    assert "usage: yuj code" in captured.out
    assert "--cwd" in captured.out
    assert "--prompt-text" in captured.out


def test_code_uses_positional_prompt_and_current_dir_by_default(tmp_path, capsys, monkeypatch):
    store = SessionStore(tmp_path / "assist-home")
    work_dir = tmp_path / "work"
    work_dir.mkdir()
    monkeypatch.chdir(work_dir)

    def fake_run_session(store_obj, record, *, resume):
        artifact_dir = Path(record.artifact_dir)
        artifact_dir.mkdir(parents=True, exist_ok=True)
        (artifact_dir / ".trace.jsonl").write_text(
            json.dumps({
                "event": "session_end",
                "session_number": 1,
                "finish_reason": "stop",
                "turns": 1,
            }) + "\n"
        )
        store_obj.update_session(record.session_id, status="completed", last_finish_reason="stop")
        return True, "stop"

    with patch("scripts.llm_assist.__main__.SessionStore", return_value=store), \
            patch(
                "scripts.llm_assist.__main__.resolve_served_model",
                return_value=("exact-served.gguf", ["exact-served.gguf"]),
            ), \
            patch("scripts.llm_assist.__main__.run_session", side_effect=fake_run_session):
        rc = main(["code", "fix", "the", "failing", "test"])

    captured = capsys.readouterr()
    assert rc == 0
    assert "starting:" in captured.out
    sessions = store.list_sessions(limit=1)
    assert sessions, "expected a persisted session record"
    assert sessions[0].cwd == str(work_dir.resolve())
    assert sessions[0].prompt_text == "fix the failing test"
    assert sessions[0].prompt_source == "inline-positional"


def test_run_persists_exact_served_model_id(tmp_path, capsys):
    store = SessionStore(tmp_path / "assist-home")
    work_dir = tmp_path / "work"
    work_dir.mkdir()

    def fake_run_session(store_obj, record, *, resume):
        artifact_dir = Path(record.artifact_dir)
        artifact_dir.mkdir(parents=True, exist_ok=True)
        (artifact_dir / ".trace.jsonl").write_text(
            json.dumps({
                "event": "session_end",
                "session_number": 1,
                "finish_reason": "stop",
                "turns": 1,
            }) + "\n"
        )
        store_obj.update_session(record.session_id, status="completed", last_finish_reason="stop")
        return True, "stop"

    with patch("scripts.llm_assist.__main__.SessionStore", return_value=store), \
            patch(
                "scripts.llm_assist.__main__.resolve_served_model",
                return_value=("exact-served-id.gguf", ["exact-served-id.gguf"]),
            ), \
            patch("scripts.llm_assist.__main__.run_session", side_effect=fake_run_session):
        rc = main([
            "run",
            "--cwd", str(work_dir),
            "--prompt-text", "do the thing",
            "--model", "qwen3-8b",
        ])

    assert rc == 0
    sessions = store.list_sessions(limit=1)
    assert sessions, "expected a persisted session record"
    assert sessions[0].model == "exact-served-id.gguf"


def test_show_without_id_prefers_latest_session_in_current_cwd(tmp_path, capsys, monkeypatch):
    store = SessionStore(tmp_path / "assist-home")
    repo_a = tmp_path / "repo-a"
    repo_b = tmp_path / "repo-b"
    repo_a.mkdir()
    repo_b.mkdir()
    monkeypatch.chdir(repo_a)

    first = store.create_session(
        cwd=repo_b,
        model="other-model",
        prompt_text="other repo",
        prompt_source="inline",
        context_mode="full",
        system_prompt_path=None,
        config_paths=[],
    )
    second = store.create_session(
        cwd=repo_a,
        model="local-model",
        prompt_text="current repo",
        prompt_source="inline",
        context_mode="full",
        system_prompt_path=None,
        config_paths=[],
    )
    artifact_dir = Path(second.artifact_dir)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    (artifact_dir / ".trace.jsonl").write_text(
        json.dumps({
            "event": "session_end",
            "session_number": 1,
            "finish_reason": "stop",
            "turns": 1,
        }) + "\n"
    )

    with patch("scripts.llm_assist.__main__.SessionStore", return_value=store):
        rc = main(["show", "--trace-lines", "0", "--turns", "0"])

    captured = capsys.readouterr()
    assert rc == 0
    assert first.session_id not in captured.out
    assert f"session_id: {second.session_id}" in captured.out


def test_resume_without_id_prefers_latest_resumable_session_in_current_cwd(tmp_path, monkeypatch):
    store = SessionStore(tmp_path / "assist-home")
    repo_a = tmp_path / "repo-a"
    repo_b = tmp_path / "repo-b"
    repo_a.mkdir()
    repo_b.mkdir()
    monkeypatch.chdir(repo_a)

    local_completed = store.create_session(
        cwd=repo_a,
        model="done-model",
        prompt_text="done",
        prompt_source="inline",
        context_mode="full",
        system_prompt_path=None,
        config_paths=[],
    )
    store.update_session(local_completed.session_id, status="completed", last_finish_reason="stop")

    local_paused = store.create_session(
        cwd=repo_a,
        model="paused-model",
        prompt_text="resume me",
        prompt_source="inline",
        context_mode="full",
        system_prompt_path=None,
        config_paths=[],
    )
    store.update_session(local_paused.session_id, status="paused", last_finish_reason="max_turns")

    remote_paused = store.create_session(
        cwd=repo_b,
        model="remote-model",
        prompt_text="remote",
        prompt_source="inline",
        context_mode="full",
        system_prompt_path=None,
        config_paths=[],
    )
    store.update_session(remote_paused.session_id, status="paused", last_finish_reason="max_turns")

    selected: list[str] = []

    def fake_run_session(store_obj, record, *, resume):
        selected.append(record.session_id)
        artifact_dir = Path(record.artifact_dir)
        artifact_dir.mkdir(parents=True, exist_ok=True)
        (artifact_dir / ".trace.jsonl").write_text(
            json.dumps({
                "event": "session_end",
                "session_number": 2,
                "finish_reason": "stop",
                "turns": 2,
            }) + "\n"
        )
        store_obj.update_session(record.session_id, status="completed", last_finish_reason="stop")
        return True, "stop"

    with patch("scripts.llm_assist.__main__.SessionStore", return_value=store), \
            patch("scripts.llm_assist.__main__.run_session", side_effect=fake_run_session):
        rc = main(["resume"])

    assert rc == 0
    assert selected == [local_paused.session_id]


def test_approve_without_id_prefers_latest_pending_request_in_current_cwd(tmp_path, capsys, monkeypatch):
    store = SessionStore(tmp_path / "assist-home")
    repo_a = tmp_path / "repo-a"
    repo_b = tmp_path / "repo-b"
    repo_a.mkdir()
    repo_b.mkdir()
    monkeypatch.chdir(repo_a)

    local_record = store.create_session(
        cwd=repo_a,
        model="local-model",
        prompt_text="local",
        prompt_source="inline",
        context_mode="full",
        system_prompt_path=None,
        config_paths=[],
    )
    remote_record = store.create_session(
        cwd=repo_b,
        model="remote-model",
        prompt_text="remote",
        prompt_source="inline",
        context_mode="full",
        system_prompt_path=None,
        config_paths=[],
    )
    save_approval_request(
        Path(local_record.artifact_dir),
        {
            "status": "pending",
            "tool_name": "bash",
            "cmd": "rm -rf build",
            "args_summary": "cmd='rm -rf build'",
            "reason": "destructive file deletion via rm",
        },
    )
    save_approval_request(
        Path(remote_record.artifact_dir),
        {
            "status": "pending",
            "tool_name": "bash",
            "cmd": "rm -rf other",
            "args_summary": "cmd='rm -rf other'",
            "reason": "destructive file deletion via rm",
        },
    )

    with patch("scripts.llm_assist.__main__.SessionStore", return_value=store):
        rc = main(["approve"])

    captured = capsys.readouterr()
    approval = load_approval_request(Path(local_record.artifact_dir))
    assert rc == 0
    assert approval is not None
    assert approval["status"] == "approved"
    assert f"approved: {local_record.session_id}" in captured.out


def test_smoke_command_fails_when_repo_not_fixed(tmp_path, capsys):
    store = SessionStore(tmp_path / "assist-home")
    smoke_root = tmp_path / "repo"

    def fake_run_session(store_obj, record, *, resume):
        artifact_dir = Path(record.artifact_dir)
        artifact_dir.mkdir(parents=True, exist_ok=True)
        (artifact_dir / ".trace.jsonl").write_text(
            json.dumps({
                "event": "session_end",
                "session_number": 1,
                "finish_reason": "stop",
                "turns": 1,
            }) + "\n"
        )
        store_obj.update_session(record.session_id, status="completed", last_finish_reason="stop")
        return True, "stop"

    with patch("scripts.llm_assist.__main__.SessionStore", return_value=store), \
            patch("scripts.llm_assist.__main__.resolve_smoke_model", return_value=("exact-model", ["exact-model"])), \
            patch("scripts.llm_assist.__main__.run_session", side_effect=fake_run_session):
        rc = main([
            "smoke",
            "--root", str(smoke_root),
            "--assist-home", str(tmp_path / "assist-home"),
        ])

    captured = capsys.readouterr()
    assert rc == 1
    assert "smoke acceptance failed" in captured.out
    assert "calc.py does not contain the fixed 'return a + b' body" in captured.out
    assert f"smoke_repo: {smoke_root.resolve()}" in captured.out
    assert "session_id:" in captured.out


def test_smoke_command_succeeds_when_repo_fixed_and_tests_pass(tmp_path, capsys):
    store = SessionStore(tmp_path / "assist-home")
    smoke_root = tmp_path / "repo"

    def fake_run_session(store_obj, record, *, resume):
        artifact_dir = Path(record.artifact_dir)
        artifact_dir.mkdir(parents=True, exist_ok=True)
        (artifact_dir / ".trace.jsonl").write_text(
            json.dumps({
                "event": "session_end",
                "session_number": 1,
                "finish_reason": "stop",
                "turns": 1,
            }) + "\n"
        )
        # Apply the fix the assistant would have applied.
        (Path(record.cwd) / "calc.py").write_text(
            "def add(a, b):\n"
            "    return a + b\n"
        )
        store_obj.update_session(record.session_id, status="completed", last_finish_reason="stop")
        return True, "stop"

    with patch("scripts.llm_assist.__main__.SessionStore", return_value=store), \
            patch("scripts.llm_assist.__main__.resolve_smoke_model", return_value=("exact-model", ["exact-model"])), \
            patch("scripts.llm_assist.__main__.run_session", side_effect=fake_run_session):
        rc = main([
            "smoke",
            "--root", str(smoke_root),
            "--assist-home", str(tmp_path / "assist-home"),
        ])

    captured = capsys.readouterr()
    assert rc == 0
    assert "smoke acceptance failed" not in captured.out
    assert "status: completed" in captured.out


def test_smoke_command_fails_when_pending_approval_exists(tmp_path, capsys):
    store = SessionStore(tmp_path / "assist-home")
    smoke_root = tmp_path / "repo"

    def fake_run_session(store_obj, record, *, resume):
        artifact_dir = Path(record.artifact_dir)
        artifact_dir.mkdir(parents=True, exist_ok=True)
        (artifact_dir / ".trace.jsonl").write_text(
            json.dumps({
                "event": "session_end",
                "session_number": 1,
                "finish_reason": "approval_required",
                "turns": 1,
            }) + "\n"
        )
        (Path(record.cwd) / "calc.py").write_text(
            "def add(a, b):\n"
            "    return a + b\n"
        )
        save_approval_request(
            Path(record.artifact_dir),
            {
                "status": "pending",
                "tool_name": "bash",
                "cmd": "rm -rf build",
                "args_summary": "cmd='rm -rf build'",
                "reason": "destructive file deletion via rm",
            },
        )
        store_obj.update_session(record.session_id, status="paused", last_finish_reason="approval_required")
        return False, "approval_required"

    with patch("scripts.llm_assist.__main__.SessionStore", return_value=store), \
            patch("scripts.llm_assist.__main__.resolve_smoke_model", return_value=("exact-model", ["exact-model"])), \
            patch("scripts.llm_assist.__main__.run_session", side_effect=fake_run_session):
        rc = main([
            "smoke",
            "--root", str(smoke_root),
            "--assist-home", str(tmp_path / "assist-home"),
        ])

    captured = capsys.readouterr()
    assert rc == 1
    assert "smoke acceptance failed" in captured.out
    assert "pending approval request" in captured.out


def test_smoke_command_bootstraps_repo_and_runs_session(tmp_path, capsys):
    store = SessionStore(tmp_path / "assist-home")
    smoke_root = tmp_path / "repo"

    def fake_run_session(store_obj, record, *, resume):
        artifact_dir = Path(record.artifact_dir)
        artifact_dir.mkdir(parents=True, exist_ok=True)
        (artifact_dir / ".trace.jsonl").write_text(
            json.dumps({
                "event": "session_end",
                "session_number": 1,
                "finish_reason": "stop",
                "turns": 3,
            }) + "\n"
        )
        (Path(record.cwd) / "calc.py").write_text(
            "def add(a, b):\n"
            "    return a + b\n"
        )
        store_obj.update_session(record.session_id, status="completed", last_finish_reason="stop")
        return True, "stop"

    with patch("scripts.llm_assist.__main__.SessionStore", return_value=store), \
            patch("scripts.llm_assist.__main__.resolve_smoke_model", return_value=("exact-model", ["exact-model"])), \
            patch("scripts.llm_assist.__main__.run_session", side_effect=fake_run_session):
        rc = main([
            "smoke",
            "--root", str(smoke_root),
            "--assist-home", str(tmp_path / "assist-home"),
        ])

    captured = capsys.readouterr()
    assert rc == 0
    assert f"smoke_repo: {smoke_root.resolve()}" in captured.out
    assert "served_models: exact-model" in captured.out
    assert "status: completed" in captured.out
    assert (smoke_root / "calc.py").exists()
