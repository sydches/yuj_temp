"""Assistant CLI entrypoint for the yuj coding shell."""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from .progress import TraceFollower
from .runner import (
    approval_request_path,
    create_session,
    derive_live_state,
    load_approval_request,
    prepare_smoke_repo,
    resolve_served_model,
    resolve_smoke_model,
    run_session,
    save_approval_request,
    session_trace_tail,
    session_turn_count,
    session_turn_tail,
)
from .store import SessionStore

CLI_NAME = "yuj"
_LATEST_SESSION_TOKENS = {"latest", "last"}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog=CLI_NAME,
        description="Assistant-mode coding shell for the local harness",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    def _attach_run_args(p: argparse.ArgumentParser) -> None:
        p.add_argument(
            "task",
            nargs="*",
            help="task prompt text; common path if --prompt-text/--prompt-file is omitted",
        )
        p.add_argument(
            "--cwd",
            type=Path,
            default=Path.cwd(),
            help="working directory to edit (default: current directory)",
        )
        p.add_argument("--prompt-text", help="literal task prompt")
        p.add_argument("--prompt-file", type=Path, help="read task prompt from file")
        p.add_argument("--model", "-m", help="model name or short alias")
        p.add_argument("--config", "-c", type=Path, action="append", default=[],
                       help="extra config TOML overlay; pass multiple times")
        p.add_argument("--system-prompt", type=Path, default=None,
                       help="file to prepend to the system prompt")
        p.add_argument("--context", default="full",
                       help="context strategy name (default: full)")
        p.set_defaults(func=cmd_run)

    run_parser = sub.add_parser("run", help="start a new assistant session")
    _attach_run_args(run_parser)

    code_parser = sub.add_parser(
        "code",
        help="start a new assistant coding session (alias of run)",
    )
    _attach_run_args(code_parser)

    smoke_parser = sub.add_parser("smoke", help="run an end-to-end assistant smoke task")
    smoke_parser.add_argument("--root", type=Path, default=None,
                              help="throwaway repo root (default: temp dir)")
    smoke_parser.add_argument("--assist-home", type=Path, default=None,
                              help="assistant artifact root (default: normal assist home)")
    smoke_parser.add_argument("--model", "-m", help="preferred model alias or exact model id")
    smoke_parser.add_argument("--config", "-c", type=Path, action="append", default=[],
                              help="extra config TOML overlay; pass multiple times")
    smoke_parser.add_argument("--system-prompt", type=Path, default=None,
                              help="file to prepend to the system prompt")
    smoke_parser.add_argument("--context", default="full",
                              help="context strategy name (default: full)")
    smoke_parser.set_defaults(func=cmd_smoke)

    resume_parser = sub.add_parser("resume", help="resume a prior assistant session")
    resume_parser.add_argument(
        "session_id",
        nargs="?",
        default="latest",
        help="assistant session id or 'latest' (default: latest resumable session)",
    )
    resume_parser.set_defaults(func=cmd_resume)

    approve_parser = sub.add_parser("approve", help="approve a pending assistant action")
    approve_parser.add_argument(
        "session_id",
        nargs="?",
        default="latest",
        help="assistant session id or 'latest' (default: latest pending approval)",
    )
    approve_parser.set_defaults(func=cmd_approve)

    sessions_parser = sub.add_parser("sessions", help="list known assistant sessions")
    sessions_parser.add_argument("--limit", type=int, default=20)
    sessions_parser.set_defaults(func=cmd_sessions)

    show_parser = sub.add_parser("show", help="inspect one assistant session")
    show_parser.add_argument(
        "session_id",
        nargs="?",
        default="latest",
        help="assistant session id or 'latest' (default: latest session)",
    )
    show_parser.add_argument("--turns", type=int, default=5,
                             help="number of recent turns to render")
    show_parser.add_argument("--trace-lines", type=int, default=10,
                             help="number of recent trace events to show")
    show_parser.set_defaults(func=cmd_show)

    inspect_parser = sub.add_parser("inspect", help="query knobs and presets")
    inspect_sub = inspect_parser.add_subparsers(dest="inspect_command", required=True)
    inspect_knobs = inspect_sub.add_parser("knobs", help="search or list knobs")
    inspect_knobs.add_argument("query", nargs="?", default=None)
    inspect_knobs.set_defaults(func=cmd_inspect_knobs)
    inspect_presets = inspect_sub.add_parser("presets", help="list presets")
    inspect_presets.set_defaults(func=cmd_inspect_presets)
    inspect_preset = inspect_sub.add_parser("preset", help="show one preset")
    inspect_preset.add_argument("name")
    inspect_preset.set_defaults(func=cmd_inspect_preset)

    args = parser.parse_args(argv)
    return args.func(args)


def cmd_run(args) -> int:
    prompt_text, prompt_source = _resolve_prompt_input(args)

    store = SessionStore()
    model, served = resolve_served_model(args.config, requested_model=args.model)
    record = create_session(
        store,
        cwd=args.cwd.resolve(),
        prompt_text=prompt_text,
        prompt_source=prompt_source,
        model=model,
        config_paths=args.config,
        system_prompt_path=args.system_prompt.resolve() if args.system_prompt else None,
        context_mode=args.context,
    )
    _print_session_start(
        record,
        action="starting",
        served_models=served,
    )
    with TraceFollower(record.artifact_path):
        success, finish_reason = run_session(store, record, resume=False)
    refreshed = store.get_session(record.session_id)
    _print_session_result(refreshed or record, success, finish_reason)
    return 0 if success else 1


def cmd_smoke(args) -> int:
    smoke_root = prepare_smoke_repo(args.root)
    model, served = resolve_smoke_model(args.config, requested_model=args.model)
    prompt_text = (
        "Fix the bug in calc.py so tests/test_calc.py passes. "
        "Make the smallest correct code change, run the relevant test, then finish."
    )
    store = SessionStore(args.assist_home.resolve()) if args.assist_home else SessionStore()
    record = create_session(
        store,
        cwd=smoke_root,
        prompt_text=prompt_text,
        prompt_source="smoke",
        model=model,
        config_paths=args.config,
        system_prompt_path=args.system_prompt.resolve() if args.system_prompt else None,
        context_mode=args.context,
    )
    print(f"smoke_repo: {smoke_root}")
    _print_session_start(
        record,
        action="starting smoke session",
        served_models=served,
    )
    success, finish_reason = run_session(store, record, resume=False)
    refreshed = store.get_session(record.session_id)
    final_record = refreshed or record
    _print_session_result(final_record, success, finish_reason)

    acceptance_ok, reasons = _smoke_acceptance_check(smoke_root, final_record)
    if not acceptance_ok:
        print("smoke acceptance failed:")
        for reason in reasons:
            print(f"  - {reason}")
        print(f"smoke_repo: {smoke_root}")
        print(f"session_id: {final_record.session_id}")
        print(f"artifacts: {final_record.artifact_dir}")
        print(f"status: {final_record.status}")
        if finish_reason:
            print(f"finish_reason: {finish_reason}")
        return 1
    return 0 if success else 1


def _smoke_acceptance_check(smoke_root: Path, record) -> tuple[bool, list[str]]:
    """Return (ok, reasons). Checks: fix present, tests pass, no pending approval."""
    reasons: list[str] = []

    calc_path = Path(smoke_root) / "calc.py"
    if not calc_path.is_file():
        reasons.append(f"{calc_path} is missing")
    else:
        contents = calc_path.read_text()
        if "return a + b" not in contents:
            reasons.append(f"{calc_path} does not contain the fixed 'return a + b' body")

    if not reasons:
        test_result = subprocess.run(
            [sys.executable, "-m", "pytest", "tests/test_calc.py", "-q"],
            cwd=str(smoke_root),
            capture_output=True,
            text=True,
            timeout=60,
        )
        if test_result.returncode != 0:
            tail = (test_result.stdout or test_result.stderr).strip().splitlines()[-5:]
            reasons.append("tests/test_calc.py failed: " + " | ".join(tail))

    approval = load_approval_request(record.artifact_path)
    if approval is not None and approval.get("status") == "pending":
        reasons.append("session has a pending approval request")

    return (not reasons, reasons)


def cmd_resume(args) -> int:
    store = SessionStore()
    record = _resolve_session_record(store, args.session_id, selector="resumable")
    approval = load_approval_request(record.artifact_path)
    if approval is not None and approval.get("status") == "pending":
        raise SystemExit(
            "session has a pending approval request; run "
            f"{CLI_NAME} approve {record.session_id} first"
        )
    if record.status == "completed":
        _print_session_result(record, True, record.last_finish_reason)
        return 0
    _print_session_start(record, action="resuming")
    with TraceFollower(record.artifact_path):
        success, finish_reason = run_session(store, record, resume=True)
    refreshed = store.get_session(record.session_id)
    _print_session_result(refreshed or record, success, finish_reason)
    return 0 if success else 1


def cmd_sessions(args) -> int:
    store = SessionStore()
    sessions = store.list_sessions(limit=args.limit)
    if not sessions:
        print("(no assistant sessions)")
        return 0
    for record in sessions:
        print(
            f"{record.session_id}  {record.status:9s}  "
            f"model={record.model}  cwd={record.cwd}"
        )
        if record.last_finish_reason:
            print(f"    last_finish_reason={record.last_finish_reason}")
    return 0


def cmd_approve(args) -> int:
    store = SessionStore()
    record = _resolve_session_record(store, args.session_id, selector="pending_approval")
    approval = load_approval_request(record.artifact_path)
    if approval is None or approval.get("status") != "pending":
        raise SystemExit(f"no pending approval request for session: {record.session_id}")
    approval["status"] = "approved"
    save_approval_request(record.artifact_path, approval)
    print(f"approved: {record.session_id}")
    print(f"request_file: {approval_request_path(record.artifact_path)}")
    print(f"resume with: {CLI_NAME} resume {record.session_id}")
    return 0


def cmd_show(args) -> int:
    store = SessionStore()
    record = _resolve_session_record(store, args.session_id, selector="latest")
    turns = session_turn_count(record.artifact_path)
    live = derive_live_state(record.artifact_path)
    status = live.status or record.status
    finish_reason = live.finish_reason if live.status else record.last_finish_reason
    print(f"session_id: {record.session_id}")
    print(f"status: {status}")
    if live.session_number:
        print(f"current_session: {live.session_number}")
    print(f"created_at: {record.created_at}")
    print(f"updated_at: {record.updated_at}")
    print(f"cwd: {record.cwd}")
    print(f"artifacts: {record.artifact_dir}")
    print(f"model: {record.model}")
    print(f"context: {record.context_mode}")
    print(f"prompt_source: {record.prompt_source}")
    if record.system_prompt_path:
        print(f"system_prompt: {record.system_prompt_path}")
    if finish_reason:
        print(f"finish_reason: {finish_reason}")
    print(f"turns: {turns}")
    approval = load_approval_request(record.artifact_path)
    if approval is None:
        print("approval: none")
    else:
        print(f"approval: {approval.get('status')}")
        print(f"approval_reason: {approval.get('reason')}")
        print(f"approval_action: {approval.get('tool_name')}({approval.get('args_summary') or approval.get('cmd') or ''})")
    turn_lines = session_turn_tail(record.artifact_path, limit=args.turns)
    if not turn_lines:
        print("recent_turns: (empty)")
    else:
        print("recent_turns:")
        for line in turn_lines:
            print(f"  {line}")
    trace_lines = session_trace_tail(record.artifact_path, limit=args.trace_lines)
    if not trace_lines:
        print("trace_tail: (empty)")
        return 0
    print("trace_tail:")
    for line in trace_lines:
        print(f"  {line}")
    return 0


def cmd_inspect_knobs(args) -> int:
    if args.query:
        return _run_knob_command(["search", args.query])
    return _run_knob_command(["list", "--mode", "assistant"])


def cmd_inspect_presets(args) -> int:
    return _run_knob_command(["presets"])


def cmd_inspect_preset(args) -> int:
    return _run_knob_command(["preset", args.name])


def _run_knob_command(extra_args: list[str]) -> int:
    cmd = [sys.executable, "-m", "scripts.knob", *extra_args]
    return subprocess.run(cmd, check=False).returncode


def _resolve_prompt_input(args) -> tuple[str, str]:
    has_prompt_text = args.prompt_text is not None
    has_prompt_file = args.prompt_file is not None
    has_task = bool(args.task)
    provided = int(has_prompt_text) + int(has_prompt_file) + int(has_task)
    if provided != 1:
        raise SystemExit(
            "provide exactly one prompt source: positional task text, --prompt-text, or --prompt-file"
        )
    if has_prompt_file:
        prompt_path = args.prompt_file.resolve()
        return prompt_path.read_text(), str(prompt_path)
    if has_prompt_text:
        return args.prompt_text, "inline"
    return " ".join(args.task).strip(), "inline-positional"


def _print_session_start(
    record,
    *,
    action: str,
    served_models: list[str] | None = None,
) -> None:
    print(f"{action}: {record.session_id}")
    print(f"cwd: {record.cwd}")
    print(f"model: {record.model}")
    print(f"artifacts: {record.artifact_dir}")
    if served_models is not None:
        print(f"served_models: {', '.join(served_models)}")


def _print_session_result(record, success: bool, finish_reason: str | None) -> None:
    turns = session_turn_count(record.artifact_path)
    print(f"session_id: {record.session_id}")
    print(f"status: {record.status}")
    print(f"cwd: {record.cwd}")
    print(f"artifacts: {record.artifact_dir}")
    print(f"model: {record.model}")
    if finish_reason:
        print(f"finish_reason: {finish_reason}")
    print(f"turns: {turns}")
    if not success and record.status != "completed":
        print(f"resume with: {CLI_NAME} resume {record.session_id}")


def _resolve_session_record(store: SessionStore, session_ref: str, *, selector: str):
    if session_ref.lower() not in _LATEST_SESSION_TOKENS:
        record = store.get_session(session_ref)
        if record is None:
            raise SystemExit(f"unknown session: {session_ref}")
        return record

    current_cwd = str(Path.cwd().resolve())
    sessions = store.list_sessions(limit=200)
    if not sessions:
        raise SystemExit("no assistant sessions found")

    scoped = [record for record in sessions if record.cwd == current_cwd]

    if selector == "resumable":
        for records in (scoped, sessions):
            resumable = [record for record in records if record.status != "completed"]
            if resumable:
                return resumable[0]
        raise SystemExit("no resumable assistant session found")

    if selector == "pending_approval":
        for records in (scoped, sessions):
            for record in records:
                approval = load_approval_request(record.artifact_path)
                if approval is not None and approval.get("status") == "pending":
                    return record
        raise SystemExit("no pending approval request found")

    candidates = scoped or sessions
    return candidates[0]


if __name__ == "__main__":
    sys.exit(main())
