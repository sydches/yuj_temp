"""Assistant CLI entrypoint for the yuj coding shell."""
from __future__ import annotations

import argparse
import contextlib
import os
import subprocess
import sys
from pathlib import Path

from ..llm_solver.config import get_server_base_url
from .progress import TraceFollower
from .runner import (
    approval_request_path,
    create_session,
    derive_live_state,
    load_approval_request,
    load_interrupt_marker,
    mark_session_interrupted,
    prepare_smoke_repo,
    resolve_served_model,
    resolve_smoke_model,
    run_session,
    save_approval_request,
    session_compact_summary,
    session_trace_tail,
    session_turn_count,
    session_turn_tail,
)
from .store import AmbiguousSessionRefError, SessionLockedError, SessionStore

CLI_NAME = "yuj"
_LATEST_SESSION_TOKENS = {"latest", "last"}
_PROVIDER_PRESETS = {
    "local": {"provider": "openai-compatible"},
    "openai": {
        "provider": "openai-compatible",
        "base_url": "https://api.openai.com/v1",
        "api_key": "$ENV:OPENAI_API_KEY",
    },
    "openrouter": {
        "provider": "openai-compatible",
        "base_url": "https://openrouter.ai/api/v1",
        "api_key": "$ENV:OPENROUTER_API_KEY",
    },
    "zai": {
        "provider": "openai-compatible",
        "base_url": "https://api.z.ai/api/paas/v4",
        "api_key": "$ENV:ZAI_API_KEY",
    },
    "anthropic": {
        "provider": "anthropic",
        "base_url": "https://api.anthropic.com/v1",
        "api_key": "$ENV:ANTHROPIC_API_KEY",
    },
    "custom": {"provider": "openai-compatible"},
}


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
        p.add_argument(
            "--provider",
            choices=sorted(_PROVIDER_PRESETS),
            help="model supplier preset: local, openai, anthropic, zai, openrouter, or custom",
        )
        p.add_argument(
            "--base-url",
            help="OpenAI-compatible or Anthropic API base URL; overrides provider preset",
        )
        p.add_argument(
            "--api-key-env",
            help="environment variable containing the API key; stored as an env reference, not the key",
        )
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
    smoke_parser.add_argument(
        "--provider",
        choices=sorted(_PROVIDER_PRESETS),
        help="model supplier preset: local, openai, anthropic, zai, openrouter, or custom",
    )
    smoke_parser.add_argument(
        "--base-url",
        help="OpenAI-compatible or Anthropic API base URL; overrides provider preset",
    )
    smoke_parser.add_argument(
        "--api-key-env",
        help="environment variable containing the API key; stored as an env reference, not the key",
    )
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

    status_parser = sub.add_parser("status", help="show concise status for one session")
    status_parser.add_argument(
        "session_id",
        nargs="?",
        default="latest",
        help="assistant session id/ref or 'latest' (default: latest session)",
    )
    status_parser.set_defaults(func=cmd_status)
    current_parser = sub.add_parser(
        "current",
        help="show concise status for the current/active session (alias of status latest)",
    )
    current_parser.set_defaults(func=cmd_current)

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
    transport_overrides = _transport_overrides_from_args(args)
    model, served = _resolve_model_or_exit(
        args.config,
        requested_model=args.model,
        config_overrides=transport_overrides,
    )
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
    record = _persist_session_config_overlay(
        store,
        record,
        base_config_paths=args.config,
        transport_overrides=transport_overrides,
    )
    _print_session_start(
        record,
        action="starting",
        served_models=served,
    )
    try:
        with _session_lock(store, record), TraceFollower(record.artifact_path):
            success, finish_reason = run_session(store, record, resume=False)
    except KeyboardInterrupt:
        return _handle_keyboard_interrupt(store, record)
    refreshed = store.get_session(record.session_id)
    _print_session_result(refreshed or record, success, finish_reason)
    return 0 if success else 1


def cmd_smoke(args) -> int:
    smoke_root = prepare_smoke_repo(args.root)
    transport_overrides = _transport_overrides_from_args(args)
    model, served = _resolve_smoke_model_or_exit(
        args.config,
        requested_model=args.model,
        config_overrides=transport_overrides,
    )
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
    record = _persist_session_config_overlay(
        store,
        record,
        base_config_paths=args.config,
        transport_overrides=transport_overrides,
    )
    print(f"smoke_repo: {smoke_root}")
    _print_session_start(
        record,
        action="starting smoke session",
        served_models=served,
    )
    try:
        with _session_lock(store, record):
            success, finish_reason = run_session(store, record, resume=False)
    except KeyboardInterrupt:
        return _handle_keyboard_interrupt(store, record)
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
    store.set_active_session(record.cwd, record.session_id)
    _print_session_start(record, action="resuming")
    try:
        with _session_lock(store, record), TraceFollower(record.artifact_path):
            success, finish_reason = run_session(store, record, resume=True)
    except KeyboardInterrupt:
        return _handle_keyboard_interrupt(store, record)
    refreshed = store.get_session(record.session_id)
    _print_session_result(refreshed or record, success, finish_reason)
    return 0 if success else 1


def cmd_sessions(args) -> int:
    store = SessionStore()
    sessions = store.list_sessions(limit=args.limit)
    if not sessions:
        print("(no assistant sessions)")
        return 0
    current_cwd = str(Path.cwd().resolve())
    active_ids = store.list_active_session_ids()
    locked_ids = store.list_locked_session_ids()
    print("session_id                             status     ref       flags               model  cwd")
    for record in sessions:
        flags: list[str] = []
        if record.session_id in active_ids:
            flags.append("active")
        if record.session_id in locked_ids:
            flags.append("locked")
        if record.cwd == current_cwd:
            flags.append("cwd")
        flag_text = ",".join(flags) if flags else "-"
        print(
            f"{record.session_id}  {record.status:9s}  "
            f"{record.short_id:8s}  {flag_text:18s}  {record.model}  {record.cwd}"
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
    print(f"session_ref: {record.short_id}")
    print(f"request_file: {approval_request_path(record.artifact_path)}")
    print(f"resume with: {CLI_NAME} resume {record.short_id}")
    return 0


def cmd_status(args) -> int:
    store = SessionStore()
    record = _resolve_session_record(store, args.session_id, selector="latest")
    live = derive_live_state(record.artifact_path)
    status = live.status or record.status
    finish_reason = live.finish_reason if live.status else record.last_finish_reason
    turns = session_turn_count(record.artifact_path)
    approval = load_approval_request(record.artifact_path)
    lock = store.get_session_lock(record.session_id)
    interrupt = load_interrupt_marker(record.artifact_path)

    print(f"session_id: {record.session_id}")
    print(f"session_ref: {record.short_id}")
    print(f"status: {status}")
    if finish_reason:
        print(f"finish_reason: {finish_reason}")
    print(f"turns: {turns}")
    print(f"cwd: {record.cwd}")
    print(f"model: {record.model}")
    if approval is not None and approval.get("status") == "pending":
        print("approval: pending")
    else:
        print("approval: none")
    if lock is not None:
        print(f"lock: pid={lock.owner_pid} host={lock.owner_host}")
    else:
        print("lock: none")
    if interrupt is not None:
        print("interrupt: interrupted")
    else:
        print("interrupt: none")

    if approval is not None and approval.get("status") == "pending":
        print(f"next: {CLI_NAME} approve {record.short_id}")
    elif status in {"paused", "approval_pending"}:
        print(f"next: {CLI_NAME} resume {record.short_id}")
    elif status == "running":
        print(f"next: {CLI_NAME} show {record.short_id}")
    else:
        print("next: none")
    return 0


def cmd_current(_args) -> int:
    # Mirror `status latest` explicitly for a faster operator path.
    return cmd_status(argparse.Namespace(session_id="latest"))


def cmd_show(args) -> int:
    store = SessionStore()
    record = _resolve_session_record(store, args.session_id, selector="latest")
    turns = session_turn_count(record.artifact_path)
    live = derive_live_state(record.artifact_path)
    status = live.status or record.status
    finish_reason = live.finish_reason if live.status else record.last_finish_reason
    print(f"session_id: {record.session_id}")
    print(f"session_ref: {record.short_id}")
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
    lock = store.get_session_lock(record.session_id)
    interrupt = load_interrupt_marker(record.artifact_path)
    if approval is None:
        print("approval: none")
    else:
        print(f"approval: {approval.get('status')}")
        print(f"approval_reason: {approval.get('reason')}")
        print(f"approval_action: {approval.get('tool_name')}({approval.get('args_summary') or approval.get('cmd') or ''})")
    if lock is None:
        print("lock: none")
    else:
        print(f"lock: pid={lock.owner_pid} host={lock.owner_host} since={lock.acquired_at}")
    if interrupt is None:
        print("interrupt: none")
    else:
        print(f"interrupt: {interrupt.get('finish_reason')} at {interrupt.get('interrupted_at')}")
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


def _transport_overrides_from_args(args) -> dict:
    provider = getattr(args, "provider", None)
    base_url = getattr(args, "base_url", None)
    api_key_env = getattr(args, "api_key_env", None)
    if not provider and not base_url and not api_key_env:
        return {}
    if provider == "custom" and not base_url:
        raise SystemExit("--provider custom requires --base-url")

    overrides = dict(_PROVIDER_PRESETS.get(provider or "custom", {}))
    if base_url:
        overrides["base_url"] = base_url
    if api_key_env:
        overrides["api_key"] = f"$ENV:{api_key_env}"
    if provider and provider != "local" and overrides.get("api_key", "").startswith("$ENV:"):
        env_name = overrides["api_key"].split(":", 1)[1]
        if env_name not in os.environ:
            raise SystemExit(
                f"--provider {provider} expects {env_name}; set it or pass --api-key-env"
            )
    return overrides


def _persist_session_config_overlay(
    store: SessionStore,
    record,
    *,
    base_config_paths: list[Path],
    transport_overrides: dict,
):
    if not transport_overrides:
        return record
    overlay_path = record.artifact_path / "provider.toml"
    overlay_path.parent.mkdir(parents=True, exist_ok=True)
    overlay_path.write_text(_render_provider_overlay(transport_overrides))
    config_paths = [*base_config_paths, overlay_path]
    store.update_session_config_paths(record.session_id, config_paths)
    return store.get_session(record.session_id) or record


def _render_provider_overlay(overrides: dict) -> str:
    lines = ["[server]"]
    for key in ("provider", "base_url", "api_key"):
        if key in overrides and overrides[key] is not None:
            value = str(overrides[key]).replace("\\", "\\\\").replace('"', '\\"')
            lines.append(f'{key} = "{value}"')
    return "\n".join(lines) + "\n"


def _print_session_start(
    record,
    *,
    action: str,
    served_models: list[str] | None = None,
) -> None:
    print(f"{action}: {record.session_id}")
    print(f"ref: {record.short_id}")
    print(f"cwd: {record.cwd}")
    print(f"model: {record.model}")
    print(f"artifacts: {record.artifact_dir}")
    if served_models is not None:
        print(f"served_models: {', '.join(served_models)}")


def _print_session_result(record, success: bool, finish_reason: str | None) -> None:
    turns = session_turn_count(record.artifact_path)
    print(f"session_id: {record.session_id}")
    print(f"session_ref: {record.short_id}")
    print(f"status: {record.status}")
    print(f"cwd: {record.cwd}")
    print(f"artifacts: {record.artifact_dir}")
    print(f"model: {record.model}")
    if finish_reason:
        print(f"finish_reason: {finish_reason}")
    print(f"turns: {turns}")
    _print_run_compact_summary(record)
    if not success and record.status != "completed":
        print(f"resume with: {CLI_NAME} resume {record.short_id}")


def _print_run_compact_summary(record) -> None:
    summary = session_compact_summary(record.artifact_path)
    changed_files = list(summary.get("changed_files", []))
    last_test_cmd = str(summary.get("last_test_cmd") or "")
    last_test_result = str(summary.get("last_test_result") or "unknown")

    if not changed_files and not last_test_cmd:
        return

    print("summary:")
    if changed_files:
        shown = changed_files[:5]
        tail = " ..." if len(changed_files) > 5 else ""
        print(f"  changed_files: {', '.join(shown)}{tail}")
    else:
        print("  changed_files: none observed")

    if last_test_cmd:
        print(f"  last_test: {last_test_cmd}")
        print(f"  last_test_result: {last_test_result}")
    else:
        print("  last_test: none observed")


def _handle_keyboard_interrupt(store: SessionStore, record) -> int:
    mark_session_interrupted(record.artifact_path)
    store.update_session(
        record.session_id,
        status="paused",
        last_finish_reason="interrupted",
    )
    refreshed = store.get_session(record.session_id)
    final_record = refreshed or record
    print("interrupted: session paused cleanly")
    _print_session_result(final_record, False, "interrupted")
    return 130


def _resolve_model_or_exit(
    config_paths: list[Path],
    *,
    requested_model: str | None,
    config_overrides: dict | None = None,
):
    try:
        return resolve_served_model(
            config_paths,
            requested_model=requested_model,
            config_overrides=config_overrides,
        )
    except Exception as exc:
        friendly = _friendly_model_resolution_error(exc)
        if friendly is None:
            raise
        raise SystemExit(friendly) from exc


def _resolve_smoke_model_or_exit(
    config_paths: list[Path],
    *,
    requested_model: str | None,
    config_overrides: dict | None = None,
):
    try:
        return resolve_smoke_model(
            config_paths,
            requested_model=requested_model,
            config_overrides=config_overrides,
        )
    except Exception as exc:
        friendly = _friendly_model_resolution_error(exc)
        if friendly is None:
            raise
        raise SystemExit(friendly) from exc


def _friendly_model_resolution_error(exc: Exception) -> str | None:
    base_url = get_server_base_url()
    if isinstance(exc, KeyError) and "environment variable" in str(exc):
        return str(exc)
    if exc.__class__.__name__ in {"APIConnectionError", "APITimeoutError"}:
        return (
            f"could not reach the local model server at {base_url} while resolving /v1/models; "
            "start the server or fix the base_url setting"
        )
    if isinstance(exc, RuntimeError):
        return f"could not resolve a served model from {base_url}: {exc}"
    return None


@contextlib.contextmanager
def _session_lock(store: SessionStore, record):
    try:
        store.acquire_session_lock(record.session_id)
    except SessionLockedError as exc:
        raise SystemExit(
            f"session {record.session_id} is already locked by "
            f"pid {exc.lock.owner_pid} on {exc.lock.owner_host} since {exc.lock.acquired_at}"
        ) from exc
    try:
        yield
    finally:
        store.release_session_lock(record.session_id)


def _resolve_session_record(store: SessionStore, session_ref: str, *, selector: str):
    if session_ref.lower() not in _LATEST_SESSION_TOKENS:
        try:
            record = store.resolve_session_ref(session_ref)
        except AmbiguousSessionRefError as exc:
            raise SystemExit(str(exc)) from exc
        if record is None:
            raise SystemExit(f"unknown session: {session_ref}")
        return record

    current_cwd = Path.cwd().resolve()
    sessions = store.list_sessions(limit=200)
    if not sessions:
        raise SystemExit("no assistant sessions found")

    active_record = store.get_active_session(current_cwd)
    if selector == "latest" and active_record is not None:
        return active_record
    if selector == "resumable" and active_record is not None and active_record.status != "completed":
        return active_record
    if selector == "pending_approval" and active_record is not None:
        approval = load_approval_request(active_record.artifact_path)
        if approval is not None and approval.get("status") == "pending":
            return active_record

    current_cwd_str = str(current_cwd)
    scoped = [record for record in sessions if record.cwd == current_cwd_str]

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
