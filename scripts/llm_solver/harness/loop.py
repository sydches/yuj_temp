"""Agentic loop — Session (inner) + solve_task (outer)."""
from collections import deque
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
import inspect
import json
import logging
import re
import shlex
import subprocess
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import IO

# Tools that only read filesystem/process state; safe to dispatch
# concurrently when the flag is set. Mutating tools (write/edit/bash)
# are always serialized to avoid race conditions on the writable
# task cwd.
_READONLY_TOOLS = frozenset({"read", "glob", "grep"})

import openai

from ..config import Config
from ._shell_patterns import TEST_COMMAND_RE as _TEST_COMMAND_RE
from .context import ContextManager
from .context_strategies import SolverStateContext
from .injections import (
    InjectionState,
    fire_candidates,
    load_injections,
    record_fire,
)
from .guardrails import (
    Action,
    GuardrailState,
    GuardrailRegistry,
    build_guardrail_registry,
    init_guardrail_state,
    validate_guardrail_registry,
)
from .schemas import get_tool_schemas
from .solver import build_system_prompt, collect_provenance, write_checkpoint, write_run_metrics
from .state_writer import write_state_from_events, write_state_from_trace
from .tools import ToolRegistry, build_tool_registry, dispatch, validate_tool_handlers

# Module-level constants — avoid chr(10) calls in hot paths.
_NEWLINE = "\n"


# ── Bash command normalization for duplicate detection ──────────────────
# Strips trailing pipe chains and stderr redirects so trivial variants
# like `cmd | tail -60` and `cmd | tail -80` compare as identical.
# Content-blind: operates on bash syntax structure, not on what the
# command does.  Only used for the duplicate_abort signature — the
# actual command executes unmodified.
_TRAILING_PIPE_RE = re.compile(
    r"""
    \s*                          # optional leading whitespace before pipe
    (?:                          # group: one pipe segment
        \|                       # the pipe character
        \s*                      # optional whitespace after pipe
        (?:head|tail|grep|cat|sort|uniq|wc|tee|less|more)  # common filter commands
        (?:\s+[^\|]*)?)          # their arguments (up to next pipe or end)
    +                            # one or more trailing pipe segments
    $                            # anchored at end
    """,
    re.VERBOSE,
)
_STDERR_REDIRECT_RE = re.compile(r"\s*2>&1\s*")
_BASH_READ_TARGET_RE = re.compile(
    r"^\s*(cat|head|tail|less|more|file)\s+([^\s|;&<>`$()]+)\s*$"
)
_PATH_SUFFIXES = (
    ".py", ".pyi", ".c", ".cc", ".cpp", ".cxx", ".h", ".hpp", ".hh",
    ".rs", ".go", ".java", ".js", ".jsx", ".ts", ".tsx",
)
_SHELL_SEPARATORS = frozenset({"&&", "||", "|", ";"})
_APPROVAL_REQUEST_FILE = "approval_request.json"
_APPROVAL_DECISIONS_FILE = "approval_decisions.json"


def _dedup_signature(tc) -> tuple[str, str]:
    """Build a normalized (name, args) signature for duplicate detection.

    For bash calls, normalizes the command to strip trivial variants.
    For all other tools, uses the raw arguments as-is.
    """
    args = tc.arguments
    if tc.name == "bash" and "cmd" in args:
        normalized = dict(args)
        normalized["cmd"] = _normalize_bash_for_dedup(args["cmd"])
        return (tc.name, json.dumps(normalized, sort_keys=True))
    return (tc.name, json.dumps(args, sort_keys=True))


def _normalize_bash_for_dedup(cmd: str) -> str:
    """Normalize a bash command for duplicate detection.

    Strips trailing pipe chains (| head, | tail, | grep, etc.) and
    stderr redirects (2>&1) so the model can't evade duplicate_abort by
    appending different tail/head limits to the same command.

    The normalized form is used ONLY for dedup comparison.  The actual
    command executes unmodified.
    """
    # Strip 2>&1 first (can appear before or after pipes)
    cmd = _STDERR_REDIRECT_RE.sub(" ", cmd).strip()
    # Strip trailing pipe chains
    cmd = _TRAILING_PIPE_RE.sub("", cmd).strip()
    return cmd


def _focus_signature(tc, args_summary: str, cwd: str) -> tuple[str, str]:
    """Content-blind focus target used by rumination guardrails.

    Goal: detect repeated inspection of the same file or same normalized bash
    command without waiting for the coarse duplicate-abort threshold.
    """
    if tc.name in {"read", "write", "edit"}:
        path = tc.arguments.get("path") or tc.arguments.get("file_path")
        if isinstance(path, str) and path:
            return _encode_focus_path(path, cwd)
    if tc.name == "bash":
        cmd = tc.arguments.get("cmd", "")
        if isinstance(cmd, str) and cmd.strip():
            normalized = _normalize_bash_for_dedup(cmd)
            if not normalized:
                return "", ""
            focus = _extract_bash_focus_target(normalized, cwd)
            if focus is not None:
                return focus
            return f"bash:{normalized}", _truncate_focus_display(normalized)
    if tc.arguments:
        raw = json.dumps(tc.arguments, sort_keys=True)
        return f"{tc.name}:{raw}", f"{tc.name}({args_summary})"
    return "", ""


def _canon_focus_path(path: str) -> str:
    if not path:
        return ""
    if path.startswith("/"):
        return path.rstrip("/") or "/"
    stripped = path.lstrip("./").rstrip("/")
    return stripped or "."


def _truncate_focus_display(text: str, max_chars: int = 96) -> str:
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 3] + "..."


def _encode_focus_path(path: str, cwd: str) -> tuple[str, str]:
    canon = _canon_focus_path(path)
    if path.startswith("/") and not _path_within_cwd(path, cwd):
        return f"outside:{canon}", path
    return f"file:{canon}", path


def _encode_focus_target(key_base: str, display: str, *, root_path: str, cwd: str) -> tuple[str, str]:
    if root_path.startswith("/") and not _path_within_cwd(root_path, cwd):
        return f"outside:{key_base}", display
    return f"bash:{key_base}", display


def _path_within_cwd(path: str, cwd: str) -> bool:
    try:
        path_res = Path(path).resolve()
        cwd_res = Path(cwd).resolve()
        path_res.relative_to(cwd_res)
        return True
    except (ValueError, OSError):
        return False


def _split_bash_segments(cmd: str) -> list[list[str]]:
    try:
        tokens = shlex.split(cmd, posix=True)
    except ValueError:
        return []
    segments: list[list[str]] = []
    current: list[str] = []
    for token in tokens:
        if token in _SHELL_SEPARATORS:
            if current:
                segments.append(current)
                current = []
            continue
        current.append(token)
    if current:
        segments.append(current)
    return segments


def _looks_like_path_token(token: str) -> bool:
    if not token or token.startswith("-"):
        return False
    base = token.split("::", 1)[0].rstrip(",")
    if base in {".", "..", "tests", "test"}:
        return True
    if "/" in base or base.startswith("/"):
        return True
    if base.endswith("/"):
        return True
    if base.lower().startswith("test"):
        return True
    return base.endswith(_PATH_SUFFIXES)


def _extract_test_target_from_command(cmd: str) -> str:
    if not _TEST_COMMAND_RE.search(cmd or ""):
        return ""
    try:
        tokens = shlex.split(cmd, posix=True)
    except ValueError:
        return ""
    for token in tokens:
        candidate = token.split("::", 1)[0].rstrip(",")
        if _looks_like_path_token(candidate) and "test" in candidate.lower():
            return candidate
    return ""


def _extract_bash_focus_target(cmd: str, cwd: str) -> tuple[str, str] | None:
    m = _BASH_READ_TARGET_RE.match(cmd)
    if m:
        return _encode_focus_path(m.group(2), cwd)

    test_target = _extract_test_target_from_command(cmd)
    if test_target:
        return _encode_focus_path(test_target, cwd)

    segments = _split_bash_segments(cmd)
    for segment in reversed(segments):
        if not segment:
            continue
        name = segment[0]
        rest = segment[1:]
        if name in {"ls", "tree", "du"}:
            for token in reversed(rest):
                if _looks_like_path_token(token):
                    return _encode_focus_path(token, cwd)
            return _encode_focus_path(".", cwd)
        if name == "find":
            root = "."
            pattern = ""
            path_filter = ""
            saw_root = False
            for i, token in enumerate(rest):
                if token in {"-name", "-iname", "-path", "-wholename"} and i + 1 < len(rest):
                    if token in {"-name", "-iname"}:
                        pattern = rest[i + 1]
                    else:
                        path_filter = rest[i + 1]
                if token in {"2>/dev/null", "1>/dev/null"}:
                    continue
                if not saw_root and not token.startswith("-"):
                    root = token
                    saw_root = True
                    continue
            if pattern:
                display = f"{pattern} under {root}"
                if path_filter:
                    display += f" matching {path_filter}"
                key_base = f"{_canon_focus_path(root)}::{pattern}"
                if path_filter:
                    key_base += f"::{path_filter}"
                return _encode_focus_target(key_base, display, root_path=root, cwd=cwd)
            return _encode_focus_path(root, cwd)
        if name in {"grep", "rg", "fd"}:
            for token in reversed(rest):
                if _looks_like_path_token(token):
                    return _encode_focus_path(token, cwd)
    return None


def _approval_request_path(trace_path: Path | None) -> Path | None:
    if trace_path is None:
        return None
    return Path(trace_path).parent / _APPROVAL_REQUEST_FILE


def _approval_decisions_path(trace_path: Path | None) -> Path | None:
    if trace_path is None:
        return None
    return Path(trace_path).parent / _APPROVAL_DECISIONS_FILE


def _load_approval_request(trace_path: Path | None) -> dict | None:
    req_path = _approval_request_path(trace_path)
    if req_path is None or not req_path.is_file():
        return None
    try:
        return json.loads(req_path.read_text())
    except (OSError, json.JSONDecodeError):
        return None


def _write_approval_request(trace_path: Path | None, payload: dict) -> None:
    req_path = _approval_request_path(trace_path)
    if req_path is None:
        return
    req_path.write_text(json.dumps(payload, indent=2) + "\n")


def _clear_approval_request(trace_path: Path | None) -> None:
    req_path = _approval_request_path(trace_path)
    if req_path is None or not req_path.exists():
        return
    try:
        req_path.unlink()
    except OSError:
        pass


def _load_approval_decisions(trace_path: Path | None) -> dict:
    path = _approval_decisions_path(trace_path)
    if path is None or not path.is_file():
        return {}
    try:
        data = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return {}
    return data if isinstance(data, dict) else {}


def _approval_reason_for_bash(cmd: str, cwd: str | None = None) -> str | None:
    cmd_s = (cmd or "").strip()
    if not cmd_s:
        return None
    segments = _split_bash_segments(cmd_s)
    if not segments:
        segments = [[cmd_s]]
    for segment in segments:
        if not segment:
            continue
        head = segment[0]
        rest = segment[1:]
        if head == "rm":
            return "destructive file deletion via rm"
        if head == "git" and rest[:2] == ["reset", "--hard"]:
            return "destructive git reset --hard"
        if head == "git" and rest and rest[0] == "clean":
            return "destructive git clean"
        if head == "git" and rest[:2] == ["checkout", "--"]:
            return "destructive git checkout --"
        if head == "chmod":
            return "permission change via chmod"
        if head == "chown":
            return "ownership change via chown"
        if head in {"mv", "cp"} and cwd:
            if _segment_has_external_path(rest, cwd):
                return f"{head} crosses the repo root"
    return None


def _segment_has_external_path(rest: list[str], cwd: str) -> bool:
    """Return True if any positional path token resolves outside ``cwd``.

    Flag tokens (``-r``, ``--preserve``) are skipped. A literal ``--``
    marks the end of flags; tokens after it are always treated as paths.
    Content-blind: syntax-based pre-dispatch classifier only.
    """
    saw_end_of_flags = False
    for token in rest:
        if not token:
            continue
        if not saw_end_of_flags:
            if token == "--":
                saw_end_of_flags = True
                continue
            if token.startswith("-"):
                continue
        if _path_outside_task_cwd(token, cwd):
            return True
    return False


def _path_outside_task_cwd(path: str, cwd: str) -> bool:
    """True when ``path`` resolves outside ``cwd`` (task repo root).

    Relative paths resolve against ``cwd`` rather than the process cwd
    so classification matches what the bash dispatch will see.
    """
    try:
        cwd_res = Path(cwd).resolve()
        candidate = Path(path)
        if not candidate.is_absolute():
            candidate = cwd_res / candidate
        resolved = candidate.resolve()
        resolved.relative_to(cwd_res)
        return False
    except (ValueError, OSError):
        return True

log = logging.getLogger(__name__)

# ── Error taxonomy ───────────────────────────────────────────────────────

NORMAL_LIFECYCLE = frozenset({"context_full", "length"})
MODEL_STUCK = frozenset({"duplicate_abort", "max_turns"})
_TRANSIENT_ERRORS = (openai.APIConnectionError, openai.APITimeoutError)


def _resolve_token_estimator(client) -> Callable[[list[dict]], int] | None:
    """Return the profile token estimator when the client explicitly carries one."""
    profile = _resolve_profile(client)
    estimator = getattr(profile, "estimate_tokens", None)
    if callable(estimator):
        return estimator
    return None


def _resolve_profile(client):
    """Return the profile object only when explicitly present on the client."""
    return getattr(client, "__dict__", {}).get("profile")


def _apply_profile_tool_cap(tool_schemas: list[dict], client) -> list[dict]:
    """Apply profile max_tools cap to the declared tool surface."""
    profile = _resolve_profile(client)
    if profile is None:
        return tool_schemas
    max_tools = int(getattr(profile, "max_tools", 0) or 0)
    if max_tools <= 0 or len(tool_schemas) <= max_tools:
        return tool_schemas
    log.info("Profile tool cap: sending first %d/%d tools", max_tools, len(tool_schemas))
    return tool_schemas[:max_tools]


def _simplify_tool_schema(schema: dict) -> dict:
    """Return a schema copy with description-like fields removed recursively."""
    if isinstance(schema, dict):
        out = {}
        for key, value in schema.items():
            if key in {"description", "examples"}:
                continue
            out[key] = _simplify_tool_schema(value)
        return out
    if isinstance(schema, list):
        return [_simplify_tool_schema(item) for item in schema]
    return schema


def _apply_profile_schema_simplify(tool_schemas: list[dict], client) -> list[dict]:
    """Apply profile simplify_schemas knob to tool schemas."""
    profile = _resolve_profile(client)
    if profile is None or not bool(getattr(profile, "simplify_schemas", False)):
        return tool_schemas
    log.info("Profile simplify_schemas enabled: stripping schema descriptions")
    return [_simplify_tool_schema(schema) for schema in tool_schemas]


def _apply_profile_preamble(system_prompt: str, client) -> str:
    """Apply profile preamble as a prefixed system-prompt block."""
    profile = _resolve_profile(client)
    if profile is None:
        return system_prompt
    preamble = str(getattr(profile, "preamble", "") or "").strip()
    if not preamble:
        return system_prompt
    return preamble + "\n\n" + system_prompt


@dataclass
class SessionResult:
    turns: int
    finish_reason: str  # stop, max_turns, context_full, duplicate_abort, error
    done: bool
    total_prompt_tokens: int = 0
    total_completion_tokens: int = 0


@dataclass(frozen=True)
class TaskSpec:
    """Task substrate input for solve_task (repo layout is only one source)."""

    prompt_text: str
    pretest_script: Path | None = None


class Session:
    """One context window — multi-turn tool calling until done or limit."""

    def __init__(
        self,
        cfg: Config,
        client,
        system_prompt: str,
        initial_message: str,
        cwd: str,
        context_manager: ContextManager | None = None,
        trace_file: IO | None = None,
        session_number: int = 0,
        trace_path: Path | None = None,
        state_path: Path | None = None,
        output_control=None,
        universal_rewrites=None,
        output_parser=None,
        pretest_parsed: dict | None = None,
        guardrail_registry: GuardrailRegistry | None = None,
        tool_registry: ToolRegistry | None = None,
    ):
        self.cfg = cfg
        self.client = client
        self.cwd = cwd
        self.output_control = output_control
        self.universal_rewrites = universal_rewrites
        self.output_parser = output_parser
        self.pretest_parsed = pretest_parsed
        # Monotonic bash counter for sink filenames (.tool_output/<sess>_<N>.log)
        self._sink_counter: int = 0
        self._tool_schemas = _apply_profile_tool_cap(
            _apply_profile_schema_simplify(get_tool_schemas(cfg.tool_desc), client),
            client,
        )
        self._tool_registry = tool_registry or build_tool_registry()
        schema_names = [s["function"]["name"] for s in self._tool_schemas]
        validate_tool_handlers(schema_names, registry=self._tool_registry)
        if context_manager is not None:
            self.context: ContextManager = context_manager
        else:
            context_kwargs = {
                "cwd": cwd,
                "original_prompt": initial_message,
                "trace_lines": cfg.solver_trace_lines,
                "evidence_lines": cfg.solver_evidence_lines,
                "inference_lines": cfg.solver_inference_lines,
                "recent_tool_results_chars": cfg.recent_tool_results_chars,
                "trace_stub_chars": cfg.trace_stub_chars,
                "min_turns": cfg.min_turns_before_context,
                "suffix": cfg.state_context_suffix,
            }
            token_estimator = _resolve_token_estimator(client)
            if token_estimator is not None:
                context_kwargs["token_estimator"] = token_estimator
            self.context = SolverStateContext(
                **context_kwargs,
            )
        self.context.add_system(system_prompt)
        self.context.add_user(initial_message)
        # All thrash-control state lives in one place. See harness/guardrails.py.
        # Session is the orchestrator; the guardrails own their own state
        # machines and expose a uniform Decision interface to the turn loop.
        self._guards: GuardrailState = init_guardrail_state(cfg)
        self._guardrail_registry = guardrail_registry or build_guardrail_registry()
        validate_guardrail_registry(self._guardrail_registry)
        # In-memory mirror of the trace file for this task. Seeded at
        # session __init__ from any prior-session events (trace is appended
        # across sessions). Appended to by _write_trace. Consumed by
        # _refresh_state, avoiding a per-tool-call re-read + JSON parse of
        # the full trace file — was O(T^2) across a session.
        self._trace_events: list[dict] = []
        if trace_path is not None and trace_path.is_file():
            try:
                with open(trace_path) as _f:
                    for _line in _f:
                        _line = _line.strip()
                        if _line:
                            self._trace_events.append(json.loads(_line))
            except (OSError, json.JSONDecodeError):
                # Best-effort; a partially-corrupt trace falls back to
                # file-based rebuild at session boundaries.
                self._trace_events = []
        # Seed pretest parity from session 1's parsed pretest verdict (passed
        # as a dict with 'failing' and 'passing' sets). Later sessions inherit
        # the baseline from session 1 via the same mechanism (caller passes
        # the same dict every time). No-op when pretest was not parseable.
        if pretest_parsed:
            self._guards.pretest_failing_tests = set(pretest_parsed.get("failing") or ())
            self._guards.pretest_passing_tests = set(pretest_parsed.get("passing") or ())
        self._last_fill: float = 0.0
        self._tool_log: list[tuple[str, str]] = []  # (name, args_summary)
        self._trace_file = trace_file
        self._session_number = session_number
        # Adaptive phase state (config-driven runtime switch).
        self._adaptive_phase = "base"
        self._adaptive_switched = False
        self._observed_test_signal = False
        window = max(0, int(getattr(cfg, "adaptive_low_pressure_window", 0) or 0))
        self._pressure_events = deque(maxlen=window if window > 0 else 1)
        # Mechanical state.json writer — harness side, not model side.
        # Gated on state_path being non-None. Stateless runs never seed
        # .solver/state.json and therefore get no state writes.
        self._trace_path = trace_path
        self._state_path = state_path
        # Injection subsystem (harness/injections.py). Off-by-default;
        # when enabled, loads markdown fragments from
        # <cwd>/<cfg.injections_dir> at session start. Fire state is
        # per-session so fire_once fragments inject at most once.
        self._injections = []
        self._injection_state = InjectionState()
        if cfg.injections_enabled:
            inj_dir = Path(self.cwd) / cfg.injections_dir
            self._injections = load_injections(inj_dir)

    @property
    def last_tool_calls(self) -> list[tuple[str, str]]:
        """Last N tool calls as (name, args_summary) pairs."""
        return self._tool_log[-self.cfg.duplicate_abort:]

    @property
    def context_fill_ratio(self) -> float:
        """Last known context fill ratio (0.0–1.0)."""
        return self._last_fill

    def _apply_injections(self) -> None:
        """Fire matching injections against the latest user/tool text.

        No-op when the subsystem is disabled or no fragments loaded.
        For each fragment that fires, appends a new user-role message
        containing its ``<injected-fragment source=NAME>`` block so
        the model sees it inline on the next API call, and records a
        per-fire event on the savings ledger (bucket=``injection``,
        mechanism=fragment name).
        """
        if not self._injections:
            return
        messages = self.context.get_messages()
        last_text = ""
        for m in reversed(messages):
            if m.get("role") in ("user", "tool"):
                c = m.get("content", "")
                last_text = c if isinstance(c, str) else str(c)
                break
        fired = fire_candidates(
            self._injections, text=last_text, state=self._injection_state,
        )
        for inj in fired:
            block = inj.format_block()
            self.context.add_user(block)
            record_fire(
                inj.name, body_chars=len(block), match_mode=inj.trigger,
            )

    def _chat_with_retry(self, turn: int):
        """Call client.chat(), retrying on transient errors. Returns None on fatal."""
        max_retries = self.cfg.max_transient_retries
        backoff = self.cfg.retry_backoff
        for attempt in range(max_retries + 1):
            try:
                return self.client.chat(
                    self.context.get_messages(), self._tool_schemas, turn=turn
                )
            except _TRANSIENT_ERRORS as e:
                if attempt < max_retries:
                    delay = backoff[attempt] if attempt < len(backoff) else backoff[-1]
                    log.warning(
                        "Transient error on turn %d, retry %d/%d: %s",
                        turn, attempt + 1, max_retries, e,
                    )
                    time.sleep(delay)
                else:
                    log.error(
                        "Transient error on turn %d, retries exhausted: %s", turn, e
                    )
                    return None
            except Exception as e:
                log.error("Fatal API error on turn %d: %s", turn, e)
                return None

    def _write_trace(self, entry: dict) -> None:
        """Write a single JSON line to the trace file, if open.

        After writing, triggers the mechanical state writer if a state.json
        target is configured. Uses the in-memory trace
        events list so the refresh does not re-read + re-parse the whole
        trace file each call.
        """
        if self._trace_file is not None:
            self._trace_file.write(json.dumps(entry) + "\n")
            self._trace_file.flush()
        self._trace_events.append(entry)
        self._refresh_state()

    def _refresh_state(self) -> None:
        """Rebuild .solver/state.json from the in-memory trace event list.

        No-op if state_path was not provided. The events list
        is kept in sync with the on-disk trace by _write_trace, so the
        projection is equivalent to re-reading the file — without the
        O(T) file read + JSON parse per call.
        """
        if self._state_path is None:
            return
        write_state_from_events(self._trace_events, self._state_path,
                                max_result_chars=self.cfg.max_output_chars)

    def _project_and_sink(self, tc_name: str, cmd: str, result: str, turn: int) -> str:
        """Apply structured output projection + sink-and-surface.

        Order:
          1. Structured output (when enabled + parser present + test cmd):
             Parse raw output, render digest, write raw to
             .tool_output/<session>_<sink_counter>.log in cwd, replace
             result with "digest\\n[raw output: <path>, <chars>, <lines>]".
             Preserves the full raw output on disk so the model can read
             it via the existing `read` tool; projects a compact digest
             into the context window.

          2. Sink-only (when result > sink_threshold_chars and step 1
             did not fire): write raw to same location, replace body
             with head/tail head/tail + pointer.

        Applies only to bash results; all other tools return unchanged.
        """
        if tc_name != "bash" or not result:
            return result

        projected = False
        pointer_line = ""
        raw_input_chars = len(result)
        if (self.cfg.bash_transforms_structured_output_enabled
                and self.output_parser is not None
                and self.output_control is not None):
            # Only project for test commands — other bash invocations
            # (build, config, etc.) don't have meaningful structured
            # output and raw is better.
            from ..bash_quirks.transforms import _is_test_command
            if _is_test_command(cmd, self.output_control):
                from ..bash_quirks import parse_structured, render_digest
                parsed = parse_structured(result, self.output_parser)
                digest = render_digest(parsed)
                # Even when the digest is empty (unparseable), the parsed
                # dict still carries whatever per-test records matched.
                # Update the parity-streak state from the parsed record.
                self._update_parity_from_parsed(parsed)
                if digest:
                    pointer_line = self._sink_to_disk(result, turn)
                    result = digest + ("\n" + pointer_line if pointer_line else "")
                    projected = True
                    # Token accounting: exact raw-vs-digest delta.
                    from .savings import get_ledger
                    get_ledger().record(
                        bucket="structured_projection",
                        layer="L2_bash_quirks",
                        mechanism=f"{self.cfg.analysis_task_format}_digest",
                        input_chars=raw_input_chars,
                        output_chars=len(result),
                        measure_type="exact",
                        ctx={"cmd": cmd[:120],
                             "n_tests_parsed": len(parsed.get("tests") or {}),
                             "summary": parsed.get("summary")},
                    )

        if (not projected
                and self.cfg.bash_transforms_sink_threshold_chars > 0
                and len(result) > self.cfg.bash_transforms_sink_threshold_chars):
            pointer_line = self._sink_to_disk(result, turn)
            if pointer_line:
                # Keep a short head+tail preview so the model still sees
                # SOMETHING without needing to open the file. Slice sizes
                # and the body-truncated marker text live in cfg (no
                # prompt literal in harness code).
                head = result[:self.cfg.sink_head_bytes]
                tail = result[-self.cfg.sink_tail_bytes:]
                result = (
                    f"{head}\n{self.cfg.sink_body_marker}\n{tail}\n"
                    f"{pointer_line}"
                )
                # Token accounting: exact raw-vs-preview delta.
                from .savings import get_ledger
                get_ledger().record(
                    bucket="sink_surface",
                    layer="L2_bash_quirks",
                    mechanism="head_tail_with_pointer",
                    input_chars=raw_input_chars,
                    output_chars=len(result),
                    measure_type="exact",
                    ctx={"cmd": cmd[:120],
                         "threshold": self.cfg.bash_transforms_sink_threshold_chars},
                )

        return result

    def _update_parity_from_parsed(self, parsed: dict) -> None:
        """Process a parsed test run: regression detection + parity update.

        Regression detection runs unconditionally (observability only —
        no gate, no termination). For every test in the prior run's
        verdicts that was PASSED and is now FAILED/ERROR, with at
        least one intervening mutation, a trace event of type
        ``regression`` is written for post-hoc analysis.

        Parity update runs only when ``cfg.done_require_pretest_parity``
        is set AND the pretest baseline was captured. Checks whether
        every pretest-failing test is now PASSED and no pretest-passing
        test has regressed; increments green_parity_streak on match.
        """
        tests = parsed.get("tests") or {}
        if not tests:
            return

        # ── Regression observability (always on when we have a parse) ──
        prev = self._guards.prev_test_parsed
        mutations_between = (
            self._guards.mutation_count - self._guards.mutation_count_at_prev_test
        )
        if prev and mutations_between > 0:
            regressed = [
                tid for tid, prev_v in prev.items()
                if prev_v == "PASSED"
                and tests.get(tid) in ("FAILED", "ERROR")
            ]
            if regressed:
                log.info("Regression detected: %d tests (mutations_between=%d)",
                         len(regressed), mutations_between)
                self._write_trace({
                    "event": "regression",
                    "session_number": self._session_number,
                    "tests_regressed": sorted(regressed)[:20],
                    "n_regressed": len(regressed),
                    "mutations_between": mutations_between,
                })
        # Update prior-state trackers for the next call.
        self._guards.prev_test_parsed = dict(tests)
        self._guards.mutation_count_at_prev_test = self._guards.mutation_count

        # ── Pretest-parity update (opt-in) ─────────────────────────────
        if not getattr(self.cfg, "done_require_pretest_parity", False):
            return
        if not self._guards.pretest_failing_tests:
            return
        self._guards.latest_test_parsed = dict(tests)
        passed_now = {t for t, v in tests.items() if v in ("PASSED", "PASS")}
        targets_hit = self._guards.pretest_failing_tests.issubset(passed_now)
        regressed_parity = any(
            tests.get(t) not in (None, "PASSED", "PASS")
            for t in self._guards.pretest_passing_tests
        )
        if targets_hit and not regressed_parity:
            self._guards.green_parity_streak += 1
        else:
            self._guards.green_parity_streak = 0

    def _sink_to_disk(self, raw: str, turn: int) -> str:
        """Write raw bash output to .tool_output/<session>_<counter>.log.

        Returns a one-line pointer to append to the model-visible result,
        or empty string on failure (sink is best-effort; never blocks
        the loop).
        """
        self._sink_counter += 1
        try:
            sink_dir = Path(self.cwd) / ".tool_output"
            sink_dir.mkdir(parents=True, exist_ok=True)
            sink_name = f"{self._session_number}_{self._sink_counter:04d}_t{turn}.log"
            sink_path = sink_dir / sink_name
            # write_bytes skips the codec-negotiation layer that write_text
            # runs on every call; raw is already-decoded subprocess output.
            sink_path.write_bytes(raw.encode("utf-8", errors="replace"))
        except OSError as e:
            log.debug("Sink write failed: %s", e)
            return ""
        rel = sink_path.relative_to(self.cwd)
        return self.cfg.sink_pointer.format(
            path=rel,
            chars=len(raw),
            lines=raw.count(_NEWLINE) + 1,
        )

    def run(self) -> SessionResult:
        """Drive one session's turn loop.

        Phase order (see docs/separation_of_concerns.md "Harness sub-concern:
        Guardrails"). Each guardrail returns a uniform ``Decision`` — the
        handling below is the ONLY place Session decides how to act on it.

            1. API call (with transient-error retry)
            2. context fill                                     END
            3. intent_gate         (turn-level, pre-dispatch)   BLOCK / END
            4. stop check          (natural exit)
            5. duplicate_guard     (turn-level, pre-dispatch)   WARN / END
            6. per tool call:
               6a. done_guard        (tc-level, pre-dispatch)    BLOCK (or accept→END)
               6b. rumination_gate   (tc-level, pre-dispatch)    WARN-grace / BLOCK / END
               6c. dispatch          (when not blocked)
               6d. error_ladder      (tc-level, post-dispatch)   WARN / END
               6e. rumination_ladder (tc-level, post-dispatch)   WARN + ARM
               6f. append turn-level WARN; trace; record
            7. max_turns                                         END
        """
        total_prompt = 0
        total_completion = 0
        turn_pre = self._guardrail_registry.turn_pre_dispatch
        tool_pre = self._guardrail_registry.tool_pre_dispatch
        tool_post = self._guardrail_registry.tool_post_dispatch
        observers = self._guardrail_registry.observers
        from .savings import get_ledger
        for turn in range(self.cfg.max_turns):
            # Stamp the savings ledger with (session, turn) so every record
            # written by transforms downstream carries the turn context.
            get_ledger().set_turn(self._session_number, turn)
            # Inject keyword-triggered fragments (harness/injections.py)
            # against the latest user/tool content before the API call.
            # No-op when the subsystem is disabled or no fragments load.
            self._apply_injections()
            # ─── 1. API call (with transient-error retry) ────────────────
            chat_result = self._chat_with_retry(turn)
            if chat_result is None:
                return SessionResult(turn, "error", done=False, total_prompt_tokens=total_prompt, total_completion_tokens=total_completion)
            content = chat_result.content
            tool_calls = chat_result.tool_calls
            reason = chat_result.finish_reason
            prompt_tokens = chat_result.usage.prompt_tokens
            completion_tokens = chat_result.usage.completion_tokens
            total_prompt += prompt_tokens
            total_completion += completion_tokens
            self.context.add_assistant(
                self.client.build_assistant_message(content, tool_calls)
            )

            # ─── 2. GUARDRAIL: context fill (END tier) ───────────────────
            if self.cfg.context_size > 0:
                fill = self.context.estimate_tokens() / self.cfg.context_size
                self._last_fill = fill
                if fill > self.cfg.context_fill_ratio:
                    log.info("Context %.0f%% full at turn %d, ending session", fill * 100, turn)
                    return SessionResult(turn, "context_full", done=False, total_prompt_tokens=total_prompt, total_completion_tokens=total_completion)

            # ─── 3. GUARDRAIL: intent_gate (BLOCK / END tiers) ───────────
            intent_decision = turn_pre["intent_gate"](
                self._guards, self.cfg,
                turn=turn, content=content, tool_calls=tool_calls,
            )
            if intent_decision.action in (Action.BLOCK, Action.END):
                self._record_pressure_event(True)
                log.info("Intent gate: rejecting silent tool call at turn %d "
                         "(block #%d, consecutive %d)", turn,
                         self._guards.intent_block_count,
                         self._guards.consecutive_intent_rejections)
                for tc in tool_calls:
                    args_summary = _summarize_args(tc.arguments, self.cfg.args_summary_chars)
                    self.context.add_tool_result(tc.id, intent_decision.text,
                                                 tool_name=tc.name, cmd_signature="",
                                                 gate_blocked=True)
                    self._write_trace({
                        "event": "tool_call",
                        "session_number": self._session_number,
                        "turn_number": turn,
                        "tool_name": tc.name,
                        "args_summary": args_summary,
                        "result_summary": intent_decision.text,
                        "reasoning": "",
                        "gate_blocked": True,
                        "prompt_tokens": prompt_tokens,
                        "completion_tokens": completion_tokens,
                    })
                if intent_decision.action == Action.END:
                    log.warning("Intent abort: %d consecutive silent rejections",
                                self._guards.consecutive_intent_rejections)
                    return SessionResult(turn, intent_decision.reason, done=False,
                                         total_prompt_tokens=total_prompt,
                                         total_completion_tokens=total_completion)
                continue

            # ─── 4. Stop check (natural exit) ────────────────────────────
            if not tool_calls:
                if reason == "length":
                    log.info("Response truncated at turn %d (max_tokens hit), ending session", turn)
                    return SessionResult(turn, "length", done=False, total_prompt_tokens=total_prompt, total_completion_tokens=total_completion)
                log.info("Model stopped at turn %d (reason=%s)", turn, reason)
                return SessionResult(turn, "stop", done=True, total_prompt_tokens=total_prompt, total_completion_tokens=total_completion)

            # ─── 5. GUARDRAIL: duplicate_guard (WARN / END tiers) ────────
            sig = tuple(_dedup_signature(tc) for tc in tool_calls)
            dup_decision = turn_pre["duplicate_guard"](
                self._guards, self.cfg, tool_calls_sig=sig
            )
            if dup_decision.action == Action.END:
                self._record_pressure_event(True)
                log.warning("Duplicate tool calls detected, aborting at turn %d", turn)
                return SessionResult(turn, dup_decision.reason, done=False,
                                     total_prompt_tokens=total_prompt,
                                     total_completion_tokens=total_completion)
            turn_warn_text = dup_decision.text if dup_decision.action == Action.WARN else ""
            turn_had_pressure = bool(turn_warn_text)

            # ─── 5b. GUARDRAIL: loop_detect (WARN / END tiers) ───────────
            # Tighter than duplicate_guard: fires at N consecutive identical
            # signatures (default 5) with a single recovery-inject before
            # hard abort. See guardrails.loop_detect for the contract.
            loop_decision = turn_pre["loop_detect"](
                self._guards, self.cfg, tool_calls_sig=sig
            )
            if loop_decision.action == Action.END:
                self._record_pressure_event(True)
                log.warning("Loop detected, aborting at turn %d", turn)
                return SessionResult(turn, loop_decision.reason, done=False,
                                     total_prompt_tokens=total_prompt,
                                     total_completion_tokens=total_completion)
            if loop_decision.action == Action.WARN:
                # Compose with any duplicate-guard warn already queued.
                turn_warn_text = (
                    f"{turn_warn_text}\n\n{loop_decision.text}"
                    if turn_warn_text else loop_decision.text
                )
                turn_had_pressure = True

            # ─── 6. Dispatch loop (per tool call) ────────────────────────
            # Optional parallel pre-execute for all-read-only turns.
            # Guardrails and post-dispatch state still run sequentially
            # per-tc below; this only concurrent-izes the file-I/O
            # dispatch() work for turns that emit multiple independent
            # read/glob/grep calls. Mutating tools (write/edit/bash)
            # always run sequentially — they never enter this path.
            preexecuted: dict[str, str] = {}
            if (
                self.cfg.parallel_readonly_enabled
                and len(tool_calls) > 1
                and all(tc.name in _READONLY_TOOLS for tc in tool_calls)
            ):
                effective_output_control = (
                    self.output_control if self.cfg.bash_transforms_task_format_enabled else None
                )
                effective_universal_rewrites = (
                    self.universal_rewrites if self.cfg.bash_transforms_universal_enabled else None
                )
                with ThreadPoolExecutor(
                    max_workers=max(1, self.cfg.parallel_max_workers)
                ) as _ex:
                    futures = {
                        tc.id: _ex.submit(
                            dispatch, tc.name, tc.arguments,
                            cwd=self.cwd, cfg=self.cfg,
                            output_control=effective_output_control,
                            universal_rewrites=effective_universal_rewrites,
                            tool_registry=self._tool_registry,
                        )
                        for tc in tool_calls
                    }
                    for tc_id, fut in futures.items():
                        try:
                            preexecuted[tc_id] = fut.result()
                        except Exception as e:
                            preexecuted[tc_id] = f"ERROR: {e}"

            for tc in tool_calls:
                args_summary = _summarize_args(tc.arguments, self.cfg.args_summary_chars)
                focus_key, focus_display = _focus_signature(tc, args_summary, self.cwd)
                log.info("turn=%d pt=%d %s(%s)", turn, prompt_tokens, tc.name, args_summary)
                self._tool_log.append((tc.name, args_summary))

                approval_allowed, approval_reason = self._approval_decision(
                    tc.name, tc.arguments, args_summary
                )
                if not approval_allowed:
                    turn_had_pressure = True
                    approval_text = (
                        "APPROVAL REQUIRED: This tool call was not executed. "
                        f"Reason: {approval_reason}. "
                        "Review it with `yuj show` and approve it with "
                        "`yuj approve <session_id>`, then resume the session."
                    )
                    self.context.add_tool_result(
                        tc.id,
                        approval_text,
                        tool_name=tc.name,
                        gate_blocked=True,
                    )
                    self._write_trace({
                        "event": "approval_request",
                        "session_number": self._session_number,
                        "turn_number": turn,
                        "tool_name": tc.name,
                        "args_summary": _truncate_for_trace(args_summary, self.cfg.trace_args_summary_chars),
                        "reason": approval_reason,
                        "reasoning": _truncate_for_trace(content or "", self.cfg.trace_reasoning_store_chars),
                        "prompt_tokens": prompt_tokens,
                        "completion_tokens": completion_tokens,
                    })
                    return SessionResult(
                        turn,
                        "approval_required",
                        done=False,
                        total_prompt_tokens=total_prompt,
                        total_completion_tokens=total_completion,
                    )

                # 6a. done_guard — accept path ends session; otherwise BLOCK.
                if tc.name == "done":
                    done_decision = tool_pre["done_guard"](
                        self._guards, self.cfg, tc_name=tc.name
                    )
                    if done_decision.action == Action.PASS:
                        log.info("Model called done() at turn %d", turn)
                        self.context.add_tool_result(tc.id, "Session ended by model.", tool_name="done")
                        self._write_trace({
                            "event": "tool_call",
                            "session_number": self._session_number,
                            "turn_number": turn,
                            "tool_name": "done",
                            "args_summary": tc.arguments.get("message", ""),
                            "result_summary": "Session ended by model.",
                            "reasoning": _truncate_for_trace(content or "", self.cfg.trace_reasoning_store_chars),
                            "gate_blocked": False,
                            "prompt_tokens": prompt_tokens,
                            "completion_tokens": completion_tokens,
                        })
                        return SessionResult(turn, "model_done", done=True, total_prompt_tokens=total_prompt, total_completion_tokens=total_completion)
                    # BLOCK: store rejection and continue to next tc
                    self.context.add_tool_result(tc.id, done_decision.text, tool_name="done")
                    self._write_trace({
                        "event": "tool_call",
                        "session_number": self._session_number,
                        "turn_number": turn,
                        "tool_name": "done",
                        "args_summary": tc.arguments.get("message", ""),
                        "result_summary": done_decision.text,
                        "reasoning": _truncate_for_trace(content or "", self.cfg.trace_reasoning_store_chars),
                        "gate_blocked": False,
                        "prompt_tokens": prompt_tokens,
                        "completion_tokens": completion_tokens,
                    })
                    continue

                # 6b. mutation_repeat_guard — stop repeated identical edits.
                mutation_decision = tool_pre["mutation_repeat_guard"](
                    self._guards, self.cfg,
                    tc_name=tc.name,
                    tc_args=tc.arguments,
                    focus_display=focus_display,
                )
                mutation_warn_text = ""
                gate_blocked_flag = False
                gate_intercepted = False
                if mutation_decision.action == Action.END:
                    turn_had_pressure = True
                    self.context.add_tool_result(tc.id, mutation_decision.text,
                                                 tool_name=tc.name, gate_blocked=True)
                    self._write_trace({
                        "event": "tool_call",
                        "session_number": self._session_number,
                        "turn_number": turn,
                        "tool_name": tc.name,
                        "args_summary": _truncate_for_trace(args_summary, self.cfg.trace_args_summary_chars),
                        "result_summary": mutation_decision.text,
                        "reasoning": _truncate_for_trace(content or "", self.cfg.trace_reasoning_store_chars),
                        "gate_blocked": True,
                        "prompt_tokens": prompt_tokens,
                        "completion_tokens": completion_tokens,
                    })
                    return SessionResult(turn, mutation_decision.reason, done=False,
                                         total_prompt_tokens=total_prompt,
                                         total_completion_tokens=total_completion)
                if mutation_decision.action == Action.BLOCK:
                    turn_had_pressure = True
                    log.info("Mutation repeat guard blocked %s", tc.name)
                    result = mutation_decision.text
                    gate_blocked_flag = True
                    gate_intercepted = True
                elif mutation_decision.action == Action.WARN:
                    mutation_warn_text = mutation_decision.text

                # 6c. contract_gate — warn/block broad exploration once a
                # tighter commit/recovery contract is active.
                contract_warn_text = ""
                if not gate_blocked_flag:
                    contract_decision = tool_pre["contract_gate"](
                        self._guards, self.cfg,
                        tc_name=tc.name,
                        tc_args=tc.arguments,
                        focus_key=focus_key,
                        focus_display=focus_display,
                    )
                    if contract_decision.action == Action.END:
                        turn_had_pressure = True
                        self.context.add_tool_result(tc.id, contract_decision.text,
                                                     tool_name=tc.name, gate_blocked=True)
                        self._write_trace({
                            "event": "tool_call",
                            "session_number": self._session_number,
                            "turn_number": turn,
                            "tool_name": tc.name,
                            "args_summary": _truncate_for_trace(args_summary, self.cfg.trace_args_summary_chars),
                            "result_summary": contract_decision.text,
                            "reasoning": _truncate_for_trace(content or "", self.cfg.trace_reasoning_store_chars),
                            "gate_blocked": True,
                            "prompt_tokens": prompt_tokens,
                            "completion_tokens": completion_tokens,
                        })
                        return SessionResult(turn, contract_decision.reason, done=False,
                                             total_prompt_tokens=total_prompt,
                                             total_completion_tokens=total_completion)
                    if contract_decision.action == Action.BLOCK:
                        turn_had_pressure = True
                        log.info("Contract gate blocked %s", tc.name)
                        result = contract_decision.text
                        gate_blocked_flag = True
                        gate_intercepted = True
                    elif contract_decision.action == Action.WARN:
                        contract_warn_text = contract_decision.text

                if not gate_blocked_flag:
                    if mutation_warn_text and contract_warn_text:
                        contract_warn_text = mutation_warn_text + "\n" + contract_warn_text
                    elif mutation_warn_text:
                        contract_warn_text = mutation_warn_text

                    # 6d. rumination_gate — grace (WARN+dispatch) / BLOCK / END.
                    gate_decision = tool_pre["rumination_gate"](
                        self._guards, self.cfg, tc_name=tc.name
                    )
                    if gate_decision.action == Action.END:
                        turn_had_pressure = True
                        log.info("Gate escalation: %d blocks, ending session",
                                 self._guards.gate_block_count)
                        self.context.add_tool_result(tc.id, gate_decision.text,
                                                     tool_name=tc.name, gate_blocked=True)
                        self._write_trace({
                            "event": "tool_call",
                            "session_number": self._session_number,
                            "turn_number": turn,
                            "tool_name": tc.name,
                            "args_summary": _truncate_for_trace(args_summary, self.cfg.trace_args_summary_chars),
                            "result_summary": gate_decision.text,
                            "reasoning": _truncate_for_trace(content or "", self.cfg.trace_reasoning_store_chars),
                            "gate_blocked": True,
                            "prompt_tokens": prompt_tokens,
                            "completion_tokens": completion_tokens,
                        })
                        return SessionResult(turn, gate_decision.reason, done=False,
                                             total_prompt_tokens=total_prompt,
                                             total_completion_tokens=total_completion)
                    if gate_decision.action == Action.BLOCK:
                        turn_had_pressure = True
                        log.info("Rumination gate blocked %s", tc.name)
                        result = gate_decision.text
                        gate_blocked_flag = True
                        gate_intercepted = True
                    elif gate_decision.action == Action.WARN:
                        # GRACE: dispatch + append gate warning prefix
                        log.info("Rumination gate grace used for %s (%d remaining)",
                                 tc.name, self._guards.rumination_gate_grace)
                        effective_output_control = (
                            self.output_control if self.cfg.bash_transforms_task_format_enabled else None
                        )
                        effective_universal_rewrites = (
                            self.universal_rewrites if self.cfg.bash_transforms_universal_enabled else None
                        )
                        result = preexecuted.get(tc.id)
                        if result is None:
                            result = dispatch(tc.name, tc.arguments, cwd=self.cwd, cfg=self.cfg,
                                              output_control=effective_output_control,
                                              universal_rewrites=effective_universal_rewrites,
                                              tool_registry=self._tool_registry)
                        result += "\n\n" + gate_decision.text
                        gate_intercepted = True
                    else:
                        # 6d. Dispatch.
                        effective_output_control = (
                            self.output_control if self.cfg.bash_transforms_task_format_enabled else None
                        )
                        effective_universal_rewrites = (
                            self.universal_rewrites if self.cfg.bash_transforms_universal_enabled else None
                        )
                        result = preexecuted.get(tc.id)
                        if result is None:
                            result = dispatch(tc.name, tc.arguments, cwd=self.cwd, cfg=self.cfg,
                                              output_control=effective_output_control,
                                              universal_rewrites=effective_universal_rewrites,
                                              tool_registry=self._tool_registry)
                        if tc.name == "bash":
                            self._observe_test_signal(tc.arguments.get("cmd", ""), result)
                        # 6e. error_ladder (WARN / END tiers). Log every error
                        # for trace visibility; the ladder decides escalation.
                        err_decision = tool_post["error_ladder"](
                            self._guards, self.cfg, tc_name=tc.name, result=result
                        )
                        if result.startswith("ERROR:"):
                            turn_had_pressure = True
                            log.info("Tool error: %s consecutive=%d",
                                     tc.name, self._guards.consecutive_errors.get(tc.name, 0))
                        if err_decision.action == Action.END:
                            turn_had_pressure = True
                            log.warning("Error abort: %s consecutive=%d", tc.name,
                                        self._guards.consecutive_errors.get(tc.name, 0))
                            self.context.add_tool_result(tc.id, result, tool_name=tc.name,
                                                         cmd_signature="", gate_blocked=False)
                            self._write_trace({
                                "event": "tool_call",
                                "session_number": self._session_number,
                                "turn_number": turn,
                                "tool_name": tc.name,
                                "args_summary": _truncate_for_trace(args_summary, self.cfg.trace_args_summary_chars),
                                "result_summary": result,
                                "reasoning": _truncate_for_trace(content or "", self.cfg.trace_reasoning_store_chars),
                                "gate_blocked": False,
                                "prompt_tokens": prompt_tokens,
                                "completion_tokens": completion_tokens,
                            })
                            return SessionResult(turn, err_decision.reason, done=False,
                                                 total_prompt_tokens=total_prompt,
                                                 total_completion_tokens=total_completion)
                        if err_decision.action == Action.WARN:
                            result += "\n\n" + err_decision.text

                        # Post-dispatch: structured output projection + sink.
                        # Only for bash, only on non-error results, only when
                        # the relevant cfg flag is on. Sink writes raw output
                        # to .tool_output/<session>_<N>_t<turn>.log; the
                        # model can read the file when it wants the full
                        # content. Structured projection replaces raw with a
                        # compact digest for test commands.
                        if tc.name == "bash" and not result.startswith("ERROR:"):
                            cmd = tc.arguments.get("cmd", "")
                            result = self._project_and_sink(tc.name, cmd, result, turn)

                    if contract_warn_text and not gate_blocked_flag:
                        result += "\n\n" + contract_warn_text

                # 6f. rumination_ladder (WARN + ARM). Runs for every tc —
                # it owns the counter increment, nudge emission, and gate
                # arming. When the gate already intercepted this call, the
                # ladder skips the counter bump to avoid double-counting.
                test_read_decision = tool_post["test_read_ladder"](
                    self._guards, self.cfg,
                    tc_name=tc.name, result=result,
                    gate_blocked=gate_blocked_flag,
                    tc_args=tc.arguments,
                )
                if test_read_decision.action == Action.WARN:
                    result += "\n\n" + test_read_decision.text

                rum_decision = tool_post["rumination_ladder"](
                    self._guards, self.cfg,
                    tc_name=tc.name, result=result,
                    gate_blocked=gate_blocked_flag,
                    already_blocked_this_turn=gate_intercepted,
                    focus_key=focus_key,
                    focus_display=focus_display,
                )
                if rum_decision.action == Action.WARN:
                    result += "\n\n" + rum_decision.text

                # Context-side dedup reset on a successful write/edit. The
                # guardrail state is reset inside rumination_ladder; this is
                # the context's own signal (separate concern — stateful
                # compaction, not thrash control).
                if (tc.name in ("write", "edit")
                        and not result.startswith("ERROR:")
                        and hasattr(self.context, "reset_dedup_counts")):
                    self.context.reset_dedup_counts()

                # Content-blind "verified since mutation" signal for the
                # done guard.
                observers["mark_bash_verified"](
                    self._guards, self.cfg,
                    tc_name=tc.name, result=result,
                    gate_blocked=gate_blocked_flag,
                )
                observers["observe_test_file_read"](
                    self._guards, self.cfg,
                    tc_name=tc.name, result=result,
                    gate_blocked=gate_blocked_flag,
                    tc_args=tc.arguments,
                    focus_key=focus_key,
                    focus_display=focus_display,
                )
                observers["observe_contract_state"](
                    self._guards, self.cfg,
                    tc_name=tc.name, result=result,
                    gate_blocked=gate_blocked_flag,
                    tc_args=tc.arguments,
                    focus_key=focus_key,
                    focus_display=focus_display,
                )

                # Append turn-level WARN (from duplicate ladder) after all
                # tc-level appends so it reads last.
                if turn_warn_text:
                    result += "\n\n" + turn_warn_text

                # 6f. Trace + record.
                self._write_trace({
                    "event": "tool_call",
                    "session_number": self._session_number,
                    "turn_number": turn,
                    "tool_name": tc.name,
                    "args_summary": _truncate_for_trace(args_summary, self.cfg.trace_args_summary_chars),
                    "result_summary": _truncate_for_trace(result, self.cfg.max_output_chars),
                    "reasoning": _truncate_for_trace(content or "", self.cfg.trace_reasoning_store_chars),
                    "gate_blocked": gate_blocked_flag,
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                })

                # Gate-blocked calls were never executed — don't store a
                # cmd_signature (prevents later calls of the same command
                # from matching against a non-execution).
                cmd_sig = ""
                if not gate_blocked_flag:
                    cmd_sig = _dedup_signature(tc)[1] if tc.name == "bash" else ""
                self.context.add_tool_result(tc.id, result, tool_name=tc.name,
                                             cmd_signature=cmd_sig,
                                             gate_blocked=gate_blocked_flag)
            self._record_pressure_event(turn_had_pressure)
            self._maybe_switch_adaptive_phase(turn)

        # ─── 7. GUARDRAIL: max_turns (hard cap, END tier) ────────────────
        return SessionResult(self.cfg.max_turns, "max_turns", done=False, total_prompt_tokens=total_prompt, total_completion_tokens=total_completion)

    def _record_pressure_event(self, had_pressure: bool) -> None:
        """Track whether this turn had loop pressure events (errors/blocks/warns)."""
        self._pressure_events.append(bool(had_pressure))

    def _observe_test_signal(self, cmd: str, result: str) -> None:
        """Mark that we observed a non-trivial test command signal."""
        if result.startswith("ERROR:"):
            return
        cmd_s = (cmd or "").strip()
        if not cmd_s:
            return
        is_test = False
        if self.output_control is not None:
            try:
                from ..bash_quirks.transforms import _is_test_command
                is_test = _is_test_command(cmd_s, self.output_control)
            except Exception:
                is_test = False
        if not is_test:
            is_test = bool(re.search(r"\b(pytest|unittest|cargo test|go test|ctest|npm test|pnpm test|yarn test)\b", cmd_s))
        if not is_test:
            return
        if "[exit code:" in result:
            self._observed_test_signal = True
            return
        if result.strip():
            self._observed_test_signal = True

    def _approval_decision(self, tc_name: str, tc_args: dict, args_summary: str) -> tuple[bool, str | None]:
        """Return (allowed, reason) for assistant-mode approval-gated actions."""
        if getattr(self.cfg, "runtime_mode", "measurement") != "assistant":
            return True, None
        if tc_name != "bash":
            return True, None
        cmd = str(tc_args.get("cmd") or "")
        reason = _approval_reason_for_bash(cmd, self.cwd)
        if reason is None:
            return True, None
        decision = _load_approval_decisions(self._trace_path).get(f"{tc_name}:{cmd}")
        if decision == "approved":
            return True, None
        if decision == "rejected":
            return False, f"{reason}; previously rejected by operator"
        approval = _load_approval_request(self._trace_path)
        if (
            approval
            and approval.get("status") == "approved"
            and approval.get("tool_name") == tc_name
            and approval.get("cmd") == cmd
        ):
            _clear_approval_request(self._trace_path)
            return True, None
        if (
            approval
            and approval.get("status") == "rejected"
            and approval.get("tool_name") == tc_name
            and approval.get("cmd") == cmd
        ):
            return False, approval.get("rejection_reason") or f"{reason}; rejected by operator"
        payload = {
            "status": "pending",
            "tool_name": tc_name,
            "cmd": cmd,
            "args_summary": args_summary,
            "reason": reason,
            "requested_at": time.time(),
        }
        _write_approval_request(self._trace_path, payload)
        return False, reason

    def _maybe_switch_adaptive_phase(self, turn: int) -> None:
        """Switch from base to phase2 policy when configured conditions are met."""
        if self._adaptive_switched or not getattr(self.cfg, "adaptive_policy_enabled", False):
            return
        if turn < int(getattr(self.cfg, "adaptive_switch_min_turn", 0) or 0):
            return
        if getattr(self.cfg, "adaptive_requires_mutation", True) and not self._guards.has_mutated:
            return
        if getattr(self.cfg, "adaptive_requires_test_signal", True) and not self._observed_test_signal:
            return

        window = int(getattr(self.cfg, "adaptive_low_pressure_window", 0) or 0)
        max_events = int(getattr(self.cfg, "adaptive_low_pressure_max_events", 0) or 0)
        if window > 0:
            if len(self._pressure_events) < window:
                return
            recent = list(self._pressure_events)[-window:]
            if sum(1 for x in recent if x) > max_events:
                return

        self.cfg = replace(
            self.cfg,
            done_guard_enabled=bool(getattr(self.cfg, "adaptive_phase2_done_guard_enabled", True)),
            bash_transforms_task_format_enabled=bool(
                getattr(self.cfg, "adaptive_phase2_bash_task_format_enabled", True)
            ),
            bash_transforms_structured_output_enabled=bool(
                getattr(self.cfg, "adaptive_phase2_bash_structured_output_enabled", True)
            ),
            bash_transforms_sink_threshold_chars=int(
                getattr(self.cfg, "adaptive_phase2_bash_sink_threshold_chars", 0) or 0
            ),
        )
        self._adaptive_phase = "phase2"
        self._adaptive_switched = True
        self._write_trace({
            "event": "adaptive_phase_switch",
            "session_number": self._session_number,
            "turn_number": turn,
            "phase": self._adaptive_phase,
            "done_guard_enabled": self.cfg.done_guard_enabled,
            "bash_task_format_enabled": self.cfg.bash_transforms_task_format_enabled,
            "bash_structured_output_enabled": self.cfg.bash_transforms_structured_output_enabled,
            "bash_sink_threshold_chars": self.cfg.bash_transforms_sink_threshold_chars,
        })
        log.info(
            "Adaptive policy switched to phase2 at turn %d (done_guard=%s, bash_task_format=%s, structured=%s)",
            turn,
            self.cfg.done_guard_enabled,
            self.cfg.bash_transforms_task_format_enabled,
            self.cfg.bash_transforms_structured_output_enabled,
        )


def _summarize_args(args: dict, max_chars: int) -> str:
    """Short summary of tool arguments for logging.

    max_chars is required (wired from cfg.args_summary_chars). No default:
    shadow defaults in harness code drift silently from config.
    """
    parts = []
    for k, v in args.items():
        s = str(v)
        if len(s) > max_chars:
            s = s[:max_chars - 3] + "..."
        parts.append(f"{k}={s!r}")
    return ", ".join(parts)


def _truncate_for_trace(s: str, maxlen: int) -> str:
    """Truncate a string for trace logging.

    Now takes an explicit max length — callers pass 200 for the action/args
    summary (where a short repr is always enough) and the tools.py output
    cap (20000 by default, from Config.max_output_chars) for the result.
    Previously defaulted to 200 for both, which silently stubbed file
    reads and broke the stateful context strategy.
    """
    if len(s) <= maxlen:
        return s
    return s[:maxlen - 3] + "..."


def _auto_commit(repo_dir: Path, session_num: int, finish_reason: str) -> None:
    """Commit changes in repo_dir if working tree is dirty. Local only."""
    status = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=str(repo_dir), capture_output=True, text=True,
    )
    if not status.stdout.strip():
        return
    subprocess.run(["git", "add", "-A"], cwd=str(repo_dir), check=True)
    msg = f"harness: session {session_num} checkpoint ({finish_reason})"
    subprocess.run(
        ["git", "commit", "-m", msg],
        cwd=str(repo_dir), capture_output=True, check=True,
    )
    log.info("Auto-commit: session %d (%s)", session_num, finish_reason)


# Resume base prompt now lives in config.toml [prompts] resume_base.


# ── Pretest injection (issue #64) ────────────────────────────────────────

# Pretest truncation limits are now in config.toml [output] section:
#   pretest_head_chars, pretest_tail_chars
# The _truncate_pretest_output function below receives them as arguments.
#
# The harness does NOT rewrite task-environment-specific paths (e.g.
# container mount points) that appear in pretest output. Environment setup is
# responsible for sanitizing its own output before returning. The harness sees
# already-clean text.

_STATUS_WORD_RE = re.compile(
    r'\b(?:passed|failed|error|warnings?|deselected|no tests ran|no tests collected)\b'
)
_TIMING_RE = re.compile(r'\s*in\s+\d+\.\d+s')


def _sanitize_runner_timing(output: str) -> str:
    """Strip wall-clock timing from pytest/unittest summary lines.

    Pytest embeds sub-second timing (``13 failed in 1.55s``) that varies
    per invocation.  Under deterministic inference (temp=0, top-k=1) a
    single changed character flips the sampled path.  Stripping timing
    makes the pretest block byte-identical across runs of the same task.

    Operates per-line so ANSI color codes between the status word and
    the timing fragment don't defeat the match.
    """
    out_lines = []
    for line in output.split('\n'):
        if _STATUS_WORD_RE.search(line):
            line = _TIMING_RE.sub('', line)
        out_lines.append(line)
    return '\n'.join(out_lines)


def _normalize_repo_timestamps(repo_dir: Path) -> None:
    """Set every file/dir mtime under repo_dir (except .git/) to a fixed epoch.

    Removes wall-clock leakage that appears in the agent's first
    ``ls -la`` and would otherwise flip the model's path under
    deterministic inference (temp=0, top-k=1).
    """
    epoch = "2020-01-01T00:00:00"
    try:
        # Exclude .git contents (corrupts index timestamps) but touch
        # the .git directory itself so its entry in `ls -la` is stable.
        subprocess.run(
            ["find", str(repo_dir), "-not", "-path", f"{repo_dir}/.git/*",
             "-exec", "touch", "-d", epoch, "{}", "+"],
            check=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=30,
        )
    except (subprocess.SubprocessError, FileNotFoundError, PermissionError):
        # Non-fatal: determinism degrades but the run continues.
        pass


def _truncate_pretest_output(output: str, head_chars: int, tail_chars: int) -> str:
    """Head + tail slice with a middle-drop marker. No-op if below limit."""
    limit = head_chars + tail_chars
    if len(output) <= limit:
        return output
    dropped = len(output) - limit
    head = output[:head_chars]
    tail = output[-tail_chars:]
    return f"{head}\n\n... [truncated {dropped} chars] ...\n\n{tail}"


def run_pretest(repo_dir: Path, *, pretest_script: Path | None = None, pretest_timeout: int,
                pretest_head_chars: int, pretest_tail_chars: int) -> str:
    """Run the per-task pretest script and format the verdict for prepending.

    The harness knows nothing about the task environment that produced the
    script. The environment layer is responsible for materializing an
    executable shell command that produces the current failing-check verdict
    and for sanitizing any environment-specific paths before returning. The
    harness simply runs the script and injects its stdout verbatim — no
    rewriting, no content inspection.

    Location convention for measurement runs:
    ``<run_root>/pretest/<task_id>.sh`` where ``run_root`` is
    ``repo_dir.parent.parent`` and ``<task_id>`` is the repo dir's name.
    The script lives outside the agent's cwd so a plain ``find .`` in the
    repo cannot discover it.

    No pretest script → return ``""`` and the harness degrades to classic
    cold-start orientation. Script crash, timeout, or non-shell error are
    captured as verdict text rather than raising — per issue #64 the pretest
    must never break the agentic loop.
    """
    script = pretest_script
    if script is None:
        run_dir = repo_dir.parent.parent
        script = run_dir / "pretest" / f"{repo_dir.name}.sh"
    if not script.exists():
        return ""

    try:
        result = subprocess.run(
            ["bash", str(script.resolve())],
            cwd=str(repo_dir),
            capture_output=True,
            text=True,
            timeout=pretest_timeout,
        )
    except subprocess.TimeoutExpired:
        return (
            "## Current test state\n"
            f"(pretest timed out after {pretest_timeout}s — exceeded budget)\n"
        )
    except (subprocess.SubprocessError, FileNotFoundError, PermissionError) as e:
        return f"## Current test state\n(pretest crashed: {e})\n"

    merged = (result.stdout or "") + (result.stderr or "")
    merged = _sanitize_runner_timing(merged)
    truncated = _truncate_pretest_output(merged, pretest_head_chars, pretest_tail_chars)
    return (
        "## Current test state\n"
        "```\n"
        f"{truncated}\n"
        "```\n\n"
        f"exit code: {result.returncode}\n"
    )


def _pretest_is_green(block: str) -> bool:
    """True if the pretest block indicates a clean green run (exit code 0)."""
    return bool(block) and "\nexit code: 0\n" in block


def build_resume_prompt(
    prev_result: SessionResult,
    prev_session: Session,
    cfg: Config,
    task_description: str = "",
) -> str:
    """Build a context-rich resume prompt from the previous session's outcome."""
    parts = []

    if task_description:
        parts.append(f"Task:\n{task_description}")

    # Session summary
    parts.append(
        f"Previous session ended after {prev_result.turns} turns: "
        f"{prev_result.finish_reason}. "
        f"Consumed {prev_result.total_prompt_tokens} prompt tokens."
    )

    reason = prev_result.finish_reason

    if reason == "duplicate_abort":
        calls = prev_session.last_tool_calls
        if calls:
            name, args = calls[-1]
            parts.append(cfg.resume_duplicate_abort.format(
                n=len(calls), call=f"{name}({args})"))

    elif reason == "context_full":
        pct = int(prev_session.context_fill_ratio * 100)
        parts.append(cfg.resume_context_full.format(pct=pct))

    elif reason == "max_turns":
        calls = prev_session.last_tool_calls[-cfg.resume_last_n_actions:]
        if calls:
            summaries = "; ".join(f"{n}({a})" for n, a in calls)
            parts.append(cfg.resume_max_turns.format(
                n=len(calls), actions=summaries))

    elif reason == "gate_escalation":
        parts.append(cfg.resume_gate_escalation.format(n=5))

    elif reason == "length":
        parts.append(cfg.resume_length)

    parts.append(cfg.resume_base)
    return "\n\n".join(parts)


def _load_trace_events(trace_path: Path) -> list[dict]:
    """Load trace events from an append-only JSONL trace file."""
    trace_path = Path(trace_path)
    if not trace_path.is_file():
        return []
    events: list[dict] = []
    with open(trace_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            events.append(json.loads(line))
    return events


def _next_session_number(trace_path: Path) -> int:
    """Return the next session number for a trace-backed task."""
    session_numbers = [
        int(ev.get("session_number", 0) or 0)
        for ev in _load_trace_events(trace_path)
        if ev.get("event") in {"session_start", "session_end", "tool_call"}
    ]
    return (max(session_numbers) + 1) if session_numbers else 1


def build_resume_prompt_from_trace(
    trace_path: Path,
    cfg: Config,
    task_description: str = "",
) -> str | None:
    """Reconstruct a resume prompt from persisted trace artifacts.

    This is the disk-backed companion to ``build_resume_prompt``. It enables
    assistant-mode session resume across separate process invocations without
    forking a second loop implementation.
    """
    events = _load_trace_events(trace_path)
    if not events:
        return None
    last_end = next(
        (ev for ev in reversed(events) if ev.get("event") == "session_end"),
        None,
    )
    if last_end is None:
        return None

    session_number = int(last_end.get("session_number", 0) or 0)
    calls = [
        (
            str(ev.get("tool_name") or "?"),
            str(ev.get("args_summary") or ""),
        )
        for ev in events
        if ev.get("event") == "tool_call"
        and int(ev.get("session_number", 0) or 0) == session_number
    ]

    parts = []
    if task_description:
        parts.append(f"Task:\n{task_description}")

    finish_reason = str(last_end.get("finish_reason") or "?")
    turns = int(last_end.get("turns", 0) or 0)
    prompt_tokens = int(last_end.get("total_prompt_tokens", 0) or 0)
    parts.append(
        f"Previous session ended after {turns} turns: "
        f"{finish_reason}. Consumed {prompt_tokens} prompt tokens."
    )

    if finish_reason == "duplicate_abort" and calls:
        name, args = calls[-1]
        parts.append(
            cfg.resume_duplicate_abort.format(
                n=len(calls), call=f"{name}({args})"
            )
        )
    elif finish_reason == "context_full":
        pct = int(cfg.context_fill_ratio * 100)
        parts.append(cfg.resume_context_full.format(pct=pct))
    elif finish_reason == "max_turns" and calls:
        recent = calls[-cfg.resume_last_n_actions:]
        summaries = "; ".join(f"{name}({args})" for name, args in recent)
        parts.append(
            cfg.resume_max_turns.format(n=len(recent), actions=summaries)
        )
    elif finish_reason == "gate_escalation":
        parts.append(cfg.resume_gate_escalation.format(n=5))
    elif finish_reason == "length":
        parts.append(cfg.resume_length)

    parts.append(cfg.resume_base)
    return "\n\n".join(parts)


def _record_session_start_costs(cfg: Config, client, system_prompt: str,
                                 system_prompt_file: Path | None) -> None:
    """Record one-time per-task costs on the savings ledger.

    Captures:
      system_prompt   — tokens paid by the configured system_header.
      tool_surface    — tokens paid by tool schemas at the active
                        tool_desc mode.
      protocol_commandments — tokens paid by --system-prompt content
                              (zero when no file given).
      profile_behavioral    — tokens paid by the profile's behavioral
                              suffix (probe: run denormalize on a
                              minimal Commandments-tagged message and
                              measure the content delta).

    All records positive delta_chars (cost paid).
    """
    from .savings import get_ledger
    ledger = get_ledger()

    # System header (base system prompt before any --system-prompt append).
    ledger.record(
        bucket="system_prompt",
        layer="config",
        mechanism="system_header",
        input_chars=0,
        output_chars=len(cfg.system_header),
        measure_type="exact",
        ctx={"tool_desc": cfg.tool_desc},
    )

    # Tool surface: schema JSON emitted to the model (post profile knobs).
    try:
        schemas = _apply_profile_tool_cap(
            _apply_profile_schema_simplify(get_tool_schemas(cfg.tool_desc), client),
            client,
        )
        schema_chars = sum(len(json.dumps(s, default=str)) for s in schemas)
        ledger.record(
            bucket="tool_surface",
            layer="harness",
            mechanism=f"tool_schemas:{cfg.tool_desc}:effective",
            input_chars=0,
            output_chars=schema_chars,
            measure_type="exact",
            ctx={"n_tools": len(schemas)},
        )
    except Exception as e:
        log.debug("Tool-surface cost record skipped: %s", e)

    # Protocol commandments: content of the --system-prompt file, if any.
    if system_prompt_file is not None and Path(system_prompt_file).is_file():
        try:
            commandments_chars = len(Path(system_prompt_file).read_text())
            ledger.record(
                bucket="protocol_commandments",
                layer="L4_protocol",
                mechanism=Path(system_prompt_file).name,
                input_chars=0,
                output_chars=commandments_chars,
                measure_type="exact",
                ctx={"path": str(system_prompt_file)},
            )
        except OSError as e:
            log.debug("Protocol commandments cost record skipped: %s", e)

    # Profile behavioral: probe the denormalize pipeline on a minimal
    # system message marked with "Commandments" so any gated behavioral
    # modules fire. Delta = after-content minus before-content.
    profile = getattr(client, "profile", None)
    if profile is not None and hasattr(profile, "denormalize_messages"):
        try:
            probe = [{"role": "system", "content": "Commandments\n"}]
            before_chars = len(probe[0]["content"])
            after = profile.denormalize_messages([dict(m) for m in probe])
            after_content = after[0].get("content", "") if after else ""
            after_chars = len(after_content)
            if after_chars != before_chars:
                ledger.record(
                    bucket="profile_behavioral",
                    layer="L1_model_quirks",
                    mechanism=f"{profile.name}_behavioral_suffix",
                    input_chars=before_chars,
                    output_chars=after_chars,
                    measure_type="exact",
                    ctx={"profile": profile.name},
                )
        except Exception as e:
            log.debug("Profile-behavioral cost probe skipped: %s", e)
    if profile is not None:
        preamble = str(getattr(profile, "preamble", "") or "")
        if preamble.strip():
            profile_name = str(getattr(profile, "name", "unknown_profile"))
            ledger.record(
                bucket="profile_preamble",
                layer="L1_model_quirks",
                mechanism=f"{profile_name}_capacity_preamble",
                input_chars=0,
                output_chars=len(preamble),
                measure_type="exact",
                ctx={"profile": profile_name},
            )


def _load_bash_transforms(cfg: Config, *, force_load_all: bool = False):
    """Load the three bash transform layers respected by Session.

    Each layer is gated by its own enabled flag so ablation configs can
    disable them in isolation:
      1. Universal rewrites (pip -q, npm --loglevel=error, make -s)
      2. Task-format output control (pytest --tb=short, condense PASSED)
      3. Task-format output parser (structured test-run digest)

    Returns (output_control, universal_rewrites, output_parser). Any
    field can be None if the corresponding layer is disabled or
    misconfigured. Errors during load are swallowed to a debug log —
    the harness continues with raw-bash semantics.
    """
    output_control = None
    universal_rewrites = None
    output_parser = None
    try:
        from ..bash_quirks import (
            load_output_control,
            load_output_parser,
            load_universal_rewrites,
        )
        if cfg.bash_transforms_universal_enabled or force_load_all:
            universal_rewrites = load_universal_rewrites()
            if universal_rewrites:
                log.info("Loaded %d universal bash rewrites", len(universal_rewrites))
        else:
            log.info("Universal bash rewrites disabled via config")
        if cfg.bash_transforms_task_format_enabled or force_load_all:
            _analysis_fmt = cfg.analysis_task_format if hasattr(cfg, "analysis_task_format") else None
            if _analysis_fmt:
                from ..language_quirks import FORMATS_DIR
                fmt_path = FORMATS_DIR / f"{_analysis_fmt}.toml"
                output_control = load_output_control(fmt_path)
                if output_control:
                    log.info("Loaded output control: %s (flag=%r, passed=%r)",
                             _analysis_fmt, output_control.failure_only_flag, output_control.passed_marker)
                if cfg.bash_transforms_structured_output_enabled or force_load_all:
                    output_parser = load_output_parser(fmt_path)
                    if output_parser:
                        log.info("Loaded output parser: %s (summary_fields=%d, per_test=%s)",
                                 _analysis_fmt,
                                 len(output_parser.summary_fields),
                                 output_parser.per_test_regex is not None)
                    else:
                        log.info("Structured output enabled but no [output_parser] block in %s.toml", _analysis_fmt)
        else:
            log.info("Task-format output control disabled via config")
    except Exception as e:
        log.debug("Could not load bash transforms: %s", e)
    return output_control, universal_rewrites, output_parser


def solve_task(
    repo_dir: Path, cfg: Config, client,
    system_prompt_file: Path | None = None,
    context_class: type[ContextManager] | None = None,
    profile_path: Path | None = None,
    initial_prompt: str | None = None,
    task_spec: TaskSpec | None = None,
    artifacts_dir: Path | None = None,
    resume_from_artifacts: bool = False,
) -> bool:
    """Outer loop: run sessions until done or max_sessions exhausted.

    client: any object with chat() and build_assistant_message() methods
           (e.g. LlamaClient from server layer). Injected for swappability.
    system_prompt_file: optional file whose content is prepended to the system prompt.
    context_class: if provided, instantiate this instead of default SolverStateContext.
    profile_path: optional path to profile.toml for provenance hashing.
    """
    work_dir = Path(repo_dir)
    artifact_dir = Path(artifacts_dir) if artifacts_dir is not None else work_dir
    prompt_file = artifact_dir / "prompt.txt"
    pretest_script = task_spec.pretest_script if task_spec is not None else None
    task_prompt = initial_prompt
    if task_prompt is None and task_spec is not None:
        task_prompt = task_spec.prompt_text
    if task_prompt is None:
        if not prompt_file.exists():
            log.error("No prompt.txt in %s", artifact_dir)
            return False
        task_prompt = prompt_file.read_text()

    start_time = time.time()
    provenance = collect_provenance(cfg, profile_path)
    if cfg.variant_name:
        provenance["variant_name"] = cfg.variant_name
        provenance["prompt_addendum"] = cfg.prompt_addendum

    system_prompt = _apply_profile_preamble(
        build_system_prompt(cfg.system_header, system_prompt_file),
        client,
    )
    prev_session: Session | None = None
    prev_result: SessionResult | None = None

    agg_prompt = 0
    agg_completion = 0
    agg_turns = 0
    sessions_used = 0
    success = False

    trace_path = artifact_dir / ".trace.jsonl"
    task_description = task_prompt

    # Mechanical state.json writer: active iff .solver/state.json was seeded
    # before solve_task starts. If .solver/state.json is absent, the writer
    # stays off and solve_task remains compatible with stateless runs.
    state_json_path = artifact_dir / ".solver" / "state.json"
    state_path: Path | None = state_json_path if state_json_path.is_file() else None

    # Open the savings ledger for this task. Always-on; Bucket A
    # observability. In measurement mode the ledger stays outside the working
    # directory for determinism. In assistant mode, a dedicated artifact dir
    # already isolates it from the edited repo, so keep the ledger local to the
    # session bundle.
    from .savings import close_ledger, open_ledger
    if artifacts_dir is None:
        _savings_root = work_dir.parent.parent / "savings"
        _savings_root.mkdir(parents=True, exist_ok=True)
        savings_path = _savings_root / f"{work_dir.name}.jsonl"
    else:
        artifact_dir.mkdir(parents=True, exist_ok=True)
        savings_path = artifact_dir / "savings.jsonl"
    open_ledger(savings_path)
    _record_session_start_costs(cfg, client, system_prompt, system_prompt_file)

    # Verbatim transcript of every HTTP exchange — forensics file at the
    # server-client chokepoint. Truncated per task; counter monotonic across
    # all sessions for this task.
    # Keep transcripts outside the working tree the model edits. Measurement
    # runs store them at the run-dir level; assistant sessions keep them in the
    # dedicated artifact bundle.
    if hasattr(client, "set_transcript"):
        if artifacts_dir is None:
            run_dir = work_dir.parent.parent  # run_dir/repos/<iid>/
            transcript_dir = run_dir / "transcripts"
            transcript_dir.mkdir(parents=True, exist_ok=True)
            transcript_path = transcript_dir / f"{work_dir.name}.log"
        else:
            transcript_path = artifact_dir / "transcript.log"
        client.set_transcript(transcript_path)

    output_control, universal_rewrites, output_parser = _load_bash_transforms(
        cfg,
        force_load_all=bool(getattr(cfg, "adaptive_policy_enabled", False)),
    )
    token_estimator = _resolve_token_estimator(client)

    # Pretest-parity baseline captured at session 1 (when parser is loaded).
    # Holds {'failing': set, 'passing': set}; passed unchanged to every
    # Session so subsequent sessions inherit the baseline. None when not
    # populated or not applicable (no parser / empty pretest).
    pretest_parsed_verdict: dict | None = None
    start_session_num = _next_session_number(trace_path) if resume_from_artifacts else 1
    end_session_num = start_session_num + cfg.max_sessions - 1

    with open(trace_path, "a") as trace_file:
        for session_num in range(start_session_num, end_session_num + 1):
            # Pretest: run failing tests BEFORE every session. Verdict becomes
            # the first block of the session's first user message. On sessions
            # 2+ we short-circuit to success if the pretest already exits
            # green — no model invocation needed.
            pretest_block = run_pretest(
                work_dir,
                pretest_script=pretest_script,
                pretest_timeout=cfg.pretest_timeout,
                pretest_head_chars=cfg.pretest_head_chars,
                pretest_tail_chars=cfg.pretest_tail_chars,
            )
            # Parse pretest output on session 1 ONLY — this is the task's
            # ground-truth baseline. Subsequent sessions inherit the same
            # baseline (pretest on session 2+ may look different after
            # mid-task progress but should not re-seed).
            if (session_num == start_session_num
                    and output_parser is not None
                    and pretest_block
                    and cfg.done_require_pretest_parity):
                try:
                    from ..bash_quirks import parse_structured
                    pre_parsed = parse_structured(pretest_block, output_parser)
                    tests = pre_parsed.get("tests") or {}
                    failing = {t for t, v in tests.items() if v in ("FAILED", "FAIL", "ERROR")}
                    passing = {t for t, v in tests.items() if v in ("PASSED", "PASS")}
                    if failing or passing:
                        pretest_parsed_verdict = {"failing": failing, "passing": passing}
                        log.info("Pretest parity baseline: %d failing, %d passing",
                                 len(failing), len(passing))
                    else:
                        log.info("Pretest not structurally parseable — done_guard falls back to heuristic")
                except Exception as e:
                    log.debug("Pretest parse failed: %s", e)
            if session_num > 1 and _pretest_is_green(pretest_block):
                log.info("Pretest exited green at session start — short-circuiting.")
                write_checkpoint(artifact_dir, cfg.model, "completed")
                success = True
                sessions_used = session_num - start_session_num
                break

            # Cross-run determinism: re-normalize file timestamps right
            # Stabilize timestamps only before the first session of this
            # invocation. Later sessions must reflect the agent's own edits.
            if session_num == start_session_num:
                _normalize_repo_timestamps(work_dir)

            if session_num == start_session_num:
                if resume_from_artifacts:
                    initial = build_resume_prompt_from_trace(
                        trace_path, cfg, task_description
                    ) or task_prompt
                else:
                    initial = task_prompt
                if cfg.prompt_addendum and not resume_from_artifacts:
                    initial = initial.rstrip() + "\n\n" + cfg.prompt_addendum
            else:
                initial = build_resume_prompt(prev_result, prev_session, cfg, task_description)

            if pretest_block:
                initial = pretest_block + "\n" + initial

            log.info("[session %d/%d] %s", session_num, end_session_num, work_dir.name)

            # Trace: session start
            trace_file.write(json.dumps({
                "event": "session_start",
                "session_number": session_num,
            }) + "\n")
            trace_file.flush()
            if state_path is not None:
                write_state_from_trace(trace_path, state_path,
                                       max_result_chars=cfg.max_output_chars)

            # Instantiate the selected context class. Different classes
            # take different constructor arguments — FullTranscript takes
            # none, CompactTranscript/YujTranscript take original_prompt,
            # SolverStateContext and its subclasses (CompoundContext) also
            # require cwd and accept trace_lines/evidence_lines. Use
            # introspection so new context classes can be added without
            # editing this dispatch.
            if context_class is not None:
                sig = inspect.signature(context_class.__init__)
                kwargs: dict = {"original_prompt": initial}
                # Map Config fields → constructor kwargs (introspection-driven).
                _cfg_map = {
                    "cwd": str(work_dir),
                    "trace_lines": cfg.solver_trace_lines,
                    "evidence_lines": cfg.solver_evidence_lines,
                    "inference_lines": cfg.solver_inference_lines,
                    "recent_tool_results_chars": cfg.recent_tool_results_chars,
                    "recent_results_chars": cfg.recent_tool_results_chars,
                    "trace_stub_chars": cfg.trace_stub_chars,
                    "trace_reasoning_chars": cfg.trace_reasoning_chars,
                    "min_turns": cfg.min_turns_before_context,
                    "args_summary_chars": cfg.args_summary_chars,
                    "inspect_repeat_threshold": cfg.context_inspect_repeat_threshold,
                    "recovery_same_target_threshold": cfg.contract_recovery_same_target_threshold,
                    "recovery_verify_repeat_threshold": cfg.contract_recovery_verify_repeat_threshold,
                    "slot_max_candidates": cfg.context_slot_max_candidates,
                    "slot_inline_files": cfg.context_slot_inline_files,
                    "focused_trace_lines": cfg.focused_compound_trace_lines,
                    "focused_evidence_lines": cfg.focused_compound_evidence_lines,
                    "focused_recent_tool_results_chars": cfg.focused_compound_recent_tool_results_chars,
                    "focused_include_resolved_evidence": cfg.focused_compound_include_resolved_evidence,
                    "selective_trace_lines": cfg.compound_selective_trace_lines,
                    "selective_unresolved_evidence_lines": cfg.compound_selective_unresolved_evidence_lines,
                    "selective_resolved_evidence_lines": cfg.compound_selective_resolved_evidence_lines,
                    "selective_resolved_evidence_stub_chars": cfg.compound_selective_resolved_evidence_stub_chars,
                    "selective_recent_tool_results_chars": cfg.compound_selective_recent_tool_results_chars,
                    "selective_trace_action_repeat_cap": cfg.compound_selective_trace_action_repeat_cap,
                    "selective_resolved_action_repeat_cap": cfg.compound_selective_resolved_action_repeat_cap,
                    "selective_trace_anchor_lines": cfg.compound_selective_trace_anchor_lines,
                    "selective_resolved_anchor_lines": cfg.compound_selective_resolved_anchor_lines,
                    "selective_trace_source_anchor_lines": cfg.compound_selective_trace_source_anchor_lines,
                    "selective_trace_test_anchor_lines": cfg.compound_selective_trace_test_anchor_lines,
                    "selective_resolved_source_anchor_lines": cfg.compound_selective_resolved_source_anchor_lines,
                    "selective_resolved_test_anchor_lines": cfg.compound_selective_resolved_test_anchor_lines,
                    "suffix": cfg.state_context_suffix,
                }
                if token_estimator is not None:
                    _cfg_map["token_estimator"] = token_estimator
                for param, value in _cfg_map.items():
                    if param in sig.parameters:
                        kwargs[param] = value
                ctx = context_class(**kwargs)
                # Session 2+: pre-populate the rolling tool-result window
                # with files modified in prior sessions so the model doesn't
                # edit from stale memory.  Only SolverStateContext subclasses
                # (stateful, compound) have this method; others skip silently.
                if session_num > 1 and hasattr(ctx, "prepopulate_from_trace"):
                    n_files = ctx.prepopulate_from_trace()
                    if n_files:
                        log.info("Pre-populated rolling window with %d file(s) from prior sessions", n_files)
            else:
                ctx = None
            session = Session(
                cfg, client, system_prompt, initial, str(work_dir),
                context_manager=ctx, trace_file=trace_file, session_number=session_num,
                trace_path=trace_path, state_path=state_path,
                output_control=output_control,
                universal_rewrites=universal_rewrites,
                output_parser=output_parser,
                pretest_parsed=pretest_parsed_verdict,
            )
            result = session.run()
            log.info(
                "Session ended: %s (turns=%d, prompt_tokens=%d)",
                result.finish_reason, result.turns, result.total_prompt_tokens,
            )

            # Aggregate metrics
            agg_prompt += result.total_prompt_tokens
            agg_completion += result.total_completion_tokens
            agg_turns += result.turns
            sessions_used = session_num - start_session_num + 1

            # Trace: session end
            trace_file.write(json.dumps({
                "event": "session_end",
                "session_number": session_num,
                "finish_reason": result.finish_reason,
                "turns": result.turns,
                "total_prompt_tokens": result.total_prompt_tokens,
            }) + "\n")
            trace_file.flush()
            if state_path is not None:
                write_state_from_trace(trace_path, state_path,
                                       max_result_chars=cfg.max_output_chars)

            if result.done:
                _auto_commit(work_dir, session_num, result.finish_reason)
                write_checkpoint(artifact_dir, cfg.model, "completed")
                success = True
                break

            if result.finish_reason == "error":
                write_checkpoint(artifact_dir, cfg.model, "error")
                break
            if result.finish_reason == "approval_required":
                write_checkpoint(artifact_dir, cfg.model, "paused")
                break

            # Auto-commit for non-error sessions (#19)
            _auto_commit(work_dir, session_num, result.finish_reason)

            # NORMAL_LIFECYCLE and MODEL_STUCK → continue to next session
            # (different preamble generated by build_resume_prompt)
            prev_session = session
            prev_result = result
        else:
            log.warning("Max sessions (%d) exhausted for %s", cfg.max_sessions, work_dir.name)
            write_checkpoint(artifact_dir, cfg.model, "error")

    # Write run metrics (#57, #60)
    wall_clock = time.time() - start_time
    total_tokens = agg_prompt + agg_completion
    metrics: dict = {
        "total_prompt_tokens": agg_prompt,
        "total_completion_tokens": agg_completion,
        "total_tokens": total_tokens,
        "wall_clock_seconds": round(wall_clock, 2),
        "sessions_used": sessions_used,
        "total_turns": agg_turns,
    }
    if sessions_used > 0:
        metrics["time_per_session_seconds"] = round(wall_clock / sessions_used, 2)
    if agg_turns > 0:
        metrics["tokens_per_turn"] = round(total_tokens / agg_turns, 2)

    write_run_metrics(artifact_dir, metrics, provenance)

    close_ledger()
    return success
