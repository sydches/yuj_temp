"""SolverStateContext — build messages from .solver/state.json, not raw history.

Every turn, constructs a two-message prompt:
  [system] system prompt (static, cached)
  [user]   .solver/ state + recent tool results (bounded, partially cached)

The model never sees conversation history. .solver/state.json IS the memory.
Recent tool results are preserved in a rolling char-budget window so that
code reads remain visible across the 2-3 turns it takes to go from "read
the file" to "edit the file" — otherwise the model would see a 200-char
stub of its own read by the next decision turn.
"""
import json
import re
from collections import deque
from collections.abc import Callable
from pathlib import Path

from ..context import ContextManager, chars_div_4


# ── Command-type classification for dedup messages ──────────────

_TEST_PREFIXES = ("pytest", "python -m pytest", "python -m unittest", "python -m py.test")
_READ_PREFIXES = ("cat ", "head ", "tail ", "less ", "more ", "wc ")
_SEARCH_PREFIXES = ("grep ", "rg ", "find ", "ag ", "fd ")


def _classify_cmd(cmd: str) -> str:
    """Classify a normalized bash command as 'test', 'read', 'search', or 'other'."""
    stripped = cmd.lstrip()
    for pfx in _TEST_PREFIXES:
        if stripped.startswith(pfx):
            return "test"
    for pfx in _READ_PREFIXES:
        if stripped.startswith(pfx):
            return "read"
    for pfx in _SEARCH_PREFIXES:
        if stripped.startswith(pfx):
            return "search"
    return "other"


_PYTEST_ERROR_RE = re.compile(r"^E\s+.+", re.MULTILINE)


def _extract_error_snippet(prev_content: str, max_chars: int = 200) -> str:
    """Extract the key error line from a previous tool result.

    For pytest output, grabs the last `E   ...` line (the actual assertion
    or exception). Falls back to the last non-empty line.
    """
    # Pytest E-lines
    matches = _PYTEST_ERROR_RE.findall(prev_content)
    if matches:
        snippet = matches[-1].strip()
        if len(snippet) > max_chars:
            snippet = snippet[:max_chars - 3] + "..."
        return snippet
    # Fallback: last non-empty line
    for line in reversed(prev_content.splitlines()):
        line = line.strip()
        if line and not line.startswith("="):
            if len(line) > max_chars:
                line = line[:max_chars - 3] + "..."
            return line
    return ""


def _dedup_message(cmd: str, prev_content: str, count: int, turn_ref: int) -> str:
    """Build a context-aware dedup message.

    Uses command type to pick the right framing and echoes the previous
    error so the model knows what to fix.
    """
    cmd_type = _classify_cmd(cmd)
    snippet = _extract_error_snippet(prev_content)
    blocked = count >= 2

    if cmd_type == "test":
        if blocked:
            msg = f"ERROR: BLOCKED — `{cmd}` ran {count + 1} times (turn {turn_ref})."
        else:
            msg = f"WARNING: You already ran `{cmd}` (turn {turn_ref})."
        if snippet:
            msg += f"\nPrevious failure: {snippet}"
        msg += "\nYour last edit didn't fix this. Read the error and make a different change."
    elif cmd_type == "read":
        if blocked:
            msg = f"ERROR: BLOCKED — `{cmd}` ran {count + 1} times (turn {turn_ref}).\nStop reading this file. Edit it or work on something else."
        else:
            msg = f"WARNING: You already ran `{cmd}` (turn {turn_ref}).\nYou already have this content. Edit the file or move on."
    elif cmd_type == "search":
        if blocked:
            msg = f"ERROR: BLOCKED — `{cmd}` ran {count + 1} times (turn {turn_ref}).\nYou already searched for this. Act on what you found."
        else:
            msg = f"WARNING: You already ran `{cmd}` (turn {turn_ref}).\nYou already have these results. Act on them."
    else:
        if blocked:
            msg = f"ERROR: BLOCKED — `{cmd}` ran {count + 1} times (turn {turn_ref}).\nChange your approach — this command will not produce new information."
        else:
            msg = f"WARNING: You already ran `{cmd}` (turn {turn_ref}).\nRe-running will not help — change your approach."
    return msg


class SolverStateContext(ContextManager):
    """Builds messages from .solver/ state each turn.

    After .solver/ exists, every prompt is exactly two messages:
      system: static prompt (always cached)
      user: .solver/ state + last tool results (partially cached)

    Falls back to append-only behavior until .solver/ files exist (turns 0-1).

    All numeric tunables are required kwargs — no module-level shadow
    defaults. The harness wires them from config.toml through Config.
    """

    def __init__(
        self,
        cwd: str,
        original_prompt: str,
        *,
        trace_lines: int,
        evidence_lines: int,
        inference_lines: int,
        recent_tool_results_chars: int,
        trace_stub_chars: int,
        min_turns: int,
        suffix: str,
        token_estimator: Callable[[list[dict]], int] = chars_div_4,
    ):
        super().__init__(token_estimator)
        self._cwd = Path(cwd)
        self._original_prompt = original_prompt
        self._trace_lines = trace_lines
        self._evidence_lines = evidence_lines
        self._inference_lines = inference_lines
        self._recent_tool_results_chars = recent_tool_results_chars
        self._trace_stub_chars = trace_stub_chars
        self._min_turns = min_turns
        self._suffix = suffix

        # Internal state
        self._system_content: str = ""
        self._all_messages: list[dict] = []  # raw append log (fallback only)
        # Rolling window of recent tool results (newest append). No per-turn
        # reset — a code read at turn 3 must still be visible at turn 5 when
        # the model decides to edit. Bounded by self._recent_tool_results_chars
        # in _format_tool_results, not by entry count.
        self._recent_tool_results: deque[dict] = deque()
        self._turn_count: int = 0
        self._file_cache: dict[str, str] | None = None  # cached .solver/ file contents
        # Raw-parse cache shared with subclasses that re-read state.json
        # (CompoundContext does this for reasoning-aware trace + fail-
        # first evidence rendering). Invalidated in lockstep with
        # _file_cache — a new tool result means state.json has been
        # regenerated and the parse is stale.
        self._raw_state_cache: dict | None = None
        # Escalating dedup: tracks how many times each unique output has
        # been deduplicated. Keyed by hash(content). Escalation:
        #   1st dedup (2nd attempt): behavioral warning
        #   2nd+ dedup (3rd+ attempt): hard block
        # Cleared on successful write/edit (code change invalidates the
        # assumption that repeated commands produce identical output).
        self._dedup_counts: dict[int, int] = {}
        self._dedup_epoch: int = 0
        # Per-turn message + token caches. _build_from_solver rebuilds the
        # full user-message payload (reads state.json, formats trace,
        # splits evidence). Both get_messages and estimate_tokens are
        # called once per turn, so the cache pays back as soon as
        # estimate_tokens runs after get_messages within the same turn.
        # Every add_* invalidates both. The file-level _file_cache still
        # exists and is invalidated in add_tool_result; the message cache
        # is a superset layered on top.
        self._msg_cache: list[dict] | None = None
        self._tok_cache: int | None = None

    def add_system(self, content: str) -> None:
        self._system_content = content
        self._all_messages.append({"role": "system", "content": content})
        self._msg_cache = None
        self._tok_cache = None

    def add_user(self, content: str) -> None:
        self._all_messages.append({"role": "user", "content": content})
        self._msg_cache = None
        self._tok_cache = None

    def add_assistant(self, message: dict) -> None:
        self._all_messages.append(message)
        self._msg_cache = None
        self._tok_cache = None
        # Do NOT clear _recent_tool_results here. The original design cleared
        # them on every new turn, which meant: at turn N the model could only
        # see the tool result from turn N-1. A code read at turn 3 was
        # invisible by turn 5, so the model could never make an edit decision
        # grounded in a file it had read 2+ turns earlier. Now the rolling
        # window is trimmed only by char budget in _format_tool_results.
        self._turn_count += 1

    def reset_dedup_counts(self) -> None:
        """Clear dedup escalation state.

        Called by the session loop after a successful write/edit — a code
        change invalidates the assumption that repeated commands will produce
        identical output.
        """
        self._dedup_counts.clear()
        # Increment epoch so both dedup tiers skip pre-edit entries.
        # Replaces the old cmd_sig stripping approach — epoch handles
        # both cmd-sig (tier 1) and content (tier 2) in one shot.
        self._dedup_epoch += 1

    def add_tool_result(self, tool_call_id: str, content: str, *, tool_name: str = "", cmd_signature: str = "", gate_blocked: bool = False) -> None:
        # Two-tier escalating dedup:
        #   Tier 1 — command-signature dedup (bash only): catches pipe
        #     variations like `cat file` vs `cat file | head -100` that
        #     produce different output but read the same data.
        #   Tier 2 — content dedup (all tools): catches byte-identical
        #     output from any tool.
        # Both tiers share the same _dedup_counts escalation state and
        # reset on successful write/edit.
        #
        # Exempt: `read` tool (compound context shows stubs — the model
        # genuinely needs to re-read files before editing).
        _DEDUP_EXEMPT = frozenset({"read"})
        dedup_fired = False
        dedup_tier = ""
        original_chars = len(content)

        # Tier 1: command-signature dedup for bash
        if cmd_signature and tool_name not in _DEDUP_EXEMPT and len(content) > 200:
            for i, existing in enumerate(self._recent_tool_results):
                if existing.get("_epoch") != self._dedup_epoch:
                    continue
                if existing.get("_cmd_sig") == cmd_signature:
                    turn_ref = self._turn_count - (len(self._recent_tool_results) - i)
                    sig_key = hash(("cmd", cmd_signature))
                    self._dedup_counts[sig_key] = self._dedup_counts.get(sig_key, 0) + 1
                    count = self._dedup_counts[sig_key]
                    try:
                        _cmd = json.loads(cmd_signature).get("cmd", cmd_signature)
                    except (ValueError, TypeError):
                        _cmd = cmd_signature
                    content = _dedup_message(_cmd, existing["content"], count, turn_ref)
                    dedup_fired = True
                    dedup_tier = "tier1_cmd_signature"
                    break

        # Tier 2: byte-identical content dedup (all non-exempt tools)
        if not dedup_fired and tool_name not in _DEDUP_EXEMPT:
            for i, existing in enumerate(self._recent_tool_results):
                if existing.get("_epoch") != self._dedup_epoch:
                    continue
                if existing["content"] == content and len(content) > 200:
                    turn_ref = self._turn_count - (len(self._recent_tool_results) - i)
                    content_key = hash(content)
                    self._dedup_counts[content_key] = self._dedup_counts.get(content_key, 0) + 1
                    count = self._dedup_counts[content_key]
                    if count >= 2:
                        content = (
                            f"ERROR: BLOCKED — identical output {count + 1} times (see turn {turn_ref}).\n"
                            f"REASON: Re-running will not produce new information.\n"
                            f"ACTION REQUIRED: You MUST change your approach — "
                            f"edit code, read a different file, or try a different command."
                        )
                    else:
                        content = (
                            f"WARNING: Same output as turn {turn_ref}.\n"
                            f"Re-running will not help — change your approach."
                        )
                    dedup_fired = True
                    dedup_tier = "tier2_byte_identical"
                    break

        # Token accounting: record exact dedup savings when either tier fires.
        if dedup_fired and original_chars != len(content):
            from ..savings import get_ledger
            get_ledger().record(
                bucket="dedup",
                layer="context_strategy",
                mechanism=dedup_tier,
                input_chars=original_chars,
                output_chars=len(content),
                measure_type="exact",
                ctx={"tool_name": tool_name, "gate_blocked": gate_blocked},
            )

        # Gate-blocked entries get epoch -1 so dedup tiers never match
        # against them — the tool was never executed, the content is a
        # gate message, not real output.
        entry_epoch = -1 if gate_blocked else self._dedup_epoch
        msg = {"role": "tool", "tool_call_id": tool_call_id, "content": content, "_cmd_sig": cmd_signature, "_epoch": entry_epoch}
        self._all_messages.append(msg)
        self._recent_tool_results.append(msg)
        self._file_cache = None  # tool execution may have written to .solver/state.json
        self._raw_state_cache = None
        self._msg_cache = None
        self._tok_cache = None

    def get_messages(self) -> list[dict]:
        """Build messages from .solver/state.json if available, else fall back to full list."""
        if self._msg_cache is not None:
            return self._msg_cache
        solver_dir = self._cwd / ".solver"
        if not (solver_dir / "state.json").is_file() or self._turn_count < self._min_turns:
            self._msg_cache = self._all_messages
        else:
            self._msg_cache = self._build_from_solver(solver_dir)
            # Token accounting: solver-state projection vs. full append log.
            from ..savings import get_ledger
            full_chars = sum(len(str(m)) for m in self._all_messages)
            actual_chars = sum(len(str(m)) for m in self._msg_cache)
            get_ledger().record(
                bucket="context_projection",
                layer="context_strategy",
                mechanism=type(self).__name__,
                input_chars=full_chars,
                output_chars=actual_chars,
                measure_type="exact",
                ctx={"turn_count": self._turn_count,
                     "messages": len(self._msg_cache)},
            )
        return self._msg_cache

    def estimate_tokens(self) -> int:
        if self._tok_cache is None:
            self._tok_cache = self._token_estimator(self.get_messages())
        return self._tok_cache

    def message_count(self) -> int:
        return len(self._all_messages)

    # ── Internal ──────────────────────────────────────────

    _EMPTY_SECTIONS = {
        "state": "",
        "trace": "",
        "gates": "",
        "evidence": "",
        "inference": "",
    }

    def _get_solver_files(self, solver_dir: Path) -> dict[str, str]:
        """Read .solver/state.json and format each section as text.

        Cache is invalidated by add_tool_result() — the only point where
        tool execution may have written to .solver/state.json. Between tool
        results, multiple get_messages()/estimate_tokens() calls reuse the
        same read.

        Missing or empty file → empty sections (expected on first turn).
        Malformed JSON → JSONDecodeError surfaces. The model wrote garbage;
        the failure is evidence per the protocol.
        """
        if self._file_cache is not None:
            return self._file_cache

        state_path = solver_dir / "state.json"
        if not state_path.is_file():
            self._file_cache = dict(self._EMPTY_SECTIONS)
            return self._file_cache

        raw = state_path.read_text().strip()
        if not raw:
            self._file_cache = dict(self._EMPTY_SECTIONS)
            return self._file_cache

        data = json.loads(raw)
        self._file_cache = {
            "state": self._format_state(data.get("state")),
            "trace": self._format_trace(data.get("trace", []), self._trace_lines),
            "gates": self._format_gates(data.get("gates", [])),
            "evidence": self._format_list(data.get("evidence", []), self._evidence_lines),
            "inference": self._format_list(data.get("inference", []), self._inference_lines),
        }
        return self._file_cache

    def _format_state(self, state) -> str:
        if not state:
            return ""
        if isinstance(state, str):
            return state.strip()
        if isinstance(state, dict):
            parts = []
            for k in ("current_attempt", "last_verify", "next_action"):
                v = state.get(k)
                if v:
                    label = k.replace("_", " ").capitalize()
                    parts.append(f"{label}: {v}")
            return "\n".join(parts)
        return str(state)

    def _format_trace(self, trace, max_entries: int) -> str:
        """Format the tail of the trace as action/outcome stubs only.

        The trace section is the structural breadcrumb — it shows WHAT
        happened, not the raw payload. Full tool-result content lives in
        the rolling _format_tool_results window, which has its own 30K
        char budget and handles the recency cap.

        Previously this function rendered full results up to a 20K
        budget, which DUPLICATED content with the rolling window: the
        same file read appeared 2-6 times in one user message at cell 4
        stateful (observed at turn 50 of attempt_002). Trace + rolling
        window were both writing full content to the same prompt,
        wasting 30-50K chars per turn.

        Now every trace entry gets a short stub (self._trace_stub_chars,
        default 200) regardless of recency. The model reads the trace
        section to remember what it DID; it reads the rolling window to
        see the raw RESULT of recent actions. No overlap.
        """
        if not isinstance(trace, list) or not trace:
            return ""
        tail = trace[-max_entries:]
        lines: list[str] = []
        for entry in tail:
            if isinstance(entry, dict):
                step = entry.get("step", "?")
                action = entry.get("action", "")
                result = entry.get("result", "")
                nxt = entry.get("next", "")
            else:
                step, action, result, nxt = "?", str(entry), "", ""
            stub_result = (
                result[: self._trace_stub_chars - 3] + "..."
                if len(result) > self._trace_stub_chars
                else result
            )
            lines.append(f"{step} | {action} | {stub_result} | {nxt}")
        return "\n".join(lines)

    def _format_gates(self, gates) -> str:
        if not isinstance(gates, list) or not gates:
            return ""
        lines = []
        for g in gates:
            if isinstance(g, dict):
                name = g.get("name", "?")
                status = g.get("status", "?")
                notes = g.get("notes", "")
                line = f"{name}: {status}"
                if notes:
                    line += f" — {notes}"
                lines.append(line)
            else:
                lines.append(str(g))
        return "\n".join(lines)

    def _format_list(self, items, max_items: int) -> str:
        if not isinstance(items, list) or not items:
            return ""
        tail = items[-max_items:]
        lines = []
        for x in tail:
            if isinstance(x, dict):
                # Structured evidence entry (DRY schema).
                lines.append(f"step {x['step']}: {x['action']} → {x.get('result', '')}")
            else:
                lines.append(str(x))
        return "\n".join(lines)

    def prepopulate_from_trace(self) -> int:
        """Pre-populate the rolling window from files modified in prior sessions.

        Reads state.json trace, finds the most recent write/edit actions,
        re-reads those files from disk, and injects them into the rolling
        window. This closes the context cliff at session boundaries: without
        it, session 2+ starts with an empty rolling window and the model
        edits files from stale memory.

        Returns the number of files injected.
        """
        state_path = self._cwd / ".solver" / "state.json"
        if not state_path.is_file():
            return 0
        try:
            state = json.loads(state_path.read_text())
        except (json.JSONDecodeError, OSError):
            return 0
        trace = state.get("trace", [])
        if not trace:
            return 0

        # Collect unique file paths from the most recent write/edit actions.
        # Walk backward, stop after collecting enough to fill the window.
        # Use the full budget — new tool results during the session will
        # push these out naturally via the char-budget trim in
        # _format_tool_results.
        seen: set[str] = set()
        files_to_read: list[tuple[str, str]] = []
        budget = self._recent_tool_results_chars
        chars_used = 0
        for entry in reversed(trace):
            action = entry.get("action", "")
            if not (action.startswith("write(") or action.startswith("edit(")):
                continue
            # Extract path from action string: write(path='foo.py', ...) or edit(path='foo.py', ...)
            m = re.search(r"path='([^']+)'", action)
            if not m:
                continue
            fpath = m.group(1)
            if fpath in seen or fpath.endswith("state.json"):
                continue
            seen.add(fpath)
            target = self._cwd / fpath.lstrip("./")
            if not target.is_file():
                continue
            try:
                content = target.read_text()
            except OSError:
                continue
            if chars_used + len(content) > budget:
                break
            files_to_read.append((fpath, content))
            chars_used += len(content)

        # Inject in chronological order (oldest first = least recent mutation first).
        injected = 0
        for fpath, content in reversed(files_to_read):
            numbered = "\n".join(
                f"{i+1}: {line}" for i, line in enumerate(content.splitlines())
            )
            synthetic = {
                "role": "tool",
                "tool_call_id": f"prepopulate-{fpath}",
                "content": numbered,
            }
            self._recent_tool_results.append(synthetic)
            injected += 1
        return injected

    def _format_tool_results(self) -> str:
        """Format recent tool results for injection into user message.

        Walks the rolling window newest-first, accumulating full contents
        until recent_tool_results_chars is exhausted. Older results drop
        out. The result is rendered oldest-to-newest so the model reads
        it in chronological order.

        Also trims the deque as a side-effect: anything that didn't fit
        in this turn's window is evicted permanently so the memory
        footprint stays bounded across long runs.
        """
        if not self._recent_tool_results:
            return ""
        # Walk newest to oldest, keep within char budget.
        kept_rev: list[dict] = []
        chars_used = 0
        for tr in reversed(self._recent_tool_results):
            content = tr.get("content") or ""
            if chars_used + len(content) > self._recent_tool_results_chars and kept_rev:
                break
            kept_rev.append(tr)
            chars_used += len(content)
        # Evict anything beyond what we kept so the deque doesn't grow
        # without bound over a long session.
        while len(self._recent_tool_results) > len(kept_rev):
            self._recent_tool_results.popleft()
        parts = [tr["content"] for tr in reversed(kept_rev)]
        results = "\n---\n".join(parts)
        label = (
            f"=== Tool results (last {len(kept_rev)}, newest last) ==="
            if len(kept_rev) > 1
            else "=== Tool result from your last action ==="
        )
        return f"{label}\n{results}"

    def _build_from_solver(self, solver_dir: Path) -> list[dict]:
        """Build a two-message prompt: system + user.

        User message contains .solver/state.json sections + last tool results.
        No conversation history. No assistant messages. Bounded size.
        """
        files = self._get_solver_files(solver_dir)

        # Build the context summary
        parts = [f"Task: {self._original_prompt}"]

        if files["state"]:
            parts.append(f"=== Current state ===\n{files['state']}")
        if files["trace"]:
            parts.append(f"=== Progress trace (recent) ===\n{files['trace']}")
        if files["gates"]:
            parts.append(f"=== Constraints ===\n{files['gates']}")
        if files["evidence"]:
            parts.append(f"=== Evidence ===\n{files['evidence']}")
        if files["inference"]:
            parts.append(f"=== Hypotheses ===\n{files['inference']}")

        # Inject last tool results — the one-turn blind spot
        tool_results = self._format_tool_results()
        if tool_results:
            parts.append(tool_results)

        if self._suffix:
            parts.append(self._suffix)

        return [
            {"role": "system", "content": self._system_content},
            {"role": "user", "content": "\n\n".join(parts)},
        ]


CONTEXT_MODE = "stateful"
CONTEXT_CLASS = SolverStateContext
