"""CompactTranscript — bounded context via harness-built summary.

Each turn, records the model's reasoning (assistant content) and tool
outcome (name, args, success/error).  get_messages() builds a 2-message
prompt: system + synthesized user containing the original task, a
compressed progress log, and a char-budgeted window of recent full
tool results.

No model cooperation required.  No disk I/O.  No protocol dependency.
See docs/compact_transcript.md for design rationale.
"""
from __future__ import annotations

from collections import deque
from collections.abc import Callable
from dataclasses import dataclass

from ..context import ContextManager, chars_div_4


@dataclass
class TurnEntry:
    """One turn's compressed record."""
    turn: int
    reasoning: str  # assistant content (verbatim, typically 1-2 sentences)
    tool_name: str
    args_summary: str
    outcome: str  # "OK" or "FAIL" — content-blind, from exit-code markers only


# Canonical outcome classifier — single source of truth in _shared.classification.
# Re-exported here as _classify_outcome so yuj_transcript.py's import is unchanged.
from ..._shared.classification import classify_outcome as _classify_outcome


class CompactTranscript(ContextManager):
    """Bounded context built from turn-level summaries.

    After min_turns, every prompt is exactly 2 messages:
      system: static prompt (server-cached)
      user: task + progress log + last N full tool results

    The progress log preserves:
      - Model's reasoning per turn (semantic signal)
      - Tool call + outcome per turn (structural signal, content-blind)
    Full tool result payloads are kept only for the last N turns.

    All numeric tunables are required kwargs — no module-level shadow
    defaults. The harness wires them from config.toml through Config.
    """

    def __init__(
        self,
        original_prompt: str,
        *,
        recent_results_chars: int,
        trace_reasoning_chars: int,
        min_turns: int,
        args_summary_chars: int,
        token_estimator: Callable[[list[dict]], int] = chars_div_4,
    ):
        super().__init__(token_estimator)
        self._original_prompt = original_prompt
        self._trace_reasoning_chars = trace_reasoning_chars
        self._min_turns = min_turns
        self._args_summary_chars = args_summary_chars
        self._system_content: str = ""
        self._turn_entries: list[TurnEntry] = []
        # Unbounded deque — trimmed by char budget in _build_compact, not
        # by entry count. The `recent_results` kwarg is kept for backward
        # compatibility but unused (previously `deque(maxlen=3)`).
        self._recent_results: deque[str] = deque()
        self._recent_results_chars: int = recent_results_chars
        self._all_messages: list[dict] = []  # raw log (fallback for turns 0-1)
        self._turn_count: int = 0
        self._last_assistant_msg: dict | None = None  # buffer between add_assistant and first add_tool_result
        self._prev_assistant_msg: dict | None = None  # retained for multi-tool lookups within same turn
        # Per-turn message + token caches. get_messages rebuilds a compact
        # representation that is identical within a turn; estimate_tokens
        # scans that representation. Caching eliminates the second build
        # when estimate_tokens is called after get_messages within the
        # same turn, and avoids a fresh compact-build if both are called
        # multiple times without a mutation in between. Invalidated by
        # every add_* method.
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
        self._last_assistant_msg = message
        self._turn_count += 1
        self._msg_cache = None
        self._tok_cache = None

    def add_tool_result(self, tool_call_id: str, content: str, *, tool_name: str = "", cmd_signature: str = "", gate_blocked: bool = False) -> None:
        self._all_messages.append({
            "role": "tool", "tool_call_id": tool_call_id, "content": content,
        })
        self._recent_results.append(content)
        self._msg_cache = None
        self._tok_cache = None

        # Build turn entry from the assistant message + this result.
        # First tool result for a turn gets the reasoning; subsequent ones
        # (multi-tool calls) get empty reasoning but still record the outcome.
        assistant_msg = self._last_assistant_msg or self._prev_assistant_msg
        if assistant_msg is not None:
            reasoning = ""
            if self._last_assistant_msg is not None:
                reasoning = self._last_assistant_msg.get("content") or ""
                self._prev_assistant_msg = self._last_assistant_msg
                self._last_assistant_msg = None
            tool_name, args_summary = self._extract_tool_info(
                assistant_msg, tool_call_id,
            )
            self._turn_entries.append(TurnEntry(
                turn=self._turn_count,
                reasoning=reasoning,
                tool_name=tool_name,
                args_summary=args_summary,
                outcome=_classify_outcome(content),
            ))

    def get_messages(self) -> list[dict]:
        if self._msg_cache is not None:
            return self._msg_cache
        if self._turn_count < self._min_turns:
            # Fallback path returns the raw-log reference; cache holds the
            # same reference so a subsequent estimate_tokens sees identical
            # data. Mutation goes through add_* which invalidates.
            self._msg_cache = self._all_messages
        else:
            self._msg_cache = self._build_compact()
            # Token accounting: the projection replaces the full append log
            # with a compact summary. Record the exact delta vs. what a
            # FullTranscript would have emitted for the same turn state.
            from ..savings import get_ledger
            full_chars = sum(len(str(m)) for m in self._all_messages)
            actual_chars = sum(len(str(m)) for m in self._msg_cache)
            get_ledger().record(
                bucket="context_projection",
                layer="context_strategy",
                mechanism="compact_transcript",
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

    def _extract_tool_info(self, assistant_msg: dict, tool_call_id: str) -> tuple[str, str]:
        """Extract tool name and args summary from an assistant message by tool_call_id."""
        tool_calls = assistant_msg.get("tool_calls", [])
        for tc in tool_calls:
            if tc.get("id") == tool_call_id:
                func = tc.get("function", {})
                name = func.get("name", "?")
                args = func.get("arguments", "")
                # Summarize args: truncate long values
                if isinstance(args, str) and len(args) > self._args_summary_chars:
                    args = args[:self._args_summary_chars - 3] + "..."
                return name, args
        return "?", ""

    def _build_compact(self) -> list[dict]:
        """Build 2-message prompt from turn entries + recent results."""
        parts = [f"Task: {self._original_prompt}"]

        # Progress log
        if self._turn_entries:
            lines = []
            for e in self._turn_entries:
                reason = e.reasoning.replace("\n", " ").strip()
                if len(reason) > self._trace_reasoning_chars:
                    reason = reason[:self._trace_reasoning_chars - 3] + "..."
                if reason:
                    lines.append(
                        f"- T{e.turn}: \"{reason}\" → {e.tool_name}({e.args_summary}) {e.outcome}"
                    )
                else:
                    lines.append(
                        f"- T{e.turn}: {e.tool_name}({e.args_summary}) {e.outcome}"
                    )
            parts.append("Progress:\n" + "\n".join(lines))

        # Rolling tool-result window, char-budgeted newest-first.
        # Walk from the most recent result backward, keeping entries until
        # the char budget is exhausted; drop older ones permanently so the
        # deque stays bounded across long runs.
        if self._recent_results:
            kept_rev: list[str] = []
            chars_used = 0
            for content in reversed(self._recent_results):
                if chars_used + len(content) > self._recent_results_chars and kept_rev:
                    break
                kept_rev.append(content)
                chars_used += len(content)
            while len(self._recent_results) > len(kept_rev):
                self._recent_results.popleft()
            results = "\n---\n".join(reversed(kept_rev))
            parts.append(
                f"Last {len(kept_rev)} tool results (full, newest last):\n{results}"
            )

        parts.append(f"Turn: {self._turn_count}")

        return [
            {"role": "system", "content": self._system_content},
            {"role": "user", "content": "\n\n".join(parts)},
        ]


CONTEXT_MODE = "compact"
CONTEXT_CLASS = CompactTranscript
