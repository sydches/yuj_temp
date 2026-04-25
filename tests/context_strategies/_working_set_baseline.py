"""Shared concise/yconcise baseline on top of WorkingSet.

The first concise/yconcise attempt overfit to memory compression and grew
multiple ad hoc prompt assemblers. This module replaces that with one
shared, explicit contract:

- keep only the state the model can act on now
- surface one blocking output body, not many competing payloads
- show the current file working set, not a rolling transcript of reads
- bound the prompt as a whole, not section-by-section with overlapping caps

Both concise variants share the same ingestion and rendering machinery.
They differ only in section labels and whether they consult
``.solver/state.json`` for state/trace/evidence continuity.
"""
from __future__ import annotations

import json
import re
import shlex
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from ..context import ContextManager, chars_div_4
from .._shell_patterns import TEST_COMMAND_RE as _TEST_COMMAND_RE
from ..._shared.classification import classify_outcome as _classify_outcome
from ._working_set import WorkingSet, GateSlot


_PATH_KEYS = ("path", "file_path")
_READ_TOOLS = frozenset({"read"})
_WRITE_TOOLS = frozenset({"edit", "write"})
_BASH_READ_RE = re.compile(
    r"^\s*(cat|head|tail|less|more|file)\s+([^\s|;&<>`$()]+)\s*$"
)
_ACTION_PATH_RE = re.compile(
    r"(?:path|file_path)='([^']+)'|"
    r"(?:path|file_path)=\"([^\"]+)\"|"
    r"\"(?:path|file_path)\"\s*:\s*\"([^\"]+)\""
)
_ACTION_CMD_RE = re.compile(
    r"cmd='([^']+)'|cmd=\"([^\"]+)\"|\"cmd\"\s*:\s*\"([^\"]+)\""
)
_INSPECT_CMD_PREFIXES = ("ls", "find", "grep", "rg", "fd", "tree", "cat", "head", "tail")
_PATH_SUFFIXES = (
    ".py", ".pyi", ".c", ".cc", ".cpp", ".cxx", ".h", ".hpp", ".hh",
    ".rs", ".go", ".java", ".js", ".jsx", ".ts", ".tsx",
)


@dataclass(frozen=True)
class TurnEntry:
    turn: int
    reasoning: str
    tool_name: str
    args_summary: str
    outcome: str


@dataclass(frozen=True)
class TraceRecord:
    turn: int
    reasoning: str
    action: str
    outcome: str


@dataclass(frozen=True)
class EvidenceRecord:
    turn: int
    action: str
    verdict: str
    content: str
    first_turn: int | None = None
    repeat_count: int = 0


@dataclass(frozen=True)
class SectionSpec:
    title: str
    weight: int
    renderer: Callable[[int], str]


class WorkingSetBaselineContext(ContextManager):
    """Shared action-oriented baseline for concise/yconcise modes."""

    def __init__(
        self,
        *,
        cwd: str,
        original_prompt: str,
        recent_results_chars: int,
        trace_reasoning_chars: int,
        min_turns: int,
        args_summary_chars: int,
        trace_lines: int | None = None,
        evidence_lines: int | None = None,
        suffix: str = "",
        use_solver_state: bool = False,
        style: str = "generic",
        contract: str = "baseline",
        inspect_repeat_threshold: int = 0,
        recovery_same_target_threshold: int = 0,
        recovery_verify_repeat_threshold: int = 0,
        slot_max_candidates: int = 1,
        slot_inline_files: int = 1,
        savings_mechanism: str,
        token_estimator: Callable[[list[dict]], int] = chars_div_4,
    ):
        super().__init__(token_estimator)
        self._cwd = Path(cwd)
        self._original_prompt = original_prompt
        self._char_budget = recent_results_chars
        self._trace_reasoning_chars = trace_reasoning_chars
        self._min_turns = min_turns
        self._args_summary_chars = args_summary_chars
        self._trace_lines = trace_lines
        self._evidence_lines = evidence_lines
        self._suffix = suffix
        self._use_solver_state = use_solver_state
        self._style = style
        self._contract = contract
        self._inspect_repeat_threshold = inspect_repeat_threshold
        self._recovery_same_target_threshold = recovery_same_target_threshold
        self._recovery_verify_repeat_threshold = recovery_verify_repeat_threshold
        self._slot_max_candidates = max(1, int(slot_max_candidates or 1))
        self._slot_inline_files = max(1, int(slot_inline_files or 1))
        self._savings_mechanism = savings_mechanism

        self._system_content: str = ""
        self._all_messages: list[dict] = []
        self._turn_entries: list[TurnEntry] = []
        self._ws = WorkingSet(cwd=self._cwd)
        self._turn_count = 0
        self._last_assistant_msg: dict | None = None
        self._prev_assistant_msg: dict | None = None

        self._msg_cache: list[dict] | None = None
        self._tok_cache: int | None = None
        self._raw_state_cache: dict | None = None

    # -- ingestion -------------------------------------------------

    def add_system(self, content: str) -> None:
        self._system_content = content
        self._all_messages.append({"role": "system", "content": content})
        self._invalidate()

    def add_user(self, content: str) -> None:
        self._all_messages.append({"role": "user", "content": content})
        self._invalidate()

    def add_assistant(self, message: dict) -> None:
        self._all_messages.append(message)
        self._last_assistant_msg = message
        self._turn_count += 1
        self._invalidate()

    def reset_dedup_counts(self) -> None:
        """No-op for interface parity with SolverStateContext."""
        return

    def add_tool_result(
        self,
        tool_call_id: str,
        content: str,
        *,
        tool_name: str = "",
        cmd_signature: str = "",
        gate_blocked: bool = False,
    ) -> None:
        self._all_messages.append({
            "role": "tool", "tool_call_id": tool_call_id, "content": content,
        })
        self._invalidate()

        assistant_msg = self._last_assistant_msg or self._prev_assistant_msg
        tool_args: dict = {}
        reasoning = ""
        args_summary = ""
        resolved_name = tool_name
        if assistant_msg is not None:
            if self._last_assistant_msg is not None:
                reasoning = self._last_assistant_msg.get("content") or ""
                self._prev_assistant_msg = self._last_assistant_msg
                self._last_assistant_msg = None
            resolved_name, args_summary, tool_args = self._extract_tool_info(
                assistant_msg, tool_call_id,
            )
            if not tool_name:
                tool_name = resolved_name

        outcome = "BLOCKED" if gate_blocked else _classify_outcome(content)

        if gate_blocked:
            pass
        elif tool_name in _READ_TOOLS:
            path = _pick_path(tool_args)
            if path and outcome == "OK":
                self._ws.record_read(path, content, self._turn_count)
            else:
                self._ws.record_artifact(tool_name, args_summary, content,
                                         self._turn_count)
        elif tool_name in _WRITE_TOOLS:
            path = _pick_path(tool_args)
            if path and outcome == "OK":
                self._ws.record_mutation(path, self._turn_count)
            else:
                self._ws.record_artifact(tool_name, args_summary, content,
                                         self._turn_count)
        elif tool_name == "bash":
            effective_sig = cmd_signature
            cmd_text = tool_args.get("cmd") if isinstance(tool_args, dict) else None
            if not effective_sig and isinstance(cmd_text, str) and cmd_text:
                effective_sig = json.dumps({"cmd": cmd_text}, sort_keys=True)
            cmd_display = _cmd_display(effective_sig, args_summary)
            cmd_str = _cmd_text(effective_sig)
            bash_path = _path_from_read_cmd(cmd_str)
            if bash_path and outcome == "OK":
                self._ws.record_read(bash_path, content, self._turn_count)
            elif effective_sig:
                self._ws.record_gate(effective_sig, cmd_display, content,
                                     self._turn_count, outcome)
            else:
                self._ws.record_artifact(tool_name or "?", args_summary, content,
                                         self._turn_count)
        else:
            self._ws.record_artifact(tool_name or "?", args_summary, content,
                                     self._turn_count)

        self._turn_entries.append(TurnEntry(
            turn=self._turn_count,
            reasoning=reasoning,
            tool_name=tool_name or resolved_name or "?",
            args_summary=args_summary,
            outcome=outcome,
        ))

    # -- projection -----------------------------------------------

    def get_messages(self) -> list[dict]:
        if self._msg_cache is not None:
            return self._msg_cache
        if self._turn_count < self._min_turns:
            self._msg_cache = self._all_messages
            return self._msg_cache
        self._msg_cache = self._build()
        from ..savings import get_ledger
        full_chars = sum(len(str(m)) for m in self._all_messages)
        actual_chars = sum(len(str(m)) for m in self._msg_cache)
        get_ledger().record(
            bucket="context_projection",
            layer="context_strategy",
            mechanism=self._savings_mechanism,
            input_chars=full_chars,
            output_chars=actual_chars,
            measure_type="exact",
            ctx={"turn_count": self._turn_count, "messages": len(self._msg_cache)},
        )
        return self._msg_cache

    def estimate_tokens(self) -> int:
        if self._tok_cache is None:
            self._tok_cache = self._token_estimator(self.get_messages())
        return self._tok_cache

    def message_count(self) -> int:
        return len(self._all_messages)

    def prepopulate_from_trace(self) -> int:
        state_path = self._cwd / ".solver" / "state.json"
        return self._ws.seed_from_state_trace(state_path, turn=0)

    # -- internal -------------------------------------------------

    def _invalidate(self) -> None:
        self._msg_cache = None
        self._tok_cache = None
        self._raw_state_cache = None

    def _extract_tool_info(
        self,
        assistant_msg: dict,
        tool_call_id: str,
    ) -> tuple[str, str, dict]:
        for tc in assistant_msg.get("tool_calls", []):
            if tc.get("id") != tool_call_id:
                continue
            func = tc.get("function", {})
            name = func.get("name", "?")
            raw = func.get("arguments", "")
            args_summary = raw if isinstance(raw, str) else json.dumps(raw)
            if len(args_summary) > self._args_summary_chars:
                args_summary = args_summary[: self._args_summary_chars - 3] + "..."
            parsed: dict = {}
            if isinstance(raw, dict):
                parsed = raw
            elif isinstance(raw, str) and raw.strip().startswith("{"):
                try:
                    parsed = json.loads(raw)
                except (ValueError, TypeError):
                    parsed = {}
            return name, args_summary, parsed
        return "?", "", {}

    def _load_state_json(self) -> dict:
        if self._raw_state_cache is not None:
            return self._raw_state_cache
        state_path = self._cwd / ".solver" / "state.json"
        if not state_path.is_file():
            self._raw_state_cache = {}
            return self._raw_state_cache
        try:
            self._raw_state_cache = json.loads(state_path.read_text())
        except (json.JSONDecodeError, OSError):
            self._raw_state_cache = {}
        return self._raw_state_cache

    def _has_solver_state(self) -> bool:
        return self._use_solver_state and bool(self._load_state_json())

    def _state_text(self, max_chars: int) -> str:
        if self._contract == "slot":
            return self._slot_state_text(max_chars)
        lines: list[str] = []
        state_block = self._load_state_json().get("state") if self._has_solver_state() else None
        if isinstance(state_block, dict):
            for key in ("current_attempt", "last_verify", "next_action"):
                value = state_block.get(key)
                if value:
                    label = key.replace("_", " ").capitalize()
                    lines.append(f"{label}: {value}")

        lines.append("Working root: . (current directory already set)")
        lines.append(f"Phase: {self._phase_text()}")

        blocker = self._blocking_record()
        if blocker is not None:
            lines.append(f"Blocking command: {self._summary_line(blocker)}")

        focus = self._focus_files_text()
        if focus:
            lines.append(f"Focus files: {focus}")

        test_target = self._test_target_text()
        if test_target:
            lines.append(f"Test target: {test_target}")

        obligation = self._obligation_text()
        if obligation:
            lines.append(f"Next obligation: {obligation}")

        last_action = self._last_action_text()
        if last_action and not any(line.startswith("Current attempt:") for line in lines):
            lines.append(f"Last action: {last_action}")

        changed = self._format_path_list(self._changed_paths())
        if changed:
            lines.append(f"Files changed: {changed}")

        in_view = self._format_path_list(self._visible_paths())
        if in_view:
            lines.append(f"Files in view: {in_view}")

        if not lines or not any(line.startswith("Turn:") for line in lines):
            lines.append(f"Turn: {self._turn_count}")
        return _fit_lines(lines, max_chars)

    def _phase_text(self) -> str:
        recovery = self._recovery_state()
        if recovery is not None:
            return "recovery"
        changed = self._changed_paths()
        visible = self._focus_candidates()
        blocker = self._blocking_record()
        repeated = self._latest_repeated_trace_run()
        needs_test = self._needs_test_read()
        if changed:
            if blocker is not None and blocker.verdict.startswith("FAIL"):
                return "verify the latest change against the active blocker"
            return "verify or refine the latest change"
        if repeated is not None:
            return "leave inspection and choose a concrete file or check"
        if needs_test:
            return "read the focused test before another verification run"
        if blocker is not None and visible:
            return "prepare a targeted edit"
        if blocker is not None:
            return "investigate the active blocker"
        if visible:
            return "inspect files in view"
        return "orient"

    def _slot_state_text(self, max_chars: int) -> str:
        recovery = self._recovery_state()
        lines: list[str] = [f"repo_root: .", f"phase: {self._phase_text()}"]

        candidate_source = self._format_path_list(self._candidate_source_paths(), limit=self._slot_max_candidates)
        candidate_test = self._format_path_list(self._candidate_test_paths(), limit=self._slot_max_candidates)
        edited = self._format_path_list(self._changed_paths(), limit=1)
        last_verdict = self._last_verdict_text()
        disallowed = self._disallowed_repeat_text()

        if recovery is not None:
            reason, target = recovery
            lines.append(f"stuck_reason: {reason}")
            if target:
                lines.append(f"focused_target: {target}")
            if candidate_source:
                lines.append(f"candidate_source: {candidate_source}")
            if candidate_test:
                lines.append(f"candidate_test: {candidate_test}")
            if last_verdict:
                lines.append(f"last_verdict: {last_verdict}")
            lines.append(
                "allowed_moves: read a concrete file | edit/write | run verification"
            )
            return _fit_lines(lines, max_chars)

        if candidate_source:
            lines.append(f"candidate_source: {candidate_source}")
        if candidate_test:
            lines.append(f"candidate_test: {candidate_test}")
        if edited:
            lines.append(f"edited_file: {edited}")
        if last_verdict:
            lines.append(f"last_verdict: {last_verdict}")
        if disallowed:
            lines.append(f"disallowed_repeat: {disallowed}")
        lines.append(f"next_action: {self._slot_next_action_text()}")
        return _fit_lines(lines, max_chars)

    def _focus_candidates(self) -> list[str]:
        focus = self._changed_paths()
        if focus:
            return focus
        focus = self._visible_paths()
        if focus:
            return focus
        return self._recent_action_targets()

    def _candidate_source_paths(self) -> list[str]:
        candidates: list[str] = []
        for path in self._focus_candidates():
            if (
                _looks_like_test_path(path)
                or path in candidates
                or not self._is_repo_file_candidate(path)
            ):
                continue
            candidates.append(path)
            if len(candidates) >= self._slot_max_candidates:
                break
        return candidates

    def _candidate_test_paths(self) -> list[str]:
        targets: list[str] = []
        for target in self._candidate_test_targets():
            if target in targets:
                continue
            targets.append(target)
            if len(targets) >= self._slot_max_candidates:
                break
        return targets

    def _candidate_test_targets(self) -> list[str]:
        targets: list[str] = []
        for rec in reversed(self._trace_records()):
            target = _extract_test_target_from_action(rec.action)
            if not target or target in targets:
                continue
            targets.append(target)
            if len(targets) >= max(2, self._slot_max_candidates):
                break
        return list(reversed(targets))

    def _focus_files_text(self) -> str:
        return self._format_path_list(self._focus_candidates(), limit=3)

    def _obligation_text(self) -> str:
        changed = self._changed_paths()
        focus = self._focus_files_text()
        blocker = self._blocking_record()
        test_target = self._test_target_text()
        repeated = self._latest_repeated_trace_run()
        if changed and blocker is not None:
            return (
                f"verify or extend {focus} against the blocker before more exploration"
                if focus else
                "verify or extend the changed file before more exploration"
            )
        if changed:
            return (
                f"run the next focused verification on {focus}"
                if focus else
                "run the next focused verification"
            )
        if repeated is not None:
            repeated_target = self._repeated_target_text(repeated[0].action)
            if test_target and self._needs_test_read():
                return f"stop repeating {repeated_target}; read {test_target} before more checks"
            if focus:
                return f"stop repeating {repeated_target}; read or edit one focus target ({focus})"
            return f"stop repeating {repeated_target}; choose a new target or make an edit"
        if test_target and self._needs_test_read():
            return f"read {test_target} before another verification run"
        if blocker is not None and focus:
            return f"edit one focus file ({focus}) or change checks; do not repeat the same blocker unchanged"
        if blocker is not None:
            return "pick one concrete target before repeating checks"
        if focus:
            return f"choose one focus file ({focus}) and make the next concrete move"
        return "identify one concrete file or command target"

    def _last_action_text(self) -> str:
        if self._turn_entries:
            e = self._turn_entries[-1]
            return f"{e.tool_name}({e.args_summary}) → {e.outcome}"
        latest = self._latest_evidence_record()
        if latest is not None:
            return f"{latest.action} → {latest.verdict}"
        return ""

    def _changed_paths(self) -> list[str]:
        changed = [slot for slot in self._ws.files.values() if slot.epoch > 0]
        changed.sort(key=lambda slot: slot.last_access_turn, reverse=True)
        return [slot.path for slot in changed]

    def _visible_paths(self) -> list[str]:
        visible = sorted(
            self._ws.files.values(),
            key=lambda slot: slot.last_access_turn,
            reverse=True,
        )
        return [slot.path for slot in visible]

    def _recent_action_targets(self) -> list[str]:
        targets: list[str] = []
        for rec in reversed(self._trace_records()):
            target = _extract_action_target(rec.action)
            if not target or target in targets:
                continue
            targets.append(target)
            if len(targets) >= 4:
                break
        return targets

    def _test_target_text(self) -> str:
        return self._format_path_list(self._candidate_test_targets(), limit=2)

    def _is_repo_file_candidate(self, path: str) -> bool:
        if not path or path in {".", ".."}:
            return False
        if path.endswith("/"):
            return False
        candidate = Path(path)
        if candidate.is_absolute():
            try:
                candidate.relative_to(self._cwd)
            except ValueError:
                return False
        else:
            candidate = (self._cwd / candidate).resolve(strict=False)
        try:
            if candidate.exists():
                return candidate.is_file()
        except OSError:
            return False
        name = candidate.name
        return bool(name) and "." in name and not name.startswith(".")

    def _needs_test_read(self) -> bool:
        test_target = self._test_target_text()
        if not test_target:
            return False
        seen_paths = self._changed_paths() + self._visible_paths()
        return not any(_looks_like_test_path(path) for path in seen_paths)

    def _last_verdict_text(self) -> str:
        rec = self._blocking_record() or self._latest_evidence_record()
        if rec is None:
            return ""
        return self._summary_line(rec)

    def _disallowed_repeat_text(self) -> str:
        recovery = self._recovery_state()
        if recovery is not None:
            reason, target = recovery
            return f"{reason}: {target}" if target else reason
        repeated = self._latest_repeated_trace_run()
        if repeated is None:
            return ""
        return self._repeated_target_text(repeated[0].action)

    def _slot_next_action_text(self) -> str:
        candidate_source = self._format_path_list(
            self._candidate_source_paths(),
            limit=self._slot_max_candidates,
        )
        candidate_test = self._format_path_list(
            self._candidate_test_paths(),
            limit=self._slot_max_candidates,
        )
        changed = self._format_path_list(self._changed_paths(), limit=1)
        if changed:
            return f"run verification on {changed} or refine it"
        if candidate_test and self._needs_test_read():
            if candidate_source:
                return f"read {candidate_test} or edit {candidate_source}"
            return f"read {candidate_test} or run verification"
        if candidate_source:
            if candidate_test:
                return f"edit {candidate_source}, read {candidate_test}, or run verification"
            return f"edit {candidate_source} or run verification"
        if candidate_test:
            return f"read {candidate_test} or run verification"
        return "read one concrete file, edit/write, or run verification"

    def _repeated_verify_run(self) -> tuple[TraceRecord, int, int] | None:
        threshold = int(self._recovery_verify_repeat_threshold or 0)
        if threshold <= 1:
            return None
        records = self._trace_records()
        i = 0
        latest: tuple[TraceRecord, int, int] | None = None
        while i < len(records):
            rec = records[i]
            j = i + 1
            while (
                j < len(records)
                and records[j].action == rec.action
                and records[j].outcome == rec.outcome
            ):
                j += 1
            run_len = j - i
            if run_len >= threshold and _extract_test_target_from_action(rec.action):
                latest = (rec, run_len, records[j - 1].turn)
            i = j
        return latest

    def _recovery_state(self) -> tuple[str, str] | None:
        verify_repeat = self._repeated_verify_run()
        if verify_repeat is not None:
            target = _extract_test_target_from_action(verify_repeat[0].action)
            return ("repeated verification without refinement", target)
        repeated = self._latest_repeated_trace_run()
        if repeated is None:
            return None
        if self._recovery_same_target_threshold > 0 and repeated[1] >= self._recovery_same_target_threshold:
            target = self._repeated_target_text(repeated[0].action)
            if target.startswith("/") or " under /" in target:
                return ("repeated inspection outside repo root", target)
            return ("repeated same-target inspection", target)
        return None

    def _latest_repeated_trace_run(self) -> tuple[TraceRecord, int, int] | None:
        threshold = max(
            int(self._inspect_repeat_threshold or 0),
            int(self._recovery_same_target_threshold or 0),
        )
        if threshold <= 1:
            return None
        records = self._trace_records()
        i = 0
        latest: tuple[TraceRecord, int, int] | None = None
        while i < len(records):
            rec = records[i]
            j = i + 1
            while (
                j < len(records)
                and records[j].action == rec.action
                and records[j].outcome == rec.outcome
            ):
                j += 1
            run_len = j - i
            if run_len >= threshold and _is_inspection_action(rec.action):
                latest = (rec, run_len, records[j - 1].turn)
            i = j
        return latest

    def _repeated_target_text(self, action: str) -> str:
        target = _extract_action_target(action)
        if target:
            return target
        return action

    def _format_path_list(self, paths: list[str], limit: int = 4) -> str:
        if not paths:
            return ""
        head = paths[:limit]
        suffix = ""
        if len(paths) > limit:
            suffix = f" (+{len(paths) - limit} more)"
        return ", ".join(head) + suffix

    def _trace_records(self) -> list[TraceRecord]:
        if self._has_solver_state():
            records: list[TraceRecord] = []
            for entry in self._load_state_json().get("trace", []):
                if not isinstance(entry, dict):
                    continue
                turn = entry.get("turn")
                try:
                    turn_num = int(turn)
                except (TypeError, ValueError):
                    turn_num = self._turn_count
                result = str(entry.get("result", ""))
                gate_blocked = bool(entry.get("gate_blocked"))
                outcome = "BLOCKED" if gate_blocked else _classify_outcome(result)
                records.append(TraceRecord(
                    turn=turn_num,
                    reasoning=str(entry.get("reasoning", "") or ""),
                    action=str(entry.get("action", "") or "?"),
                    outcome=outcome,
                ))
            return records

        return [
            TraceRecord(
                turn=e.turn,
                reasoning=e.reasoning,
                action=f"{e.tool_name}({e.args_summary})",
                outcome=e.outcome,
            )
            for e in self._turn_entries
        ]

    def _evidence_records(self) -> list[EvidenceRecord]:
        if self._has_solver_state():
            latest_by_action: dict[str, EvidenceRecord] = {}
            for item in self._load_state_json().get("evidence", []):
                if not isinstance(item, dict):
                    continue
                action = str(item.get("action", "") or "?")
                step = item.get("step")
                try:
                    turn = int(step)
                except (TypeError, ValueError):
                    turn = self._turn_count
                latest_by_action[action] = EvidenceRecord(
                    turn=turn,
                    action=action,
                    verdict=str(item.get("verdict", "") or "OK"),
                    content=str(item.get("result", "") or ""),
                    first_turn=turn,
                    repeat_count=0,
                )
            return sorted(latest_by_action.values(), key=lambda rec: rec.turn)

        records: list[EvidenceRecord] = []
        for gate in self._ws.gate_latest.values():
            records.append(EvidenceRecord(
                turn=gate.turn,
                action=gate.cmd_display,
                verdict=gate.verdict,
                content=gate.content,
                first_turn=gate.first_turn,
                repeat_count=gate.repeat_count,
            ))
        records.sort(key=lambda rec: rec.turn)
        return records

    def _latest_evidence_record(self) -> EvidenceRecord | None:
        records = self._evidence_records()
        if not records:
            return None
        return max(records, key=lambda rec: rec.turn)

    def _blocking_record(self) -> EvidenceRecord | None:
        records = self._evidence_records()
        fails = [rec for rec in records if rec.verdict.startswith("FAIL")]
        if fails:
            return max(fails, key=lambda rec: rec.turn)
        if records:
            return max(records, key=lambda rec: rec.turn)
        return None

    def _gate_payload_text(self, max_chars: int) -> str:
        rec = self._blocking_record()
        if rec is None:
            return ""
        header = f"{rec.action} → {rec.verdict} (T{rec.turn})"
        if max_chars <= len(header):
            return header
        body_budget = max_chars - len(header) - 1
        body = _truncate_text(rec.content, body_budget)
        return f"{header}\n{body}" if body else header

    def _summary_line(self, rec: EvidenceRecord) -> str:
        if rec.repeat_count > 0 and rec.first_turn is not None:
            suffix = " (unchanged)" if rec.repeat_count >= 2 else ""
            return (
                f"T{rec.first_turn}-{rec.turn}: {rec.action} → {rec.verdict} "
                f"×{rec.repeat_count + 1}{suffix}"
            )
        return f"T{rec.turn}: {rec.action} → {rec.verdict}"

    def _checks_text(self, max_chars: int) -> str:
        records = self._evidence_records()
        if not records:
            return ""
        fails = [self._summary_line(rec) for rec in records if rec.verdict.startswith("FAIL")]
        passes = [self._summary_line(rec) for rec in records if not rec.verdict.startswith("FAIL")]
        lines = fails + passes
        return _fit_lines(lines, max_chars, max_lines=self._evidence_lines)

    def _evidence_text(self, max_chars: int) -> str:
        records = self._evidence_records()
        if not records:
            return ""
        fails = [self._summary_line(rec) for rec in records if rec.verdict.startswith("FAIL")]
        passes = [self._summary_line(rec) for rec in records if not rec.verdict.startswith("FAIL")]
        lines: list[str] = []
        if fails:
            lines.append("-- unresolved --")
            lines.extend(fails[-self._evidence_lines:] if self._evidence_lines else fails)
        if passes:
            lines.append("-- resolved --")
            lines.extend(passes[-self._evidence_lines:] if self._evidence_lines else passes)
        return _fit_lines(lines, max_chars)

    def _trace_text(self, max_chars: int) -> str:
        records = self._trace_records()
        if not records:
            return ""

        blocks: list[str] = []
        prev_reasoning: str | None = None
        i = 0
        while i < len(records):
            rec = records[i]
            j = i + 1
            while (
                j < len(records)
                and records[j].action == rec.action
                and records[j].outcome == rec.outcome
            ):
                j += 1
            run_len = j - i
            if self._style == "yuj":
                blocks.append(
                    self._trace_block_yuj(
                        rec,
                        prev_reasoning=prev_reasoning,
                        run_len=run_len,
                        last_turn=records[j - 1].turn,
                    )
                )
                reasoning = _clean_reasoning(rec.reasoning, self._trace_reasoning_chars)
                if reasoning:
                    prev_reasoning = reasoning
            else:
                blocks.append(
                    self._trace_block_generic(
                        rec,
                        run_len=run_len,
                        last_turn=records[j - 1].turn,
                    )
                )
            i = j

        max_entries = self._trace_lines
        return _fit_blocks(blocks, max_chars, max_entries=max_entries)

    def _trace_block_generic(self, rec: TraceRecord, *, run_len: int, last_turn: int) -> str:
        reasoning = _clean_reasoning(rec.reasoning, self._trace_reasoning_chars)
        if run_len > 1:
            return f"- T{rec.turn}-{last_turn}: {rec.action} ×{run_len} → {rec.outcome}"
        if reasoning:
            return f"- T{rec.turn}: \"{reasoning}\" → {rec.action} {rec.outcome}"
        return f"- T{rec.turn}: {rec.action} {rec.outcome}"

    def _trace_block_yuj(
        self,
        rec: TraceRecord,
        *,
        prev_reasoning: str | None,
        run_len: int,
        last_turn: int,
    ) -> str:
        lines: list[str] = []
        reasoning = _clean_reasoning(rec.reasoning, self._trace_reasoning_chars)
        if reasoning and reasoning != prev_reasoning:
            lines.append(f"T{rec.turn} [{reasoning}]")
        if run_len > 1:
            lines.append(
                f"    → {rec.action} ×{run_len} (T{rec.turn}-{last_turn}) → {rec.outcome}"
            )
        else:
            lines.append(f"    → {rec.action} → {rec.outcome}")
        return "\n".join(lines)

    def _files_text(self, max_chars: int) -> str:
        if self._contract == "slot":
            selected = self._candidate_source_paths()[: self._slot_inline_files]
            if not selected:
                selected = self._changed_paths()[: self._slot_inline_files]
            rendered, elided = self._ws.project_selected_files(selected, max_chars)
            if not rendered and not elided:
                return ""
            lines = [rendered] if rendered else []
            if elided:
                lines.append("Files elided for budget: " + ", ".join(elided))
            return "\n".join(lines)
        rendered, elided = self._ws.project_files(max_chars)
        if not rendered and not elided:
            return ""
        lines = [rendered] if rendered else []
        if elided:
            lines.append("Files elided for budget: " + ", ".join(elided))
        return "\n".join(lines)

    def _artifacts_text(self, max_chars: int) -> str:
        return self._ws.project_artifacts(max_chars)

    def _generic_sections(self) -> list[SectionSpec]:
        if self._contract == "slot":
            return [
                SectionSpec("State:", 40, self._state_text),
                SectionSpec("Candidate file:", 38, self._files_text),
                SectionSpec("Blocking output:", 22, self._gate_payload_text),
            ]
        return [
            SectionSpec("State:", 18, self._state_text),
            SectionSpec("Blocking output:", 22, self._gate_payload_text),
            SectionSpec("Files (current content):", 30, self._files_text),
            SectionSpec("Checks:", 15, self._checks_text),
            SectionSpec("Progress:", 10, self._trace_text),
            SectionSpec("Recent outputs:", 5, self._artifacts_text),
        ]

    def _yuj_sections(self) -> list[SectionSpec]:
        if self._contract == "slot":
            return [
                SectionSpec("=== State ===", 40, self._state_text),
                SectionSpec("=== Candidate File ===", 38, self._files_text),
                SectionSpec("=== Gate (blocking) ===", 22, self._gate_payload_text),
            ]
        return [
            SectionSpec("=== State ===", 18, self._state_text),
            SectionSpec("=== Gate (blocking) ===", 22, self._gate_payload_text),
            SectionSpec("=== Evidence ===", 15, self._evidence_text),
            SectionSpec("=== Files ===", 30, self._files_text),
            SectionSpec("=== Trace ===", 15, self._trace_text),
            SectionSpec("=== Recent outputs ===", 5, self._artifacts_text),
        ]

    def _build(self) -> list[dict]:
        parts = [f"Task: {self._original_prompt}"]
        sections = self._yuj_sections() if self._style == "yuj" else self._generic_sections()
        remaining = self._char_budget
        remaining_weight = sum(section.weight for section in sections)

        for idx, section in enumerate(sections):
            if remaining <= 0:
                break
            if idx == len(sections) - 1 or remaining_weight <= 0:
                allocated = remaining
            else:
                allocated = max(256, int(remaining * section.weight / remaining_weight))
                allocated = min(remaining, allocated)
            text = section.renderer(allocated)
            remaining_weight -= section.weight
            if not text:
                continue
            parts.append(f"{section.title}\n{text}")
            remaining -= min(len(text), allocated)

        if self._suffix:
            parts.append(self._suffix)

        return [
            {"role": "system", "content": self._system_content},
            {"role": "user", "content": "\n\n".join(parts)},
        ]


def _pick_path(args: dict) -> str:
    for key in _PATH_KEYS:
        value = args.get(key)
        if isinstance(value, str) and value:
            return value
    return ""


def _path_from_read_cmd(cmd: str) -> str:
    if not isinstance(cmd, str):
        return ""
    match = _BASH_READ_RE.match(cmd)
    if not match:
        return ""
    return match.group(2)


def _cmd_display(cmd_sig: str, fallback: str) -> str:
    if not cmd_sig:
        return fallback
    try:
        blob = json.loads(cmd_sig)
        cmd = blob.get("cmd") if isinstance(blob, dict) else None
        if isinstance(cmd, str) and cmd:
            return cmd
    except (ValueError, TypeError):
        pass
    return fallback or cmd_sig


def _cmd_text(cmd_sig: str) -> str:
    try:
        blob = json.loads(cmd_sig)
        if isinstance(blob, dict):
            cmd = blob.get("cmd")
            if isinstance(cmd, str):
                return cmd
    except (ValueError, TypeError):
        pass
    return ""


def _looks_like_test_path(path: str) -> bool:
    return bool(path) and "test" in path.lower()


def _extract_action_target(action: str) -> str:
    match = _ACTION_PATH_RE.search(action)
    if match:
        return next((group for group in match.groups() if group), "")
    cmd_match = _ACTION_CMD_RE.search(action)
    if not cmd_match:
        return ""
    cmd = next((group for group in cmd_match.groups() if group), "")
    return _extract_focus_target_from_command(cmd)


def _extract_test_target_from_action(action: str) -> str:
    match = _ACTION_PATH_RE.search(action)
    if match:
        path = next((group for group in match.groups() if group), "")
        return path if _looks_like_test_path(path) else ""
    cmd_match = _ACTION_CMD_RE.search(action)
    if not cmd_match:
        return ""
    cmd = next((group for group in cmd_match.groups() if group), "")
    return _extract_test_target_from_command(cmd)


def _extract_test_target_from_command(cmd: str) -> str:
    if not _TEST_COMMAND_RE.search(cmd or ""):
        return ""
    try:
        tokens = shlex.split(cmd, posix=True)
    except ValueError:
        return ""
    for token in tokens:
        candidate = token.split("::", 1)[0].rstrip(",")
        if _looks_like_test_path(candidate) and _looks_like_path_token(candidate):
            return candidate
    return ""


def _extract_focus_target_from_command(cmd: str) -> str:
    test_target = _extract_test_target_from_command(cmd)
    if test_target:
        return test_target
    read_target = _path_from_read_cmd(cmd)
    if read_target:
        return read_target
    try:
        tokens = shlex.split(cmd, posix=True)
    except ValueError:
        return ""
    if not tokens:
        return ""
    name = tokens[0]
    rest = tokens[1:]
    if name in {"ls", "tree", "du"}:
        for token in reversed(rest):
            if _looks_like_path_token(token):
                return token
        return "."
    if name == "find":
        root = "."
        pattern = ""
        for i, token in enumerate(rest):
            if token in {"-name", "-iname", "-path", "-wholename"} and i + 1 < len(rest):
                pattern = rest[i + 1]
            if token.startswith("-"):
                continue
            root = token
            break
        if pattern:
            return f"{pattern} under {root}"
        return root
    if name in {"grep", "rg", "fd"}:
        for token in reversed(rest):
            if _looks_like_path_token(token):
                return token
    return ""


def _looks_like_path_token(token: str) -> bool:
    if not token or token.startswith("-"):
        return False
    candidate = token.split("::", 1)[0].rstrip(",")
    if candidate in {".", "..", "tests", "test"}:
        return True
    if "/" in candidate or candidate.startswith("/") or candidate.endswith("/"):
        return True
    if candidate.lower().startswith("test"):
        return True
    return candidate.endswith(_PATH_SUFFIXES)


def _is_inspection_action(action: str) -> bool:
    if action.startswith(("read(", "glob(", "grep(")):
        return True
    cmd_match = _ACTION_CMD_RE.search(action)
    if not cmd_match:
        return False
    cmd = next((group for group in cmd_match.groups() if group), "")
    return any(cmd.startswith(prefix) for prefix in _INSPECT_CMD_PREFIXES)


def _clean_reasoning(reasoning: str, limit: int) -> str:
    short = (reasoning or "").replace("\n", " ").strip()
    if len(short) > limit:
        return short[: limit - 3] + "..."
    return short


def _truncate_text(text: str, max_chars: int) -> str:
    if max_chars <= 0:
        return ""
    if len(text) <= max_chars:
        return text
    if max_chars <= 24:
        return text[:max_chars]
    return text[: max_chars - 20] + f"\n[... +{len(text) - max_chars + 20} chars]"


def _fit_lines(lines: list[str], max_chars: int, max_lines: int | None = None) -> str:
    if not lines or max_chars <= 0:
        return ""
    chosen = lines[-max_lines:] if max_lines else lines
    kept_rev: list[str] = []
    used = 0
    for line in reversed(chosen):
        add = len(line) + (1 if kept_rev else 0)
        if used + add > max_chars and kept_rev:
            break
        kept_rev.append(line)
        used += add
    return "\n".join(reversed(kept_rev))


def _fit_blocks(blocks: list[str], max_chars: int, max_entries: int | None = None) -> str:
    if not blocks or max_chars <= 0:
        return ""
    chosen = blocks[-max_entries:] if max_entries else blocks
    kept_rev: list[str] = []
    used = 0
    for block in reversed(chosen):
        add = len(block) + (2 if kept_rev else 0)
        if used + add > max_chars and kept_rev:
            break
        kept_rev.append(block)
        used += add
    return "\n".join(reversed(kept_rev))
