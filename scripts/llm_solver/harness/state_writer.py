"""Mechanical .solver/state.json writer — harness-side, not model-side.

The model never maintains `.solver/state.json`. The harness does. This module
projects a sequence of `.trace.jsonl` events into a content-blind state schema
that `SolverStateContext` reads back into the prompt.

Write path: rebuild-from-trace. On each invocation, re-read the full trace
file and rewrite state.json from scratch. No in-memory accumulator, no drift.
state.json is a *view* over `.trace.jsonl`, nothing more.

Schema (target of the projection, consumed by SolverStateContext):

    {
      "state":     {"current_attempt": str, "last_verify": str, "next_action": str},
      "trace":     [{"step": int, "session": int, "turn": int, "reasoning": str,
                     "action": str, "result": str, "next": str,
                     "gate_blocked": bool}, ...],
      "gates":     [],
      "evidence":  [{"step": int, "action": str, "result": str,
                     "verdict": "OK"|"FAIL", "gate_blocked": bool}, ...],
      "inference": []
    }

Content-blind by construction: the projection never inspects tool results
for task-format patterns (pytest nodeids, compiler error formats, lint
summary lines, or any other task-specific output shape). The only markers
it reads are harness-generated wire format: the `ERROR:` wrapper emitted
by `tools.py` on exception, the `[exit code: N]` suffix appended by
`bash()` on non-zero exit, and the `[harness gate]` prefix on gate-blocked
results. A harness that derived intelligence from task output would be
moving task-specific capability from the model into the loop.

Evidence population is filtered to bash calls because bash is the
subprocess execution surface where exit-code verdicts originate. Read,
write, edit, glob, grep return harness I/O status ("wrote N bytes",
"file not found"), not gate verdicts on task state. The filter is
structural (which tool was invoked), not content-based. The verdict
field is derived from the content-blind `classify_outcome`, which reads
only the harness's own exit-code marker and error wrapper — never
task output format.

The `reasoning` field is the model's pre-tool assistant text for that
turn. All trace entries within a single (session, turn) share the same
reasoning; renderers that care about deduplication group by turn.

`gates` and `inference` stay empty: neither has a content-blind population
rule today. They remain in the schema as protocol placeholders for the
model to read.

Replay usage (offline, against any historical trace):

    from scripts.llm_solver.harness.state_writer import project_from_trace
    state = project_from_trace(Path("results/.../repos/<task>/.trace.jsonl"))

Live usage (during a solve loop):

    from scripts.llm_solver.harness.state_writer import write_state_from_trace
    write_state_from_trace(repo_dir / ".trace.jsonl", repo_dir / ".solver" / "state.json")
"""
from __future__ import annotations

import json
from pathlib import Path

from .._shared.classification import classify_outcome, is_gate_blocked

# Per-entry cap for the `action` column. `action` is `tool(args_summary)`;
# args are already bounded by loop.py's _summarize_args, so this is a
# safety net, never hit in practice.
_MAX_ACTION_CHARS = 120

# Evidence result cap — tighter than the full trace result cap because
# evidence entries are rendered into the model's context window on every
# turn. Enough for a short tail of a failing verification run without
# blowing the budget. Rendering still applies the larger rolling window
# for the full raw output; evidence is the compressed index.
_MAX_EVIDENCE_CHARS = 500


def project(events: list[dict], *, max_result_chars: int) -> dict:
    """Project a list of trace events into the state.json schema.

    Deterministic, pure. Same input → same output. Content-blind.
    max_result_chars must be supplied by the caller (wired from
    cfg.max_output_chars) so the trace stores exactly what the model
    saw live.
    """
    state: dict = {}
    trace: list[dict] = []
    evidence: list[dict] = []

    step = 0
    for ev in events:
        et = ev.get("event")
        if et == "tool_call":
            step += 1
            tool = ev.get("tool_name") or "?"
            args = _truncate(ev.get("args_summary") or "", _MAX_ACTION_CHARS)
            result = _truncate(ev.get("result_summary") or "", max_result_chars)
            reasoning = ev.get("reasoning") or ""
            action = f"{tool}({args})"
            # gate_blocked: prefer the event field (set by loop.py) with
            # fallback to wire-format detection for old traces that lack
            # it. Recognising the harness-generated gate marker is not
            # task parsing — the harness wrote it.
            blocked = ev.get("gate_blocked", is_gate_blocked(result))
            trace.append({
                "step": step,
                "session": ev.get("session_number"),
                "turn": ev.get("turn_number"),
                "reasoning": reasoning,
                "action": action,
                "result": result,
                "next": "",
                "gate_blocked": blocked,
            })
            state["current_attempt"] = action
            # Evidence: every bash call that actually ran (not gate-blocked)
            # is a verification attempt. The verdict comes from the content-
            # blind classify_outcome, which reads only the harness's own
            # exit-code marker and ERROR: wrapper. No task-format parsing.
            # Filter to bash because bash is the subprocess boundary — other
            # tools are harness I/O, not gate verdicts on task state.
            if tool == "bash" and not blocked:
                evidence.append({
                    "step": step,
                    "action": action,
                    "result": _truncate(result, _MAX_EVIDENCE_CHARS),
                    "verdict": classify_outcome(result),
                    "gate_blocked": False,
                })
        elif et == "session_end":
            fr = ev.get("finish_reason") or "?"
            sn = ev.get("session_number")
            turns = ev.get("turns") or 0
            state["last_verify"] = f"session {sn} ended: {fr} after {turns} turns"
        # session_start: no state mutation.

    state.setdefault("current_attempt", "")
    state.setdefault("last_verify", "")
    state.setdefault("next_action", "")

    return {
        "state": state,
        "trace": trace,
        "gates": [],
        "evidence": evidence,
        "inference": [],
    }


def project_from_trace(trace_path: Path, *, max_result_chars: int) -> dict:
    """Load `.trace.jsonl` and project it. Missing file → empty schema."""
    trace_path = Path(trace_path)
    if not trace_path.is_file():
        return {"state": {"current_attempt": "", "last_verify": "", "next_action": ""},
                "trace": [], "gates": [], "evidence": [], "inference": []}
    events = []
    with open(trace_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            events.append(json.loads(line))
    return project(events, max_result_chars=max_result_chars)


def write_state_from_events(events: list[dict], state_path: Path, *,
                            max_result_chars: int) -> None:
    """Rebuild state.json from an in-memory list of trace events.

    Fast path used by the harness loop: Session accumulates trace
    entries in memory as it writes them to disk, so per-turn state
    refresh avoids a re-read + JSON parse of the whole trace file
    (which would scale O(T^2) in trace-length across a session).
    """
    state_path = Path(state_path)
    state = project(events, max_result_chars=max_result_chars)
    state_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = state_path.with_suffix(state_path.suffix + ".tmp")
    with open(tmp, "w") as f:
        json.dump(state, f, indent=2)
    tmp.replace(state_path)


def write_state_from_trace(trace_path: Path, state_path: Path, *,
                           max_result_chars: int) -> None:
    """Rebuild state.json from the current contents of `.trace.jsonl`.

    Slow path used at session boundaries and by any caller without an
    in-memory events list. Re-reads the full trace file each call;
    O(T) per invocation. Prefer write_state_from_events when a
    session-local events list is available.
    """
    trace_path = Path(trace_path)
    state_path = Path(state_path)
    state = project_from_trace(trace_path, max_result_chars=max_result_chars)
    state_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = state_path.with_suffix(state_path.suffix + ".tmp")
    tmp.write_text(json.dumps(state, indent=2))
    tmp.replace(state_path)


def _truncate(s: str, n: int) -> str:
    if len(s) <= n:
        return s
    return s[: n - 3] + "..."


__all__ = [
    "project",
    "project_from_trace",
    "write_state_from_trace",
]
