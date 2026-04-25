"""Content-blind classifiers for tool results.

The harness must not derive intelligence from task output. These helpers
observe only harness-generated markers (the ERROR: wrapper emitted by
tools.py and the [exit code: N] suffix appended by bash()). They never
inspect task content.

Task-format-aware helpers (pytest detection, test-verdict parsing) live
in scripts/llm_solver/analysis/_task_format.py and are available only to
forensic analysis tools, never to the harness.
"""


def classify_outcome(result: str) -> str:
    """Return "OK" or "FAIL" from harness-generated markers only.

    Content-blind: inspects only strings the harness itself writes into
    the tool result. Never parses task output.

    Markers:
      - Empty result                               → OK
      - result starts with "ERROR:"                → FAIL (tools.py wraps
                                                     tool exceptions this way)
      - "[exit code: N]" present, N != 0          → FAIL (bash() appends
                                                     this suffix on non-zero)
      - Otherwise                                  → OK
    """
    if not result:
        return "OK"
    if result.startswith("ERROR:"):
        return "FAIL"
    if "[exit code:" in result and "[exit code: 0]" not in result:
        return "FAIL"
    return "OK"


def is_gate_blocked(result: str) -> bool:
    """True if the result is a harness gate message (tool was not executed)."""
    return result.startswith("[harness gate]")


__all__ = ["classify_outcome", "is_gate_blocked"]
