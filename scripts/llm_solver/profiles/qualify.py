"""Qualification framework — reads scenario results, determines if a model can drive the harness.

Usage:
    python -m llm_solver.profiles.qualify profiles/qwen3-8b-q4/_samples
    python -m llm_solver.profiles.qualify --all       # all profiles with _samples/
"""
import argparse
import json
import sys
from pathlib import Path

PROFILES_DIR = Path(__file__).resolve().parent.parent.parent.parent / "profiles"

# Gate categories — a model must pass ALL gates in "required" to qualify.
# "diagnostic" gates are informational (gauntlets, edge cases).
GATES = {
    # Core: can the model produce structured tool calls at all?
    "required": [
        "single_tool_call",
        "edit_tool",
        "write_tool",
        "grep_tool",
        "glob_tool",
        "full_toolset",
        "simple_text",
        "tool_call_turn2",
        "chained_tool_calls",
        "three_step_chain",
        "system_prompt_handling",
        "tool_followup",
        "error_recovery",
    ],
    # Important but non-blocking — failures here mean the profile needs workarounds.
    "expected": [
        "behavioral_system_prompt",
        "behavioral_constraint_adherence",
        "complex_arguments",
        "empty_content",
        "language_mixing",
        "long_edit_strings",
        "long_output",
        "mid_context_recall",
        "multi_turn_tool_availability",
        "no_system_prompt",
        "phantom_tool_call",
        "python_tag_emission",
        "schema_confusion",
        "tool_call_as_text",
        "tool_call_in_content",
        "unclosed_think_tag",
        "thinking_elicitation",
        "thinking_suppression",
        "always_on_thinking",
        "system_prompt_with_reasoning_model",
        "text_with_tool_calls",
        "duplicate_tool_loop",
        "repetition_loop",
        "structured_output_code_fence",
        "grep_argument_swap",
        "invented_parameters",
        "missing_required_param",
    ],
    # Hard scenarios — informational only.
    "diagnostic": [
        "multi_tool_call",
        "parallel_full_toolset",
        "discover_and_edit",
        "gauntlet_pattern_wire",
        "gauntlet_root_cause",
        "gauntlet_verbatim_edit",
    ],
    # No meaningful eval criteria — just data collection.
    "observational": [
        "malformed_request",
    ],
}

# Flatten for lookup
_CATEGORY = {}
for cat, ids in GATES.items():
    for sid in ids:
        _CATEGORY[sid] = cat


def load_results(samples_dir: Path) -> list[dict]:
    p = samples_dir / "_all_results.json"
    if not p.exists():
        raise FileNotFoundError(f"No results at {p}")
    return json.loads(p.read_text())


def _check_detail(evaluated: dict, sid: str) -> dict:
    """Extract check-level detail for a scenario."""
    if sid in evaluated:
        ev = evaluated[sid]["evaluation"]
        passed_checks = sum(ev["checks"].values())
        total_checks = len(ev["checks"])
        return {
            "id": sid,
            "passed": ev["passed"],
            "passed_checks": passed_checks,
            "total_checks": total_checks,
            "score": f"{passed_checks}/{total_checks}",
        }
    return {"id": sid, "passed": None, "passed_checks": 0, "total_checks": 0, "score": "N/A"}


def _diagnostic_partial_credit(categories: dict) -> float:
    """Compute partial credit across diagnostic scenarios (0.0–1.0).

    Each diagnostic scenario contributes its check-level pass ratio,
    averaged across all diagnostic scenarios that were evaluated.
    """
    scored = [
        r for r in categories["diagnostic"]
        if r["total_checks"] > 0
    ]
    if not scored:
        return 0.0
    return sum(r["passed_checks"] / r["total_checks"] for r in scored) / len(scored)


def assign_tier(q: dict) -> tuple[int, str]:
    """Assign a tier based on qualification results.

    Tier 0: Failed — cannot drive the harness.
    Tier 1: Functional — passes required gate. Needs heavy profiling.
    Tier 2: Reliable — passes required + expected. Minimal profiling.
    Tier 3: Capable — reliable + meaningful diagnostic partial credit.
    """
    if not q["qualified"]:
        return 0, "failed"
    if q["expected"]["failures"] > 0:
        return 1, "functional"
    # Tier 3 threshold: >50% partial credit on diagnostic checks
    if q["diagnostic_credit"] > 0.5:
        return 3, "capable"
    return 2, "reliable"


def qualify(results: list[dict]) -> dict:
    """Produce qualification verdict from scenario results."""
    errors = [r for r in results if r.get("error")]
    evaluated = {r["scenario_id"]: r for r in results if r.get("evaluation") and not r["evaluation"].get("skipped")}

    # Per-category tallies
    categories = {}
    for cat in GATES:
        categories[cat] = [_check_detail(evaluated, sid) for sid in GATES[cat]]

    # Qualification gate: all required scenarios must pass eval
    required_pass = all(
        r["passed"] for r in categories["required"] if r["passed"] is not None
    )
    required_evaluated = any(
        r["passed"] is not None for r in categories["required"]
    )
    has_errors = len(errors)

    # Count failures per category
    def count_failures(cat):
        return sum(1 for r in categories[cat] if r["passed"] is False)

    qualified = required_pass and required_evaluated and has_errors == 0
    diag_credit = _diagnostic_partial_credit(categories)

    q = {
        "qualified": qualified,
        "errors": has_errors,
        "total_scenarios": len(results),
        "required": {
            "passed": required_pass and required_evaluated,
            "failures": [r["id"] for r in categories["required"] if r["passed"] is False],
            "detail": categories["required"],
        },
        "expected": {
            "failures": count_failures("expected"),
            "total": len(GATES["expected"]),
            "detail": categories["expected"],
        },
        "diagnostic": {
            "failures": count_failures("diagnostic"),
            "total": len(GATES["diagnostic"]),
            "detail": categories["diagnostic"],
        },
        "diagnostic_credit": round(diag_credit, 3),
    }
    q["tier"], q["tier_name"] = assign_tier(q)
    return q


def format_report(model: str, q: dict) -> str:
    tier_labels = {0: "FAILED", 1: "T1-FUNCTIONAL", 2: "T2-RELIABLE", 3: "T3-CAPABLE"}
    lines = []
    lines.append(f"{model}: {tier_labels[q['tier']]}")

    if q["errors"]:
        lines.append(f"  {q['errors']}/{q['total_scenarios']} scenarios had HTTP/runtime errors")

    # Required
    req = q["required"]
    if req["passed"]:
        lines.append(f"  required: ALL PASS ({len(GATES['required'])} scenarios)")
    else:
        lines.append(f"  required: FAIL — {req['failures']}")

    # Expected
    exp = q["expected"]
    if exp["failures"] == 0:
        lines.append(f"  expected: ALL PASS ({exp['total']} scenarios)")
    else:
        failed = [r["id"] for r in exp["detail"] if r["passed"] is False]
        lines.append(f"  expected: {exp['failures']}/{exp['total']} failed — {failed}")

    # Diagnostic
    diag = q["diagnostic"]
    lines.append(f"  diagnostic: {q['diagnostic_credit']:.0%} partial credit")
    for r in diag["detail"]:
        tag = "PASS" if r["passed"] else "FAIL" if r["passed"] is False else "N/A"
        lines.append(f"    {r['id']}: {tag} ({r['score']})")

    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Qualify models from scenario results")
    parser.add_argument("samples_dir", nargs="?", type=Path, help="Path to _samples/ directory")
    parser.add_argument("--all", action="store_true", help="Qualify all profiles with _samples/")
    parser.add_argument("--json", action="store_true", help="Output JSON instead of text")
    args = parser.parse_args(argv)

    if not args.all and not args.samples_dir:
        parser.error("Provide a samples_dir or --all")

    targets = []
    if args.all:
        for d in sorted(PROFILES_DIR.iterdir()):
            if d.is_dir() and d.name != "_base" and (d / "_samples" / "_all_results.json").exists():
                targets.append((d.name, d / "_samples"))
    else:
        model = args.samples_dir.parent.name if args.samples_dir.name == "_samples" else args.samples_dir.name
        targets.append((model, args.samples_dir))

    all_results = {}
    any_failed = False
    for model, samples_dir in targets:
        results = load_results(samples_dir)
        q = qualify(results)
        all_results[model] = q
        if not q["qualified"]:
            any_failed = True

    if args.json:
        # Strip detail from JSON output for readability
        slim = {}
        for model, q in all_results.items():
            slim[model] = {
                "tier": q["tier"],
                "tier_name": q["tier_name"],
                "qualified": q["qualified"],
                "errors": q["errors"],
                "required_pass": q["required"]["passed"],
                "required_failures": q["required"]["failures"],
                "expected_failures": q["expected"]["failures"],
                "expected_total": q["expected"]["total"],
                "diagnostic_credit": q["diagnostic_credit"],
            }
        print(json.dumps(slim, indent=2))
    else:
        for model, q in all_results.items():
            print(format_report(model, q))
            print()

    return 1 if any_failed else 0


if __name__ == "__main__":
    sys.exit(main())
