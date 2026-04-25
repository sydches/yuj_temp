"""Bash command rewriting and output filtering.

Two layers of transforms, loaded from separate configs:

1. **Universal rewrites** (rewrites.toml): pip -q, npm --loglevel=error,
   make -s, etc. Apply to every run regardless of task format. Reduce output
   volume from noisy commands.

2. **Task-format transforms** (language_quirks/*.toml [output_control]): test
   runner flags (--tb=short) and output condensation (strip PASSED lines).
   Apply only when a task format is configured.

The harness core calls:
    rewrite_command(cmd, oc, universal_rewrites) -> cmd
    condense_output(output, cmd, oc) -> output
"""
from __future__ import annotations

import functools
import logging
import re
from dataclasses import dataclass
from pathlib import Path

log = logging.getLogger(__name__)

# ── Universal rewrites ──────────────────────────────────────────────────

@dataclass(frozen=True)
class RewriteRule:
    """One command rewrite rule from rewrites.toml."""
    name: str
    pattern: re.Pattern
    flag: str
    skip_if: tuple[str, ...]


def load_universal_rewrites(path: Path | None = None) -> list[RewriteRule]:
    """Load [[rewrite]] entries from rewrites.toml.

    Defaults to the rewrites.toml next to this module.
    Returns empty list if file is missing.
    """
    from .._shared.toml_compat import tomllib

    if path is None:
        path = Path(__file__).parent / "rewrites.toml"
    if not path.is_file():
        return []

    with open(path, "rb") as f:
        data = tomllib.load(f)

    rules = []
    for entry in data.get("rewrite", []):
        rules.append(RewriteRule(
            name=entry["name"],
            pattern=re.compile(entry["pattern"], re.IGNORECASE),
            flag=entry["flag"],
            skip_if=tuple(entry.get("skip_if", [])),
        ))
    return rules


# ── Task-format output control ──────────────────────────────────────────

@dataclass(frozen=True)
class OutputControl:
    """Runner-specific output control loaded from language_quirks/*.toml."""
    failure_only_flag: str
    passed_marker: str
    failed_marker: str
    verification_patterns: tuple[re.Pattern, ...]


@dataclass(frozen=True)
class OutputParser:
    """Runner-specific structured-output parser loaded from language_quirks/*.toml.

    Each runner's TOML may declare:
      [output_parser.summary]   — field: passed / failed / errors (each a regex)
      [output_parser.per_test]
        regex with named captures: test_id, verdict

    Using one regex per summary field (rather than a single alternation)
    makes parsing order-agnostic: pytest sometimes prints
    "2 failed, 8 passed" and sometimes "8 passed, 2 failed", and either
    needs to work. Per-test uses a single regex because the format is
    uniform (verdict + test_id at line start).

    All fields are optional — absent keys simply produce partial
    parses.
    """
    summary_fields: dict[str, re.Pattern]
    per_test_regex: re.Pattern | None


def load_output_control(task_format_path) -> OutputControl | None:
    """Load [output_control] from a task format TOML file.

    Returns None if the file has no [output_control] section.
    """
    from .._shared.toml_compat import tomllib

    path = Path(task_format_path)
    if not path.is_file():
        return None

    with open(path, "rb") as f:
        data = tomllib.load(f)

    oc = data.get("output_control")
    if not oc:
        return None

    patterns = tuple(
        re.compile(p, re.IGNORECASE | re.VERBOSE)
        for p in data.get("verification_patterns", [])
    )

    return OutputControl(
        failure_only_flag=oc.get("failure_only_flag", ""),
        passed_marker=oc.get("passed_marker", ""),
        failed_marker=oc.get("failed_marker", ""),
        verification_patterns=patterns,
    )


def load_output_parser(task_format_path) -> OutputParser | None:
    """Load [output_parser] from a task format TOML file.

    Returns None when the file is absent or has no [output_parser] section.
    Individual fields may be missing; the parser tolerates partial config.
    """
    from .._shared.toml_compat import tomllib

    path = Path(task_format_path)
    if not path.is_file():
        return None

    with open(path, "rb") as f:
        data = tomllib.load(f)

    op = data.get("output_parser")
    if not op:
        return None

    # Summary fields: case-insensitive (runners emit "passed" / "PASSED"
    # variably; human-readable summary words). MULTILINE + VERBOSE for
    # formatted pattern strings in the TOML.
    summary_flags = re.MULTILINE | re.VERBOSE | re.IGNORECASE
    summary_fields: dict[str, re.Pattern] = {}
    for field_name, pattern in (op.get("summary") or {}).items():
        if pattern:
            summary_fields[field_name] = re.compile(pattern, summary_flags)

    # Per-test verdicts: the TOML literal is uppercase (PASSED|FAILED|ERROR).
    # IGNORECASE on a regex matching uppercase literals against a large log
    # adds Unicode case-folding overhead to every line-start position for
    # no match gain — drop it. MULTILINE must stay for the ^ anchor.
    per_test_flags = re.MULTILINE | re.VERBOSE
    per_test_cfg = op.get("per_test") or {}
    per_test_regex = None
    if per_test_cfg.get("regex"):
        per_test_regex = re.compile(per_test_cfg["regex"], per_test_flags)

    if not summary_fields and per_test_regex is None:
        return None

    return OutputParser(
        summary_fields=summary_fields,
        per_test_regex=per_test_regex,
    )


# Summary scan window: pytest and most runners emit their terminal
# summary line in the last few hundred chars. Searching only the tail
# avoids iterating hundreds of intermediate "N passed" matches on a
# 100K+ pytest log. Fall back to full-scan if the tail window misses
# (rare — happens when the whole output is short enough that tail ==
# output, or when the runner emits the summary mid-stream).
_SUMMARY_TAIL_CHARS = 4000


# Canonical verdict map — normalizes runner-specific PASS/FAIL tokens
# so downstream consumers (done-parity, render_digest, run_summary)
# don't need to know each runner's vocabulary. Unknown verdicts are
# passed through uppercased (safe default: done-parity checks treat
# them as non-passing).
_VERDICT_NORMALIZE = {
    # canonical
    "PASSED": "PASSED", "FAILED": "FAILED", "ERROR": "ERROR",
    "SKIPPED": "SKIPPED",
    # short forms (pytest -rA, jest CI)
    "PASS": "PASSED", "FAIL": "FAILED", "SKIP": "SKIPPED",
    # cargo
    "OK": "PASSED", "IGNORED": "SKIPPED",
    # jest default reporter (unicode marks)
    "✓": "PASSED", "✕": "FAILED",
    # go
    # (PASS/FAIL already covered)
}


def _normalize_verdict(raw: str) -> str:
    """Map a runner-emitted verdict literal to a canonical PASSED/FAILED/etc.

    Unknown tokens are returned uppercased unchanged. Downstream code
    treats non-canonical tokens as non-passing, so the failure mode of
    an unrecognized verdict is a false negative on done-parity
    checks, not a false positive.
    """
    key = (raw or "").strip().upper()
    # Handle unicode marks before uppercase (✓ is already canonical).
    if raw in _VERDICT_NORMALIZE:
        return _VERDICT_NORMALIZE[raw]
    return _VERDICT_NORMALIZE.get(key, key)


def parse_structured(output: str, parser: OutputParser) -> dict:
    """Extract {summary: {passed, failed, errors, ...}, tests: {id: verdict}}.

    Per-field regexes run independently — order-agnostic, so pytest's
    "2 failed, 8 passed, 1 error" parses the same as "8 passed, 2
    failed, 1 error". Last numeric match per field wins (runners
    sometimes emit the tally twice; the second instance is the
    terminal summary). Scan is bounded to the tail of the output when
    the output is large, to avoid sweeping intermediate lines.

    Per-test verdicts are normalized to canonical PASSED/FAILED/
    ERROR/SKIPPED via _normalize_verdict. Callers depending on
    canonical verdict strings (done-parity, regression detection,
    render_digest) work uniformly across runners.
    """
    tail = output if len(output) <= _SUMMARY_TAIL_CHARS else output[-_SUMMARY_TAIL_CHARS:]
    summary: dict[str, int] = {}
    for field_name, rx in parser.summary_fields.items():
        match = None
        for match in rx.finditer(tail):
            pass  # keep the last match
        if match is None and tail is not output:
            # Fall back to full-scan only when the tail missed — covers
            # runners that emit their summary mid-stream.
            for match in rx.finditer(output):
                pass
        if match is None:
            continue
        try:
            summary[field_name] = int(match.group(1))
        except (IndexError, ValueError, TypeError):
            continue

    tests: dict[str, str] = {}
    if parser.per_test_regex:
        for m in parser.per_test_regex.finditer(output):
            tid = m.groupdict().get("test_id")
            verdict = m.groupdict().get("verdict")
            if tid and verdict:
                tests[tid] = _normalize_verdict(verdict)

    return {"summary": summary or None, "tests": tests}


def render_digest(parsed: dict, *, max_failures_shown: int = 10) -> str:
    """Render a parsed test-run record as compact text for the model."""
    lines: list[str] = []
    summary = parsed.get("summary") or {}
    if summary:
        parts: list[str] = []
        # Preserve stable ordering for the model: passed, failed, errors, skipped,
        # then any runner-specific fields alphabetically.
        for k in ("passed", "failed", "errors", "skipped"):
            if k in summary:
                parts.append(f"{summary[k]} {k}")
        for k in sorted(summary):
            if k in ("passed", "failed", "errors", "skipped"):
                continue
            parts.append(f"{summary[k]} {k}")
        if parts:
            lines.append("[digest] " + ", ".join(parts))
    tests = parsed.get("tests") or {}
    failing = [tid for tid, v in tests.items() if v in ("FAILED", "FAIL", "ERROR")]
    if failing:
        shown = failing[:max_failures_shown]
        lines.append(f"[digest] failing ({len(failing)}): " + ", ".join(shown))
        if len(failing) > max_failures_shown:
            lines.append(f"[digest] ... {len(failing) - max_failures_shown} more failing tests")
    if not lines:
        return ""
    return "\n".join(lines)


# ── Transform functions (called by dispatch) ────────────────────────────

@functools.lru_cache(maxsize=256)
def _is_test_command(cmd: str, oc: OutputControl) -> bool:
    """Cached: OutputControl is frozen/hashable, cmds repeat across turns.

    ``rewrite_command``, ``condense_output``, and the harness's
    ``_project_and_sink`` each check whether a bash cmd is a
    verification gate; without caching that loops 6 regexes per
    check × multiple checks per tool call. Cache key is (cmd, oc);
    size 256 covers a long task without unbounded growth.
    """
    return any(p.search(cmd) for p in oc.verification_patterns)


def rewrite_command(cmd: str, oc: OutputControl | None,
                    universal_rewrites: list[RewriteRule] | None = None) -> str:
    """Rewrite a bash command to reduce output volume.

    Applies universal rewrites (pip -q, npm --loglevel=error, etc.) and
    task-format-specific flags (--tb=short for pytest). No-op for commands
    that don't match any pattern or already have the flag.
    """
    # Universal rewrites — always apply.
    if universal_rewrites:
        for rule in universal_rewrites:
            if not rule.pattern.search(cmd):
                continue
            if any(skip in cmd for skip in rule.skip_if):
                continue
            # Append flag before trailing pipe chain.
            pipe_idx = cmd.find("|")
            if pipe_idx > 0:
                cmd = cmd[:pipe_idx].rstrip() + " " + rule.flag + " " + cmd[pipe_idx:]
            else:
                cmd = cmd.rstrip() + " " + rule.flag
            break  # one rewrite per command

    # Task-format rewrite — test runner flags.
    if oc and oc.failure_only_flag:
        if _is_test_command(cmd, oc) and oc.failure_only_flag not in cmd:
            pipe_idx = cmd.find("|")
            if pipe_idx > 0:
                cmd = cmd[:pipe_idx].rstrip() + " " + oc.failure_only_flag + " " + cmd[pipe_idx:]
            else:
                cmd = cmd.rstrip() + " " + oc.failure_only_flag

    return cmd


def condense_output(output: str, cmd: str, oc: OutputControl | None) -> str:
    """Strip passing-test lines from test command output.

    Replaces lines containing passed_marker (but not failed_marker) with
    a single count summary. No-op when oc is None, markers are empty,
    or the command isn't a test invocation.
    """
    if not oc or not oc.passed_marker:
        return output
    if not _is_test_command(cmd, oc):
        return output

    lines = output.split("\n")
    kept: list[str] = []
    passed_count = 0
    for line in lines:
        if oc.passed_marker in line and oc.failed_marker not in line:
            passed_count += 1
            continue
        kept.append(line)
    if passed_count:
        kept.insert(max(len(kept) - 1, 0), f"[{passed_count} tests passed]")
    result = "\n".join(kept)
    if passed_count:
        # Token accounting: record exact savings (raw chars vs. stripped).
        from ..harness.savings import get_ledger
        get_ledger().record(
            bucket="bash_output_condense",
            layer="L2_bash_quirks",
            mechanism="passed_line_stripping",
            input_chars=len(output),
            output_chars=len(result),
            measure_type="exact",
            ctx={"cmd": cmd[:120], "passed_stripped": passed_count,
                 "passed_marker": oc.passed_marker},
        )
    return result
