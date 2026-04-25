"""Rule refinement from post-normalization verification failures.

Deterministic — no LLM. Analyzes dirty content patterns from verification
failures and generates improved strip rules / code modules.

Usage:
    from llm_solver.profiles.refiner import refine_rules
    result = refine_rules(verification_result, profile_dir)
"""
import ast
import json
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path

from .._shared.paths import project_root
from .._shared.toml_compat import tomllib
from .verify import VerificationResult, CheckFailure

PROJECT_ROOT = project_root()

log = logging.getLogger(__name__)

# ── Known token patterns ─────────────────────────────────────────

# Block-level token pairs: (opening_literal, closing_literal, rule_name, regex_pattern)
KNOWN_BLOCK_PAIRS = [
    ("<|channel>", "<channel|>", "channel_block", r"<\|channel>.*?<channel\|>"),
    ("<think>", "</think>", "thinking_block", r"<think>.*?</think>"),
]

# Bare token cleanup: (rule_name, regex_pattern)
# Catches fragments left after block stripping — token + rest of line
KNOWN_BARE_TOKENS = [
    ("channel_token", r"<\|?channel\|?>[^\n]*\n?"),
    ("think_tag", r"</?think>[^\n]*\n?"),
]

# Code module template for arguments-as-dict quirk
_CODE_FIX_ARGUMENTS = '''\
"""Fix arguments-as-dict quirk (llama-server bug #20198)."""
import json


def apply(response: dict) -> dict:
    for tc in response.get("tool_calls", []):
        func = tc.get("function", {})
        args = func.get("arguments")
        if isinstance(args, dict):
            func["arguments"] = json.dumps(args)
    return response
'''


# ── Data classes ─────────────────────────────────────────────────

@dataclass
class RuleChange:
    """A single change to the rule set."""
    action: str       # "widen", "add", "report"
    rule_name: str
    old_pattern: str  # empty for "add"
    new_pattern: str
    reason: str


@dataclass
class RefinementResult:
    """Result of rule refinement for a profile."""
    profile_name: str
    changes: list[RuleChange] = field(default_factory=list)
    rules_path: Path | None = None

    @property
    def changed(self) -> bool:
        return any(c.action in ("widen", "add") for c in self.changes)


# ── Helpers ──────────────────────────────────────────────────────

def _parse_actual_content(actual: str) -> str:
    """Extract raw string from CheckFailure.actual (which is repr'd)."""
    try:
        return ast.literal_eval(actual)
    except (ValueError, SyntaxError):
        return actual


def _load_strip_rules(rules_path: Path) -> list[dict]:
    """Load [[strip]] rules from a TOML file."""
    if not rules_path.is_file():
        return []
    with open(rules_path, "rb") as f:
        data = tomllib.load(f)
    return list(data.get("strip", []))


def _write_strip_rules(rules: list[dict], rules_path: Path, profile_name: str) -> None:
    """Write [[strip]] rules to a TOML file."""
    lines = [f"# Normalize rules — refined by refiner"]
    lines.append(f"# Profile: {profile_name}")
    lines.append("")

    for rule in rules:
        comment = rule.get("_comment")
        if comment:
            lines.append(f"# {comment}")
        lines.append("[[strip]]")
        lines.append(f'name = "{rule["name"]}"')
        lines.append(f"pattern = '{rule['pattern']}'")
        lines.append(f'target = "{rule["target"]}"')
        lines.append(f'replace = "{rule.get("replace", "")}"')
        lines.append("")

    rules_path.parent.mkdir(parents=True, exist_ok=True)
    rules_path.write_text("\n".join(lines))


def _detect_block_pattern(dirty: str) -> tuple[str, str, str, str] | None:
    """Check if dirty content contains a block pattern (paired tokens).

    Returns (block_name, block_pattern, bare_name, bare_pattern) or None.
    """
    for open_lit, close_lit, block_name, block_pat in KNOWN_BLOCK_PAIRS:
        if open_lit in dirty and close_lit in dirty:
            bare_name = block_name.replace("_block", "_token")
            # Find matching bare token pattern
            for bname, bpat in KNOWN_BARE_TOKENS:
                if bname == bare_name:
                    return block_name, block_pat, bare_name, bpat
            # Fallback: bare pattern from opening token
            open_esc = re.escape(open_lit)
            return block_name, block_pat, bare_name, open_esc
    return None


def _find_matching_rule(existing: list[dict], dirty: str) -> dict | None:
    """Find an existing strip rule whose compiled pattern matches part of the dirty content."""
    for rule in existing:
        try:
            pat = re.compile(rule["pattern"], re.DOTALL)
            if pat.search(dirty):
                return rule
        except re.error:
            continue
    return None


# ── Leaked token refinement ──────────────────────────────────────

def _collect_leaked_failures(result: VerificationResult) -> list[CheckFailure]:
    """Collect all content_clean failures about leaked tokens."""
    failures = []
    for scenario in result.scenarios:
        for f in scenario.failures:
            if f.check == "content_clean" and "leaked" in f.message:
                failures.append(f)
    return failures


def _refine_leaked_tokens(
    failures: list[CheckFailure],
    existing: list[dict],
) -> tuple[list[dict], list[RuleChange]]:
    """Analyze leaked token failures and produce updated strip rules.

    Returns (updated_rules, changes).
    """
    changes: list[RuleChange] = []
    existing_names = {r["name"] for r in existing}
    rules = [dict(r) for r in existing]  # deep-ish copy

    # Parse all dirty content from failures
    dirty_contents = [_parse_actual_content(f.actual) for f in failures]
    combined_dirty = "\n".join(dirty_contents)

    # Check for block-level patterns
    block_info = _detect_block_pattern(combined_dirty)

    if block_info:
        block_name, block_pat, bare_name, bare_pat = block_info
        if block_name in existing_names:
            return rules, changes  # already have block rule

        # Find existing narrow rule that partially matches
        matching = _find_matching_rule(existing, combined_dirty)

        if matching:
            # Widen: remove narrow rule, add block + bare
            old_pattern = matching["pattern"]
            rules = [r for r in rules if r["name"] != matching["name"]]
            existing_names.discard(matching["name"])
            changes.append(RuleChange(
                action="widen",
                rule_name=block_name,
                old_pattern=old_pattern,
                new_pattern=block_pat,
                reason="Dirty content shows full block pattern, not just bare tokens",
            ))
        else:
            changes.append(RuleChange(
                action="add",
                rule_name=block_name,
                old_pattern="",
                new_pattern=block_pat,
                reason="Dirty content contains paired tokens forming a block",
            ))

        # Insert block rule at top (block before bare)
        rules.insert(0, {
            "name": block_name,
            "pattern": block_pat,
            "target": "content",
            "replace": "",
            "_comment": "Strip full blocks (paired tokens with content between)",
        })

        # Add bare token cleanup if not already present
        if bare_name not in existing_names:
            rules.append({
                "name": bare_name,
                "pattern": bare_pat,
                "target": "content",
                "replace": "",
                "_comment": "Strip bare/partial tokens that survive after block stripping",
            })
            changes.append(RuleChange(
                action="add",
                rule_name=bare_name,
                old_pattern="",
                new_pattern=bare_pat,
                reason="Catch fragments left after block stripping",
            ))

        return rules, changes

    # No block pattern — add bare token rules from dirty content
    for dirty in dirty_contents:
        for bare_name, bare_pat in KNOWN_BARE_TOKENS:
            if re.search(bare_pat, dirty) and bare_name not in existing_names:
                rules.append({
                    "name": bare_name,
                    "pattern": bare_pat,
                    "target": "content",
                    "replace": "",
                })
                existing_names.add(bare_name)
                changes.append(RuleChange(
                    action="add",
                    rule_name=bare_name,
                    old_pattern="",
                    new_pattern=bare_pat,
                    reason=f"Leaked token detected in normalized output",
                ))
                break
        else:
            # Unknown token — create rule from literal
            token_match = re.search(r"<[^>]{1,30}>", dirty)
            if token_match:
                token = token_match.group()
                escaped = re.escape(token)
                rule_name = f"leaked_{token.strip('<>').replace('|', '').replace('/', '')}"
                if rule_name not in existing_names:
                    rules.append({
                        "name": rule_name,
                        "pattern": escaped,
                        "target": "content",
                        "replace": "",
                    })
                    existing_names.add(rule_name)
                    changes.append(RuleChange(
                        action="add",
                        rule_name=rule_name,
                        old_pattern="",
                        new_pattern=escaped,
                        reason=f"Unknown leaked token: {token}",
                    ))

    return rules, changes


# ── Tool call refinement ─────────────────────────────────────────

def _collect_tool_call_failures(result: VerificationResult) -> dict[str, list[CheckFailure]]:
    """Collect tool call failures grouped by check type."""
    by_check: dict[str, list[CheckFailure]] = {}
    for scenario in result.scenarios:
        for f in scenario.failures:
            if f.check.startswith("tool_call_"):
                by_check.setdefault(f.check, []).append(f)
    return by_check


def _refine_tool_calls(
    failures_by_check: dict[str, list[CheckFailure]],
    profile_dir: Path,
) -> list[RuleChange]:
    """Handle tool call failures: generate code modules or TOML rules."""
    changes: list[RuleChange] = []
    norm_dir = profile_dir / "normalize"
    norm_dir.mkdir(parents=True, exist_ok=True)

    # arguments-as-dict → fix_arguments.py code module
    if "tool_call_arguments" in failures_by_check:
        args_failures = failures_by_check["tool_call_arguments"]
        has_dict = any("dict" in f.actual for f in args_failures)
        if has_dict:
            module_path = norm_dir / "fix_arguments.py"
            if not module_path.is_file():
                module_path.write_text(_CODE_FIX_ARGUMENTS)
                _add_module_to_profile(profile_dir, "fix_arguments.py")
                changes.append(RuleChange(
                    action="add",
                    rule_name="fix_arguments",
                    old_pattern="",
                    new_pattern="fix_arguments.py",
                    reason="Tool call arguments returned as dict, not JSON string",
                ))

    # missing tool_call_id → TOML rule
    if "tool_call_id" in failures_by_check:
        rules_path = norm_dir / "rules.toml"
        existing = _load_strip_rules(rules_path)
        # Check if fix_tool_call rule exists (different TOML section)
        if rules_path.is_file():
            with open(rules_path, "rb") as f:
                data = tomllib.load(f)
            if not data.get("fix_tool_call"):
                # Append fix_tool_call rule
                with open(rules_path, "a") as f:
                    f.write("\n[[fix_tool_call]]\n")
                    f.write('guard = { finish_reason = ["tool_calls", "tool"] }\n')
                    f.write('when = "id_missing"\n')
                    f.write('strategy = "generate"\n')
                changes.append(RuleChange(
                    action="add",
                    rule_name="fix_tool_call",
                    old_pattern="",
                    new_pattern="id_missing/generate",
                    reason="Tool calls missing ID field",
                ))

    return changes


def _add_module_to_profile(profile_dir: Path, module_filename: str) -> None:
    """Add a code module to profile.toml's normalize.modules list."""
    profile_toml = profile_dir / "profile.toml"
    if not profile_toml.is_file():
        return
    text = profile_toml.read_text()
    # Parse to check current modules
    with open(profile_toml, "rb") as f:
        data = tomllib.load(f)
    modules = data.get("normalize", {}).get("modules", [])
    if module_filename in modules:
        return
    modules.append(module_filename)
    # Replace modules line in text
    old_line = f'modules = {json.dumps(modules[:-1])}'
    new_line = f'modules = {json.dumps(modules)}'
    if old_line in text:
        text = text.replace(old_line, new_line)
        profile_toml.write_text(text)


# ── Finish reason refinement ─────────────────────────────────────

def _collect_finish_reason_failures(result: VerificationResult) -> list[CheckFailure]:
    failures = []
    for scenario in result.scenarios:
        for f in scenario.failures:
            if f.check == "finish_reason":
                failures.append(f)
    return failures


def _refine_finish_reasons(
    failures: list[CheckFailure],
    profile_dir: Path,
) -> list[RuleChange]:
    """Add map_finish_reason rules for non-canonical finish reasons."""
    changes: list[RuleChange] = []
    rules_path = profile_dir / "normalize" / "rules.toml"

    # Extract non-canonical finish reasons from failures
    non_canonical: set[str] = set()
    for f in failures:
        # actual is repr of the finish_reason value
        val = _parse_actual_content(f.actual)
        if val:
            non_canonical.add(val)

    if not non_canonical:
        return changes

    # Check existing map_finish_reason rules
    existing_maps: set[str] = set()
    if rules_path.is_file():
        with open(rules_path, "rb") as fh:
            data = tomllib.load(fh)
        for rule in data.get("map_finish_reason", []):
            existing_maps.add(rule.get("from", ""))

    # Append new mappings
    new_mappings = non_canonical - existing_maps
    if new_mappings:
        with open(rules_path, "a") as fh:
            for from_val in sorted(new_mappings):
                to_val = "tool_calls" if "tool" in from_val else "stop"
                fh.write(f"\n[[map_finish_reason]]\n")
                fh.write(f'from = "{from_val}"\n')
                fh.write(f'to = "{to_val}"\n')
                changes.append(RuleChange(
                    action="add",
                    rule_name=f"map_{from_val}",
                    old_pattern="",
                    new_pattern=f"{from_val} → {to_val}",
                    reason=f"Non-canonical finish_reason '{from_val}'",
                ))

    return changes


# ── Main entry point ─────────────────────────────────────────────

def refine_rules(
    result: VerificationResult,
    profile_dir: Path,
) -> RefinementResult:
    """Analyze verification failures and refine normalize rules.

    Modifies rules.toml and/or code modules in profile_dir/normalize/.
    Does NOT re-run verification — the caller does that.

    Args:
        result: VerificationResult from verify.py
        profile_dir: Path to the profile directory (e.g., profiles/gemma-4-26b)

    Returns:
        RefinementResult describing what changed
    """
    if result.all_passed:
        return RefinementResult(profile_name=result.profile_name)

    rules_path = profile_dir / "normalize" / "rules.toml"
    existing_strip = _load_strip_rules(rules_path)
    all_changes: list[RuleChange] = []

    # 1. Leaked tokens → widen/add strip rules
    leaked = _collect_leaked_failures(result)
    if leaked:
        updated_rules, strip_changes = _refine_leaked_tokens(leaked, existing_strip)
        if strip_changes:
            _write_strip_rules(updated_rules, rules_path, result.profile_name)
            all_changes.extend(strip_changes)

    # 2. Tool call issues → code modules / TOML rules
    tc_failures = _collect_tool_call_failures(result)
    if tc_failures:
        all_changes.extend(_refine_tool_calls(tc_failures, profile_dir))

    # 3. Finish reason → map rules
    fr_failures = _collect_finish_reason_failures(result)
    if fr_failures:
        all_changes.extend(_refine_finish_reasons(fr_failures, profile_dir))

    # 4. Content issues (whitespace, empty) → report only, _base territory
    for scenario in result.scenarios:
        for f in scenario.failures:
            if f.check == "content_empty":
                all_changes.append(RuleChange(
                    action="report",
                    rule_name="_base",
                    old_pattern="",
                    new_pattern="",
                    reason=f"Content issue ({f.message}) should be handled by _base rules",
                ))

    refinement = RefinementResult(
        profile_name=result.profile_name,
        changes=all_changes,
        rules_path=rules_path if all_changes else None,
    )

    if refinement.changed:
        log.info(
            "Refined %s: %d changes (%s)",
            result.profile_name,
            len(all_changes),
            ", ".join(f"{c.action}:{c.rule_name}" for c in all_changes if c.action != "report"),
        )

    return refinement
