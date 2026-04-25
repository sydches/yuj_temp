"""Post-normalization verification — checks that normalize pipeline produces clean output.

Loads raw samples, runs them through the profile's normalize pipeline, and checks
the normalized output for cleanliness: no leaked tokens, well-formed tool calls,
correct content/None handling, canonical finish reasons.

Also validates [server] section completeness: model_path, ctx_size, reasoning flags.

Usage:
    python -m scripts.llm_solver.profiles.verify <profile_name> [--samples-dir <path>]
"""
import json
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path

from .._shared.paths import project_root
from .._shared.toml_compat import tomllib
from ..server.profile_loader import load_profile

PROJECT_ROOT = project_root()

log = logging.getLogger(__name__)

# Canonical finish reasons accepted by the harness
CANONICAL_FINISH_REASONS = {"stop", "tool_calls", "length"}

# Patterns that must not appear in normalized content
LEAKED_TOKEN_PATTERNS = [
    (re.compile(r"<think>", re.IGNORECASE), "leaked <think> tag"),
    (re.compile(r"</think>", re.IGNORECASE), "leaked </think> tag"),
    (re.compile(r"<\|?channel\|?>"), "leaked channel token"),
    (re.compile(r"<channel\|>"), "leaked closing channel token"),
]

# Trailing whitespace check (content should be clean after normalize)
TRAILING_WS = re.compile(r"\s+$")


@dataclass
class CheckFailure:
    """Single check failure within a scenario."""
    check: str
    message: str
    expected: str = ""
    actual: str = ""


@dataclass
class ScenarioResult:
    """Verification result for a single scenario."""
    scenario_id: str
    passed: bool
    failures: list[CheckFailure] = field(default_factory=list)


@dataclass
class VerificationResult:
    """Full verification result for a profile."""
    profile_name: str
    total: int
    passed: int
    failed: int
    scenarios: list[ScenarioResult] = field(default_factory=list)
    server_failures: list[CheckFailure] = field(default_factory=list)

    @property
    def all_passed(self) -> bool:
        return self.failed == 0 and len(self.server_failures) == 0


def _check_content_clean(response: dict) -> list[CheckFailure]:
    """Check normalized content for leaked tokens and trailing whitespace."""
    failures = []
    content = response.get("content")
    if content is None:
        return failures

    # Leaked special tokens
    for pattern, desc in LEAKED_TOKEN_PATTERNS:
        match = pattern.search(content)
        if match:
            # Show context around the match
            start = max(0, match.start() - 30)
            end = min(len(content), match.end() + 30)
            context = content[start:end]
            failures.append(CheckFailure(
                check="content_clean",
                message=desc,
                expected="no special tokens",
                actual=repr(context),
            ))

    # Trailing whitespace
    if TRAILING_WS.search(content):
        tail = repr(content[-40:]) if len(content) > 40 else repr(content)
        failures.append(CheckFailure(
            check="content_clean",
            message="trailing whitespace in content",
            expected="no trailing whitespace",
            actual=tail,
        ))

    return failures


def _check_tool_calls(response: dict) -> list[CheckFailure]:
    """Check tool calls are well-formed after normalize."""
    failures = []
    tool_calls = response.get("tool_calls")
    if not tool_calls:
        return failures

    for i, tc in enumerate(tool_calls):
        if not isinstance(tc, dict):
            failures.append(CheckFailure(
                check="tool_call_format",
                message=f"tool_call[{i}] is not a dict",
                expected="dict",
                actual=type(tc).__name__,
            ))
            continue

        # Check id present and non-empty
        tc_id = tc.get("id")
        if not tc_id:
            failures.append(CheckFailure(
                check="tool_call_id",
                message=f"tool_call[{i}] missing or empty id",
                expected="non-empty string",
                actual=repr(tc_id),
            ))

        # Check arguments is a JSON string, not a dict
        func = tc.get("function", {})
        if isinstance(func, dict):
            args = func.get("arguments")
            if isinstance(args, dict):
                failures.append(CheckFailure(
                    check="tool_call_arguments",
                    message=f"tool_call[{i}] arguments is dict, not JSON string",
                    expected="JSON string",
                    actual=f"dict with keys {list(args.keys())}",
                ))
            elif isinstance(args, str):
                # Verify it's valid JSON
                try:
                    json.loads(args)
                except json.JSONDecodeError as e:
                    failures.append(CheckFailure(
                        check="tool_call_arguments",
                        message=f"tool_call[{i}] arguments is not valid JSON",
                        expected="valid JSON string",
                        actual=repr(args[:80]) + f" (error: {e})",
                    ))

    return failures


def _check_content_none_consistency(response: dict) -> list[CheckFailure]:
    """Check content consistency: no whitespace-only content that should be empty."""
    failures = []
    content = response.get("content")

    # After normalize, content should never be whitespace-only —
    # the empty_content rule should have caught it.
    # Both None and "" are valid empty states.
    if isinstance(content, str) and content != "" and content.strip() == "":
        failures.append(CheckFailure(
            check="content_empty",
            message="content is whitespace-only after normalize (should be empty or None)",
            expected='None or ""',
            actual=repr(content),
        ))

    return failures


def _check_finish_reason(response: dict) -> list[CheckFailure]:
    """Check finish_reason is in canonical set."""
    failures = []
    reason = response.get("finish_reason")
    if reason not in CANONICAL_FINISH_REASONS:
        failures.append(CheckFailure(
            check="finish_reason",
            message=f"finish_reason '{reason}' not in canonical set",
            expected=str(CANONICAL_FINISH_REASONS),
            actual=repr(reason),
        ))
    return failures


def verify_server_config(profile_dir: Path) -> list[CheckFailure]:
    """Validate [server] section in profile.toml.

    Checks:
    - model_path is non-empty
    - ctx_size > 0
    - reasoning_disable_flag is set when reasoning_mode is "think" or "channel"
    """
    failures = []
    toml_path = profile_dir / "profile.toml"
    if not toml_path.is_file():
        failures.append(CheckFailure(
            check="server_config",
            message="profile.toml not found",
            expected=str(toml_path),
        ))
        return failures

    with open(toml_path, "rb") as f:
        parsed = tomllib.load(f)

    server = parsed.get("server", {})

    # model_path must be non-empty
    model_path = server.get("model_path", "")
    if not model_path:
        failures.append(CheckFailure(
            check="server_model_path",
            message="model_path is empty — `python -m scripts.llm_solver.server launch` will fail",
            expected="non-empty path (e.g. ~/models/model.gguf)",
            actual=repr(model_path),
        ))

    # ctx_size must be > 0
    ctx_size = server.get("ctx_size", 0)
    if not isinstance(ctx_size, int) or ctx_size <= 0:
        failures.append(CheckFailure(
            check="server_ctx_size",
            message=f"ctx_size must be positive integer, got {ctx_size!r}",
            expected="> 0",
            actual=repr(ctx_size),
        ))

    # reasoning_disable_flag must be set for think/channel modes
    reasoning_mode = server.get("reasoning_mode", "none")
    reasoning_flag = server.get("reasoning_disable_flag", "")
    if reasoning_mode in ("think", "channel") and not reasoning_flag:
        failures.append(CheckFailure(
            check="server_reasoning",
            message=f"reasoning_mode={reasoning_mode!r} requires non-empty reasoning_disable_flag",
            expected="e.g. '--reasoning-budget 0' or '--reasoning off'",
            actual=repr(reasoning_flag),
        ))

    return failures


def verify_scenario(profile, scenario: dict) -> ScenarioResult:
    """Verify a single scenario's response after normalization."""
    scenario_id = scenario["scenario_id"]
    response = dict(scenario["response"])  # shallow copy

    # Run through normalize pipeline
    normalized = profile.normalize(response)

    # Run all checks
    failures = []
    failures.extend(_check_content_clean(normalized))
    failures.extend(_check_tool_calls(normalized))
    failures.extend(_check_content_none_consistency(normalized))
    failures.extend(_check_finish_reason(normalized))

    return ScenarioResult(
        scenario_id=scenario_id,
        passed=len(failures) == 0,
        failures=failures,
    )


def verify_profile(profile_dir: Path, samples_path: Path) -> VerificationResult:
    """Load profile and samples, run verification on all scenarios.

    Args:
        profile_dir: Path to the profile directory (e.g., profiles/gemma-4-26b)
        samples_path: Path to _all_results.json

    Returns:
        VerificationResult with per-scenario pass/fail and failure details
    """
    profiles_dir = profile_dir.parent
    profile_name = profile_dir.name
    profile = load_profile(profile_name, profiles_dir)

    with open(samples_path) as f:
        samples = json.load(f)

    # Validate [server] section
    server_failures = verify_server_config(profile_dir)

    scenarios = []
    for sample in samples:
        result = verify_scenario(profile, sample)
        scenarios.append(result)

    passed = sum(1 for s in scenarios if s.passed)
    failed = len(scenarios) - passed

    return VerificationResult(
        profile_name=profile_name,
        total=len(scenarios),
        passed=passed,
        failed=failed,
        scenarios=scenarios,
        server_failures=server_failures,
    )


def format_report(result: VerificationResult) -> str:
    """Format a human-readable verification report."""
    lines = []
    status = "PASS" if result.all_passed else "FAIL"
    lines.append(f"Profile: {result.profile_name}  [{status}]")
    lines.append(f"Scenarios: {result.passed}/{result.total} passed")

    if result.server_failures:
        lines.append("")
        lines.append("Server config:")
        for f in result.server_failures:
            lines.append(f"  [{f.check}] {f.message}")
            if f.expected:
                lines.append(f"    expected: {f.expected}")
            if f.actual:
                lines.append(f"    actual:   {f.actual}")

    if result.failed > 0:
        lines.append("")
        lines.append("Failures:")
        for scenario in result.scenarios:
            if not scenario.passed:
                lines.append(f"  {scenario.scenario_id}:")
                for f in scenario.failures:
                    lines.append(f"    [{f.check}] {f.message}")
                    if f.expected:
                        lines.append(f"      expected: {f.expected}")
                    if f.actual:
                        lines.append(f"      actual:   {f.actual}")

    return "\n".join(lines)


def main() -> int:
    """CLI entry point."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Post-normalization verification for model profiles"
    )
    parser.add_argument("profile_name", help="Profile directory name (e.g., gemma-4-26b)")
    parser.add_argument(
        "--samples-dir",
        type=Path,
        default=None,
        help="Path to samples directory (default: profiles/<name>/_samples)",
    )
    parser.add_argument(
        "--profiles-dir",
        type=Path,
        default=PROJECT_ROOT / "profiles",
        help="Path to profiles directory",
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    if args.verbose:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.WARNING)

    profile_dir = args.profiles_dir / args.profile_name
    if not profile_dir.is_dir():
        print(f"Profile directory not found: {profile_dir}", file=sys.stderr)
        return 1

    samples_dir = args.samples_dir or profile_dir / "_samples"
    samples_path = samples_dir / "_all_results.json"
    if not samples_path.is_file():
        print(f"Samples file not found: {samples_path}", file=sys.stderr)
        return 1

    result = verify_profile(profile_dir, samples_path)
    print(format_report(result))

    return 0 if result.all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
