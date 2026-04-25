"""Behavioral denormalization for qwen3-8b-q4.

Suffix loaded from profile.toml [behavioral].suffix via configure()
before apply() is first called. Legacy in-file constant kept as a
fallback for environments where configure() is not wired.

Adapted from qwen3.5-9b behavioral profile — same Qwen family, same
observed failure patterns at smaller scale. Task-agnostic.
"""


_BEHAVIORAL_SUFFIX_FALLBACK = """

## Concrete application

**Friction — what counts as a gate:** The repo's own test suite is the mechanical gate
for Correctness (tests_pass). Import checks and pip install are build steps, not gates.
Before your first execute→verify cycle, find the gate: search for test files
(tests/, test/, *_test.py, Makefile, tox.ini). Read them — they define what the
verify step checks. Run the gate after every code change. Gate strength ordering:
import check < inline smoke test < repo test suite. Use the strongest available.

**Compromise — orientation budget:** "Smallest verified step" means act, don't survey.
After starting or resuming, spend at most 2 turns orienting (ls, pwd, reading existing
files). Then execute. If you wrote a file last session, don't re-read it — edit or test it.

**Friction — gate verdict required for termination:** "Verdict is the gate's, not the
agent's." Do not declare done without gate output. If you haven't run the repo's test
suite (or equivalent verification), the Correctness constraint is unverified."""


_BEHAVIORAL_SUFFIX: str = _BEHAVIORAL_SUFFIX_FALLBACK


def configure(behavioral_cfg: dict) -> None:
    """Accept the profile's [behavioral] dict from profile_loader."""
    global _BEHAVIORAL_SUFFIX
    if not isinstance(behavioral_cfg, dict):
        return
    suffix = behavioral_cfg.get("suffix", "")
    if isinstance(suffix, str) and suffix:
        _BEHAVIORAL_SUFFIX = suffix


def apply(messages: list[dict]) -> list[dict]:
    """Append behavioral instructions to the system prompt.

    Idempotent: if the suffix is already in the content, returns
    unchanged. Without this check, calling apply() twice on the same
    underlying messages list would append the suffix twice, wasting
    ~2KB/turn of context on duplicate text.
    """
    if not messages:
        return messages
    first = messages[0]
    if first.get("role") != "system":
        return messages
    content = first["content"]
    if _BEHAVIORAL_MARKER in content:
        return messages
    messages[0] = {**first, "content": content + _BEHAVIORAL_SUFFIX}
    return messages


# Distinctive phrase from the suffix for idempotency detection.
_BEHAVIORAL_MARKER = "## Concrete application"
