"""Behavioral denormalization for glm-4-flash.

Suffix loaded from profile.toml [behavioral].suffix via configure()
before apply() is first called. Legacy in-file constant kept as a
fallback for environments where configure() is not wired.

Observed failure modes (90K context, 10 sessions, 2 rounds) informed
the text but not its placement — suffix is task-agnostic.
"""


_BEHAVIORAL_SUFFIX_FALLBACK = """

## Concrete application

**Friction — what counts as a gate:** The repo's own test suite is the mechanical gate
for Correctness (tests_pass). Before your first execute→verify cycle, find the gate:
search for test files (tests/, test/, *_test.py, Makefile, tox.ini). Read them.

**Friction — test immediately after every mutation:** After every write or edit to a
code file, your very next action must be to run the test suite. Do not interpose
python3 -c checks, pip install, ls, cat, or file reads between a code change and
the test run. python3 -c "import X" is not the gate — it tells you nothing about
correctness. The test suite is the only gate.

**BareMetal — working directory is already correct:** Your working directory is set by
the harness. Do not cd to absolute paths. All commands run from the current directory.
If a command fails, the issue is not your directory.

**Compromise — orientation budget of 2 turns:** After starting or resuming, spend at
most 2 turns reading files or listing directories. Then act. If you wrote a file
last session, do not re-read it — edit it or run the tests.

**Compromise — iterate, don't rewrite:** When tests fail, read the error and make a
targeted edit. Do not delete files and start over. Do not rewrite a file from scratch
when a single function needs fixing. Reading the same file twice without changing it
is surveying, not descending. Each session builds on the last.

**Ratchet — do not repeat commands:** If you ran a command and got output, use that
output. Do not run the same command again expecting different results. If the output
was not what you expected, change your approach — do not retry the same action.

**Ratchet — each session builds on the last:** When resuming, your prior work is in
the working directory. Check what exists (2 turns max), then continue from where you
left off. Do not restart implementation from scratch. The test suite will tell you
what still needs fixing.

**Friction — gate verdict required for termination:** Do not declare done without
test suite output showing which tests pass and which fail."""


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

    Only activates when the commandments are present (with_yuj modes).
    wo_yuj modes are the control arm — no behavioral injection.

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
    if "Commandments" not in content:
        return messages
    if _BEHAVIORAL_MARKER in content:
        return messages
    messages[0] = {**first, "content": content + _BEHAVIORAL_SUFFIX}
    return messages


# Distinctive phrase from the suffix for idempotency detection. Update
# this marker in lockstep with any edit to _BEHAVIORAL_SUFFIX that
# removes this substring.
_BEHAVIORAL_MARKER = "## Concrete application"
