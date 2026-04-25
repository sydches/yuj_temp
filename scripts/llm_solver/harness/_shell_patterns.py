"""Shared shell-surface regexes used by multiple harness modules."""
import re

TEST_COMMAND_RE = re.compile(
    r"\b(pytest|py\.test|python\s+-m\s+pytest|python3\s+-m\s+pytest|"
    r"unittest|cargo test|go test|ctest|npm test|pnpm test|yarn test)\b",
    re.IGNORECASE,
)
