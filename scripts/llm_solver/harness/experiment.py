"""Prompt experiment utilities — variant loading and iteration."""
from pathlib import Path

from .._shared.toml_compat import tomllib


def load_variants(path: Path) -> dict[str, dict]:
    """Load named prompt variants from a TOML file.

    Expected format:
        [variants.baseline]
        prompt_addendum = ""

        [variants.read-tests]
        prompt_addendum = "Always read the test file before implementing."

    Returns dict mapping variant name → variant config dict.
    Raises FileNotFoundError if path does not exist.
    """
    if not path.is_file():
        raise FileNotFoundError(f"Variants file not found: {path}")
    with open(path, "rb") as f:
        data = tomllib.load(f)
    variants = data.get("variants", {})
    if not variants:
        raise ValueError(f"No [variants.*] sections in {path}")
    return dict(variants)
