"""Language quirks — per-runner/per-language semantics in declarative TOML.

Each ``<runner>.toml`` describes one test runner (pytest, cargo, jest, go,
ctest, …): invocation patterns, verdict markers, output-control flags.
Consumed by bash quirks (for output condensation) and by analysis
detectors (for format-conditional tagging).

Adding a new language/runner = adding a TOML file here. No code change in
harness, analysis, or any other layer.
"""
from pathlib import Path

FORMATS_DIR = Path(__file__).parent

__all__ = ["FORMATS_DIR"]
