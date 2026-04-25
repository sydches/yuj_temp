"""Data structures for ProfileAnalyzer outputs."""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class QuirkResult:
    name: str
    description: str
    affected_scenarios: list[str]
    rule_toml: str = ""         # TOML snippet for rules.toml
    code_module: str = ""       # Python code for a .py module
    module_filename: str = ""   # e.g. "fix_arguments.py"
    fixture: dict = field(default_factory=dict)  # {"description": ..., "cases": [...]}


__all__ = ["QuirkResult"]
