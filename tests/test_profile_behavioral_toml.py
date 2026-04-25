"""Tests for reading behavioral config from profile.toml.

Verifies that qwen3.5-9b's behavioral.py pulls its suffix from the
sibling profile.toml [behavioral].suffix field, and that the shared
loader handles both populated and absent TOML sections.
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from llm_solver._shared.profile_behavioral import (
    BehavioralConfig,
    load_profile_behavioral,
)


class TestLoader:

    def test_loads_populated_section(self, tmp_path):
        profile = tmp_path / "profile.toml"
        profile.write_text(
            '[behavioral]\n'
            'suffix = "hello"\n'
            'tool_call_style = "qwen_xml"\n'
            'reminder_placement = "user_tail"\n'
        )
        module_file = tmp_path / "denormalize" / "behavioral.py"
        module_file.parent.mkdir()
        module_file.write_text("")
        cfg = load_profile_behavioral(str(module_file))
        assert cfg.suffix == "hello"
        assert cfg.tool_call_style == "qwen_xml"
        assert cfg.reminder_placement == "user_tail"

    def test_empty_behavioral_section_returns_defaults(self, tmp_path):
        profile = tmp_path / "profile.toml"
        profile.write_text("[profile]\nname = 'x'\n")
        module_file = tmp_path / "denormalize" / "behavioral.py"
        module_file.parent.mkdir()
        module_file.write_text("")
        cfg = load_profile_behavioral(str(module_file))
        assert cfg.suffix == ""
        assert cfg.tool_call_style == "openai_json"
        assert cfg.reminder_placement == "system"

    def test_missing_profile_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_profile_behavioral(str(tmp_path / "behavioral.py"))


class TestQwen35Integration:
    """End-to-end: the shipping qwen3.5-9b profile must source its
    suffix from profile.toml, not its legacy in-file constant."""

    def test_suffix_comes_from_profile_toml(self):
        from llm_solver._shared.profile_behavioral import load_profile_behavioral
        behavioral_path = (
            PROJECT_ROOT / "profiles" / "qwen3.5-9b"
            / "denormalize" / "behavioral.py"
        )
        cfg = load_profile_behavioral(str(behavioral_path))
        # The shipping profile.toml has a populated suffix.
        assert cfg.suffix != ""
        # Must contain one of the operational-rule markers; this is a
        # content-blind sanity probe that survives wording changes.
        assert "Operational rules" in cfg.suffix

    def test_behavioral_py_accepts_configure_from_toml(self):
        """Verify the behavioral module's runtime constant equals the
        TOML value after configure() is called with the profile's
        [behavioral] section."""
        import importlib.util
        from llm_solver._shared.profile_behavioral import load_profile_behavioral
        behavioral_path = (
            PROJECT_ROOT / "profiles" / "qwen3.5-9b"
            / "denormalize" / "behavioral.py"
        )
        spec = importlib.util.spec_from_file_location(
            "qwen35_9b_behavioral", behavioral_path,
        )
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        cfg = load_profile_behavioral(str(behavioral_path))
        # Before configure(), the fallback constant is in use.
        assert mod._BEHAVIORAL_SUFFIX == mod._BEHAVIORAL_SUFFIX_FALLBACK
        mod.configure({"suffix": cfg.suffix})
        assert mod._BEHAVIORAL_SUFFIX == cfg.suffix

    def test_configure_ignores_non_dict(self):
        import importlib.util
        behavioral_path = (
            PROJECT_ROOT / "profiles" / "qwen3.5-9b"
            / "denormalize" / "behavioral.py"
        )
        spec = importlib.util.spec_from_file_location(
            "qwen35_9b_behavioral_b", behavioral_path,
        )
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        before = mod._BEHAVIORAL_SUFFIX
        mod.configure("not a dict")
        assert mod._BEHAVIORAL_SUFFIX == before
        mod.configure({})
        assert mod._BEHAVIORAL_SUFFIX == before
        mod.configure({"suffix": ""})
        assert mod._BEHAVIORAL_SUFFIX == before


class TestInheritFromBase:
    """Placeholder smoke-test: loader walks up parents to find
    profile.toml when a sub-directory contains the caller."""

    def test_walks_up_to_find_profile(self, tmp_path):
        profile = tmp_path / "profile.toml"
        profile.write_text("[behavioral]\nsuffix = 'x'\n")
        deep = tmp_path / "a" / "b" / "c" / "mod.py"
        deep.parent.mkdir(parents=True)
        deep.write_text("")
        cfg = load_profile_behavioral(str(deep))
        assert cfg.suffix == "x"
