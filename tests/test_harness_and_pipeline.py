"""Tests for harness loop, tools, solver, generate pipeline, config, and end-to-end integration."""
import json
import os
import subprocess as _subprocess
import sys
import tempfile
import time
from pathlib import Path
from unittest.mock import MagicMock, patch, PropertyMock

import openai
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
sys.path.insert(0, str(PROJECT_ROOT / "tests"))

from llm_solver.server.types import TurnResult, Usage, ToolCall
from llm_solver.config import Config, load_config, MODEL_MAP, _deep_merge, get_sdk_config


# ──────────────────────────────────────────────
# Helper: build a Config without loading TOML
# ──────────────────────────────────────────────

from _config_helpers import make_config  # centralized defaults — see tests/_config_helpers.py


def make_turn_result(content=None, tool_calls=None, finish_reason="stop", prompt_tokens=10):
    return TurnResult(
        content=content,
        tool_calls=tool_calls or [],
        finish_reason=finish_reason,
        usage=Usage(prompt_tokens=prompt_tokens, completion_tokens=5),
    )


# ──────────────────────────────────────────────
# 1. Config loading
# ──────────────────────────────────────────────

class TestConfig:

    def test_load_config_returns_config(self):
        cfg = load_config()
        assert isinstance(cfg, Config)
        assert cfg.base_url  # should have a value

    def test_model_map_has_aliases(self):
        assert "haiku" in MODEL_MAP
        assert "sonnet" in MODEL_MAP
        assert "qwen3-vl" in MODEL_MAP

    def test_deep_merge(self):
        base = {"a": 1, "b": {"c": 2, "d": 3}}
        overlay = {"b": {"c": 99}, "e": 5}
        result = _deep_merge(base, overlay)
        assert result["a"] == 1
        assert result["b"]["c"] == 99
        assert result["b"]["d"] == 3
        assert result["e"] == 5

    def test_load_config_with_overrides(self):
        cfg = load_config(overrides={"model": "test-override"})
        assert cfg.model == "test-override"

    def test_load_config_resolves_api_key_env_reference(self, monkeypatch):
        monkeypatch.setenv("YUJ_TEST_API_KEY", "resolved-secret")
        cfg = load_config(overrides={"api_key": "$ENV:YUJ_TEST_API_KEY"})
        assert cfg.api_key == "resolved-secret"

    def test_load_config_reports_missing_api_key_env_reference(self, monkeypatch):
        monkeypatch.delenv("YUJ_TEST_API_KEY", raising=False)
        with pytest.raises(KeyError, match="YUJ_TEST_API_KEY"):
            load_config(overrides={"api_key": "$ENV:YUJ_TEST_API_KEY"})

    def test_anthropic_adapter_preserves_tool_turns(self):
        from scripts.llm_solver.server.client import (
            _anthropic_to_openai_response,
            _to_anthropic_payload,
        )

        payload = {
            "model": "claude-sonnet-4-5",
            "messages": [
                {"role": "system", "content": "Use tools."},
                {"role": "user", "content": "Read calc.py"},
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [{
                        "id": "call_0_0",
                        "type": "function",
                        "function": {
                            "name": "read",
                            "arguments": '{"path": "calc.py"}',
                        },
                    }],
                },
                {
                    "role": "tool",
                    "tool_call_id": "call_0_0",
                    "content": "def add(a, b): return a - b",
                },
            ],
            "tools": [{
                "type": "function",
                "function": {
                    "name": "read",
                    "description": "Read a file",
                    "parameters": {
                        "type": "object",
                        "properties": {"path": {"type": "string"}},
                    },
                },
            }],
            "max_tokens": 128,
        }

        anthropic_payload = _to_anthropic_payload(payload)
        assert anthropic_payload["system"] == "Use tools."
        assert anthropic_payload["tools"][0]["input_schema"]["properties"]["path"]["type"] == "string"
        assert anthropic_payload["messages"][1]["content"][0]["type"] == "tool_use"
        assert anthropic_payload["messages"][2]["content"][0]["type"] == "tool_result"

        compat = _anthropic_to_openai_response({
            "content": [{
                "type": "tool_use",
                "id": "toolu_1",
                "name": "edit",
                "input": {"path": "calc.py", "old_str": "-", "new_str": "+"},
            }],
            "stop_reason": "tool_use",
            "usage": {"input_tokens": 10, "output_tokens": 5},
        })
        assert compat.choices[0].finish_reason == "tool_calls"
        tc = compat.choices[0].message.tool_calls[0]
        assert tc.id == "toolu_1"
        assert tc.function.name == "edit"
        assert json.loads(tc.function.arguments)["path"] == "calc.py"

    def test_get_sdk_config(self):
        sdk = get_sdk_config()
        assert "tools" in sdk or "default_model" in sdk

    def test_config_frozen(self):
        cfg = make_config()
        with pytest.raises(AttributeError):
            cfg.model = "changed"


# ──────────────────────────────────────────────
# 3. Harness tools
# ──────────────────────────────────────────────

class TestHarnessTools:

    def test_truncate_output_short(self):
        from llm_solver.harness.tools import truncate_output
        cfg = make_config(max_output_chars=1000)
        text = "short text"
        assert truncate_output(text, cfg) == text

    def test_truncate_output_long(self):
        from llm_solver.harness.tools import truncate_output
        cfg = make_config(max_output_chars=100, truncate_head_lines=2, truncate_tail_lines=2)
        lines = [f"line {i}" for i in range(200)]
        text = "\n".join(lines)
        result = truncate_output(text, cfg)
        assert "line 0" in result
        assert "line 1" in result
        assert "line 199" in result
        assert "omitted" in result

    def test_truncate_output_huge_single_line(self):
        """Regression: a single massive line must still be char-capped.

        Cell 2 of the smallest-task run hit a 3.26 MB input at HTTP 71
        because `cat transcript.log` returned a multi-megabyte payload
        on few lines and the old line-based slice kept it whole.
        """
        from llm_solver.harness.tools import truncate_output
        cfg = make_config(
            max_output_chars=1000,
            truncate_head_lines=100,
            truncate_tail_lines=50,
        )
        # One line, 100k chars. Line count < head_lines + tail_lines, so
        # the line-based path leaves the whole thing intact. Char cap
        # must kick in.
        text = "x" * 100_000
        result = truncate_output(text, cfg)
        assert len(result) <= cfg.max_output_chars + 200  # cap + bookkeeping text
        assert "chars omitted" in result

    def test_truncate_output_few_massive_lines(self):
        """Each of a few lines is individually larger than max_output_chars."""
        from llm_solver.harness.tools import truncate_output
        cfg = make_config(
            max_output_chars=1000,
            truncate_head_lines=100,
            truncate_tail_lines=50,
        )
        # 5 lines × 50k chars = 250k. Line count fits under head+tail,
        # so line-slice returns the whole thing. Char cap must bound it.
        text = "\n".join("y" * 50_000 for _ in range(5))
        result = truncate_output(text, cfg)
        assert len(result) <= cfg.max_output_chars + 200

    def test_collapse_duplicate_lines_compresses_runs(self):
        from llm_solver.harness.tools import _collapse_duplicate_lines
        text = "a\nb\nb\nb\nc\n"
        out = _collapse_duplicate_lines(text)
        assert out == "a\nb [×3]\nc\n"

    def test_collapse_duplicate_lines_unique_lines_pass_through(self):
        from llm_solver.harness.tools import _collapse_duplicate_lines
        text = "one\ntwo\nthree\n"
        assert _collapse_duplicate_lines(text) == text

    def test_collapse_duplicate_lines_is_content_blind(self):
        # The compressor operates on byte equality. It does not know what
        # the lines represent — retry-loop spam, progress-bar repeats,
        # identical status lines from any tool all collapse the same way.
        from llm_solver.harness.tools import _collapse_duplicate_lines
        spam = "connection refused\n" * 5 + "ok\n"
        out = _collapse_duplicate_lines(spam)
        assert "connection refused [×5]" in out
        assert "ok" in out

    def test_collapse_duplicate_lines_respects_intervening_differences(self):
        from llm_solver.harness.tools import _collapse_duplicate_lines
        text = "x\nx\ny\nx\nx\nx\n"
        out = _collapse_duplicate_lines(text)
        # Two separate runs of x — each compressed independently.
        assert out == "x [×2]\ny\nx [×3]\n"

    # ── Skeleton-based similar-line collapsing ──────────────────────

    def test_line_skeleton_same_template(self):
        from llm_solver.harness.tools import _line_skeleton
        a = _line_skeleton("tests/test_foo.py::test_bar PASSED  [ 3%]")
        b = _line_skeleton("tests/test_foo.py::test_baz PASSED  [ 4%]")
        assert a == b

    def test_line_skeleton_different_punctuation(self):
        from llm_solver.harness.tools import _line_skeleton
        assert _line_skeleton("a::b") != _line_skeleton("a.b")

    def test_collapse_similar_lines_bulk_template_collapses(self):
        """Dominant skeleton (>50% of lines) collapses; rare lines survive."""
        from llm_solver.harness.tools import _collapse_similar_lines
        # 30 PASSED lines (dominant) + 3 FAILED lines (rare) + header/footer
        passed = [f"tests/test_foo.py::test_{i:03d} PASSED  [{i}%]" for i in range(30)]
        failed = [
            "FAILED tests/test_foo.py::test_broken - AssertionError",
            "FAILED tests/test_foo.py::test_other - ValueError",
            "FAILED tests/test_foo.py::test_third - TypeError",
        ]
        header = ["===== test session starts =====", "collected 33 items", ""]
        footer = ["", "===== 3 failed, 30 passed ====="]
        text = "\n".join(header + passed + failed + footer)
        out = _collapse_similar_lines(text)
        # PASSED lines collapsed
        assert "[×30 similar lines]" in out
        # FAILED lines survive individually
        for f in failed:
            assert f in out
        # Header/footer survive
        assert "test session starts" in out
        assert "3 failed, 30 passed" in out

    def test_collapse_similar_lines_no_dominant_template(self):
        """Output with no dominant template passes through unchanged."""
        from llm_solver.harness.tools import _collapse_similar_lines
        lines = [
            "drwxr-xr-x  2 syd syd  4096 Apr 12 16:48 ci",
            "-rw-r--r--  1 syd syd   156 Apr 12 16:48 .gitignore",
            "drwxr-xr-x  3 syd syd  4096 Apr 12 16:48 .github",
            "-rw-r--r--  1 syd syd  3519 Apr 12 16:48 README.md",
            "drwxr-xr-x  8 syd syd  4096 Apr 12 16:48 seaborn",
            "-rw-r--r--  1 syd syd   584 Apr 12 16:48 setup.cfg",
            "drwxr-xr-x  6 syd syd  4096 Apr 12 16:48 tests",
            "-rw-r--r--  1 syd syd   512 Apr 12 16:48 CITATION.cff",
            "-rw-r--r--  1 syd syd  1491 Apr 12 16:48 LICENSE.md",
            "-rw-r--r--  1 syd syd   219 Apr 12 16:48 Makefile",
        ]
        text = "\n".join(lines)
        out = _collapse_similar_lines(text)
        assert out == text  # no collapse — no single skeleton > 50%

    def test_collapse_similar_lines_small_output_skips(self):
        """Fewer than 10 non-blank lines → no collapse."""
        from llm_solver.harness.tools import _collapse_similar_lines
        lines = [f"tests/test_foo.py::test_{i} PASSED" for i in range(5)]
        text = "\n".join(lines)
        assert _collapse_similar_lines(text) == text

    def test_collapse_similar_lines_preserves_blank_separators(self):
        """Blank lines break consecutive runs of bulk lines."""
        from llm_solver.harness.tools import _collapse_similar_lines
        passed_a = [f"tests/test_a.py::test_{i} PASSED  [{i}%]" for i in range(15)]
        passed_b = [f"tests/test_b.py::test_{i} PASSED  [{50+i}%]" for i in range(15)]
        text = "\n".join(passed_a) + "\n\n" + "\n".join(passed_b)
        out = _collapse_similar_lines(text)
        # Both groups collapse independently; blank line preserved
        assert out.count("[×15 similar lines]") == 2

    def test_collapse_similar_lines_content_blind(self):
        """Compiler warnings collapse the same way as test output."""
        from llm_solver.harness.tools import _collapse_similar_lines
        # 20 warnings (dominant) + 2 errors (rare)
        warnings = [f"src/{chr(97+i)}.c:{i*10}: warning: unused variable" for i in range(20)]
        errors = [
            "src/main.c:5: error: undefined reference to 'foo'",
            "src/main.c:12: error: incompatible types",
        ]
        text = "\n".join(warnings + errors)
        out = _collapse_similar_lines(text)
        assert "[×20 similar lines]" in out
        for e in errors:
            assert e in out

    def test_filter_bash_output_skeleton_collapse_integrated(self):
        from llm_solver.harness.tools import _filter_bash_output
        cfg = make_config(
            strip_ansi=False, collapse_blank_lines=False,
            collapse_duplicate_lines=False, collapse_similar_lines=True,
            max_output_chars=200,
        )
        # 30 lines with header — dominant template collapses
        header = ["collected 30 items", ""]
        passed = [f"tests/test_foo.py::test_{i:03d} PASSED  [{i}%]" for i in range(30)]
        out = _filter_bash_output("\n".join(header + passed), "pytest", cfg)
        assert "[×30 similar lines]" in out

    def test_filter_bash_output_skeleton_skips_small_output(self):
        from llm_solver.harness.tools import _filter_bash_output
        cfg = make_config(
            strip_ansi=False, collapse_blank_lines=False,
            collapse_duplicate_lines=False, collapse_similar_lines=True,
            max_output_chars=20000,
        )
        lines = [f"tests/test_foo.py::test_{i} PASSED  [{i:2d}%]" for i in range(10)]
        text = "\n".join(lines)
        out = _filter_bash_output(text, "pytest", cfg)
        assert out == text  # no collapse — output too small

    def test_pipeline_byte_identical_before_skeleton(self):
        from llm_solver.harness.tools import _filter_bash_output
        cfg = make_config(
            strip_ansi=False, collapse_blank_lines=False,
            collapse_duplicate_lines=True, collapse_similar_lines=True,
        )
        a = "tests/test_foo.py::test_a PASSED  [ 1%]"
        b = "tests/test_foo.py::test_b PASSED  [ 2%]"
        text = "\n".join([a] * 5 + [b] * 5)
        out = _filter_bash_output(text, "pytest", cfg)
        # Byte-identical collapser fires first on each group
        assert f"{a} [×5]" in out
        assert f"{b} [×5]" in out

    # ── Bash command normalization for duplicate detection ──────────

    def test_normalize_bash_strips_trailing_tail(self):
        from llm_solver.harness.loop import _normalize_bash_for_dedup
        a = _normalize_bash_for_dedup("pytest tests/ -v 2>&1 | tail -60")
        b = _normalize_bash_for_dedup("pytest tests/ -v 2>&1 | tail -80")
        assert a == b

    def test_normalize_bash_strips_trailing_head(self):
        from llm_solver.harness.loop import _normalize_bash_for_dedup
        a = _normalize_bash_for_dedup("pytest tests/ -v | head -100")
        b = _normalize_bash_for_dedup("pytest tests/ -v | head -200")
        assert a == b

    def test_normalize_bash_strips_stderr_redirect(self):
        from llm_solver.harness.loop import _normalize_bash_for_dedup
        a = _normalize_bash_for_dedup("pytest tests/ -v 2>&1")
        b = _normalize_bash_for_dedup("pytest tests/ -v")
        assert a == b

    def test_normalize_bash_strips_chained_pipes(self):
        from llm_solver.harness.loop import _normalize_bash_for_dedup
        a = _normalize_bash_for_dedup("make 2>&1 | tail -50 | head -20")
        b = _normalize_bash_for_dedup("make")
        assert a == b

    def test_normalize_bash_preserves_meaningful_pipes(self):
        from llm_solver.harness.loop import _normalize_bash_for_dedup
        # Pipes to non-filter commands should be preserved
        a = _normalize_bash_for_dedup("echo hello | python3 -c 'import sys; print(sys.stdin.read())'")
        b = _normalize_bash_for_dedup("echo hello")
        assert a != b

    def test_normalize_bash_preserves_non_bash(self):
        from llm_solver.harness.loop import _normalize_bash_for_dedup
        # No pipes → passthrough
        cmd = "python3 -m pytest tests/test_foo.py -v"
        assert _normalize_bash_for_dedup(cmd) == cmd

    def test_dedup_signature_normalizes_bash(self):
        from llm_solver.harness.loop import _dedup_signature
        from llm_solver.server.types import ToolCall
        tc1 = ToolCall(id="1", name="bash", arguments={"cmd": "pytest -v | tail -60"})
        tc2 = ToolCall(id="2", name="bash", arguments={"cmd": "pytest -v | tail -80"})
        assert _dedup_signature(tc1) == _dedup_signature(tc2)

    def test_dedup_signature_non_bash_unchanged(self):
        from llm_solver.harness.loop import _dedup_signature
        from llm_solver.server.types import ToolCall
        tc1 = ToolCall(id="1", name="read", arguments={"path": "foo.py"})
        tc2 = ToolCall(id="2", name="read", arguments={"path": "foo.py"})
        assert _dedup_signature(tc1) == _dedup_signature(tc2)

    def test_focus_signature_extracts_outside_cwd_find_target(self):
        from llm_solver.harness.loop import _focus_signature

        tc = ToolCall(
            id="1",
            name="bash",
            arguments={"cmd": 'find /opt/miniconda3 -name "generic.py" -path "*/groupby/*" 2>/dev/null'},
        )
        key, display = _focus_signature(tc, "cmd='find /opt/miniconda3 ...'", "/tmp/task")
        assert key.startswith("outside:")
        assert "generic.py" in display
        assert "/opt/miniconda3" in display

    def test_dispatch_unknown_tool(self):
        from llm_solver.harness.tools import dispatch
        cfg = make_config()
        result = dispatch("nonexistent_tool", {}, cwd="/tmp", cfg=cfg)
        assert "ERROR: unknown tool" in result

    def test_dispatch_bad_args(self):
        from llm_solver.harness.tools import dispatch
        cfg = make_config()
        result = dispatch("bash", {}, cwd="/tmp", cfg=cfg)  # missing "cmd"
        assert "ERROR" in result

    def test_bash_tool(self, tmp_path):
        from llm_solver.harness.tools import bash
        result = bash("echo hello", cwd=str(tmp_path), timeout=10)
        assert "hello" in result

    def test_bash_timeout(self, tmp_path):
        from llm_solver.harness.tools import bash
        result = bash("sleep 10", cwd=str(tmp_path), timeout=1)
        assert "timed out" in result

    def test_read_tool(self, tmp_path):
        from llm_solver.harness.tools import read
        (tmp_path / "test.txt").write_text("line1\nline2\nline3\n")
        result = read("test.txt", cwd=str(tmp_path))
        assert "1: line1" in result
        assert "2: line2" in result

    def test_read_not_found(self, tmp_path):
        from llm_solver.harness.tools import read
        result = read("nonexistent.txt", cwd=str(tmp_path))
        assert "ERROR: file not found" in result

    def test_write_tool(self, tmp_path):
        from llm_solver.harness.tools import write
        result = write("new.txt", "hello world", cwd=str(tmp_path))
        assert "OK" in result
        assert (tmp_path / "new.txt").read_text() == "hello world"

    def test_edit_tool(self, tmp_path):
        from llm_solver.harness.tools import edit
        (tmp_path / "file.txt").write_text("old text here")
        result = edit("file.txt", "old", "new", cwd=str(tmp_path))
        assert result == "OK"
        assert (tmp_path / "file.txt").read_text() == "new text here"

    def test_edit_not_found(self, tmp_path):
        from llm_solver.harness.tools import edit
        result = edit("file.txt", "old", "new", cwd=str(tmp_path))
        assert "ERROR" in result

    def test_edit_whitespace_normalized_indentation(self, tmp_path):
        """Model rebuilds source from numbered read() output and gets
        indentation slightly wrong (4 spaces instead of the file's
        tabs). Exact match fails; whitespace-normalized fallback
        succeeds when the cascade DOE arm is enabled."""
        from llm_solver.harness.tools import edit
        cfg = make_config(edit_fuzzy_cascade_enabled=True,
                          edit_strict_match=False)
        src = "def foo():\n\treturn 1\n"
        (tmp_path / "f.py").write_text(src)
        # old_str uses 4 spaces instead of a tab
        result = edit(
            "f.py",
            "def foo():\n    return 1",
            "def foo():\n    return 2",
            cwd=str(tmp_path), cfg=cfg,
        )
        assert "OK" in result
        assert "whitespace-normalized" in result
        assert (tmp_path / "f.py").read_text() == "def foo():\n    return 2\n"

    def test_edit_whitespace_normalized_extra_blank_lines(self, tmp_path):
        """Cascade arm: model's old_str has an extra blank line the
        file doesn't. Normalized match collapses whitespace runs."""
        from llm_solver.harness.tools import edit
        cfg = make_config(edit_fuzzy_cascade_enabled=True,
                          edit_strict_match=False)
        src = "class X:\n    def m(self):\n        pass\n"
        (tmp_path / "x.py").write_text(src)
        result = edit(
            "x.py",
            "def m(self):\n\n    pass",  # extra blank line
            "def m(self):\n    return 42",
            cwd=str(tmp_path), cfg=cfg,
        )
        assert "OK" in result
        assert "return 42" in (tmp_path / "x.py").read_text()

    def test_edit_whitespace_fallback_preserves_non_whitespace_exactness(self, tmp_path):
        """Cascade arm: the fuzzy fallback must NOT match across a
        typo in a real identifier — we don't want to silently rewrite
        the wrong function. Only whitespace is relaxed."""
        from llm_solver.harness.tools import edit
        cfg = make_config(edit_fuzzy_cascade_enabled=True,
                          edit_strict_match=False)
        src = "def foo_bar():\n    return 1\n"
        (tmp_path / "f.py").write_text(src)
        # Typo: foo_baz instead of foo_bar
        result = edit(
            "f.py",
            "def foo_baz():\n    return 1",
            "def foo_bar():\n    return 2",
            cwd=str(tmp_path), cfg=cfg,
        )
        assert "ERROR" in result
        # File unchanged
        assert (tmp_path / "f.py").read_text() == "def foo_bar():\n    return 1\n"

    def test_edit_whitespace_fallback_uses_first_match(self, tmp_path):
        """Cascade arm: when normalized match has multiple candidates,
        use the first occurrence — same contract as the exact-match
        path."""
        from llm_solver.harness.tools import edit
        cfg = make_config(edit_fuzzy_cascade_enabled=True,
                          edit_strict_match=False)
        src = "pass\n\npass\n"
        (tmp_path / "a.py").write_text(src)
        result = edit("a.py", "pass", "yield",
                      cwd=str(tmp_path), cfg=cfg)
        assert "OK" in result
        # Only the first pass is replaced.
        assert (tmp_path / "a.py").read_text() == "yield\n\npass\n"

    def test_glob_tool(self, tmp_path):
        from llm_solver.harness.tools import glob_files
        (tmp_path / "a.py").write_text("")
        (tmp_path / "b.py").write_text("")
        (tmp_path / "c.txt").write_text("")
        result = glob_files("*.py", cwd=str(tmp_path))
        assert "a.py" in result
        assert "b.py" in result
        assert "c.txt" not in result

    def test_dispatch_routes_correctly(self, tmp_path):
        from llm_solver.harness.tools import dispatch
        cfg = make_config()
        (tmp_path / "test.txt").write_text("hello")
        result = dispatch("read", {"path": "test.txt"}, cwd=str(tmp_path), cfg=cfg)
        assert "hello" in result


# ──────────────────────────────────────────────
# 4. Harness solver
# ──────────────────────────────────────────────

class TestHarnessSolver:

    def test_collect_pending(self, tmp_path):
        from llm_solver.harness.solver import collect_pending
        repos = tmp_path / "repos"
        repos.mkdir()
        # Task 1: pending
        t1 = repos / "task1"
        t1.mkdir()
        (t1 / "prompt.txt").write_text("do something")
        # Task 2: completed
        t2 = repos / "task2"
        t2.mkdir()
        (t2 / "prompt.txt").write_text("do something else")
        (t2 / "checkpoint.json").write_text('{"status":"completed"}')
        # Task 3: no prompt
        t3 = repos / "task3"
        t3.mkdir()

        pending = collect_pending(tmp_path)
        assert len(pending) == 1
        assert pending[0].name == "task1"

    def test_collect_pending_no_repos_dir(self, tmp_path):
        from llm_solver.harness.solver import collect_pending
        with pytest.raises(FileNotFoundError):
            collect_pending(tmp_path)

    # check_done removed — completion is now model-signaled (stop with no tool calls)

    def test_write_checkpoint(self, tmp_path):
        from llm_solver.harness.solver import write_checkpoint
        write_checkpoint(tmp_path, "test-model", "completed")
        cp = json.loads((tmp_path / "checkpoint.json").read_text())
        assert cp["status"] == "completed"
        assert cp["model"] == "test-model"
        assert cp["solver"] == "llm_solver"

    def test_build_system_prompt_default(self):
        from llm_solver.harness.solver import build_system_prompt
        header = "You are a software engineering solver."
        prompt = build_system_prompt(header)
        assert "solver" in prompt.lower()
        assert prompt == header

    def test_build_system_prompt_with_file(self, tmp_path):
        from llm_solver.harness.solver import build_system_prompt
        header = "You are a software engineering solver."
        proto = tmp_path / "protocol.md"
        proto.write_text("Follow these rules.")
        prompt = build_system_prompt(header, proto)
        assert "Follow these rules." in prompt
        assert "solver" in prompt.lower()


# ──────────────────────────────────────────────
# 5. Cross-session learning (build_resume_prompt)
# ──────────────────────────────────────────────

class TestCrossSessionLearning:

    def _make_session(self, cfg=None):
        from llm_solver.harness.loop import Session, SessionResult
        cfg = cfg or make_config()
        client = MagicMock()
        client.build_assistant_message.return_value = {"role": "assistant", "content": "ok"}
        session = Session(cfg, client, "system", "initial", "/tmp")
        return session

    def test_resume_prompt_duplicate_abort(self):
        from llm_solver.harness.loop import build_resume_prompt, SessionResult
        cfg = make_config()
        session = self._make_session(cfg)
        session._tool_log = [("bash", "cmd='ls'"), ("bash", "cmd='ls'"), ("bash", "cmd='ls'")]
        result = SessionResult(turns=10, finish_reason="duplicate_abort", done=False, total_prompt_tokens=500)
        prompt = build_resume_prompt(result, session, cfg)
        assert "duplicate_abort" in prompt
        assert "identical" in prompt.lower()
        assert "bash" in prompt

    def test_resume_prompt_context_full(self):
        from llm_solver.harness.loop import build_resume_prompt, SessionResult
        cfg = make_config()
        session = self._make_session(cfg)
        session._last_fill = 0.92
        result = SessionResult(turns=30, finish_reason="context_full", done=False, total_prompt_tokens=8000)
        prompt = build_resume_prompt(result, session, cfg)
        assert "92%" in prompt
        assert "full" in prompt.lower()

    def test_resume_prompt_max_turns(self):
        from llm_solver.harness.loop import build_resume_prompt, SessionResult
        cfg = make_config()
        session = self._make_session(cfg)
        session._tool_log = [("read", "path='a.py'"), ("edit", "path='a.py'"), ("bash", "cmd='pytest'")]
        result = SessionResult(turns=60, finish_reason="max_turns", done=False, total_prompt_tokens=10000)
        prompt = build_resume_prompt(result, session, cfg)
        assert "max_turns" in prompt

    def test_resume_prompt_length(self):
        from llm_solver.harness.loop import build_resume_prompt, SessionResult
        cfg = make_config()
        session = self._make_session(cfg)
        result = SessionResult(turns=5, finish_reason="length", done=False, total_prompt_tokens=200)
        prompt = build_resume_prompt(result, session, cfg)
        assert "truncated" in prompt.lower()

    def test_resume_prompt_always_has_base(self):
        from llm_solver.harness.loop import build_resume_prompt, SessionResult
        cfg = make_config()
        session = self._make_session(cfg)
        result = SessionResult(turns=1, finish_reason="context_full", done=False)
        prompt = build_resume_prompt(result, session, cfg)
        assert cfg.resume_base in prompt


# ──────────────────────────────────────────────
# 6. Error taxonomy and transient retry
# ──────────────────────────────────────────────

class TestErrorTaxonomy:

    def test_transient_retry_succeeds(self):
        from llm_solver.harness.loop import Session
        cfg = make_config()
        client = MagicMock()
        tr = make_turn_result(content="ok")
        # Fail once, then succeed
        client.chat.side_effect = [
            openai.APIConnectionError(request=MagicMock()),
            tr,
        ]
        client.build_assistant_message.return_value = {"role": "assistant", "content": "ok"}
        session = Session(cfg, client, "sys", "user msg", "/tmp")
        with patch("llm_solver.harness.loop.time.sleep"):  # skip actual sleep
            result = session._chat_with_retry(0)
        assert result is not None
        assert result.content == "ok"

    def test_transient_retry_exhausted(self):
        from llm_solver.harness.loop import Session
        cfg = make_config()
        client = MagicMock()
        client.chat.side_effect = openai.APIConnectionError(request=MagicMock())
        client.build_assistant_message.return_value = {"role": "assistant", "content": ""}
        session = Session(cfg, client, "sys", "user msg", "/tmp")
        with patch("llm_solver.harness.loop.time.sleep"):
            result = session._chat_with_retry(0)
        assert result is None
        assert client.chat.call_count == cfg.max_transient_retries + 1

    def test_fatal_error_no_retry(self):
        from llm_solver.harness.loop import Session
        cfg = make_config()
        client = MagicMock()
        client.chat.side_effect = RuntimeError("unexpected")
        client.build_assistant_message.return_value = {"role": "assistant", "content": ""}
        session = Session(cfg, client, "sys", "user msg", "/tmp")
        result = session._chat_with_retry(0)
        assert result is None
        assert client.chat.call_count == 1  # no retry


# ──────────────────────────────────────────────
# 7. Session.run() integration
# ──────────────────────────────────────────────

class TestSessionRun:

    def test_session_stops_on_text_response(self):
        from llm_solver.harness.loop import Session
        cfg = make_config(max_turns=10)
        client = MagicMock()
        client.chat.return_value = make_turn_result(content="done", finish_reason="stop")
        client.build_assistant_message.return_value = {"role": "assistant", "content": "done"}
        session = Session(cfg, client, "sys", "prompt", "/tmp")
        result = session.run()
        assert result.done is True
        assert result.finish_reason == "stop"
        assert client.chat.call_count == 1

    def test_session_duplicate_abort(self):
        from llm_solver.harness.loop import Session
        cfg = make_config(max_turns=10, duplicate_abort=3)
        client = MagicMock()
        tc = [ToolCall(id="c1", name="bash", arguments={"cmd": "ls"})]
        client.chat.return_value = make_turn_result(tool_calls=tc, finish_reason="tool_calls")
        client.build_assistant_message.return_value = {"role": "assistant", "content": None, "tool_calls": []}

        with patch("llm_solver.harness.loop.dispatch", return_value="output"):
            session = Session(cfg, client, "sys", "prompt", "/tmp")
            result = session.run()

        assert result.finish_reason == "duplicate_abort"
        assert result.done is False

    def test_session_context_full(self):
        from llm_solver.harness.loop import Session
        cfg = make_config(max_turns=10, context_size=100, context_fill_ratio=0.5)
        client = MagicMock()
        # Return a tool call so the session continues, but context estimate will be large
        tc = [ToolCall(id="c1", name="bash", arguments={"cmd": "ls"})]
        client.chat.return_value = make_turn_result(tool_calls=tc, finish_reason="tool_calls", prompt_tokens=80)
        client.build_assistant_message.return_value = {"role": "assistant", "content": "x" * 1000}

        with patch("llm_solver.harness.loop.dispatch", return_value="y" * 1000):
            session = Session(cfg, client, "sys", "prompt", "/tmp")
            result = session.run()

        assert result.finish_reason == "context_full"

    def test_session_max_turns(self):
        from llm_solver.harness.loop import Session
        cfg = make_config(max_turns=2, duplicate_abort=10)
        client = MagicMock()
        # Different tool calls each time to avoid duplicate_abort
        call_count = [0]
        def varying_chat(*args, **kwargs):
            call_count[0] += 1
            tc = [ToolCall(id=f"c{call_count[0]}", name="bash", arguments={"cmd": f"echo {call_count[0]}"})]
            return make_turn_result(tool_calls=tc, finish_reason="tool_calls")
        client.chat.side_effect = varying_chat
        client.build_assistant_message.return_value = {"role": "assistant", "content": None}

        with patch("llm_solver.harness.loop.dispatch", return_value="ok"):
            session = Session(cfg, client, "sys", "prompt", "/tmp")
            result = session.run()

        assert result.finish_reason == "max_turns"
        assert result.turns == 2

    def test_session_length_response(self):
        from llm_solver.harness.loop import Session
        cfg = make_config(max_turns=10)
        client = MagicMock()
        client.chat.return_value = make_turn_result(content="truncated...", finish_reason="length")
        client.build_assistant_message.return_value = {"role": "assistant", "content": "truncated..."}
        session = Session(cfg, client, "sys", "prompt", "/tmp")
        result = session.run()
        assert result.finish_reason == "length"
        assert result.done is False

    def test_session_uses_profile_token_estimator(self):
        from llm_solver.harness.loop import Session

        def estimate(_messages):
            return 42

        profile = type("Profile", (), {})()
        profile.estimate_tokens = estimate

        cfg = make_config(max_turns=10)
        client = MagicMock()
        client.__dict__["profile"] = profile
        client.chat.return_value = make_turn_result(content="done", finish_reason="stop")
        client.build_assistant_message.return_value = {"role": "assistant", "content": "done"}

        session = Session(cfg, client, "sys", "prompt", "/tmp")

        assert session.context._token_estimator is estimate
        assert session.context.estimate_tokens() == 42

    def test_session_error_on_none_chat(self):
        from llm_solver.harness.loop import Session
        cfg = make_config(max_turns=10)
        client = MagicMock()
        client.chat.side_effect = RuntimeError("fatal")
        client.build_assistant_message.return_value = {"role": "assistant", "content": ""}
        session = Session(cfg, client, "sys", "prompt", "/tmp")
        result = session.run()
        assert result.finish_reason == "error"

    def test_session_fails_fast_on_tool_surface_mismatch(self):
        from llm_solver.harness.loop import Session

        cfg = make_config(max_turns=10)
        client = MagicMock()

        bad = [{"type": "function", "function": {"name": "not_a_real_tool"}}]
        with patch("llm_solver.harness.loop.get_tool_schemas", return_value=bad):
            with pytest.raises(ValueError, match="Tool surface mismatch"):
                Session(cfg, client, "sys", "prompt", "/tmp")

    def test_session_applies_profile_max_tools_cap(self):
        from llm_solver.harness.loop import Session
        from llm_solver.harness.schemas import get_tool_schemas

        cfg = make_config(max_turns=10)
        client = MagicMock()
        profile = type("Profile", (), {})()
        profile.max_tools = 3
        client.__dict__["profile"] = profile

        session = Session(cfg, client, "sys", "prompt", "/tmp")
        expected = [s["function"]["name"] for s in get_tool_schemas("minimal")[:3]]
        actual = [s["function"]["name"] for s in session._tool_schemas]
        assert actual == expected

    def test_session_applies_profile_simplify_schemas(self):
        from llm_solver.harness.loop import Session

        def _contains_description(value):
            if isinstance(value, dict):
                if "description" in value:
                    return True
                return any(_contains_description(v) for v in value.values())
            if isinstance(value, list):
                return any(_contains_description(v) for v in value)
            return False

        cfg = make_config(max_turns=10)
        client = MagicMock()
        profile = type("Profile", (), {})()
        profile.simplify_schemas = True
        client.__dict__["profile"] = profile

        session = Session(cfg, client, "sys", "prompt", "/tmp")
        assert all(not _contains_description(s) for s in session._tool_schemas)

    def test_session_uses_injected_tool_registry(self):
        from llm_solver.harness.loop import Session
        from llm_solver.harness.tools import build_tool_registry

        cfg = make_config(max_turns=1, duplicate_abort=10)
        client = MagicMock()
        tc = [ToolCall(id="c1", name="read", arguments={"path": "x.py"})]
        client.chat.return_value = make_turn_result(tool_calls=tc, finish_reason="tool_calls")
        client.build_assistant_message.return_value = {"role": "assistant", "content": None}
        registry = build_tool_registry(
            overrides={"read": lambda _args, _cwd, _cfg: "REGISTRY_READ_OK"}
        )

        tool_results = []
        session = Session(
            cfg,
            client,
            "sys",
            "prompt",
            "/tmp",
            tool_registry=registry,
        )
        orig_add = session.context.add_tool_result

        def capture(cid, result, **kwargs):
            tool_results.append(result)
            return orig_add(cid, result, **kwargs)

        session.context.add_tool_result = capture
        session.run()
        assert any("REGISTRY_READ_OK" in r for r in tool_results)

    def test_session_fails_fast_when_injected_registry_missing_handler(self):
        from llm_solver.harness.loop import Session
        from llm_solver.harness.tools import ToolRegistry

        cfg = make_config(max_turns=10)
        client = MagicMock()
        bad_registry = ToolRegistry(handlers={"bash": lambda _a, _c, _f: "ok"})

        with pytest.raises(ValueError, match="Tool surface mismatch"):
            Session(cfg, client, "sys", "prompt", "/tmp", tool_registry=bad_registry)

    def test_adaptive_policy_switches_after_mutation(self):
        from llm_solver.harness.loop import Session

        cfg = make_config(
            max_turns=3,
            duplicate_abort=10,
            done_guard_enabled=False,
            adaptive_policy_enabled=True,
            adaptive_requires_mutation=True,
            adaptive_requires_test_signal=False,
            adaptive_phase2_done_guard_enabled=True,
        )
        client = MagicMock()
        turns = iter([
            make_turn_result(tool_calls=[ToolCall(id="c1", name="write", arguments={"path": "a.py", "content": "x"})], finish_reason="tool_calls"),
            make_turn_result(content="done", tool_calls=[], finish_reason="stop"),
        ])
        client.chat.side_effect = lambda *a, **k: next(turns)
        client.build_assistant_message.return_value = {"role": "assistant", "content": None}

        with patch("llm_solver.harness.loop.dispatch", return_value="OK"):
            session = Session(cfg, client, "sys", "prompt", "/tmp")
            result = session.run()

        assert result.finish_reason == "stop"
        assert session._adaptive_switched is True
        assert session.cfg.done_guard_enabled is True

    def test_adaptive_policy_respects_test_signal_gate(self):
        from llm_solver.harness.loop import Session

        cfg = make_config(
            max_turns=3,
            duplicate_abort=10,
            done_guard_enabled=False,
            adaptive_policy_enabled=True,
            adaptive_requires_mutation=True,
            adaptive_requires_test_signal=True,
            adaptive_phase2_done_guard_enabled=True,
        )
        client = MagicMock()
        turns = iter([
            make_turn_result(tool_calls=[ToolCall(id="c1", name="write", arguments={"path": "a.py", "content": "x"})], finish_reason="tool_calls"),
            make_turn_result(content="done", tool_calls=[], finish_reason="stop"),
        ])
        client.chat.side_effect = lambda *a, **k: next(turns)
        client.build_assistant_message.return_value = {"role": "assistant", "content": None}

        with patch("llm_solver.harness.loop.dispatch", return_value="OK"):
            session = Session(cfg, client, "sys", "prompt", "/tmp")
            session.run()

        assert session._adaptive_switched is False
        assert session.cfg.done_guard_enabled is False

    def test_adaptive_policy_accepts_successful_test_without_exit_marker(self):
        from llm_solver.harness.loop import Session

        cfg = make_config(
            max_turns=3,
            duplicate_abort=10,
            done_guard_enabled=False,
            adaptive_policy_enabled=True,
            adaptive_requires_mutation=True,
            adaptive_requires_test_signal=True,
            adaptive_phase2_done_guard_enabled=True,
        )
        client = MagicMock()
        turns = iter([
            make_turn_result(
                tool_calls=[ToolCall(id="c1", name="write", arguments={"path": "a.py", "content": "x"})],
                finish_reason="tool_calls",
            ),
            make_turn_result(
                tool_calls=[ToolCall(id="c2", name="bash", arguments={"cmd": "python3 -m pytest tests/test_app.py -v"})],
                finish_reason="tool_calls",
            ),
            make_turn_result(content="done", tool_calls=[], finish_reason="stop"),
        ])
        client.chat.side_effect = lambda *a, **k: next(turns)
        client.build_assistant_message.return_value = {"role": "assistant", "content": None}

        def dispatch_stub(name, args, **kwargs):
            if name == "bash":
                return (
                    "============================= test session starts ==============================\n"
                    "platform linux -- Python 3.10.12, pytest-9.0.2 -- /usr/bin/python3\n"
                    "collecting ... collected 1 item\n\n"
                    "============================== 1 passed ===============================\n"
                )
            return "OK"

        with patch("llm_solver.harness.loop.dispatch", side_effect=dispatch_stub):
            session = Session(cfg, client, "sys", "prompt", "/tmp")
            result = session.run()

        assert result.finish_reason == "stop"
        assert session._observed_test_signal is True
        assert session._adaptive_switched is True
        assert session.cfg.done_guard_enabled is True

    def test_assistant_mode_pauses_for_risky_bash_command(self, tmp_path):
        from llm_solver.harness.loop import Session

        cfg = make_config(
            runtime_mode="assistant",
            max_turns=2,
            duplicate_abort=10,
            done_guard_enabled=False,
        )
        client = MagicMock()
        client.chat.return_value = make_turn_result(
            tool_calls=[ToolCall(id="c1", name="bash", arguments={"cmd": "rm -rf build"})],
            finish_reason="tool_calls",
        )
        client.build_assistant_message.return_value = {"role": "assistant", "content": "Clean the build output."}

        with patch("llm_solver.harness.loop.dispatch") as dispatch_mock:
            session = Session(
                cfg,
                client,
                "sys",
                "prompt",
                str(tmp_path),
                trace_path=tmp_path / ".trace.jsonl",
            )
            result = session.run()

        approval_path = tmp_path / "approval_request.json"
        assert result.finish_reason == "approval_required"
        assert dispatch_mock.called is False
        assert approval_path.exists()
        approval = json.loads(approval_path.read_text())
        assert approval["status"] == "pending"
        assert approval["cmd"] == "rm -rf build"

    def test_assistant_mode_pauses_for_external_cp_target(self, tmp_path):
        from llm_solver.harness.loop import Session

        cfg = make_config(
            runtime_mode="assistant",
            max_turns=2,
            duplicate_abort=10,
            done_guard_enabled=False,
        )
        client = MagicMock()
        external = tmp_path.parent / "escape" / "leak.txt"
        client.chat.return_value = make_turn_result(
            tool_calls=[
                ToolCall(
                    id="c1",
                    name="bash",
                    arguments={"cmd": f"cp inside.txt {external}"},
                )
            ],
            finish_reason="tool_calls",
        )
        client.build_assistant_message.return_value = {"role": "assistant", "content": "exfiltrate"}

        with patch("llm_solver.harness.loop.dispatch") as dispatch_mock:
            session = Session(
                cfg,
                client,
                "sys",
                "prompt",
                str(tmp_path),
                trace_path=tmp_path / ".trace.jsonl",
            )
            result = session.run()

        assert result.finish_reason == "approval_required"
        assert dispatch_mock.called is False
        approval = json.loads((tmp_path / "approval_request.json").read_text())
        assert approval["reason"] == "cp crosses the repo root"

    def test_assistant_mode_pauses_for_external_mv_source(self, tmp_path):
        from llm_solver.harness.loop import Session

        cfg = make_config(
            runtime_mode="assistant",
            max_turns=2,
            duplicate_abort=10,
            done_guard_enabled=False,
        )
        client = MagicMock()
        client.chat.return_value = make_turn_result(
            tool_calls=[
                ToolCall(
                    id="c1",
                    name="bash",
                    arguments={"cmd": "mv /etc/passwd stolen.txt"},
                )
            ],
            finish_reason="tool_calls",
        )
        client.build_assistant_message.return_value = {"role": "assistant", "content": "import secrets"}

        with patch("llm_solver.harness.loop.dispatch") as dispatch_mock:
            session = Session(
                cfg,
                client,
                "sys",
                "prompt",
                str(tmp_path),
                trace_path=tmp_path / ".trace.jsonl",
            )
            result = session.run()

        assert result.finish_reason == "approval_required"
        assert dispatch_mock.called is False
        approval = json.loads((tmp_path / "approval_request.json").read_text())
        assert approval["reason"] == "mv crosses the repo root"

    def test_assistant_mode_allows_in_repo_cp_and_mv(self, tmp_path):
        from llm_solver.harness.loop import Session, SessionResult

        cfg = make_config(
            runtime_mode="assistant",
            max_turns=2,
            duplicate_abort=10,
            done_guard_enabled=False,
        )
        client = MagicMock()
        turns = iter([
            make_turn_result(
                tool_calls=[
                    ToolCall(
                        id="c1",
                        name="bash",
                        arguments={"cmd": "cp -r src/ dst/"},
                    )
                ],
                finish_reason="tool_calls",
            ),
            make_turn_result(
                tool_calls=[
                    ToolCall(
                        id="c2",
                        name="bash",
                        arguments={"cmd": "mv old_name new_name"},
                    )
                ],
                finish_reason="tool_calls",
            ),
        ])
        client.chat.side_effect = lambda *a, **k: next(turns)
        client.build_assistant_message.return_value = {"role": "assistant", "content": "rearranging"}

        with patch("llm_solver.harness.loop.dispatch", return_value="ok") as dispatch_mock:
            session = Session(
                cfg,
                client,
                "sys",
                "prompt",
                str(tmp_path),
                trace_path=tmp_path / ".trace.jsonl",
            )
            result = session.run()

        assert isinstance(result, SessionResult)
        assert result.finish_reason != "approval_required"
        assert dispatch_mock.call_count == 2
        assert not (tmp_path / "approval_request.json").exists()

    def test_measurement_mode_skips_approval_for_external_cp(self, tmp_path):
        from llm_solver.harness.loop import Session

        cfg = make_config(
            runtime_mode="measurement",
            max_turns=1,
            duplicate_abort=10,
            done_guard_enabled=False,
        )
        client = MagicMock()
        external = tmp_path.parent / "out.txt"
        client.chat.return_value = make_turn_result(
            tool_calls=[
                ToolCall(
                    id="c1",
                    name="bash",
                    arguments={"cmd": f"cp inside.txt {external}"},
                )
            ],
            finish_reason="tool_calls",
        )
        client.build_assistant_message.return_value = {"role": "assistant", "content": "benchmark"}

        with patch("llm_solver.harness.loop.dispatch", return_value="ok") as dispatch_mock:
            session = Session(
                cfg,
                client,
                "sys",
                "prompt",
                str(tmp_path),
                trace_path=tmp_path / ".trace.jsonl",
            )
            session.run()

        assert dispatch_mock.called is True
        assert not (tmp_path / "approval_request.json").exists()

    def test_done_guard_accepts_successful_bash_without_exit_marker(self):
        from llm_solver.harness.loop import Session

        cfg = make_config(max_turns=4, duplicate_abort=10)
        client = MagicMock()
        turns = iter([
            make_turn_result(
                tool_calls=[ToolCall(id="c1", name="write", arguments={"path": "a.py", "content": "x"})],
                finish_reason="tool_calls",
            ),
            make_turn_result(
                tool_calls=[ToolCall(id="c2", name="bash", arguments={"cmd": "python3 -m pytest tests/test_app.py -v"})],
                finish_reason="tool_calls",
            ),
            make_turn_result(
                tool_calls=[ToolCall(id="c3", name="done", arguments={"message": "finished"})],
                finish_reason="tool_calls",
            ),
        ])
        client.chat.side_effect = lambda *a, **k: next(turns)
        client.build_assistant_message.return_value = {"role": "assistant", "content": "done"}

        def dispatch_stub(name, args, **kwargs):
            if name == "bash":
                return (
                    "============================= test session starts ==============================\n"
                    "platform linux -- Python 3.10.12, pytest-9.0.2 -- /usr/bin/python3\n"
                    "collecting ... collected 1 item\n\n"
                    "============================== 1 passed ===============================\n"
                )
            return "OK"

        with patch("llm_solver.harness.loop.dispatch", side_effect=dispatch_stub):
            session = Session(cfg, client, "sys", "prompt", "/tmp")
            result = session.run()

        assert result.finish_reason == "model_done"
        assert result.done is True

    def test_adaptive_policy_switches_task_format_transform_gating(self):
        from llm_solver.harness.loop import Session

        cfg = make_config(
            max_turns=3,
            duplicate_abort=10,
            bash_transforms_task_format_enabled=False,
            adaptive_policy_enabled=True,
            adaptive_requires_mutation=True,
            adaptive_requires_test_signal=False,
            adaptive_phase2_bash_task_format_enabled=True,
        )
        client = MagicMock()
        turns = iter([
            make_turn_result(tool_calls=[ToolCall(id="c1", name="write", arguments={"path": "a.py", "content": "x"})], finish_reason="tool_calls"),
            make_turn_result(tool_calls=[ToolCall(id="c2", name="bash", arguments={"cmd": "echo hi"})], finish_reason="tool_calls"),
            make_turn_result(content="done", tool_calls=[], finish_reason="stop"),
        ])
        client.chat.side_effect = lambda *a, **k: next(turns)
        client.build_assistant_message.return_value = {"role": "assistant", "content": None}

        seen_output_control = []

        def _capture_dispatch(_name, _arguments, *, cwd, cfg, output_control=None, universal_rewrites=None, tool_registry=None):
            seen_output_control.append(output_control)
            return "OK"

        with patch("llm_solver.harness.loop.dispatch", side_effect=_capture_dispatch):
            session = Session(
                cfg,
                client,
                "sys",
                "prompt",
                "/tmp",
                output_control=object(),
            )
            session.run()

        assert seen_output_control[0] is None  # before switch
        assert seen_output_control[1] is not None  # after switch


# ──────────────────────────────────────────────
# 8. solve_task end-to-end (mock client)
# ──────────────────────────────────────────────

class TestSolveTask:

    def test_solve_task_success_first_session(self, tmp_path):
        from llm_solver.harness.loop import solve_task
        # Setup task directory
        (tmp_path / "prompt.txt").write_text("implement feature X")

        client = MagicMock()
        client.chat.return_value = make_turn_result(content="Done!", finish_reason="stop")
        client.build_assistant_message.return_value = {"role": "assistant", "content": "Done!"}

        cfg = make_config(max_turns=10, max_sessions=3)
        ok = solve_task(tmp_path, cfg, client)
        assert ok is True
        cp = json.loads((tmp_path / "checkpoint.json").read_text())
        assert cp["status"] == "completed"

    def test_solve_task_no_prompt(self, tmp_path):
        from llm_solver.harness.loop import solve_task
        client = MagicMock()
        cfg = make_config()
        ok = solve_task(tmp_path, cfg, client)
        assert ok is False

    def test_solve_task_error_writes_checkpoint(self, tmp_path):
        from llm_solver.harness.loop import solve_task
        (tmp_path / "prompt.txt").write_text("do something")

        client = MagicMock()
        client.chat.side_effect = RuntimeError("crash")
        client.build_assistant_message.return_value = {"role": "assistant", "content": ""}

        cfg = make_config(max_turns=5, max_sessions=1)
        ok = solve_task(tmp_path, cfg, client)
        assert ok is False
        cp = json.loads((tmp_path / "checkpoint.json").read_text())
        assert cp["status"] == "error"

    def test_solve_task_accepts_explicit_prompt_without_prompt_file(self, tmp_path):
        from llm_solver.harness.loop import solve_task

        client = MagicMock()
        client.chat.return_value = make_turn_result(content="Done!", finish_reason="stop")
        client.build_assistant_message.return_value = {"role": "assistant", "content": "Done!"}

        cfg = make_config(max_turns=10, max_sessions=1)
        ok = solve_task(tmp_path, cfg, client, initial_prompt="fix issue")

        assert ok is True
        cp = json.loads((tmp_path / "checkpoint.json").read_text())
        assert cp["status"] == "completed"

    def test_solve_task_accepts_task_spec_prompt_without_prompt_file(self, tmp_path):
        from llm_solver.harness.loop import TaskSpec, solve_task

        client = MagicMock()
        client.chat.return_value = make_turn_result(content="Done!", finish_reason="stop")
        client.build_assistant_message.return_value = {"role": "assistant", "content": "Done!"}

        cfg = make_config(max_turns=10, max_sessions=1)
        ok = solve_task(
            tmp_path,
            cfg,
            client,
            task_spec=TaskSpec(prompt_text="spec prompt"),
        )
        assert ok is True
        cp = json.loads((tmp_path / "checkpoint.json").read_text())
        assert cp["status"] == "completed"

    def test_explicit_prompt_still_gets_prompt_addendum(self, tmp_path):
        from llm_solver.harness.loop import SessionResult, solve_task

        captured = {}

        class _CapturingSession:
            def __init__(self, *args, **kwargs):
                # Signature: cfg, client, system_prompt, initial_message, cwd, ...
                captured["initial"] = args[3]

            def run(self):
                return SessionResult(
                    turns=0,
                    finish_reason="stop",
                    done=True,
                    total_prompt_tokens=0,
                    total_completion_tokens=0,
                )

        client = MagicMock()
        cfg = make_config(max_turns=10, max_sessions=1, prompt_addendum="EXTRA")

        with patch("llm_solver.harness.loop.Session", _CapturingSession):
            with patch("llm_solver.harness.loop._auto_commit"):
                ok = solve_task(tmp_path, cfg, client, initial_prompt="base prompt")

        assert ok is True
        assert captured["initial"].endswith("\n\nEXTRA")

    def test_solve_task_applies_profile_preamble_to_system_prompt(self, tmp_path):
        from llm_solver.harness.loop import SessionResult, solve_task

        (tmp_path / "prompt.txt").write_text("do work")
        captured = {}

        class _CapturingSession:
            def __init__(self, *args, **kwargs):
                captured["system_prompt"] = args[2]

            def run(self):
                return SessionResult(
                    turns=0,
                    finish_reason="stop",
                    done=True,
                    total_prompt_tokens=0,
                    total_completion_tokens=0,
                )

        client = MagicMock()
        profile = type("Profile", (), {})()
        profile.preamble = "PROFILE PREAMBLE"
        client.__dict__["profile"] = profile

        cfg = make_config(max_turns=1, max_sessions=1)
        with patch("llm_solver.harness.loop.Session", _CapturingSession):
            with patch("llm_solver.harness.loop._auto_commit"):
                ok = solve_task(tmp_path, cfg, client)
        assert ok is True
        assert captured["system_prompt"].startswith("PROFILE PREAMBLE")

    def test_solve_task_passes_profile_token_estimator_to_context_class(self, tmp_path):
        from llm_solver.harness.context import FullTranscript
        from llm_solver.harness.loop import solve_task

        (tmp_path / "prompt.txt").write_text("implement feature X")

        def estimate(_messages):
            return 42

        profile = type("Profile", (), {})()
        profile.estimate_tokens = estimate

        client = MagicMock()
        client.__dict__["profile"] = profile
        client.chat.return_value = make_turn_result(content="Done!", finish_reason="stop")
        client.build_assistant_message.return_value = {"role": "assistant", "content": "Done!"}

        captured = {}

        class CapturingContext(FullTranscript):
            def __init__(self, original_prompt, token_estimator):
                captured["token_estimator"] = token_estimator
                super().__init__(
                    original_prompt=original_prompt,
                    token_estimator=token_estimator,
                )

        cfg = make_config(max_turns=10, max_sessions=1)
        with patch("llm_solver.harness.loop._auto_commit"):
            ok = solve_task(tmp_path, cfg, client, context_class=CapturingContext)

        assert ok is True
        assert captured["token_estimator"] is estimate

    def test_solve_task_multi_session_with_resume(self, tmp_path):
        from llm_solver.harness.loop import solve_task
        (tmp_path / "prompt.txt").write_text("implement feature X")

        session_count = [0]
        call_in_session = [0]
        def chat_fn(*args, **kwargs):
            call_in_session[0] += 1
            if session_count[0] == 0:
                # Session 1: one tool call, then stop with text (simulates model finishing)
                if call_in_session[0] == 1:
                    tc = [ToolCall(id="c1", name="bash", arguments={"cmd": "echo hi"})]
                    return make_turn_result(tool_calls=tc, finish_reason="tool_calls")
                # Second turn: model says "not done yet" but truncated → length → triggers new session
                return make_turn_result(content="working...", finish_reason="length")
            # Session 2: done
            return make_turn_result(content="All done!", finish_reason="stop")

        def track_sessions(*args, **kwargs):
            # Detect new session by tracking Session creation
            session_count[0] += 1
            call_in_session[0] = 0

        client = MagicMock()
        client.chat.side_effect = chat_fn
        client.build_assistant_message.return_value = {"role": "assistant", "content": "working"}

        cfg = make_config(max_turns=5, max_sessions=3, duplicate_abort=10)

        # Patch Session.__init__ to track session boundaries
        original_init = __import__("llm_solver.harness.loop", fromlist=["Session"]).Session.__init__
        def patched_init(self, *a, **kw):
            session_count[0] += 1
            call_in_session[0] = 0
            original_init(self, *a, **kw)

        with patch("llm_solver.harness.loop.dispatch", return_value="output"):
            with patch("llm_solver.harness.loop.Session.__init__", patched_init):
                ok = solve_task(tmp_path, cfg, client)

        assert ok is True

    def test_solve_task_max_sessions_exhausted(self, tmp_path):
        from llm_solver.harness.loop import solve_task
        (tmp_path / "prompt.txt").write_text("hard task")

        # Always return tool calls, never stop — will hit max_turns each session
        call_count = [0]
        def chat_fn(*args, **kwargs):
            call_count[0] += 1
            tc = [ToolCall(id=f"c{call_count[0]}", name="bash", arguments={"cmd": f"try {call_count[0]}"})]
            return make_turn_result(tool_calls=tc, finish_reason="tool_calls")

        client = MagicMock()
        client.chat.side_effect = chat_fn
        client.build_assistant_message.return_value = {"role": "assistant", "content": None}

        cfg = make_config(max_turns=2, max_sessions=2, duplicate_abort=10)
        with patch("llm_solver.harness.loop.dispatch", return_value="ok"):
            ok = solve_task(tmp_path, cfg, client)

        assert ok is False
        cp = json.loads((tmp_path / "checkpoint.json").read_text())
        assert cp["status"] == "error"


# ──────────────────────────────────────────────
# 9. Scenario runner (mock HTTP)
# ──────────────────────────────────────────────

class TestScenarioRunner:

    def test_load_scenarios(self):
        from llm_solver.profiles.run_scenarios import load_scenarios
        scenarios = load_scenarios()
        assert len(scenarios) > 0
        for s in scenarios:
            assert "messages" in s
            assert "_file" in s

    def test_load_scenarios_custom_dir(self, tmp_path):
        from llm_solver.profiles.run_scenarios import load_scenarios
        scenario = {"messages": [{"role": "user", "content": "test"}], "description": "custom"}
        (tmp_path / "test.json").write_text(json.dumps(scenario))
        scenarios = load_scenarios(tmp_path)
        assert len(scenarios) == 1
        assert scenarios[0]["description"] == "custom"

    def test_run_scenario_success(self):
        from llm_solver.profiles.run_scenarios import run_scenario
        mock_client = MagicMock()
        mock_msg = MagicMock()
        mock_msg.content = "hello"
        mock_msg.tool_calls = None
        mock_choice = MagicMock()
        mock_choice.message = mock_msg
        mock_choice.finish_reason = "stop"
        mock_usage = MagicMock()
        mock_usage.prompt_tokens = 10
        mock_usage.completion_tokens = 5
        mock_resp = MagicMock()
        mock_resp.choices = [mock_choice]
        mock_resp.usage = mock_usage
        mock_client.chat.completions.create.return_value = mock_resp

        scenario = {
            "messages": [{"role": "user", "content": "hi"}],
            "_file": "test.json",
            "description": "test",
        }
        result = run_scenario(mock_client, "model", scenario)
        assert result["error"] is None
        assert result["response"]["content"] == "hello"
        assert result["response"]["finish_reason"] == "stop"

    def test_run_scenario_error(self):
        from llm_solver.profiles.run_scenarios import run_scenario
        mock_client = MagicMock()
        mock_client.chat.completions.create.side_effect = Exception("connection failed")

        scenario = {
            "messages": [{"role": "user", "content": "hi"}],
            "_file": "test.json",
            "description": "test",
        }
        result = run_scenario(mock_client, "model", scenario)
        assert result["error"] is not None
        assert "connection failed" in result["error"]


# ──────────────────────────────────────────────
# 9b. Scenario completeness (all 6 harness tools covered)
# ──────────────────────────────────────────────

class TestScenarioCompleteness:
    """Ensure all 6 harness tools appear in at least one scenario."""

    HARNESS_TOOL_NAMES = {"bash", "read", "write", "edit", "glob", "grep"}

    def test_all_harness_tools_covered(self):
        from llm_solver.profiles.run_scenarios import load_scenarios
        scenarios = load_scenarios()
        covered = set()
        for s in scenarios:
            for tool_def in s.get("tools", []):
                name = tool_def.get("function", {}).get("name")
                if name:
                    covered.add(name)
        missing = self.HARNESS_TOOL_NAMES - covered
        assert missing == set(), (
            f"These harness tools have no scenario coverage: {sorted(missing)}"
        )

    def test_scenario_count(self):
        from llm_solver.profiles.run_scenarios import load_scenarios
        scenarios = load_scenarios()
        # 47 original - 3 archived + 3 new = 47
        assert len(scenarios) >= 44, f"Expected >= 44 scenarios, got {len(scenarios)}"

    def test_gauntlet_scenarios_have_machine_checkable_expect(self):
        from llm_solver.profiles.run_scenarios import load_scenarios
        scenarios = load_scenarios()
        gauntlets = [s for s in scenarios if s["id"].startswith("gauntlet_")]
        assert len(gauntlets) >= 3, f"Expected >= 3 gauntlet scenarios, got {len(gauntlets)}"
        for s in gauntlets:
            expect = s.get("expect", {})
            assert "args_contain" in expect or "tool_names_include" in expect, (
                f"{s['id']}: gauntlet scenario must have machine-checkable expect fields"
            )

    def test_scenario_structural_integrity(self):
        from llm_solver.profiles.run_scenarios import load_scenarios
        scenarios = load_scenarios()
        for s in scenarios:
            fname = s.get("_file", "?")
            assert "id" in s, f"{fname}: missing 'id'"
            assert "description" in s, f"{fname}: missing 'description'"
            assert "messages" in s, f"{fname}: missing 'messages'"
            assert len(s["messages"]) > 0, f"{fname}: empty 'messages'"
            # Tools array must be valid when present
            for tool_def in s.get("tools", []):
                fn = tool_def.get("function", {})
                assert "name" in fn, f"{fname}: tool missing 'name'"
                assert "parameters" in fn, f"{fname}: tool '{fn.get('name')}' missing 'parameters'"


# ──────────────────────────────────────────────
# 9b. Scenario evaluation
# ──────────────────────────────────────────────

class TestEvaluateScenario:

    def test_no_expect_block(self):
        from llm_solver.profiles.run_scenarios import evaluate_scenario
        result = evaluate_scenario({"messages": []}, {"response": {}})
        assert result["skipped"] is True
        assert result["passed"] is True

    def test_no_response(self):
        from llm_solver.profiles.run_scenarios import evaluate_scenario
        scenario = {"expect": {"has_tool_calls": True}}
        result = evaluate_scenario(scenario, {"response": None})
        assert result["passed"] is False
        assert result["checks"]["response_present"] is False

    def test_basic_checks_pass(self):
        from llm_solver.profiles.run_scenarios import evaluate_scenario
        scenario = {
            "expect": {
                "has_tool_calls": True,
                "min_tool_calls": 1,
                "max_tool_calls": 2,
                "finish_reason_in": ["tool_calls", "tool"],
                "has_content": False,
            }
        }
        result = {
            "response": {
                "content": None,
                "tool_calls": [
                    {"function": {"name": "edit", "arguments": "{\"path\": \"a.py\"}"}}
                ],
                "finish_reason": "tool_calls",
            }
        }
        ev = evaluate_scenario(scenario, result)
        assert ev["passed"] is True
        assert all(ev["checks"].values())

    def test_basic_checks_fail(self):
        from llm_solver.profiles.run_scenarios import evaluate_scenario
        scenario = {"expect": {"has_tool_calls": True, "min_tool_calls": 3}}
        result = {
            "response": {
                "content": None,
                "tool_calls": [
                    {"function": {"name": "edit", "arguments": "{}"}}
                ],
                "finish_reason": "tool_calls",
            }
        }
        ev = evaluate_scenario(scenario, result)
        assert ev["passed"] is False
        assert ev["checks"]["has_tool_calls"] is True
        assert ev["checks"]["min_tool_calls"] is False

    def test_tool_names_include(self):
        from llm_solver.profiles.run_scenarios import evaluate_scenario
        scenario = {"expect": {"tool_names_include": ["edit", "bash"]}}
        result = {
            "response": {
                "content": None,
                "tool_calls": [
                    {"function": {"name": "edit", "arguments": "{}"}},
                    {"function": {"name": "bash", "arguments": "{}"}},
                ],
                "finish_reason": "tool_calls",
            }
        }
        ev = evaluate_scenario(scenario, result)
        assert ev["checks"]["tool_names_include"] is True

    def test_args_contain(self):
        from llm_solver.profiles.run_scenarios import evaluate_scenario
        scenario = {
            "expect": {
                "args_contain": [
                    {"tool_name": "edit", "key": "path", "substring": "api.py"},
                    {"tool_name": "edit", "key": "old_str", "substring": "process_data"},
                ]
            }
        }
        result = {
            "response": {
                "content": None,
                "tool_calls": [
                    {"function": {
                        "name": "edit",
                        "arguments": "{\"path\": \"src/api.py\", \"old_str\": \"from utils import process_data\", \"new_str\": \"from utils import transform_data\"}"
                    }}
                ],
                "finish_reason": "tool_calls",
            }
        }
        ev = evaluate_scenario(scenario, result)
        assert ev["passed"] is True
        assert ev["checks"]["args_contain[0]"] is True
        assert ev["checks"]["args_contain[1]"] is True

    def test_args_exclude(self):
        from llm_solver.profiles.run_scenarios import evaluate_scenario
        scenario = {
            "expect": {
                "args_exclude": [
                    {"tool_name": "edit", "key": "path", "substring": "test_"}
                ]
            }
        }
        # Violating case: editing a test file
        result = {
            "response": {
                "content": None,
                "tool_calls": [
                    {"function": {
                        "name": "edit",
                        "arguments": "{\"path\": \"test_batch.py\", \"old_str\": \"x\", \"new_str\": \"y\"}"
                    }}
                ],
                "finish_reason": "tool_calls",
            }
        }
        ev = evaluate_scenario(scenario, result)
        assert ev["passed"] is False
        assert ev["checks"]["args_exclude[0]"] is False

    def test_malformed_arguments_json(self):
        from llm_solver.profiles.run_scenarios import evaluate_scenario
        scenario = {
            "expect": {
                "args_contain": [
                    {"tool_name": "edit", "key": "path", "substring": "foo"}
                ]
            }
        }
        result = {
            "response": {
                "content": None,
                "tool_calls": [
                    {"function": {"name": "edit", "arguments": "not json at all"}}
                ],
                "finish_reason": "tool_calls",
            }
        }
        ev = evaluate_scenario(scenario, result)
        assert ev["checks"]["args_contain[0]"] is False


# ──────────────────────────────────────────────
# 10. __main__.py CLI
# ──────────────────────────────────────────────

class TestCLI:

    def test_dry_run_parses(self, tmp_path):
        from llm_solver.__main__ import main
        # Create minimal run_dir structure
        repos = tmp_path / "repos"
        repos.mkdir()

        # --dry-run returns 0 (no SystemExit)
        ret = main([str(tmp_path), "--dry-run"])
        assert ret == 0

    def test_main_no_args_prints_help(self):
        from llm_solver.__main__ import main
        with pytest.raises(SystemExit) as exc_info:
            main([])
        assert exc_info.value.code == 2  # argparse error


# ──────────────────────────────────────────────
# 11. _summarize_args
# ──────────────────────────────────────────────

# ──────────────────────────────────────────────
# 13. Round-trip validation (samples → normalize → canonical)
# ──────────────────────────────────────────────

class TestRoundTrip:
    """Verify that scenario samples produce canonical output through the profile's normalize pipeline."""

    PROFILES_DIR = PROJECT_ROOT / "profiles"

    def _run_round_trip_for(self, profile_name: str):
        """Load profile, run its _samples/ through normalize, check canonical format."""
        from llm_solver.server.profile_loader import load_profile
        profile = load_profile(profile_name, self.PROFILES_DIR)
        samples_dir = self.PROFILES_DIR / profile_name / "_samples"
        if not samples_dir.is_dir():
            pytest.skip(f"No _samples/ for {profile_name}")

        combined = samples_dir / "_all_results.json"
        if combined.is_file():
            results = json.loads(combined.read_text())
        else:
            results = []
            for p in sorted(samples_dir.glob("*.json")):
                if p.name.startswith("_"):
                    continue
                results.append(json.loads(p.read_text()))

        failures = []
        for r in results:
            if r.get("error") or not r.get("response"):
                continue
            sid = r.get("scenario_id", "?")
            resp = dict(r["response"])
            normalized = profile.normalize(resp)

            # Check canonical requirements
            content = normalized.get("content")
            if isinstance(content, str) and "<think>" in content:
                failures.append(f"{sid}: unstripped <think> block")
            if isinstance(content, str) and content.strip() != content:
                failures.append(f"{sid}: content has leading/trailing whitespace")

            for i, tc in enumerate(normalized.get("tool_calls", [])):
                if isinstance(tc, dict):
                    if not tc.get("id"):
                        failures.append(f"{sid}: tool_call[{i}] missing id")
                    args = tc.get("function", {}).get("arguments")
                    if isinstance(args, dict):
                        failures.append(f"{sid}: tool_call[{i}].arguments is dict not string")

        return failures

    def test_qwen3_8b_round_trip(self):
        failures = self._run_round_trip_for("qwen3-8b-q4")
        assert failures == [], f"Round-trip failures: {failures}"

    def test_glm_round_trip(self):
        failures = self._run_round_trip_for("glm-4-flash")
        assert failures == [], f"Round-trip failures: {failures}"

    def test_qwen35_round_trip(self):
        failures = self._run_round_trip_for("qwen3.5-9b")
        assert failures == [], f"qwen3.5-9b round-trip failures: {failures}"

    @pytest.mark.parametrize("profile_name", ["devstral-small", "gemma-4-26b"])
    def test_round_trip_parametric(self, profile_name):
        failures = self._run_round_trip_for(profile_name)
        assert failures == [], f"Round-trip failures for {profile_name}: {failures}"


# ──────────────────────────────────────────────
# 14. Profile readiness checklist
# ──────────────────────────────────────────────

class TestProfileReadiness:
    """A profile is 'ready' when ALL of these pass."""

    PROFILES_DIR = PROJECT_ROOT / "profiles"

    def _check_readiness(self, profile_name: str) -> list[str]:
        """Return list of readiness issues (empty = ready)."""
        from llm_solver.server.profile_loader import load_profile
        from llm_solver.server.security import validate_profile as sec_validate
        issues = []
        profile_dir = self.PROFILES_DIR / profile_name

        # 1. Profile loads
        try:
            profile = load_profile(profile_name, self.PROFILES_DIR)
        except Exception as e:
            issues.append(f"load failed: {e}")
            return issues

        # 2. Security passes
        violations = sec_validate(profile_dir)
        if violations:
            issues.append(f"security violations: {violations}")

        # 3. Fixtures pass (if any)
        fixtures_dir = profile_dir / "normalize" / "fixtures"
        if fixtures_dir.is_dir():
            for fp in fixtures_dir.glob("*.json"):
                fixture = json.loads(fp.read_text())
                for i, case in enumerate(fixture.get("cases", [])):
                    actual = profile.normalize(dict(case["input"]))
                    for key, exp in case["expected"].items():
                        if actual.get(key) != exp:
                            issues.append(f"fixture {fp.name}[{i}]: {key} mismatch")

        # 4. Round-trip on samples (if any)
        samples_dir = profile_dir / "_samples"
        if samples_dir.is_dir():
            combined = samples_dir / "_all_results.json"
            if combined.is_file():
                results = json.loads(combined.read_text())
                for r in results:
                    if not r.get("response"):
                        continue
                    normalized = profile.normalize(dict(r["response"]))
                    content = normalized.get("content")
                    if isinstance(content, str) and "<think>" in content:
                        issues.append(f"round-trip {r.get('scenario_id')}: unstripped thinking")

        # 5. No-op on clean input
        clean = {"content": "Clean text.", "tool_calls": [], "finish_reason": "stop"}
        result = profile.normalize(dict(clean))
        if result["content"] != "Clean text.":
            issues.append(f"no-op fail: clean content mutated to {result['content']!r}")

        return issues

    def test_base_ready(self):
        issues = self._check_readiness("_base")
        assert issues == [], f"_base not ready: {issues}"

    def test_qwen3_8b_ready(self):
        issues = self._check_readiness("qwen3-8b-q4")
        assert issues == [], f"qwen3-8b-q4 not ready: {issues}"

    def test_glm_ready(self):
        issues = self._check_readiness("glm-4-flash")
        assert issues == [], f"glm-4-flash not ready: {issues}"

    @pytest.mark.parametrize("profile_name", ["devstral-small", "gemma-4-26b", "qwen3.5-9b"])
    def test_readiness_parametric(self, profile_name):
        issues = self._check_readiness(profile_name)
        assert issues == [], f"{profile_name} readiness issues: {issues}"


# ──────────────────────────────────────────────
# 15. SolverStateContext
# ──────────────────────────────────────────────

class TestSolverStateContext:

    def _make_solver_dir(self, tmp_path):
        """Create .solver/state.json with known content."""
        import json
        solver = tmp_path / ".solver"
        solver.mkdir(parents=True)

        (solver / "state.json").write_text(json.dumps({
            "state": {
                "current_attempt": "edited main.py line 12",
                "next_action": "run tests",
            },
            "trace": [
                {"step": 1, "action": "read main.py", "result": "found bug on line 12", "next": "edit"},
                {"step": 2, "action": "edit main.py", "result": "fixed bug", "next": "verify"},
                {"step": 3, "action": "bash(cmd='run tests')", "result": "2 failures", "next": "fix"},
            ],
            "gates": [
                {"name": "Correctness", "status": "PARTIAL", "notes": "8/10 tests pass"},
                {"name": "Compute", "status": "PASS", "notes": "within budget"},
            ],
            "evidence": [
                {"step": 3, "action": "bash(cmd='run verify')",
                 "result": "SENTINEL_FAIL_MARKER\n[exit code: 1]",
                 "verdict": "FAIL", "gate_blocked": False},
            ],
            "inference": [
                "Hypothesis: off-by-one in loop counter",
            ],
        }))
        return solver

    def test_builds_from_solver_files(self, tmp_path):
        from llm_solver.harness.context_strategies import SolverStateContext
        self._make_solver_dir(tmp_path)

        ctx = _make_solver_state(cwd=str(tmp_path), original_prompt="Fix the bug")
        ctx.add_system("You are a solver.")
        ctx.add_user("Fix the bug")
        # Simulate 2 turns so it switches to .solver/ mode
        ctx.add_assistant({"role": "assistant", "content": "I'll read the file"})
        ctx.add_tool_result("c1", "file contents here")
        ctx.add_assistant({"role": "assistant", "content": "I see the bug"})
        ctx.add_tool_result("c2", "OK")

        msgs = ctx.get_messages()

        # Exactly 2 messages: system + user (no conversation history)
        assert len(msgs) == 2
        assert msgs[0]["role"] == "system"
        assert msgs[1]["role"] == "user"
        assert "Fix the bug" in msgs[1]["content"]  # original prompt
        assert "edited main.py" in msgs[1]["content"]  # from state
        assert "8/10 tests pass" in msgs[1]["content"]  # from gates
        assert "SENTINEL_FAIL_MARKER" in msgs[1]["content"]  # from evidence
        assert "Hypothesis" in msgs[1]["content"]  # from inference
        assert "OK" in msgs[1]["content"]  # last tool result injected into user msg

    def test_keeps_rolling_window_of_tool_results(self, tmp_path):
        """Recent tool results persist across turns (rolling window, not
        one-turn-only). This is the fix for the code-reads-get-stubbed bug:
        a file read at turn N must still be visible at turn N+2 when the
        model makes its edit decision, otherwise the model is effectively
        editing blind.
        """
        from llm_solver.harness.context_strategies import SolverStateContext
        self._make_solver_dir(tmp_path)

        ctx = _make_solver_state(cwd=str(tmp_path), original_prompt="Task")
        ctx.add_system("sys")
        ctx.add_user("Task")
        ctx.add_assistant({"role": "assistant", "content": "first"})
        ctx.add_tool_result("c1", "first result")
        ctx.add_assistant({"role": "assistant", "content": "second response"})
        ctx.add_tool_result("c2", "second result")

        msgs = ctx.get_messages()

        # Exactly 2 messages: system + user (no raw conversation history)
        assert len(msgs) == 2
        assert msgs[0]["role"] == "system"
        assert msgs[1]["role"] == "user"
        # BOTH tool results must be present — the rolling window preserves
        # recent history so the model can see prior reads when deciding
        # what to do next.
        assert "first result" in msgs[1]["content"]
        assert "second result" in msgs[1]["content"]
        # No assistant messages in output
        assert not any(m["role"] == "assistant" for m in msgs)

    def test_tool_result_window_evicts_under_char_budget(self, tmp_path):
        """When the rolling window exceeds its char budget, oldest entries
        are evicted. Verifies the eviction is actually happening — without
        it the deque would grow unbounded across a long session.
        """
        self._make_solver_dir(tmp_path)

        ctx = _make_solver_state(cwd=str(tmp_path), original_prompt="Task")
        ctx.add_system("sys")
        ctx.add_user("Task")
        # Four 10000-char results → 40000 chars total, exceeds the 30000
        # budget. The oldest ("AAAA...") should evict.
        for i, tag in enumerate(["A", "B", "C", "D"]):
            ctx.add_assistant({"role": "assistant", "content": f"t{i}"})
            ctx.add_tool_result(f"c{i}", tag * 10000)

        msgs = ctx.get_messages()
        content = msgs[1]["content"]
        assert "D" * 100 in content  # newest present
        assert "C" * 100 in content  # second-newest present
        # Oldest ("A") should have been evicted to fit the budget.
        assert "A" * 10000 not in content
        # The deque itself should have been trimmed, not just the render.
        assert len(ctx._recent_tool_results) < 4

    def test_fallback_when_no_solver(self, tmp_path):
        from llm_solver.harness.context_strategies import SolverStateContext

        # No .solver/ directory
        ctx = _make_solver_state(cwd=str(tmp_path), original_prompt="Task")
        ctx.add_system("sys")
        ctx.add_user("Task")
        ctx.add_assistant({"role": "assistant", "content": "response"})

        msgs = ctx.get_messages()

        # Falls back to full history
        assert len(msgs) == 3
        assert msgs[0]["role"] == "system"
        assert msgs[1]["role"] == "user"
        assert msgs[2]["role"] == "assistant"

    def test_fallback_early_turns(self, tmp_path):
        from llm_solver.harness.context_strategies import SolverStateContext
        self._make_solver_dir(tmp_path)

        ctx = _make_solver_state(cwd=str(tmp_path), original_prompt="Task")
        ctx.add_system("sys")
        ctx.add_user("Task")
        # Only 1 turn — too early for .solver/ mode
        ctx.add_assistant({"role": "assistant", "content": "first"})

        msgs = ctx.get_messages()
        # Should be raw history, not .solver/ build
        assert len(msgs) == 3

    def test_truncates_long_trace(self, tmp_path):
        import json
        from llm_solver.harness.context_strategies import SolverStateContext
        solver = tmp_path / ".solver"
        solver.mkdir(parents=True)

        # 200 trace entries — only the last 10 should render
        trace = [
            {"step": i, "action": f"action_{i}", "result": "result", "next": "next"}
            for i in range(200)
        ]
        (solver / "state.json").write_text(json.dumps({
            "state": {"current_attempt": "working"},
            "trace": trace,
        }))

        ctx = _make_solver_state(cwd=str(tmp_path), original_prompt="Task", trace_lines=10)
        ctx.add_system("sys")
        ctx.add_user("Task")
        ctx.add_assistant({"role": "assistant", "content": "a"})
        ctx.add_tool_result("c1", "b")
        ctx.add_assistant({"role": "assistant", "content": "c"})

        msgs = ctx.get_messages()
        user_msg = msgs[1]["content"]

        # Should contain last 10 entries, not all 200
        assert "action_199" in user_msg
        assert "action_190" in user_msg
        assert "action_0 " not in user_msg
        assert "action_0\n" not in user_msg

    def test_estimate_tokens(self, tmp_path):
        from llm_solver.harness.context_strategies import SolverStateContext
        self._make_solver_dir(tmp_path)

        ctx = _make_solver_state(cwd=str(tmp_path), original_prompt="Task")
        ctx.add_system("sys")
        ctx.add_user("Task")
        ctx.add_assistant({"role": "assistant", "content": "response"})
        ctx.add_tool_result("c1", "result")
        ctx.add_assistant({"role": "assistant", "content": "next"})

        tokens = ctx.estimate_tokens()
        assert isinstance(tokens, int)
        assert tokens > 0

    def test_message_count_tracks_all(self, tmp_path):
        from llm_solver.harness.context_strategies import SolverStateContext
        ctx = _make_solver_state(cwd=str(tmp_path), original_prompt="Task")
        assert ctx.message_count() == 0
        ctx.add_system("sys")
        assert ctx.message_count() == 1
        ctx.add_user("Task")
        ctx.add_assistant({"role": "assistant", "content": "a"})
        ctx.add_tool_result("c1", "b")
        assert ctx.message_count() == 4

    def test_session_uses_solver_context_by_default(self, tmp_path):
        from llm_solver.harness.loop import Session
        from llm_solver.harness.context_strategies import SolverStateContext
        cfg = make_config()
        client = MagicMock()
        client.build_assistant_message.return_value = {"role": "assistant", "content": ""}
        session = Session(cfg, client, "sys", "prompt", str(tmp_path))
        assert isinstance(session.context, SolverStateContext)

    def test_file_cache_avoids_redundant_reads(self, tmp_path):
        from llm_solver.harness.context_strategies import SolverStateContext
        self._make_solver_dir(tmp_path)

        ctx = _make_solver_state(cwd=str(tmp_path), original_prompt="Task")
        ctx.add_system("sys")
        ctx.add_user("Task")
        ctx.add_assistant({"role": "assistant", "content": "a"})
        ctx.add_tool_result("c1", "b")
        ctx.add_assistant({"role": "assistant", "content": "c"})

        # First call populates cache
        msgs1 = ctx.get_messages()
        # Second call (e.g. estimate_tokens) should reuse cache
        cache_after_first = ctx._file_cache
        msgs2 = ctx.get_messages()
        assert ctx._file_cache is cache_after_first  # same object, not re-read

        # add_tool_result invalidates cache
        ctx.add_tool_result("c2", "d")
        assert ctx._file_cache is None

        # Next get_messages re-reads
        msgs3 = ctx.get_messages()
        assert ctx._file_cache is not None
        assert ctx._file_cache is not cache_after_first


class TestEscalatingDedup:
    """Escalating context dedup: 1st dedup = warning, 2nd+ = hard block."""

    def test_first_dedup_returns_warning(self, tmp_path):
        ctx = _make_solver_state(cwd=str(tmp_path), original_prompt="Task")
        ctx.add_system("sys")
        ctx.add_user("Task")
        ctx.add_assistant({"role": "assistant", "content": "a"})
        ctx.add_tool_result("c1", "x" * 300)  # original
        ctx.add_assistant({"role": "assistant", "content": "b"})
        ctx.add_tool_result("c2", "x" * 300)  # 1st dedup (2nd attempt)

        last = ctx._recent_tool_results[-1]["content"]
        assert "WARNING: Same output as turn" in last
        assert "change your approach" in last
        assert "BLOCKED" not in last

    def test_second_dedup_returns_hard_block(self, tmp_path):
        ctx = _make_solver_state(cwd=str(tmp_path), original_prompt="Task")
        ctx.add_system("sys")
        ctx.add_user("Task")
        ctx.add_assistant({"role": "assistant", "content": "a"})
        ctx.add_tool_result("c1", "x" * 300)  # original
        ctx.add_assistant({"role": "assistant", "content": "b"})
        ctx.add_tool_result("c2", "x" * 300)  # 1st dedup
        ctx.add_assistant({"role": "assistant", "content": "c"})
        ctx.add_tool_result("c3", "x" * 300)  # 2nd dedup (3rd attempt)

        last = ctx._recent_tool_results[-1]["content"]
        assert "ERROR: BLOCKED" in last
        assert "ACTION REQUIRED" in last

    def test_third_dedup_still_blocked(self, tmp_path):
        ctx = _make_solver_state(cwd=str(tmp_path), original_prompt="Task")
        ctx.add_system("sys")
        ctx.add_user("Task")
        ctx.add_assistant({"role": "assistant", "content": "a"})
        ctx.add_tool_result("c1", "x" * 300)
        for i in range(3):
            ctx.add_assistant({"role": "assistant", "content": f"t{i}"})
            ctx.add_tool_result(f"d{i}", "x" * 300)

        last = ctx._recent_tool_results[-1]["content"]
        assert "ERROR: BLOCKED" in last
        assert "4 times" in last

    def test_reset_clears_escalation(self, tmp_path):
        """After reset (edit), identical content from a pre-edit entry
        must NOT trigger dedup — the edit may have changed behavior."""
        ctx = _make_solver_state(cwd=str(tmp_path), original_prompt="Task")
        ctx.add_system("sys")
        ctx.add_user("Task")
        ctx.add_assistant({"role": "assistant", "content": "a"})
        ctx.add_tool_result("c1", "x" * 300)
        ctx.add_assistant({"role": "assistant", "content": "b"})
        ctx.add_tool_result("c2", "x" * 300)  # 1st dedup → warning

        # Simulate successful write/edit
        ctx.reset_dedup_counts()

        # After reset, same content must pass through — the edit
        # invalidated the assumption that output won't change.
        ctx.add_assistant({"role": "assistant", "content": "c"})
        ctx.add_tool_result("c3", "x" * 300)

        last = ctx._recent_tool_results[-1]["content"]
        assert last == "x" * 300  # no dedup

    def test_short_content_not_deduped(self, tmp_path):
        """Content under 200 chars is not deduped (likely error messages)."""
        ctx = _make_solver_state(cwd=str(tmp_path), original_prompt="Task")
        ctx.add_system("sys")
        ctx.add_user("Task")
        ctx.add_assistant({"role": "assistant", "content": "a"})
        ctx.add_tool_result("c1", "short")
        ctx.add_assistant({"role": "assistant", "content": "b"})
        ctx.add_tool_result("c2", "short")

        last = ctx._recent_tool_results[-1]["content"]
        assert last == "short"  # no dedup

    def test_different_content_not_deduped(self, tmp_path):
        ctx = _make_solver_state(cwd=str(tmp_path), original_prompt="Task")
        ctx.add_system("sys")
        ctx.add_user("Task")
        ctx.add_assistant({"role": "assistant", "content": "a"})
        ctx.add_tool_result("c1", "x" * 300)
        ctx.add_assistant({"role": "assistant", "content": "b"})
        ctx.add_tool_result("c2", "y" * 300)  # different content

        last = ctx._recent_tool_results[-1]["content"]
        assert last == "y" * 300  # no dedup


class TestCommandSignatureDedup:
    """Command-signature dedup: catches bash pipe variations reading the same data."""

    def test_read_cmd_warning(self, tmp_path):
        """cat file then cat file | head -100 → WARNING with read-specific message."""
        ctx = _make_solver_state(cwd=str(tmp_path), original_prompt="Task")
        ctx.add_system("sys")
        ctx.add_user("Task")
        ctx.add_assistant({"role": "assistant", "content": "a"})
        ctx.add_tool_result("c1", "x" * 300, tool_name="bash",
                           cmd_signature='{"cmd": "cat file.py"}')
        ctx.add_assistant({"role": "assistant", "content": "b"})
        ctx.add_tool_result("c2", "y" * 300, tool_name="bash",
                           cmd_signature='{"cmd": "cat file.py"}')

        last = ctx._recent_tool_results[-1]["content"]
        assert "WARNING: You already ran `cat file.py`" in last
        assert "Edit the file or move on" in last

    def test_read_cmd_escalates_to_block(self, tmp_path):
        """Third cat with same signature → BLOCKED with read message."""
        ctx = _make_solver_state(cwd=str(tmp_path), original_prompt="Task")
        ctx.add_system("sys")
        ctx.add_user("Task")
        ctx.add_assistant({"role": "assistant", "content": "a"})
        ctx.add_tool_result("c1", "x" * 300, tool_name="bash",
                           cmd_signature='{"cmd": "cat file.py"}')
        ctx.add_assistant({"role": "assistant", "content": "b"})
        ctx.add_tool_result("c2", "y" * 300, tool_name="bash",
                           cmd_signature='{"cmd": "cat file.py"}')
        ctx.add_assistant({"role": "assistant", "content": "c"})
        ctx.add_tool_result("c3", "z" * 300, tool_name="bash",
                           cmd_signature='{"cmd": "cat file.py"}')

        last = ctx._recent_tool_results[-1]["content"]
        assert "ERROR: BLOCKED" in last
        assert "`cat file.py` ran 3 times" in last
        assert "Stop reading this file" in last

    def test_test_cmd_echoes_previous_error(self, tmp_path):
        """pytest dedup echoes the E-line from the previous failure."""
        ctx = _make_solver_state(cwd=str(tmp_path), original_prompt="Task")
        ctx.add_system("sys")
        ctx.add_user("Task")
        prev_output = (
            "FAILURES\n"
            "tests/test_foo.py:10: in test_bar\n"
            "E   TypeError: missing 1 required positional argument: '_y'\n"
            "short test summary info\n"
            "FAILED tests/test_foo.py::test_bar\n"
            "1 failed in 0.5s\n" + "x" * 200
        )
        ctx.add_assistant({"role": "assistant", "content": "a"})
        ctx.add_tool_result("c1", prev_output, tool_name="bash",
                           cmd_signature='{"cmd": "pytest tests/test_foo.py -v"}')
        ctx.add_assistant({"role": "assistant", "content": "b"})
        ctx.add_tool_result("c2", prev_output, tool_name="bash",
                           cmd_signature='{"cmd": "pytest tests/test_foo.py -v"}')

        last = ctx._recent_tool_results[-1]["content"]
        assert "WARNING: You already ran `pytest tests/test_foo.py -v`" in last
        assert "TypeError: missing 1 required positional argument: '_y'" in last
        assert "Your last edit didn't fix this" in last

    def test_test_cmd_block_message(self, tmp_path):
        """pytest BLOCKED message includes error and test-specific framing."""
        ctx = _make_solver_state(cwd=str(tmp_path), original_prompt="Task")
        ctx.add_system("sys")
        ctx.add_user("Task")
        prev_output = "E   ValueError: bad\n" + "x" * 200
        ctx.add_assistant({"role": "assistant", "content": "a"})
        ctx.add_tool_result("c1", prev_output, tool_name="bash",
                           cmd_signature='{"cmd": "pytest tests/ -v"}')
        ctx.add_assistant({"role": "assistant", "content": "b"})
        ctx.add_tool_result("c2", "y" * 300, tool_name="bash",
                           cmd_signature='{"cmd": "pytest tests/ -v"}')
        ctx.add_assistant({"role": "assistant", "content": "c"})
        ctx.add_tool_result("c3", "z" * 300, tool_name="bash",
                           cmd_signature='{"cmd": "pytest tests/ -v"}')

        last = ctx._recent_tool_results[-1]["content"]
        assert "ERROR: BLOCKED" in last
        assert "ValueError: bad" in last
        assert "make a different change" in last

    def test_search_cmd_message(self, tmp_path):
        """grep dedup uses search-specific message."""
        ctx = _make_solver_state(cwd=str(tmp_path), original_prompt="Task")
        ctx.add_system("sys")
        ctx.add_user("Task")
        ctx.add_assistant({"role": "assistant", "content": "a"})
        ctx.add_tool_result("c1", "x" * 300, tool_name="bash",
                           cmd_signature='{"cmd": "grep -r pattern ."}')
        ctx.add_assistant({"role": "assistant", "content": "b"})
        ctx.add_tool_result("c2", "y" * 300, tool_name="bash",
                           cmd_signature='{"cmd": "grep -r pattern ."}')

        last = ctx._recent_tool_results[-1]["content"]
        assert "WARNING" in last
        assert "Act on them" in last

    def test_different_files_no_match(self, tmp_path):
        """cat foo.py then cat bar.py → no dedup (different signatures)."""
        ctx = _make_solver_state(cwd=str(tmp_path), original_prompt="Task")
        ctx.add_system("sys")
        ctx.add_user("Task")
        ctx.add_assistant({"role": "assistant", "content": "a"})
        ctx.add_tool_result("c1", "x" * 300, tool_name="bash",
                           cmd_signature='{"cmd": "cat foo.py"}')
        ctx.add_assistant({"role": "assistant", "content": "b"})
        ctx.add_tool_result("c2", "y" * 300, tool_name="bash",
                           cmd_signature='{"cmd": "cat bar.py"}')

        last = ctx._recent_tool_results[-1]["content"]
        assert last == "y" * 300  # no dedup

    def test_reset_clears_cmd_sig_escalation(self, tmp_path):
        """reset_dedup_counts clears escalation AND strips cmd_sig from
        the rolling window so post-edit reruns don't match pre-edit entries."""
        ctx = _make_solver_state(cwd=str(tmp_path), original_prompt="Task")
        ctx.add_system("sys")
        ctx.add_user("Task")
        ctx.add_assistant({"role": "assistant", "content": "a"})
        ctx.add_tool_result("c1", "x" * 300, tool_name="bash",
                           cmd_signature='{"cmd": "cat file.py"}')
        ctx.add_assistant({"role": "assistant", "content": "b"})
        ctx.add_tool_result("c2", "y" * 300, tool_name="bash",
                           cmd_signature='{"cmd": "cat file.py"}')  # WARNING

        ctx.reset_dedup_counts()

        # After reset, same command must NOT match — the edit invalidated
        # the old result. No WARNING, no BLOCKED, just the raw content.
        ctx.add_assistant({"role": "assistant", "content": "c"})
        ctx.add_tool_result("c3", "z" * 300, tool_name="bash",
                           cmd_signature='{"cmd": "cat file.py"}')

        last = ctx._recent_tool_results[-1]["content"]
        assert last == "z" * 300  # no dedup

    def test_content_dedup_still_fires_without_cmd_sig(self, tmp_path):
        """Byte-identical content dedup still works for non-bash tools."""
        ctx = _make_solver_state(cwd=str(tmp_path), original_prompt="Task")
        ctx.add_system("sys")
        ctx.add_user("Task")
        ctx.add_assistant({"role": "assistant", "content": "a"})
        ctx.add_tool_result("c1", "x" * 300, tool_name="bash")  # no cmd_signature
        ctx.add_assistant({"role": "assistant", "content": "b"})
        ctx.add_tool_result("c2", "x" * 300, tool_name="bash")  # identical content

        last = ctx._recent_tool_results[-1]["content"]
        assert "WARNING: Same output as turn" in last  # content dedup fires


class TestSummarizeArgs:

    def test_short_args(self):
        from llm_solver.harness.loop import _summarize_args
        result = _summarize_args({"cmd": "ls", "path": "/tmp"}, max_chars=80)
        assert "cmd='ls'" in result
        assert "path='/tmp'" in result

    def test_long_args_truncated(self):
        from llm_solver.harness.loop import _summarize_args
        result = _summarize_args({"content": "x" * 100}, max_chars=80)
        assert "..." in result
        assert len(result) < 200


# ──────────────────────────────────────────────
# 12. Session accessors for cross-session learning
# ──────────────────────────────────────────────

class TestSessionAccessors:

    def test_last_tool_calls(self):
        from llm_solver.harness.loop import Session
        cfg = make_config(duplicate_abort=3)
        client = MagicMock()
        client.build_assistant_message.return_value = {"role": "assistant", "content": ""}
        session = Session(cfg, client, "sys", "msg", "/tmp")
        session._tool_log = [("a", "x"), ("b", "y"), ("c", "z"), ("d", "w")]
        assert len(session.last_tool_calls) == 3
        assert session.last_tool_calls[-1] == ("d", "w")

    def test_context_fill_ratio(self):
        from llm_solver.harness.loop import Session
        cfg = make_config()
        client = MagicMock()
        client.build_assistant_message.return_value = {"role": "assistant", "content": ""}
        session = Session(cfg, client, "sys", "msg", "/tmp")
        assert session.context_fill_ratio == 0.0
        session._last_fill = 0.87
        assert session.context_fill_ratio == 0.87


# ──────────────────────────────────────────────
# CompactTranscript
# ──────────────────────────────────────────────

_COMPACT_DEFAULTS = dict(
    recent_results_chars=30000,
    trace_reasoning_chars=150,
    min_turns=2,
    args_summary_chars=80,
)


def _make_compact(**overrides):
    from llm_solver.harness.context_strategies import CompactTranscript
    kwargs = dict(_COMPACT_DEFAULTS)
    kwargs.update(overrides)
    return CompactTranscript(**kwargs)


_SOLVER_STATE_DEFAULTS = dict(
    trace_lines=50,
    evidence_lines=30,
    inference_lines=20,
    recent_tool_results_chars=30000,
    trace_stub_chars=200,
    min_turns=2,
    suffix="Continue working. Your progress is tracked in .solver/state.json — read it to see what you've already done.",
)


def _make_solver_state(**overrides):
    from llm_solver.harness.context_strategies import SolverStateContext
    kwargs = dict(_SOLVER_STATE_DEFAULTS)
    kwargs.update(overrides)
    return SolverStateContext(**kwargs)


def _make_yuj_transcript(**overrides):
    from llm_solver.harness.context_strategies import YujTranscript
    kwargs = dict(_COMPACT_DEFAULTS)
    kwargs.update(overrides)
    return YujTranscript(**kwargs)


class TestCompactTranscript:

    def _make_assistant_msg(self, content, tool_name, tool_args, call_id="c1"):
        return {
            "role": "assistant",
            "content": content,
            "tool_calls": [{
                "id": call_id,
                "type": "function",
                "function": {
                    "name": tool_name,
                    "arguments": json.dumps(tool_args),
                },
            }],
        }

    def test_fallback_early_turns(self):
        ctx = _make_compact(original_prompt="Task X")
        ctx.add_system("sys")
        ctx.add_user("Task X")
        ctx.add_assistant(self._make_assistant_msg("thinking", "bash", {"cmd": "ls"}))
        # Only 1 turn — should return raw messages
        msgs = ctx.get_messages()
        assert len(msgs) == 3
        assert msgs[0]["role"] == "system"

    def test_compact_after_two_turns(self):
        ctx = _make_compact(original_prompt="Implement feature X")
        ctx.add_system("sys prompt")
        ctx.add_user("Implement feature X")

        # Turn 1
        ctx.add_assistant(self._make_assistant_msg("Exploring repo", "bash", {"cmd": "ls"}, "c1"))
        ctx.add_tool_result("c1", "file1.py\nfile2.py")

        # Turn 2
        ctx.add_assistant(self._make_assistant_msg("Writing code", "write", {"path": "f.py", "content": "x"}, "c2"))
        ctx.add_tool_result("c2", "OK: wrote 1 bytes to f.py")

        msgs = ctx.get_messages()
        assert len(msgs) == 2
        assert msgs[0]["role"] == "system"
        assert msgs[0]["content"] == "sys prompt"
        assert msgs[1]["role"] == "user"
        user = msgs[1]["content"]
        assert "Implement feature X" in user
        assert "Progress:" in user
        assert "Exploring repo" in user
        assert "Writing code" in user
        assert "bash(" in user
        assert "write(" in user

    def test_outcome_classification(self):
        from llm_solver.harness.context_strategies.compact_transcript import _classify_outcome
        # Content-blind classification: only harness-generated markers count.
        assert _classify_outcome("file1.py\nfile2.py") == "OK"
        assert _classify_outcome("OK: wrote 10 bytes") == "OK"
        # ERROR: wrapper is harness-generated (tools.py wraps exceptions).
        assert _classify_outcome("ERROR: file not found: foo.py") == "FAIL"
        # Tracebacks are task content — harness does NOT parse them.
        assert _classify_outcome(
            "Traceback (most recent call last):\n  File...\nNameError: name 'x'"
        ) == "OK"
        # [exit code: N] is harness-generated by bash() when N != 0.
        assert _classify_outcome("output\n[exit code: 1]") == "FAIL"
        assert _classify_outcome("output\n[exit code: 0]") == "OK"

    def test_recent_results_unbounded_when_small(self):
        """With small results, the char-budget window keeps every turn.
        Previously the deque was maxlen=3 and older results were dropped
        even when they were tiny, which starved the model of recent code
        reads. Now small results accumulate and the model sees them all.
        """
        ctx = _make_compact(original_prompt="Task")
        ctx.add_system("sys")
        ctx.add_user("Task")

        for i in range(5):
            ctx.add_assistant(self._make_assistant_msg(f"step {i}", "bash", {"cmd": f"echo {i}"}, f"c{i}"))
            ctx.add_tool_result(f"c{i}", f"result_{i}")

        msgs = ctx.get_messages()
        user = msgs[1]["content"]
        # Every result is present — tiny results fit under the budget.
        for i in range(5):
            assert f"result_{i}" in user

    def test_recent_results_char_budget_evicts_oldest(self):
        """When the total char budget overflows, oldest results evict. The
        rolling window trims both the render AND the underlying deque so
        memory stays bounded across long sessions.
        """
        # 10000-char budget so 3 x 5000-char results overflow and one evicts.
        ctx = _make_compact(original_prompt="Task", recent_results_chars=10000)
        ctx.add_system("sys")
        ctx.add_user("Task")

        for i, tag in enumerate(["A", "B", "C"]):
            ctx.add_assistant(self._make_assistant_msg(f"step {i}", "bash", {"cmd": f"echo {i}"}, f"c{i}"))
            ctx.add_tool_result(f"c{i}", tag * 5000)

        msgs = ctx.get_messages()
        user = msgs[1]["content"]
        assert "C" * 100 in user  # newest
        assert "B" * 100 in user  # fits
        assert "A" * 5000 not in user  # evicted
        assert len(ctx._recent_results) < 3

    def test_progress_log_has_all_turns(self):
        ctx = _make_compact(original_prompt="Task")
        ctx.add_system("sys")
        ctx.add_user("Task")

        for i in range(10):
            ctx.add_assistant(self._make_assistant_msg(f"action {i}", "bash", {"cmd": f"cmd{i}"}, f"c{i}"))
            ctx.add_tool_result(f"c{i}", f"ok_{i}")

        msgs = ctx.get_messages()
        user = msgs[1]["content"]
        # All 10 turns in progress log
        for i in range(10):
            assert f"action {i}" in user

    def test_empty_reasoning_handled(self):
        ctx = _make_compact(original_prompt="Task")
        ctx.add_system("sys")
        ctx.add_user("Task")

        # Assistant with no content (just tool calls)
        msg = self._make_assistant_msg(None, "bash", {"cmd": "ls"}, "c1")
        msg["content"] = None
        ctx.add_assistant(msg)
        ctx.add_tool_result("c1", "files")

        ctx.add_assistant(self._make_assistant_msg("second", "bash", {"cmd": "pwd"}, "c2"))
        ctx.add_tool_result("c2", "/tmp")

        msgs = ctx.get_messages()
        user = msgs[1]["content"]
        assert "bash(" in user  # structural still present even without reasoning

    def test_estimate_tokens(self):
        ctx = _make_compact(original_prompt="Task")
        ctx.add_system("sys")
        ctx.add_user("Task")
        ctx.add_assistant(self._make_assistant_msg("a", "bash", {"cmd": "ls"}, "c1"))
        ctx.add_tool_result("c1", "ok")
        ctx.add_assistant(self._make_assistant_msg("b", "bash", {"cmd": "pwd"}, "c2"))
        ctx.add_tool_result("c2", "ok")
        tokens = ctx.estimate_tokens()
        assert isinstance(tokens, int)
        assert tokens > 0

    def test_multi_tool_calls(self):
        ctx = _make_compact(original_prompt="Task")
        ctx.add_system("sys")
        ctx.add_user("Task")

        # Assistant with two tool calls in one turn
        msg = {
            "role": "assistant",
            "content": "Reading two files",
            "tool_calls": [
                {"id": "c1", "type": "function", "function": {"name": "read", "arguments": '{"path": "a.py"}'}},
                {"id": "c2", "type": "function", "function": {"name": "read", "arguments": '{"path": "b.py"}'}},
            ],
        }
        ctx.add_assistant(msg)
        ctx.add_tool_result("c1", "contents of a")
        ctx.add_tool_result("c2", "contents of b")

        # Second turn to trigger compact mode
        ctx.add_assistant(self._make_assistant_msg("next step", "bash", {"cmd": "ls"}, "c3"))
        ctx.add_tool_result("c3", "files")

        msgs = ctx.get_messages()
        user = msgs[1]["content"]
        # Both tool calls should appear in progress
        assert "read(" in user
        assert "Reading two files" in user
        # Second tool call of same turn also recorded (without reasoning)
        assert user.count("read(") >= 2


# ──────────────────────────────────────────────
# YujTranscript
# ──────────────────────────────────────────────

class TestYujTranscript:

    def _make_assistant_msg(self, content, tool_name, tool_args, call_id="c1"):
        return {
            "role": "assistant",
            "content": content,
            "tool_calls": [{
                "id": call_id,
                "type": "function",
                "function": {
                    "name": tool_name,
                    "arguments": json.dumps(tool_args),
                },
            }],
        }

    def test_yuj_sections_present(self):
        ctx = _make_yuj_transcript(original_prompt="Fix the bug")
        ctx.add_system("sys")
        ctx.add_user("Fix the bug")

        ctx.add_assistant(self._make_assistant_msg("Reading code", "read", {"path": "a.py"}, "c1"))
        ctx.add_tool_result("c1", "def foo(): pass")

        # Non-zero exit code is harness-generated — classify_outcome will mark FAIL.
        ctx.add_assistant(self._make_assistant_msg("Running verification", "bash", {"cmd": "./verify"}, "c2"))
        ctx.add_tool_result("c2", "some output\n[exit code: 1]")

        msgs = ctx.get_messages()
        assert len(msgs) == 2
        user = msgs[1]["content"]

        # Protocol sections present
        assert "=== Trace ===" in user
        assert "=== Evidence (unresolved) ===" in user
        assert "=== Gate (blocking) ===" in user
        assert "=== State ===" in user

        # Trace has all actions
        assert "Reading code" in user
        assert "Running verification" in user

        # Evidence has only failures (content-blind: exit code was non-zero).
        assert "FAIL" in user

        # State has turn info
        assert "Turn:" in user

    def test_no_evidence_when_all_pass(self):
        ctx = _make_yuj_transcript(original_prompt="Task")
        ctx.add_system("sys")
        ctx.add_user("Task")

        ctx.add_assistant(self._make_assistant_msg("step 1", "bash", {"cmd": "ls"}, "c1"))
        ctx.add_tool_result("c1", "files")

        ctx.add_assistant(self._make_assistant_msg("step 2", "bash", {"cmd": "pwd"}, "c2"))
        ctx.add_tool_result("c2", "/tmp")

        msgs = ctx.get_messages()
        user = msgs[1]["content"]

        assert "=== Trace ===" in user
        assert "=== Evidence" not in user  # no failures
        assert "=== Gate" not in user

    def test_inherits_compact_behavior(self):
        ctx = _make_yuj_transcript(original_prompt="Task")
        ctx.add_system("sys")
        ctx.add_user("Task")

        # Fallback on turn 0
        ctx.add_assistant(self._make_assistant_msg("a", "bash", {"cmd": "ls"}, "c1"))
        msgs = ctx.get_messages()
        assert len(msgs) == 3  # raw messages, not compact

        # After turn 2, compact
        ctx.add_tool_result("c1", "ok")
        ctx.add_assistant(self._make_assistant_msg("b", "bash", {"cmd": "pwd"}, "c2"))
        ctx.add_tool_result("c2", "ok")

        msgs = ctx.get_messages()
        assert len(msgs) == 2  # compact mode


# ──────────────────────────────────────────────
# 16. Turn-level error detection (#18)
# ──────────────────────────────────────────────

class TestErrorDetection:

    def test_nudge_after_consecutive_errors(self):
        """After cfg.error_nudge_threshold consecutive errors from same tool, nudge appended."""
        from llm_solver.harness.loop import Session
        threshold = 3
        cfg = make_config(max_turns=threshold + 2, duplicate_abort=20,
                          error_nudge_threshold=threshold)
        client = MagicMock()

        call_count = [0]
        def chat_fn(*args, **kwargs):
            call_count[0] += 1
            if call_count[0] > threshold:
                return make_turn_result(content="done", finish_reason="stop")
            tc = [ToolCall(id=f"c{call_count[0]}", name="edit",
                          arguments={"path": f"f{call_count[0]}.py", "old_str": "a", "new_str": "b"})]
            return make_turn_result(tool_calls=tc, finish_reason="tool_calls")

        client.chat.side_effect = chat_fn
        client.build_assistant_message.return_value = {"role": "assistant", "content": None}

        tool_results = []
        with patch("llm_solver.harness.loop.dispatch", return_value="ERROR: old_str not found"):
            session = Session(cfg, client, "sys", "prompt", "/tmp")
            orig_add = session.context.add_tool_result
            def capture(cid, result, **kwargs):
                tool_results.append(result)
                return orig_add(cid, result, **kwargs)
            session.context.add_tool_result = capture
            session.run()

        nudge_results = [r for r in tool_results if "[harness:" in r]
        assert len(nudge_results) >= 1, f"No nudge in results: {tool_results}"
        # Original error preserved
        assert nudge_results[0].startswith("ERROR:")
        # Nudge mentions consecutive count
        assert f"{threshold} consecutive" in nudge_results[0]

    def test_error_counter_resets_on_success(self):
        """Successful dispatch resets consecutive error counter — no nudge emitted."""
        from llm_solver.harness.loop import Session
        cfg = make_config(max_turns=10, duplicate_abort=20, error_nudge_threshold=3)
        client = MagicMock()

        call_count = [0]
        def chat_fn(*args, **kwargs):
            call_count[0] += 1
            if call_count[0] > 4:
                return make_turn_result(content="done", finish_reason="stop")
            tc = [ToolCall(id=f"c{call_count[0]}", name="bash",
                          arguments={"cmd": f"cmd{call_count[0]}"})]
            return make_turn_result(tool_calls=tc, finish_reason="tool_calls")

        client.chat.side_effect = chat_fn
        client.build_assistant_message.return_value = {"role": "assistant", "content": None}

        # Alternate error/success — never reaches threshold
        dispatch_returns = iter(["ERROR: fail", "ok", "ERROR: fail", "ok"])
        tool_results = []
        with patch("llm_solver.harness.loop.dispatch", side_effect=dispatch_returns):
            session = Session(cfg, client, "sys", "prompt", "/tmp")
            orig_add = session.context.add_tool_result
            def capture(cid, result, **kwargs):
                tool_results.append(result)
                return orig_add(cid, result, **kwargs)
            session.context.add_tool_result = capture
            session.run()

        nudge_results = [r for r in tool_results if "[harness:" in r]
        assert len(nudge_results) == 0, f"Unexpected nudge: {nudge_results}"

    def test_error_detection_logs_events(self):
        """Error events are logged at INFO level."""
        from llm_solver.harness.loop import Session
        cfg = make_config(max_turns=3, duplicate_abort=20)
        client = MagicMock()

        call_count = [0]
        def chat_fn(*args, **kwargs):
            call_count[0] += 1
            if call_count[0] > 1:
                return make_turn_result(content="done", finish_reason="stop")
            tc = [ToolCall(id="c1", name="bash", arguments={"cmd": "bad"})]
            return make_turn_result(tool_calls=tc, finish_reason="tool_calls")

        client.chat.side_effect = chat_fn
        client.build_assistant_message.return_value = {"role": "assistant", "content": None}

        import logging
        with patch("llm_solver.harness.loop.dispatch", return_value="ERROR: fail"):
            session = Session(cfg, client, "sys", "prompt", "/tmp")
            with patch("llm_solver.harness.loop.log") as mock_log:
                session.run()

        # At least one INFO log about tool error
        info_calls = [str(c) for c in mock_log.info.call_args_list]
        assert any("error" in c.lower() or "ERROR" in c for c in info_calls)


class TestRuminationNudge:
    """Counterpart to error_nudge: detects model stuck reading/grepping
    without committing to a write, and appends a guidance message to the
    next tool result so the model has an off-ramp from exploration."""

    def test_nudge_after_threshold_non_write_calls(self):
        """After rumination threshold non-write tool calls in a row,
        a nudge is appended to the next tool result."""
        from llm_solver.harness.loop import Session
        threshold = 7  # at max_turns=100, 7% → 7 absolute (above floor of 6)
        cfg = make_config(max_turns=100, duplicate_abort=20,
                          rumination_nudge_threshold=threshold)
        client = MagicMock()

        call_count = [0]
        def chat_fn(*args, **kwargs):
            call_count[0] += 1
            if call_count[0] > threshold + 1:
                return make_turn_result(content="done", finish_reason="stop")
            tc = [ToolCall(id=f"c{call_count[0]}", name="bash",
                          arguments={"cmd": f"find . -name file{call_count[0]}.py"})]
            return make_turn_result(tool_calls=tc, finish_reason="tool_calls")

        client.chat.side_effect = chat_fn
        client.build_assistant_message.return_value = {"role": "assistant", "content": None}

        tool_results = []
        with patch("llm_solver.harness.loop.dispatch", return_value="some_output.py"):
            session = Session(cfg, client, "sys", "prompt", "/tmp")
            orig_add = session.context.add_tool_result
            def capture(cid, result, **kwargs):
                tool_results.append(result)
                return orig_add(cid, result, **kwargs)
            session.context.add_tool_result = capture
            session.run()

        nudge_results = [r for r in tool_results if "[harness:" in r and "non-write tool calls since" in r]
        assert len(nudge_results) >= 1, f"No rumination nudge in: {tool_results}"
        # The nudge must preserve the original tool output above it.
        assert "some_output.py" in nudge_results[0]
        # Nudge text must mention the threshold count and the hard gate.
        assert f"{threshold}" in nudge_results[0]
        assert "must be write or edit" in nudge_results[0]
        assert "rejected" in nudge_results[0]

    def test_write_resets_rumination_counter(self):
        """A write/edit call resets the non-write counter — no nudge fires if
        the model regularly alternates reads with writes."""
        from llm_solver.harness.loop import Session
        cfg = make_config(max_turns=100, duplicate_abort=20,
                          rumination_nudge_threshold=7)  # 7% of 100 → 7 absolute (above floor of 6)
        client = MagicMock()

        # Sequence: bash, bash, edit, bash, bash, stop
        # Each bash increments; the edit resets. Max streak = 2, below threshold 7.
        tool_sequence = [
            ("bash", {"cmd": "ls"}),
            ("bash", {"cmd": "cat f.py"}),
            ("edit", {"path": "f.py", "old_str": "x", "new_str": "y"}),
            ("bash", {"cmd": "ls"}),
            ("bash", {"cmd": "cat g.py"}),
        ]
        call_count = [0]
        def chat_fn(*args, **kwargs):
            call_count[0] += 1
            if call_count[0] > len(tool_sequence):
                return make_turn_result(content="done", finish_reason="stop")
            name, args = tool_sequence[call_count[0] - 1]
            tc = [ToolCall(id=f"c{call_count[0]}", name=name, arguments=args)]
            return make_turn_result(tool_calls=tc, finish_reason="tool_calls")

        client.chat.side_effect = chat_fn
        client.build_assistant_message.return_value = {"role": "assistant", "content": None}

        tool_results = []
        with patch("llm_solver.harness.loop.dispatch", return_value="ok"):
            session = Session(cfg, client, "sys", "prompt", "/tmp")
            orig_add = session.context.add_tool_result
            def capture(cid, result, **kwargs):
                tool_results.append(result)
                return orig_add(cid, result, **kwargs)
            session.context.add_tool_result = capture
            session.run()

        nudge_results = [r for r in tool_results if "tool calls since your last write" in r]
        assert len(nudge_results) == 0, f"Unexpected rumination nudge: {nudge_results}"

    def test_rumination_nudge_fires_once_per_cycle(self):
        """Nudge text is one-shot per non-write cycle. Once fired, it does
        not re-fire until a successful write/edit resets the cycle —
        otherwise every subsequent non-write call would carry an identical
        append and become noise."""
        from llm_solver.harness.loop import Session
        threshold = 7  # 7% of 100 → 7 absolute (above floor of 6)
        cfg = make_config(max_turns=100, duplicate_abort=20,
                          rumination_nudge_threshold=threshold)
        client = MagicMock()

        call_count = [0]
        def chat_fn(*args, **kwargs):
            call_count[0] += 1
            if call_count[0] > threshold * 2:
                return make_turn_result(content="done", finish_reason="stop")
            tc = [ToolCall(id=f"c{call_count[0]}", name="read",
                          arguments={"path": f"f{call_count[0]}.py"})]
            return make_turn_result(tool_calls=tc, finish_reason="tool_calls")

        client.chat.side_effect = chat_fn
        client.build_assistant_message.return_value = {"role": "assistant", "content": None}

        tool_results = []
        with patch("llm_solver.harness.loop.dispatch", return_value="file contents"):
            session = Session(cfg, client, "sys", "prompt", "/tmp")
            orig_add = session.context.add_tool_result
            def capture(cid, result, **kwargs):
                tool_results.append(result)
                return orig_add(cid, result, **kwargs)
            session.context.add_tool_result = capture
            session.run()

        nudge_results = [r for r in tool_results if "non-write tool calls since your last write" in r]
        # Nudge fires exactly once at the nudge threshold. The gate arms
        # (same turn under legacy config; a later turn when
        # rumination_gate_arm_threshold > rumination_nudge_threshold).
        # Without a successful write/edit between, the nudge does not
        # re-fire even though non-write calls continue to accumulate.
        assert len(nudge_results) == 1, f"Expected 1 nudge, got {len(nudge_results)}: {tool_results}"

    def test_gate_blocks_non_write_after_nudge_fires(self):
        """Once the nudge fires, the gate is armed. The model gets one
        grace call (dispatched with a warning), then the SECOND non-write
        call is rejected WITHOUT invoking dispatch."""
        from llm_solver.harness.loop import Session
        threshold = 7  # 7% of 100 → 7 absolute (above floor of 6)
        cfg = make_config(max_turns=100, duplicate_abort=20,
                          rumination_nudge_threshold=threshold)
        client = MagicMock()

        # threshold reads to trip the gate, then two more reads:
        # one grace (dispatched), one blocked.
        def chat_fn(*args, **kwargs):
            call_count[0] += 1
            if call_count[0] > threshold + 2:
                return make_turn_result(content="done", finish_reason="stop")
            tc = [ToolCall(id=f"c{call_count[0]}", name="read",
                          arguments={"path": f"f{call_count[0]}.py"})]
            return make_turn_result(tool_calls=tc, finish_reason="tool_calls")

        call_count = [0]
        client.chat.side_effect = chat_fn
        client.build_assistant_message.return_value = {"role": "assistant", "content": None}

        dispatch_calls = []
        def tracking_dispatch(name, args, cwd, cfg, **kwargs):
            dispatch_calls.append((name, args))
            return "dispatched-output"

        tool_results = []
        with patch("llm_solver.harness.loop.dispatch", side_effect=tracking_dispatch):
            session = Session(cfg, client, "sys", "prompt", "/tmp")
            orig_add = session.context.add_tool_result
            def capture(cid, result, **kwargs):
                tool_results.append(result)
                return orig_add(cid, result, **kwargs)
            session.context.add_tool_result = capture
            session.run()

        # threshold reads dispatched + 1 grace read dispatched = threshold+1
        assert len(dispatch_calls) == threshold + 1, (
            f"Expected {threshold + 1} dispatches (incl grace), got {len(dispatch_calls)}: {dispatch_calls}"
        )
        # Grace call result contains the warning
        grace_result = tool_results[threshold]
        assert "Gate armed" in grace_result
        assert "dispatched-output" in grace_result
        # The blocked call (threshold+2) must be the gate rejection
        assert len(tool_results) == threshold + 2
        gated = tool_results[-1]
        assert "[harness gate]" in gated
        assert "NOT executed" in gated
        assert "dispatched-output" not in gated

    def test_gate_does_not_clear_on_errored_write_or_edit(self):
        """A write/edit that returns ERROR: (e.g. old_str==new_str or
        not found) must NOT clear the gate — otherwise the model can
        game the gate with a no-op edit purely to resume exploration.
        Observed on attempt_009: the model submitted `old_str == new_str`
        at turn 14 which errored but cleared the gate, then went back
        to reads."""
        from llm_solver.harness.loop import Session
        threshold = 7  # 7% of 100 → 7 absolute (above floor of 6)
        cfg = make_config(max_turns=100, duplicate_abort=20,
                          rumination_nudge_threshold=threshold)
        client = MagicMock()

        # threshold reads → gate arms (grace=1)
        # errored edit → gate stays armed, grace still 1
        # first read after edit → grace consumed (dispatched with warning)
        # second read → must be blocked (grace=0, gate still armed)
        tool_sequence = (
            [("read", {"path": f"f{i}.py"}) for i in range(threshold)]
            + [("edit", {"path": "x.py", "old_str": "a", "new_str": "a"})]  # no-op
            + [("read", {"path": "grace.py"})]
            + [("read", {"path": "blocked.py"})]
        )
        call_count = [0]
        def chat_fn(*args, **kwargs):
            call_count[0] += 1
            if call_count[0] > len(tool_sequence):
                return make_turn_result(content="done", finish_reason="stop")
            name, args = tool_sequence[call_count[0] - 1]
            tc = [ToolCall(id=f"c{call_count[0]}", name=name, arguments=args)]
            return make_turn_result(tool_calls=tc, finish_reason="tool_calls")

        client.chat.side_effect = chat_fn
        client.build_assistant_message.return_value = {"role": "assistant", "content": None}

        dispatch_calls = []
        def errored_edit_dispatch(name, args, cwd, cfg, **kwargs):
            dispatch_calls.append(name)
            if name == "edit":
                return "ERROR: old_str not found"
            return "ok"

        tool_results = []
        with patch("llm_solver.harness.loop.dispatch", side_effect=errored_edit_dispatch):
            session = Session(cfg, client, "sys", "prompt", "/tmp")
            orig_add = session.context.add_tool_result
            def capture(cid, result, **kwargs):
                tool_results.append(result)
                return orig_add(cid, result, **kwargs)
            session.context.add_tool_result = capture
            session.run()

        # Dispatch should have received: threshold reads + 1 edit + 1 grace read.
        # The final read ("blocked.py") must NOT reach dispatch because
        # the errored edit kept the gate armed and grace was consumed.
        assert dispatch_calls == ["read"] * threshold + ["edit"] + ["read"], (
            f"Unexpected dispatches: {dispatch_calls}"
        )
        # Grace read has the warning
        grace_result = tool_results[threshold + 1]  # after threshold reads + edit
        assert "Gate armed" in grace_result
        # Last tool result is the gated read, not dispatched output.
        gated = tool_results[-1]
        assert "[harness gate]" in gated, f"Expected gate block, got: {gated}"

    def test_gate_pauses_duplicate_abort(self):
        """While the rumination gate is armed, duplicate_abort must NOT
        fire — otherwise the gate's rejection messages (which the model
        may respond to with repeated identical calls) would trip
        duplicate_abort and end the session before the gate could
        redirect the model. Observed on attempt_010: duplicate_abort
        at turn 7 stole the gate's window."""
        from llm_solver.harness.loop import Session
        threshold = 7  # 7% of 100 → 7 absolute (above floor of 6)
        # duplicate_abort=9 > threshold so varied reads don't fill the
        # deque. The model makes threshold varied reads (arms the gate),
        # then 3 identical gated calls that would have tripped
        # duplicate_abort in the old behavior. The session must NOT end
        # on duplicate_abort; the gate should handle those repeated calls.
        cfg = make_config(max_turns=100, duplicate_abort=9,
                          rumination_nudge_threshold=threshold)
        client = MagicMock()

        # First threshold calls: varied reads (trip gate without tripping
        # duplicate_abort since they're not identical). Then 3 identical
        # finds (would trip duplicate_abort if it weren't paused).
        tool_sequence = (
            [("read", {"path": f"f{i}.py"}) for i in range(threshold)]
            + [("bash", {"cmd": "find . -name x.py"}) for _ in range(3)]
        )
        call_count = [0]
        def chat_fn(*args, **kwargs):
            call_count[0] += 1
            if call_count[0] > len(tool_sequence):
                return make_turn_result(content="done", finish_reason="stop")
            name, args = tool_sequence[call_count[0] - 1]
            tc = [ToolCall(id=f"c{call_count[0]}", name=name, arguments=args)]
            return make_turn_result(tool_calls=tc, finish_reason="tool_calls")

        client.chat.side_effect = chat_fn
        client.build_assistant_message.return_value = {"role": "assistant", "content": None}

        with patch("llm_solver.harness.loop.dispatch", return_value="ok"):
            session = Session(cfg, client, "sys", "prompt", "/tmp")
            result = session.run()

        # Session must not end on duplicate_abort — the gate should be
        # handling those repeated calls.
        assert result.finish_reason != "duplicate_abort", (
            f"duplicate_abort fired while gate was armed: {result.finish_reason}"
        )

    def test_gate_clears_on_write_or_edit(self):
        """A write or edit call clears the gate and resets the counter,
        so subsequent non-write calls pass through normally until the
        next threshold crossing."""
        from llm_solver.harness.loop import Session
        threshold = 7  # 7% of 100 → 7 absolute (above floor of 6)
        cfg = make_config(max_turns=100, duplicate_abort=20,
                          rumination_nudge_threshold=threshold)
        client = MagicMock()

        # threshold reads → gate arms
        # then write → gate clears
        # then one read → dispatches normally (not gated)
        # then stop
        tool_sequence = (
            [("read", {"path": f"f{i}.py"}) for i in range(threshold)]
            + [("write", {"path": "out.py", "content": "pass"})]
            + [("read", {"path": "after.py"})]
        )
        call_count = [0]
        def chat_fn(*args, **kwargs):
            call_count[0] += 1
            if call_count[0] > len(tool_sequence):
                return make_turn_result(content="done", finish_reason="stop")
            name, args = tool_sequence[call_count[0] - 1]
            tc = [ToolCall(id=f"c{call_count[0]}", name=name, arguments=args)]
            return make_turn_result(tool_calls=tc, finish_reason="tool_calls")

        client.chat.side_effect = chat_fn
        client.build_assistant_message.return_value = {"role": "assistant", "content": None}

        dispatch_calls = []
        def tracking_dispatch(name, args, cwd, cfg, **kwargs):
            dispatch_calls.append(name)
            return "ok"

        with patch("llm_solver.harness.loop.dispatch", side_effect=tracking_dispatch):
            session = Session(cfg, client, "sys", "prompt", "/tmp")
            session.run()

        # Every call in the sequence should have reached dispatch:
        # threshold reads + 1 write + 1 read = threshold + 2
        assert dispatch_calls == ["read"] * threshold + ["write", "read"], (
            f"Gate didn't clear on write: {dispatch_calls}"
        )


class TestRuminationArmThresholdAbsolute:
    """rumination_gate_arm_threshold_abs decouples the gate from max_turns.
    When > 0, it overrides the percentage-of-max_turns calculation."""

    def test_abs_overrides_percentage(self):
        from llm_solver.harness.guardrails import init_guardrail_state
        cfg = make_config(
            max_turns=60, duplicate_abort=20,
            rumination_nudge_threshold=20,          # → nudge = 12 (20% of 60)
            rumination_gate_arm_threshold=30,        # → would-be arm = 18 (30% of 60)
            rumination_gate_arm_threshold_abs=30,    # overrides: arm = 30
        )
        state = init_guardrail_state(cfg)
        assert state.rumination_arm_threshold == 30
        assert state.rumination_nudge_threshold == 12

    def test_abs_zero_preserves_percentage_mode(self):
        from llm_solver.harness.guardrails import init_guardrail_state
        cfg = make_config(
            max_turns=90, duplicate_abort=20,
            rumination_nudge_threshold=20,
            rumination_gate_arm_threshold=30,
            rumination_gate_arm_threshold_abs=0,    # legacy mode
        )
        state = init_guardrail_state(cfg)
        assert state.rumination_arm_threshold == 27   # 30% of 90

    def test_abs_below_nudge_floor_clamps(self):
        """arm_abs must not shrink below the nudge threshold — the nudge→arm
        ordering is a precondition the rumination ladder relies on."""
        from llm_solver.harness.guardrails import init_guardrail_state
        cfg = make_config(
            max_turns=200, duplicate_abort=20,
            rumination_nudge_threshold=20,          # → nudge = 40
            rumination_gate_arm_threshold=30,
            rumination_gate_arm_threshold_abs=10,   # below nudge
        )
        state = init_guardrail_state(cfg)
        assert state.rumination_arm_threshold == 40   # clamped to nudge


class TestRuminationNudgeThresholdAbsolute:
    """rumination_nudge_threshold_abs decouples the nudge from max_turns.
    When > 0, it overrides the percentage-of-max_turns calculation."""

    def test_abs_overrides_percentage(self):
        from llm_solver.harness.guardrails import init_guardrail_state
        cfg = make_config(
            max_turns=90, duplicate_abort=20,
            rumination_nudge_threshold=20,           # → would-be nudge = 18
            rumination_nudge_threshold_abs=12,       # overrides: nudge = 12
            rumination_gate_arm_threshold=30,
        )
        state = init_guardrail_state(cfg)
        assert state.rumination_nudge_threshold == 12
        # Arm still computes from percentage unless arm_abs is also set.
        assert state.rumination_arm_threshold == 27  # 30% of 90

    def test_abs_zero_preserves_percentage_mode(self):
        from llm_solver.harness.guardrails import init_guardrail_state
        cfg = make_config(
            max_turns=90, duplicate_abort=20,
            rumination_nudge_threshold=20,
            rumination_nudge_threshold_abs=0,
            rumination_gate_arm_threshold=30,
        )
        state = init_guardrail_state(cfg)
        assert state.rumination_nudge_threshold == 18  # 20% of 90

    def test_abs_respects_min_threshold_floor(self):
        """nudge_abs must not drop below cfg.rumination_min_threshold."""
        from llm_solver.harness.guardrails import init_guardrail_state
        cfg = make_config(
            max_turns=90, duplicate_abort=20,
            rumination_nudge_threshold=20,
            rumination_nudge_threshold_abs=3,        # below min floor (default 6)
            rumination_gate_arm_threshold=30,
        )
        state = init_guardrail_state(cfg)
        assert state.rumination_nudge_threshold == 6  # clamped to min_threshold floor

    def test_both_abs_knobs_compose(self):
        """nudge_abs and arm_abs compose independently."""
        from llm_solver.harness.guardrails import init_guardrail_state
        cfg = make_config(
            max_turns=90, duplicate_abort=20,
            rumination_nudge_threshold=20,
            rumination_nudge_threshold_abs=12,
            rumination_gate_arm_threshold=30,
            rumination_gate_arm_threshold_abs=30,
        )
        state = init_guardrail_state(cfg)
        assert state.rumination_nudge_threshold == 12
        assert state.rumination_arm_threshold == 30


class TestRuminationNudgePostMutationThreshold:
    """rumination_nudge_threshold_abs_post_mutation sets a separate nudge
    threshold for after the model's first successful write/edit. Allows
    asymmetric nudge timing: aggressive pre-mutation push, baseline-like
    post-mutation nudge."""

    def _drive_ladder(self, non_writes, has_mutated_flag, pre_abs, post_abs):
        from llm_solver.harness.guardrails import (
            rumination_ladder, init_guardrail_state,
        )
        cfg = make_config(
            max_turns=90, duplicate_abort=20,
            rumination_nudge_threshold=20,
            rumination_nudge_threshold_abs=pre_abs,
            rumination_nudge_threshold_abs_post_mutation=post_abs,
            rumination_gate_arm_threshold=30,
        )
        state = init_guardrail_state(cfg)
        state.has_mutated = has_mutated_flag
        last = None
        for _ in range(non_writes):
            last = rumination_ladder(state, cfg, tc_name="bash",
                                     result="output",
                                     gate_blocked=False,
                                     already_blocked_this_turn=False)
        return state, last

    def test_post_abs_overrides_pre_after_mutation(self):
        """Pre-mut fires at 12. Post-mut threshold 18 delays it until 18 non-writes."""
        # Post-mutation, with 12 non-writes — should NOT fire (below post=18)
        state, d = self._drive_ladder(12, has_mutated_flag=True, pre_abs=12, post_abs=18)
        assert not d.text, "Post-mut nudge should not fire at count=12 when post_abs=18"

        # Post-mutation, with 18 non-writes — should fire
        state, d = self._drive_ladder(18, has_mutated_flag=True, pre_abs=12, post_abs=18)
        assert d.text, "Post-mut nudge should fire at count=18 when post_abs=18"

    def test_pre_abs_applies_before_mutation(self):
        """Pre-mutation, nudge fires at pre_abs (12) regardless of post_abs."""
        state, d = self._drive_ladder(12, has_mutated_flag=False, pre_abs=12, post_abs=18)
        assert d.text, "Pre-mut nudge should fire at pre_abs threshold"

    def test_post_abs_zero_defaults_to_pre(self):
        """When post_abs=0, post threshold equals pre threshold (symmetric)."""
        state, d = self._drive_ladder(12, has_mutated_flag=True, pre_abs=12, post_abs=0)
        assert d.text, "Post-mut nudge should fire at pre_abs when post_abs=0"

    def test_init_stores_both_thresholds(self):
        from llm_solver.harness.guardrails import init_guardrail_state
        cfg = make_config(
            max_turns=90, duplicate_abort=20,
            rumination_nudge_threshold=20,
            rumination_nudge_threshold_abs=12,
            rumination_nudge_threshold_abs_post_mutation=18,
            rumination_gate_arm_threshold=30,
        )
        state = init_guardrail_state(cfg)
        assert state.rumination_nudge_threshold == 12
        assert state.rumination_nudge_threshold_post_mutation == 18


class TestRuminationNudgeOnlyPreMutation:
    """When rumination_nudge_only_pre_mutation is True, the nudge fires
    only before the model's first successful write/edit. Post-mutation
    non-write streaks are left alone (they're productive exploration
    between edits, not stuck rumination)."""

    def _ladder(self, has_mutated_flag, only_pre_mut):
        from llm_solver.harness.guardrails import rumination_ladder, init_guardrail_state
        cfg = make_config(
            max_turns=90, duplicate_abort=20,
            rumination_nudge_threshold=20,  # nudge=18 at mt=90
            rumination_gate_arm_threshold=30,
            rumination_nudge_only_pre_mutation=only_pre_mut,
        )
        state = init_guardrail_state(cfg)
        state.has_mutated = has_mutated_flag
        # Drive 18 non-write calls to the threshold.
        last = None
        for _ in range(18):
            last = rumination_ladder(state, cfg, tc_name="bash",
                                     result="some output",
                                     gate_blocked=False,
                                     already_blocked_this_turn=False)
        return last

    def test_nudge_fires_pre_mutation_when_toggle_on(self):
        d = self._ladder(has_mutated_flag=False, only_pre_mut=True)
        assert d.text, "Nudge should fire when has_mutated=False"

    def test_nudge_suppressed_post_mutation_when_toggle_on(self):
        d = self._ladder(has_mutated_flag=True, only_pre_mut=True)
        assert not d.text, "Nudge should be suppressed when has_mutated=True"

    def test_nudge_fires_post_mutation_when_toggle_off(self):
        """Default behavior: nudge fires regardless of has_mutated state."""
        d = self._ladder(has_mutated_flag=True, only_pre_mut=False)
        assert d.text, "Nudge should fire in legacy (toggle-off) mode"


class TestRuminationSameTarget:
    """Repeated inspection of the same target should arm the existing
    rumination gate earlier than the coarse non-write threshold when the
    same-target knobs are configured."""

    def test_same_target_path_warns_and_arms_early(self):
        from llm_solver.harness.loop import Session

        cfg = make_config(
            max_turns=100,
            duplicate_abort=20,
            rumination_nudge_threshold=80,  # keep coarse nudge out of the way
            rumination_same_target_warn_count=3,
            rumination_same_target_arm_count=4,
            rumination_gate_grace_calls=1,
        )
        client = MagicMock()

        call_count = [0]

        def chat_fn(*args, **kwargs):
            call_count[0] += 1
            if call_count[0] > 6:
                return make_turn_result(content="done", finish_reason="stop")
            tc = [ToolCall(id=f"c{call_count[0]}", name="read",
                          arguments={"path": "pkg/mod.py"})]
            return make_turn_result(tool_calls=tc, finish_reason="tool_calls")

        client.chat.side_effect = chat_fn
        client.build_assistant_message.return_value = {"role": "assistant", "content": None}

        dispatch_calls = []
        tool_results = []

        def tracking_dispatch(name, args, cwd, cfg, **kwargs):
            dispatch_calls.append((name, dict(args)))
            return "module contents"

        with patch("llm_solver.harness.loop.dispatch", side_effect=tracking_dispatch):
            session = Session(cfg, client, "sys", "prompt", "/tmp")
            orig_add = session.context.add_tool_result

            def capture(cid, result, **kwargs):
                tool_results.append(result)
                return orig_add(cid, result, **kwargs)

            session.context.add_tool_result = capture
            session.run()

        assert len(dispatch_calls) == 5, dispatch_calls
        same_target_warns = [
            r for r in tool_results if "same target hit 3 times" in r and "pkg/mod.py" in r
        ]
        assert same_target_warns, tool_results
        assert "Gate armed" in tool_results[4], tool_results
        assert "[harness gate]" in tool_results[5], tool_results
        assert "module contents" not in tool_results[5]

    def test_same_target_streak_resets_when_target_changes(self):
        from llm_solver.harness.loop import Session

        cfg = make_config(
            max_turns=100,
            duplicate_abort=20,
            rumination_nudge_threshold=80,
            rumination_same_target_warn_count=3,
            rumination_same_target_arm_count=4,
        )
        client = MagicMock()

        sequence = [
            {"path": "a.py"},
            {"path": "a.py"},
            {"path": "a.py"},
            {"path": "b.py"},
            {"path": "b.py"},
        ]
        call_count = [0]

        def chat_fn(*args, **kwargs):
            call_count[0] += 1
            if call_count[0] > len(sequence):
                return make_turn_result(content="done", finish_reason="stop")
            tc = [ToolCall(id=f"c{call_count[0]}", name="read",
                          arguments=sequence[call_count[0] - 1])]
            return make_turn_result(tool_calls=tc, finish_reason="tool_calls")

        client.chat.side_effect = chat_fn
        client.build_assistant_message.return_value = {"role": "assistant", "content": None}

        tool_results = []
        with patch("llm_solver.harness.loop.dispatch", return_value="file contents"):
            session = Session(cfg, client, "sys", "prompt", "/tmp")
            orig_add = session.context.add_tool_result

            def capture(cid, result, **kwargs):
                tool_results.append(result)
                return orig_add(cid, result, **kwargs)

            session.context.add_tool_result = capture
            session.run()

        assert sum("same target hit" in r for r in tool_results) == 1, tool_results
        assert not any("[harness gate]" in r for r in tool_results), tool_results


class TestTestReadGuard:

    def test_warns_when_running_tests_without_reading_test_file(self):
        from llm_solver.harness.guardrails import test_read_ladder, init_guardrail_state

        cfg = make_config(test_read_warn_after=1, test_read_nudge="read {target} after {count}")
        state = init_guardrail_state(cfg)
        decision = test_read_ladder(
            state,
            cfg,
            tc_name="bash",
            result="failed\n[exit code: 1]",
            gate_blocked=False,
            tc_args={"cmd": "pytest -q tests/test_app.py"},
        )
        assert decision.text == "read tests/test_app.py after 1"

    def test_resets_after_test_file_is_read(self):
        from llm_solver.harness.guardrails import (
            test_read_ladder,
            observe_test_file_read,
            init_guardrail_state,
        )

        cfg = make_config(test_read_warn_after=1, test_read_nudge="read {target} after {count}")
        state = init_guardrail_state(cfg)

        first = test_read_ladder(
            state,
            cfg,
            tc_name="bash",
            result="failed\n[exit code: 1]",
            gate_blocked=False,
            tc_args={"cmd": "pytest -q tests/test_app.py"},
        )
        assert first.text == "read tests/test_app.py after 1"

        observe_test_file_read(
            state,
            cfg,
            tc_name="read",
            result="def test_app(): ...",
            gate_blocked=False,
            tc_args={"path": "tests/test_app.py"},
            focus_key="file:tests/test_app.py",
            focus_display="tests/test_app.py",
        )

        second = test_read_ladder(
            state,
            cfg,
            tc_name="bash",
            result="failed\n[exit code: 1]",
            gate_blocked=False,
            tc_args={"cmd": "pytest -q tests/test_app.py"},
        )
        assert second.action.value == "pass"


class TestContractGate:

    def test_commit_contract_warns_then_blocks_after_source_read(self):
        from llm_solver.harness.guardrails import (
            contract_gate,
            init_guardrail_state,
            observe_contract_state,
        )

        cfg = make_config(
            contract_commit_warn_after=1,
            contract_commit_block_after=2,
            contract_commit_warn="warn {source}",
            contract_commit_block="block {source}",
        )
        state = init_guardrail_state(cfg)

        observe_contract_state(
            state,
            cfg,
            tc_name="read",
            result="print('hi')",
            gate_blocked=False,
            tc_args={"path": "src/app.py"},
            focus_key="file:src/app.py",
            focus_display="src/app.py",
        )

        first = contract_gate(
            state,
            cfg,
            tc_name="bash",
            tc_args={"cmd": "ls -la src/"},
            focus_key="bash:ls -la src/",
            focus_display="src/",
        )
        second = contract_gate(
            state,
            cfg,
            tc_name="bash",
            tc_args={"cmd": "ls -la src/"},
            focus_key="bash:ls -la src/",
            focus_display="src/",
        )

        assert first.action.value == "warn"
        assert first.text == "warn src/app.py"
        assert second.action.value == "block"
        assert second.text == "block src/app.py"

    def test_commit_contract_can_block_immediately_when_configured(self):
        from llm_solver.harness.guardrails import (
            contract_gate,
            init_guardrail_state,
            observe_contract_state,
        )

        cfg = make_config(
            contract_commit_warn_after=0,
            contract_commit_block_after=1,
            contract_commit_block="block {source}",
        )
        state = init_guardrail_state(cfg)

        observe_contract_state(
            state,
            cfg,
            tc_name="read",
            result="print('hi')",
            gate_blocked=False,
            tc_args={"path": "src/app.py"},
            focus_key="file:src/app.py",
            focus_display="src/app.py",
        )

        decision = contract_gate(
            state,
            cfg,
            tc_name="bash",
            tc_args={"cmd": "python -c \"print('noop')\""},
            focus_key="bash:python -c \"print('noop')\"",
            focus_display="python -c \"print('noop')\"",
        )

        assert decision.action.value == "block"
        assert decision.text == "block src/app.py"

    def test_commit_contract_does_not_arm_on_directory_inspection(self):
        from llm_solver.harness.guardrails import (
            contract_gate,
            init_guardrail_state,
            observe_contract_state,
        )

        cfg = make_config(
            contract_commit_warn_after=1,
            contract_commit_block_after=2,
            contract_commit_warn="warn {source}",
            contract_commit_block="block {source}",
        )
        state = init_guardrail_state(cfg)

        observe_contract_state(
            state,
            cfg,
            tc_name="bash",
            result="listing\n[exit code: 0]",
            gate_blocked=False,
            tc_args={"cmd": "ls -la src/"},
            focus_key="file:src",
            focus_display="src/",
        )

        decision = contract_gate(
            state,
            cfg,
            tc_name="bash",
            tc_args={"cmd": "ls -la src/"},
            focus_key="bash:ls -la src/",
            focus_display="src/",
        )

        assert state.commit_pending is False
        assert decision.action.value == "pass"

    def test_commit_contract_does_not_arm_on_outside_root_file_reads(self):
        from llm_solver.harness.guardrails import init_guardrail_state, observe_contract_state

        cfg = make_config(
            contract_commit_warn_after=1,
            contract_commit_block_after=2,
        )
        state = init_guardrail_state(cfg)

        observe_contract_state(
            state,
            cfg,
            tc_name="bash",
            result="code\n[exit code: 0]",
            gate_blocked=False,
            tc_args={"cmd": "cat /opt/miniconda3/lib/python3.11/site-packages/pandas/core/groupby/generic.py"},
            focus_key="outside:/opt/miniconda3/lib/python3.11/site-packages/pandas/core/groupby/generic.py",
            focus_display="/opt/miniconda3/lib/python3.11/site-packages/pandas/core/groupby/generic.py",
        )

        assert state.commit_pending is False
        assert state.commit_source_path == ""

    def test_recovery_contract_arms_on_same_target_and_blocks_broad_inspection(self):
        from llm_solver.harness.guardrails import (
            contract_gate,
            init_guardrail_state,
            rumination_ladder,
        )

        cfg = make_config(
            contract_recovery_same_target_threshold=3,
            contract_recovery_block="recover {reason} {target}",
        )
        state = init_guardrail_state(cfg)

        for _ in range(3):
            rumination_ladder(
                state,
                cfg,
                tc_name="read",
                result="print('hi')",
                gate_blocked=False,
                already_blocked_this_turn=False,
                focus_key="file:src/app.py",
                focus_display="src/app.py",
            )

        decision = contract_gate(
            state,
            cfg,
            tc_name="bash",
            tc_args={"cmd": "ls -la src/"},
            focus_key="bash:ls -la src/",
            focus_display="src/",
        )
        allowed = contract_gate(
            state,
            cfg,
            tc_name="read",
            tc_args={"path": "tests/test_app.py"},
            focus_key="file:tests/test_app.py",
            focus_display="tests/test_app.py",
        )

        assert state.recovery_mode_active is True
        assert decision.action.value == "block"
        assert decision.text == "recover repeated same-target inspection src/app.py"
        assert allowed.action.value == "pass"

    def test_recovery_contract_can_abort_repeated_same_invalid_move(self):
        from llm_solver.harness.guardrails import (
            contract_gate,
            init_guardrail_state,
            rumination_ladder,
        )

        cfg = make_config(
            contract_recovery_same_target_threshold=3,
            contract_recovery_block="recover {reason} {target}",
            contract_invalid_repeat_abort_after=2,
        )
        state = init_guardrail_state(cfg)

        for _ in range(3):
            rumination_ladder(
                state,
                cfg,
                tc_name="read",
                result="print('hi')",
                gate_blocked=False,
                already_blocked_this_turn=False,
                focus_key="file:src/app.py",
                focus_display="src/app.py",
            )

        first = contract_gate(
            state,
            cfg,
            tc_name="bash",
            tc_args={"cmd": "ls -la src/"},
            focus_key="bash:ls -la src/",
            focus_display="src/",
        )
        second = contract_gate(
            state,
            cfg,
            tc_name="bash",
            tc_args={"cmd": "ls -la src/"},
            focus_key="bash:ls -la src/",
            focus_display="src/",
        )

        assert first.action.value == "block"
        assert second.action.value == "end"
        assert second.reason == "contract_recovery_abort"

    def test_recovery_contract_abort_respects_min_turn_window(self):
        from llm_solver.harness.guardrails import (
            contract_gate,
            init_guardrail_state,
            rumination_ladder,
        )

        cfg = make_config(
            contract_recovery_same_target_threshold=2,
            contract_recovery_block="recover {reason} {target}",
            contract_invalid_repeat_abort_after=2,
            contract_abort_min_turns_since_recovery_arm=3,
        )
        state = init_guardrail_state(cfg)

        for _ in range(2):
            rumination_ladder(
                state,
                cfg,
                tc_name="read",
                result="print('hi')",
                gate_blocked=False,
                already_blocked_this_turn=False,
                focus_key="file:src/app.py",
                focus_display="src/app.py",
            )

        first = contract_gate(
            state,
            cfg,
            tc_name="bash",
            tc_args={"cmd": "ls -la src/"},
            focus_key="bash:ls -la src/",
            focus_display="src/",
        )
        second = contract_gate(
            state,
            cfg,
            tc_name="bash",
            tc_args={"cmd": "ls -la src/"},
            focus_key="bash:ls -la src/",
            focus_display="src/",
        )
        third = contract_gate(
            state,
            cfg,
            tc_name="bash",
            tc_args={"cmd": "ls -la src/"},
            focus_key="bash:ls -la src/",
            focus_display="src/",
        )

        assert first.action.value == "block"
        assert second.action.value == "block"
        assert third.action.value == "end"
        assert third.reason == "contract_recovery_abort"

    def test_commit_contract_abort_respects_min_turn_window(self):
        from llm_solver.harness.guardrails import (
            contract_gate,
            init_guardrail_state,
            observe_contract_state,
        )

        cfg = make_config(
            contract_commit_warn_after=0,
            contract_commit_block_after=1,
            contract_commit_block="block {source}",
            contract_invalid_repeat_abort_after=2,
            contract_abort_min_turns_since_commit_arm=3,
        )
        state = init_guardrail_state(cfg)

        observe_contract_state(
            state,
            cfg,
            tc_name="read",
            result="print('hi')",
            gate_blocked=False,
            tc_args={"path": "src/app.py"},
            focus_key="file:src/app.py",
            focus_display="src/app.py",
        )

        first = contract_gate(
            state,
            cfg,
            tc_name="bash",
            tc_args={"cmd": "python -c \"print('noop')\""},
            focus_key="bash:python -c \"print('noop')\"",
            focus_display="python -c \"print('noop')\"",
        )
        second = contract_gate(
            state,
            cfg,
            tc_name="bash",
            tc_args={"cmd": "python -c \"print('noop')\""},
            focus_key="bash:python -c \"print('noop')\"",
            focus_display="python -c \"print('noop')\"",
        )
        third = contract_gate(
            state,
            cfg,
            tc_name="bash",
            tc_args={"cmd": "python -c \"print('noop')\""},
            focus_key="bash:python -c \"print('noop')\"",
            focus_display="python -c \"print('noop')\"",
        )

        assert first.action.value == "block"
        assert second.action.value == "block"
        assert third.action.value == "end"
        assert third.reason == "contract_commit_abort"

    def test_equivalent_contract_classes_collapse_python_c_probes(self):
        from llm_solver.harness.guardrails import (
            contract_gate,
            init_guardrail_state,
            observe_contract_state,
        )

        cfg = make_config(
            contract_commit_warn_after=0,
            contract_commit_block_after=1,
            contract_invalid_repeat_abort_after=2,
            contract_equivalent_action_classes_enabled=True,
            contract_commit_block="block {source}",
        )
        state = init_guardrail_state(cfg)

        observe_contract_state(
            state,
            cfg,
            tc_name="read",
            result="print('hi')",
            gate_blocked=False,
            tc_args={"path": "src/app.py"},
            focus_key="file:src/app.py",
            focus_display="src/app.py",
        )

        first = contract_gate(
            state,
            cfg,
            tc_name="bash",
            tc_args={"cmd": "cd /testbed && python -c \"import pandas; print(pandas.__file__)\""},
            focus_key="bash:python-c-probe-a",
            focus_display="python-probe",
        )
        second = contract_gate(
            state,
            cfg,
            tc_name="bash",
            tc_args={"cmd": "python -c \"import seaborn; print(seaborn.__file__)\""},
            focus_key="bash:python-c-probe-b",
            focus_display="python-probe",
        )

        assert first.action.value == "block"
        assert second.action.value == "end"
        assert second.reason == "contract_commit_abort"

    def test_recovery_contract_arms_on_verify_repeats_without_mutation(self):
        from llm_solver.harness.guardrails import (
            contract_gate,
            init_guardrail_state,
            observe_contract_state,
        )

        cfg = make_config(
            contract_recovery_verify_repeat_threshold=3,
            contract_recovery_block="recover {reason} {target}",
        )
        state = init_guardrail_state(cfg)

        for _ in range(3):
            observe_contract_state(
                state,
                cfg,
                tc_name="bash",
                result="failed\n[exit code: 1]",
                gate_blocked=False,
                tc_args={"cmd": "pytest -q tests/test_app.py"},
                focus_key="bash:pytest -q tests/test_app.py",
                focus_display="pytest -q tests/test_app.py",
            )

        decision = contract_gate(
            state,
            cfg,
            tc_name="grep",
            tc_args={"pattern": "foo", "path": "."},
            focus_key="grep:{\"path\": \".\", \"pattern\": \"foo\"}",
            focus_display="grep(foo)",
        )

        assert state.recovery_mode_active is True
        assert decision.action.value == "block"
        assert decision.text == (
            "recover repeated verification without refinement tests/test_app.py"
        )

    def test_mutation_repeat_guard_warns_on_repeated_identical_edit(self):
        from llm_solver.harness.guardrails import (
            init_guardrail_state,
            mutation_repeat_guard,
            observe_contract_state,
        )

        cfg = make_config(
            mutation_repeat_warn_after=2,
            mutation_repeat_block_after=3,
            mutation_repeat_warn="warn {target}",
            mutation_repeat_block="block {target}",
        )
        state = init_guardrail_state(cfg)

        observe_contract_state(
            state,
            cfg,
            tc_name="edit",
            result="OK",
            gate_blocked=False,
            tc_args={"path": "src/app.py", "old_str": "a", "new_str": "b"},
            focus_key="file:src/app.py",
            focus_display="src/app.py",
        )

        decision = mutation_repeat_guard(
            state,
            cfg,
            tc_name="edit",
            tc_args={"path": "src/app.py", "old_str": "a", "new_str": "b"},
            focus_display="src/app.py",
        )

        assert decision.action.value == "warn"
        assert decision.text == "warn src/app.py"

    def test_mutation_repeat_guard_blocks_and_can_abort(self):
        from llm_solver.harness.guardrails import (
            init_guardrail_state,
            mutation_repeat_guard,
            observe_contract_state,
        )

        cfg = make_config(
            mutation_repeat_warn_after=2,
            mutation_repeat_block_after=3,
            mutation_repeat_abort_after=2,
            mutation_repeat_warn="warn {target}",
            mutation_repeat_block="block {target}",
        )
        state = init_guardrail_state(cfg)

        observe_contract_state(
            state,
            cfg,
            tc_name="edit",
            result="OK",
            gate_blocked=False,
            tc_args={"path": "src/app.py", "old_str": "a", "new_str": "b"},
            focus_key="file:src/app.py",
            focus_display="src/app.py",
        )
        observe_contract_state(
            state,
            cfg,
            tc_name="edit",
            result="OK",
            gate_blocked=False,
            tc_args={"path": "src/app.py", "old_str": "a", "new_str": "b"},
            focus_key="file:src/app.py",
            focus_display="src/app.py",
        )

        first = mutation_repeat_guard(
            state,
            cfg,
            tc_name="edit",
            tc_args={"path": "src/app.py", "old_str": "a", "new_str": "b"},
            focus_display="src/app.py",
        )
        second = mutation_repeat_guard(
            state,
            cfg,
            tc_name="edit",
            tc_args={"path": "src/app.py", "old_str": "a", "new_str": "b"},
            focus_display="src/app.py",
        )

        assert first.action.value == "block"
        assert first.text == "block src/app.py"
        assert second.action.value == "end"
        assert second.reason == "mutation_repeat_abort"

    def test_mutation_repeat_guard_resets_after_successful_non_mutation(self):
        from llm_solver.harness.guardrails import (
            init_guardrail_state,
            mutation_repeat_guard,
            observe_contract_state,
        )

        cfg = make_config(
            mutation_repeat_warn_after=2,
            mutation_repeat_block_after=3,
            mutation_repeat_warn="warn {target}",
            mutation_repeat_block="block {target}",
        )
        state = init_guardrail_state(cfg)

        observe_contract_state(
            state,
            cfg,
            tc_name="edit",
            result="OK",
            gate_blocked=False,
            tc_args={"path": "src/app.py", "old_str": "a", "new_str": "b"},
            focus_key="file:src/app.py",
            focus_display="src/app.py",
        )
        observe_contract_state(
            state,
            cfg,
            tc_name="bash",
            result="ok\n[exit code: 0]",
            gate_blocked=False,
            tc_args={"cmd": "pytest -q tests/test_app.py"},
            focus_key="bash:pytest -q tests/test_app.py",
            focus_display="pytest -q tests/test_app.py",
        )

        decision = mutation_repeat_guard(
            state,
            cfg,
            tc_name="edit",
            tc_args={"path": "src/app.py", "old_str": "a", "new_str": "b"},
            focus_display="src/app.py",
        )

        assert decision.action.value == "pass"


# ──────────────────────────────────────────────
# 17. Auto-commit at session boundaries (#19)
# ──────────────────────────────────────────────

class TestAutoCommit:

    def _init_git_repo(self, path):
        """Initialize a git repo with an initial commit."""
        _subprocess.run(["git", "init", str(path)], capture_output=True, check=True)
        _subprocess.run(["git", "-C", str(path), "config", "user.email", "test@test.com"],
                        capture_output=True, check=True)
        _subprocess.run(["git", "-C", str(path), "config", "user.name", "test"],
                        capture_output=True, check=True)
        (path / "init.txt").write_text("init")
        _subprocess.run(["git", "-C", str(path), "add", "-A"], capture_output=True, check=True)
        _subprocess.run(["git", "-C", str(path), "commit", "-m", "init"],
                        capture_output=True, check=True)

    def test_commits_when_dirty(self, tmp_path):
        """_auto_commit creates a commit when working tree has changes."""
        from llm_solver.harness.loop import _auto_commit
        self._init_git_repo(tmp_path)
        (tmp_path / "new.txt").write_text("added")

        _auto_commit(tmp_path, 1, "stop")

        result = _subprocess.run(
            ["git", "-C", str(tmp_path), "log", "--oneline", "-1"],
            capture_output=True, text=True,
        )
        assert "harness: session 1 checkpoint (stop)" in result.stdout

    def test_skips_clean_tree(self, tmp_path):
        """_auto_commit does nothing when working tree is clean."""
        from llm_solver.harness.loop import _auto_commit
        self._init_git_repo(tmp_path)

        _auto_commit(tmp_path, 1, "stop")

        result = _subprocess.run(
            ["git", "-C", str(tmp_path), "log", "--oneline"],
            capture_output=True, text=True,
        )
        lines = result.stdout.strip().split("\n")
        assert len(lines) == 1  # only the init commit

    def test_commit_message_format(self, tmp_path):
        """Commit message follows expected format."""
        from llm_solver.harness.loop import _auto_commit
        self._init_git_repo(tmp_path)
        (tmp_path / "file.py").write_text("code")

        _auto_commit(tmp_path, 3, "context_full")

        result = _subprocess.run(
            ["git", "-C", str(tmp_path), "log", "--oneline", "-1"],
            capture_output=True, text=True,
        )
        assert "harness: session 3 checkpoint (context_full)" in result.stdout

    def test_solve_task_no_commit_on_error(self, tmp_path):
        """solve_task does NOT call _auto_commit when session ends with error."""
        from llm_solver.harness.loop import solve_task
        (tmp_path / "prompt.txt").write_text("fix bug")

        client = MagicMock()
        client.chat.side_effect = RuntimeError("fatal")
        client.build_assistant_message.return_value = {"role": "assistant", "content": ""}

        cfg = make_config(max_turns=5, max_sessions=1)

        with patch("llm_solver.harness.loop._auto_commit") as mock_commit:
            solve_task(tmp_path, cfg, client)

        mock_commit.assert_not_called()

    def test_solve_task_commits_on_success(self, tmp_path):
        """solve_task calls _auto_commit after successful session."""
        from llm_solver.harness.loop import solve_task
        (tmp_path / "prompt.txt").write_text("fix bug")

        client = MagicMock()
        client.chat.return_value = make_turn_result(content="Done!", finish_reason="stop")
        client.build_assistant_message.return_value = {"role": "assistant", "content": "Done!"}

        cfg = make_config(max_turns=10, max_sessions=1)

        with patch("llm_solver.harness.loop._auto_commit") as mock_commit:
            solve_task(tmp_path, cfg, client)

        mock_commit.assert_called_once_with(tmp_path, 1, "stop")


# ──────────────────────────────────────────────
# 18. Structured trace logging (#42)
# ──────────────────────────────────────────────

class TestTraceLogging:

    def test_trace_file_written_per_turn(self, tmp_path):
        """Session writes a trace line per tool call."""
        from llm_solver.harness.loop import Session
        cfg = make_config(max_turns=5, duplicate_abort=20)
        client = MagicMock()

        call_count = [0]
        def chat_fn(*args, **kwargs):
            call_count[0] += 1
            if call_count[0] > 2:
                return make_turn_result(content="done", finish_reason="stop")
            tc = [ToolCall(id=f"c{call_count[0]}", name="bash",
                          arguments={"cmd": f"echo {call_count[0]}"})]
            return make_turn_result(tool_calls=tc, finish_reason="tool_calls")

        client.chat.side_effect = chat_fn
        client.build_assistant_message.return_value = {"role": "assistant", "content": None}

        trace_path = tmp_path / ".trace.jsonl"
        with open(trace_path, "a") as tf:
            with patch("llm_solver.harness.loop.dispatch", return_value="ok"):
                session = Session(cfg, client, "sys", "prompt", str(tmp_path),
                                  trace_file=tf, session_number=1)
                session.run()

        lines = trace_path.read_text().strip().split("\n")
        assert len(lines) == 2  # 2 tool calls

        for line in lines:
            entry = json.loads(line)
            assert entry["event"] == "tool_call"

    def test_trace_entry_fields(self, tmp_path):
        """Each tool_call trace entry has all required fields."""
        from llm_solver.harness.loop import Session
        cfg = make_config(max_turns=5, duplicate_abort=20)
        client = MagicMock()

        call_count = [0]
        def chat_fn(*args, **kwargs):
            call_count[0] += 1
            if call_count[0] > 1:
                return make_turn_result(content="done", finish_reason="stop", prompt_tokens=42)
            tc = [ToolCall(id="c1", name="bash", arguments={"cmd": "echo hi"})]
            return make_turn_result(tool_calls=tc, finish_reason="tool_calls", prompt_tokens=100)

        client.chat.side_effect = chat_fn
        client.build_assistant_message.return_value = {"role": "assistant", "content": None}

        trace_path = tmp_path / ".trace.jsonl"
        with open(trace_path, "a") as tf:
            with patch("llm_solver.harness.loop.dispatch", return_value="hello world"):
                session = Session(cfg, client, "sys", "prompt", str(tmp_path),
                                  trace_file=tf, session_number=2)
                session.run()

        lines = trace_path.read_text().strip().split("\n")
        entry = json.loads(lines[0])
        required = {"event", "session_number", "turn_number", "tool_name",
                     "args_summary", "result_summary", "prompt_tokens", "completion_tokens"}
        assert required.issubset(entry.keys()), f"Missing: {required - entry.keys()}"
        assert entry["session_number"] == 2
        assert entry["turn_number"] == 0
        assert entry["tool_name"] == "bash"
        assert entry["prompt_tokens"] == 100

    def test_session_events_in_solve_task(self, tmp_path):
        """solve_task writes session_start and session_end events to trace."""
        from llm_solver.harness.loop import solve_task
        (tmp_path / "prompt.txt").write_text("fix bug")

        client = MagicMock()
        client.chat.return_value = make_turn_result(content="Done!", finish_reason="stop")
        client.build_assistant_message.return_value = {"role": "assistant", "content": "Done!"}

        cfg = make_config(max_turns=10, max_sessions=1)

        with patch("llm_solver.harness.loop._auto_commit"):
            solve_task(tmp_path, cfg, client)

        trace_path = tmp_path / ".trace.jsonl"
        assert trace_path.exists()
        lines = trace_path.read_text().strip().split("\n")
        events = [json.loads(l) for l in lines]

        event_types = [e["event"] for e in events]
        assert "session_start" in event_types
        assert "session_end" in event_types

        end_event = next(e for e in events if e["event"] == "session_end")
        assert end_event["finish_reason"] == "stop"
        assert end_event["session_number"] == 1

    def test_trace_result_truncated(self, tmp_path):
        """Long tool results are truncated in trace entries."""
        from llm_solver.harness.loop import Session
        cfg = make_config(max_turns=5, duplicate_abort=20)
        client = MagicMock()

        call_count = [0]
        def chat_fn(*args, **kwargs):
            call_count[0] += 1
            if call_count[0] > 1:
                return make_turn_result(content="done", finish_reason="stop")
            tc = [ToolCall(id="c1", name="bash", arguments={"cmd": "cat bigfile"})]
            return make_turn_result(tool_calls=tc, finish_reason="tool_calls")

        client.chat.side_effect = chat_fn
        client.build_assistant_message.return_value = {"role": "assistant", "content": None}

        trace_path = tmp_path / ".trace.jsonl"
        # Feed a result LARGER than the new result cap (max_output_chars,
        # default 20000) so truncation actually kicks in. A 10000-char
        # "code read" must no longer be truncated — that was the bug.
        code_result = "x" * 10000          # well under 20000 — must survive
        huge_result = "y" * 30000          # over 20000 — must be truncated
        with open(trace_path, "a") as tf:
            with patch("llm_solver.harness.loop.dispatch", return_value=code_result):
                session = Session(cfg, client, "sys", "prompt", str(tmp_path),
                                  trace_file=tf, session_number=1)
                session.run()
        entry = json.loads(trace_path.read_text().strip().split("\n")[0])
        # Code-sized results pass through untouched.
        assert len(entry["result_summary"]) == 10000
        assert entry["result_summary"] == code_result

        # Now verify that results ABOVE the cap are still truncated.
        trace_path2 = tmp_path / ".trace2.jsonl"
        call_count[0] = 0  # reset the chat closure
        with open(trace_path2, "a") as tf:
            with patch("llm_solver.harness.loop.dispatch", return_value=huge_result):
                session = Session(cfg, client, "sys", "prompt", str(tmp_path),
                                  trace_file=tf, session_number=1)
                session.run()
        entry = json.loads(trace_path2.read_text().strip().split("\n")[0])
        assert len(entry["result_summary"]) == cfg.max_output_chars
        assert entry["result_summary"].endswith("...")

    def test_trace_no_file_no_error(self):
        """Session without trace_file runs without error."""
        from llm_solver.harness.loop import Session
        cfg = make_config(max_turns=3)
        client = MagicMock()
        client.chat.return_value = make_turn_result(content="done", finish_reason="stop")
        client.build_assistant_message.return_value = {"role": "assistant", "content": "done"}
        session = Session(cfg, client, "sys", "prompt", "/tmp")
        result = session.run()
        assert result.done is True


# ──────────────────────────────────────────────
# 19. Run metrics (#57)
# ──────────────────────────────────────────────

class TestRunMetrics:

    def test_session_result_has_completion_tokens(self):
        """SessionResult tracks total_completion_tokens."""
        from llm_solver.harness.loop import SessionResult
        result = SessionResult(
            turns=5, finish_reason="stop", done=True,
            total_prompt_tokens=100, total_completion_tokens=50,
        )
        assert result.total_completion_tokens == 50

    def test_session_result_completion_tokens_default_zero(self):
        """total_completion_tokens defaults to 0."""
        from llm_solver.harness.loop import SessionResult
        result = SessionResult(turns=1, finish_reason="stop", done=True)
        assert result.total_completion_tokens == 0

    def test_session_accumulates_completion_tokens(self):
        """Session.run() accumulates completion tokens across turns."""
        from llm_solver.harness.loop import Session
        cfg = make_config(max_turns=5, duplicate_abort=20)
        client = MagicMock()

        call_count = [0]
        def chat_fn(*args, **kwargs):
            call_count[0] += 1
            if call_count[0] > 2:
                return make_turn_result(content="done", finish_reason="stop", prompt_tokens=10)
            tc = [ToolCall(id=f"c{call_count[0]}", name="bash",
                          arguments={"cmd": f"echo {call_count[0]}"})]
            return make_turn_result(tool_calls=tc, finish_reason="tool_calls", prompt_tokens=10)

        client.chat.side_effect = chat_fn
        client.build_assistant_message.return_value = {"role": "assistant", "content": None}

        with patch("llm_solver.harness.loop.dispatch", return_value="ok"):
            session = Session(cfg, client, "sys", "prompt", "/tmp")
            result = session.run()

        # 3 API calls × 5 completion tokens each (make_turn_result default) = 15
        assert result.total_completion_tokens == 15

    def test_solve_task_writes_metrics_json(self, tmp_path):
        """solve_task writes metrics.json with required fields."""
        from llm_solver.harness.loop import solve_task
        (tmp_path / "prompt.txt").write_text("fix bug")

        client = MagicMock()
        client.chat.return_value = make_turn_result(
            content="Done!", finish_reason="stop", prompt_tokens=100,
        )
        client.build_assistant_message.return_value = {"role": "assistant", "content": "Done!"}

        cfg = make_config(max_turns=10, max_sessions=1)

        with patch("llm_solver.harness.loop._auto_commit"):
            solve_task(tmp_path, cfg, client)

        metrics_path = tmp_path / "metrics.json"
        assert metrics_path.exists()
        data = json.loads(metrics_path.read_text())

        assert "metrics" in data
        m = data["metrics"]
        assert m["total_prompt_tokens"] == 100
        assert m["total_completion_tokens"] == 5
        assert m["total_tokens"] == 105
        assert m["wall_clock_seconds"] > 0
        assert m["sessions_used"] == 1
        assert m["total_turns"] == 0  # model stopped immediately, turn index = 0

    def test_metrics_multi_session_aggregation(self, tmp_path):
        """Metrics aggregate across multiple sessions."""
        from llm_solver.harness.loop import solve_task
        (tmp_path / "prompt.txt").write_text("hard task")

        call_count = [0]
        def chat_fn(*args, **kwargs):
            call_count[0] += 1
            if call_count[0] == 1:  # session 1, turn 1: tool call
                tc = [ToolCall(id="c1", name="bash", arguments={"cmd": "echo hi"})]
                return make_turn_result(tool_calls=tc, finish_reason="tool_calls", prompt_tokens=50)
            if call_count[0] == 2:  # session 1, turn 2: length → end session
                return make_turn_result(content="...", finish_reason="length", prompt_tokens=60)
            # session 2: done
            return make_turn_result(content="Done!", finish_reason="stop", prompt_tokens=40)

        client = MagicMock()
        client.chat.side_effect = chat_fn
        client.build_assistant_message.return_value = {"role": "assistant", "content": "x"}

        cfg = make_config(max_turns=5, max_sessions=3, duplicate_abort=10)
        with patch("llm_solver.harness.loop.dispatch", return_value="output"):
            with patch("llm_solver.harness.loop._auto_commit"):
                solve_task(tmp_path, cfg, client)

        data = json.loads((tmp_path / "metrics.json").read_text())
        m = data["metrics"]
        assert m["total_prompt_tokens"] == 150  # 50 + 60 + 40
        assert m["sessions_used"] == 2
        assert m["wall_clock_seconds"] > 0

    def test_metrics_derived_values(self, tmp_path):
        """Derived metrics (time_per_session, tokens_per_turn) are computed."""
        from llm_solver.harness.loop import solve_task
        (tmp_path / "prompt.txt").write_text("task")

        client = MagicMock()
        call_count = [0]
        def chat_fn(*args, **kwargs):
            call_count[0] += 1
            if call_count[0] > 3:
                return make_turn_result(content="done", finish_reason="stop", prompt_tokens=10)
            tc = [ToolCall(id=f"c{call_count[0]}", name="bash",
                          arguments={"cmd": f"echo {call_count[0]}"})]
            return make_turn_result(tool_calls=tc, finish_reason="tool_calls", prompt_tokens=10)

        client.chat.side_effect = chat_fn
        client.build_assistant_message.return_value = {"role": "assistant", "content": None}

        cfg = make_config(max_turns=10, max_sessions=1, duplicate_abort=20)
        with patch("llm_solver.harness.loop.dispatch", return_value="ok"):
            with patch("llm_solver.harness.loop._auto_commit"):
                solve_task(tmp_path, cfg, client)

        data = json.loads((tmp_path / "metrics.json").read_text())
        m = data["metrics"]
        assert m["sessions_used"] == 1
        assert m["total_turns"] == 3  # turn index when stop happened
        assert "time_per_session_seconds" in m
        assert "tokens_per_turn" in m
        expected_tpt = m["total_tokens"] / m["total_turns"]
        assert abs(m["tokens_per_turn"] - expected_tpt) < 0.1

    def test_metrics_on_error(self, tmp_path):
        """Metrics are written even when solve_task fails."""
        from llm_solver.harness.loop import solve_task
        (tmp_path / "prompt.txt").write_text("do something")

        client = MagicMock()
        client.chat.side_effect = RuntimeError("crash")
        client.build_assistant_message.return_value = {"role": "assistant", "content": ""}

        cfg = make_config(max_turns=5, max_sessions=1)
        solve_task(tmp_path, cfg, client)

        metrics_path = tmp_path / "metrics.json"
        assert metrics_path.exists()
        data = json.loads(metrics_path.read_text())
        assert data["metrics"]["sessions_used"] == 1

    def test_no_metrics_without_prompt(self, tmp_path):
        """No metrics.json when prompt.txt is missing (no run happened)."""
        from llm_solver.harness.loop import solve_task
        client = MagicMock()
        cfg = make_config()
        solve_task(tmp_path, cfg, client)
        assert not (tmp_path / "metrics.json").exists()


# ──────────────────────────────────────────────
# 20. Run provenance (#60)
# ──────────────────────────────────────────────

class TestProvenance:

    def test_collect_provenance_required_fields(self):
        """Provenance includes timestamp and model."""
        from llm_solver.harness.solver import collect_provenance
        cfg = make_config()
        prov = collect_provenance(cfg)
        assert "timestamp" in prov
        assert prov["model"] == "test-model"

    def test_collect_provenance_git_commit(self):
        """Provenance exposes git commit when the harness repo has git metadata."""
        from llm_solver.harness.solver import collect_provenance
        cfg = make_config()
        prov = collect_provenance(cfg)
        if "harness_git_commit" in prov:
            assert len(prov["harness_git_commit"]) == 40

    def test_collect_provenance_profile_hash(self, tmp_path):
        """Provenance computes sha256 of profile file when provided."""
        from llm_solver.harness.solver import collect_provenance
        profile = tmp_path / "profile.toml"
        profile.write_text("[model]\nname = 'test'\n")

        cfg = make_config()
        prov = collect_provenance(cfg, profile_path=profile)
        assert "profile_toml_sha256" in prov
        assert len(prov["profile_toml_sha256"]) == 64  # SHA256 hex

    def test_collect_provenance_no_profile(self):
        """Provenance works without profile path — field omitted."""
        from llm_solver.harness.solver import collect_provenance
        cfg = make_config()
        prov = collect_provenance(cfg)
        assert "profile_toml_sha256" not in prov

    def test_collect_provenance_nonexistent_profile(self, tmp_path):
        """Provenance handles nonexistent profile path gracefully."""
        from llm_solver.harness.solver import collect_provenance
        cfg = make_config()
        prov = collect_provenance(cfg, profile_path=tmp_path / "nope.toml")
        assert "profile_toml_sha256" not in prov

    def test_write_run_metrics_creates_file(self, tmp_path):
        """write_run_metrics creates metrics.json with correct structure."""
        from llm_solver.harness.solver import write_run_metrics
        metrics = {"total_tokens": 100, "wall_clock_seconds": 5.0}
        provenance = {"model": "test", "timestamp": "2026-04-05T00:00:00"}
        write_run_metrics(tmp_path, metrics, provenance)

        data = json.loads((tmp_path / "metrics.json").read_text())
        assert data["metrics"]["total_tokens"] == 100
        assert data["provenance"]["model"] == "test"

    def test_solve_task_includes_provenance(self, tmp_path):
        """solve_task writes provenance section in metrics.json."""
        from llm_solver.harness.loop import solve_task
        (tmp_path / "prompt.txt").write_text("fix bug")

        client = MagicMock()
        client.chat.return_value = make_turn_result(content="Done!", finish_reason="stop")
        client.build_assistant_message.return_value = {"role": "assistant", "content": "Done!"}

        cfg = make_config(max_turns=10, max_sessions=1)

        with patch("llm_solver.harness.loop._auto_commit"):
            solve_task(tmp_path, cfg, client)

        data = json.loads((tmp_path / "metrics.json").read_text())
        assert "provenance" in data
        p = data["provenance"]
        assert "timestamp" in p
        assert p["model"] == "test-model"
        if "harness_git_commit" in p:
            assert len(p["harness_git_commit"]) == 40


# ──────────────────────────────────────────────
# 18. Prompt experimentation (#59)
# ──────────────────────────────────────────────

class TestPromptExperiment:
    """Prompt addendum injection, variant naming, and variant loading."""

    def test_config_experiment_defaults(self):
        """Config has prompt_addendum='' and variant_name='' by default."""
        cfg = make_config()
        assert cfg.prompt_addendum == ""
        assert cfg.variant_name == ""

    def test_config_experiment_fields(self):
        """Config accepts prompt_addendum and variant_name."""
        cfg = make_config(prompt_addendum="Always read tests first.", variant_name="read-tests")
        assert cfg.prompt_addendum == "Always read tests first."
        assert cfg.variant_name == "read-tests"

    def test_load_config_reads_experiment_section(self, tmp_path):
        """load_config reads [experiment] section from TOML."""
        toml_content = b'[experiment]\nprompt_addendum = "Run pytest after every edit."\nvariant_name = "pytest-loop"\n'
        cfg_file = tmp_path / "experiment.toml"
        cfg_file.write_bytes(toml_content)
        cfg = load_config(user_config=cfg_file)
        assert cfg.prompt_addendum == "Run pytest after every edit."
        assert cfg.variant_name == "pytest-loop"

    def test_solve_task_appends_addendum(self, tmp_path):
        """solve_task appends prompt_addendum to the task prompt."""
        from llm_solver.harness.loop import solve_task
        (tmp_path / "prompt.txt").write_text("Fix the bug in main.py")

        captured_messages = []
        client = MagicMock()
        client.chat.return_value = make_turn_result(content="Done!", finish_reason="stop")
        client.build_assistant_message.return_value = {"role": "assistant", "content": "Done!"}

        # Capture what gets passed to Session
        original_session_init = __import__("llm_solver.harness.loop", fromlist=["Session"]).Session.__init__
        def capture_init(self, cfg, cl, system, initial, cwd, **kw):
            captured_messages.append(initial)
            original_session_init(self, cfg, cl, system, initial, cwd, **kw)

        cfg = make_config(
            max_turns=10, max_sessions=1,
            prompt_addendum="Always read the test file before implementing.",
            variant_name="read-tests",
        )

        with patch("llm_solver.harness.loop._auto_commit"):
            with patch("llm_solver.harness.loop.Session.__init__", capture_init):
                solve_task(tmp_path, cfg, client)

        assert len(captured_messages) >= 1
        initial = captured_messages[0]
        assert "Fix the bug in main.py" in initial
        assert "Always read the test file before implementing." in initial

    def test_solve_task_no_addendum_unchanged(self, tmp_path):
        """solve_task with empty addendum leaves prompt unchanged."""
        from llm_solver.harness.loop import solve_task
        (tmp_path / "prompt.txt").write_text("Fix the bug")

        captured_messages = []
        client = MagicMock()
        client.chat.return_value = make_turn_result(content="Done!", finish_reason="stop")
        client.build_assistant_message.return_value = {"role": "assistant", "content": "Done!"}

        original_session_init = __import__("llm_solver.harness.loop", fromlist=["Session"]).Session.__init__
        def capture_init(self, cfg, cl, system, initial, cwd, **kw):
            captured_messages.append(initial)
            original_session_init(self, cfg, cl, system, initial, cwd, **kw)

        cfg = make_config(max_turns=10, max_sessions=1)

        with patch("llm_solver.harness.loop._auto_commit"):
            with patch("llm_solver.harness.loop.Session.__init__", capture_init):
                solve_task(tmp_path, cfg, client)

        initial = captured_messages[0]
        assert initial == "Fix the bug"

    def test_variant_name_in_provenance(self, tmp_path):
        """variant_name appears in metrics.json provenance when set."""
        from llm_solver.harness.loop import solve_task
        (tmp_path / "prompt.txt").write_text("task")

        client = MagicMock()
        client.chat.return_value = make_turn_result(content="Done!", finish_reason="stop")
        client.build_assistant_message.return_value = {"role": "assistant", "content": "Done!"}

        cfg = make_config(
            max_turns=10, max_sessions=1,
            variant_name="read-tests",
            prompt_addendum="Read tests first.",
        )

        with patch("llm_solver.harness.loop._auto_commit"):
            solve_task(tmp_path, cfg, client)

        data = json.loads((tmp_path / "metrics.json").read_text())
        assert data["provenance"]["variant_name"] == "read-tests"
        assert data["provenance"]["prompt_addendum"] == "Read tests first."

    def test_variant_name_absent_when_empty(self, tmp_path):
        """variant_name not in provenance when empty."""
        from llm_solver.harness.loop import solve_task
        (tmp_path / "prompt.txt").write_text("task")

        client = MagicMock()
        client.chat.return_value = make_turn_result(content="Done!", finish_reason="stop")
        client.build_assistant_message.return_value = {"role": "assistant", "content": "Done!"}

        cfg = make_config(max_turns=10, max_sessions=1)

        with patch("llm_solver.harness.loop._auto_commit"):
            solve_task(tmp_path, cfg, client)

        data = json.loads((tmp_path / "metrics.json").read_text())
        assert "variant_name" not in data["provenance"]

    def test_load_variants_from_toml(self, tmp_path):
        """load_variants reads named variants from TOML file."""
        from llm_solver.harness.experiment import load_variants
        toml_content = b"""\
[variants.baseline]
prompt_addendum = ""

[variants.read-tests]
prompt_addendum = "Always read the test file before implementing."

[variants.pytest-loop]
prompt_addendum = "Run pytest after every edit to verify correctness."
"""
        vf = tmp_path / "variants.toml"
        vf.write_bytes(toml_content)
        variants = load_variants(vf)
        assert len(variants) == 3
        assert variants["baseline"]["prompt_addendum"] == ""
        assert "read the test file" in variants["read-tests"]["prompt_addendum"]
        assert "pytest" in variants["pytest-loop"]["prompt_addendum"]

    def test_load_variants_file_not_found(self):
        """load_variants raises FileNotFoundError for missing file."""
        from llm_solver.harness.experiment import load_variants
        with pytest.raises(FileNotFoundError):
            load_variants(Path("/nonexistent/variants.toml"))

    def test_cli_experiment_flags(self, tmp_path):
        """CLI accepts --prompt-addendum and --variant-name flags."""
        from llm_solver.__main__ import main
        repos = tmp_path / "repos"
        repos.mkdir()
        ret = main([
            str(tmp_path), "--dry-run",
            "--prompt-addendum", "Test hint",
            "--variant-name", "test-hint",
        ])
        assert ret == 0
