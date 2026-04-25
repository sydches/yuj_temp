"""Tests for post-normalization verification (scripts/llm_solver/profiles/verify.py).

Two test categories:
1. Real data: run verification against all 4 existing model profiles — all must pass.
2. Dirty data: inject known-bad patterns into responses and verify the checker catches them.
"""
import copy
import json
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from llm_solver.profiles.verify import (
    CANONICAL_FINISH_REASONS,
    CheckFailure,
    ScenarioResult,
    VerificationResult,
    _check_content_clean,
    _check_content_none_consistency,
    _check_finish_reason,
    _check_tool_calls,
    verify_server_config,
    format_report,
    verify_profile,
    verify_scenario,
)
from llm_solver.server.profile_loader import load_profile

PROFILES_DIR = PROJECT_ROOT / "profiles"
# Active models only — gemma-4-26b deprecated (samples removed).
MODEL_NAMES = ["qwen3-8b-q4", "qwen3.5-9b", "glm-4-flash"]


# ──────────────────────────────────────────────
# 1. Real data — all models must pass
# ──────────────────────────────────────────────


class TestRealData:
    """Verify against real sample data from all 4 model profiles."""

    @pytest.fixture(params=MODEL_NAMES)
    def model_name(self, request):
        return request.param

    def test_all_scenarios_pass(self, model_name):
        """Normalize pipeline produces clean output for all scenarios."""
        profile_dir = PROFILES_DIR / model_name
        samples_path = profile_dir / "_samples" / "_all_results.json"
        if not samples_path.is_file():
            pytest.skip(f"No samples for {model_name}")

        result = verify_profile(profile_dir, samples_path)
        if result.failed > 0:
            failures = []
            for s in result.scenarios:
                for f in s.failures:
                    failures.append(f"{s.scenario_id}: [{f.check}] {f.message}")
            pytest.fail(
                f"{model_name}: {result.failed}/{result.total} scenario failures:\n"
                + "\n".join(failures)
            )

    def test_scenario_count(self, model_name):
        profile_dir = PROFILES_DIR / model_name
        samples_path = profile_dir / "_samples" / "_all_results.json"
        if not samples_path.is_file():
            pytest.skip(f"No samples for {model_name}")

        result = verify_profile(profile_dir, samples_path)
        assert result.total == 47, f"Expected 47 scenarios, got {result.total}"


# ──────────────────────────────────────────────
# 2. Dirty data — verify checker catches issues
# ──────────────────────────────────────────────


def _make_clean_response():
    """Return a minimal clean response dict."""
    return {
        "content": "Hello, world!",
        "tool_calls": None,
        "finish_reason": "stop",
        "usage": {"prompt_tokens": 10, "completion_tokens": 5},
    }


def _make_tool_call_response():
    """Return a clean response with tool calls."""
    return {
        "content": "",
        "tool_calls": [
            {
                "id": "call_0_0",
                "type": "function",
                "function": {
                    "name": "bash",
                    "arguments": '{"cmd": "pwd"}',
                },
            }
        ],
        "finish_reason": "tool_calls",
        "usage": {"prompt_tokens": 10, "completion_tokens": 5},
    }


class TestContentClean:
    """Test _check_content_clean catches leaked tokens."""

    def test_clean_content_passes(self):
        resp = _make_clean_response()
        assert _check_content_clean(resp) == []

    def test_none_content_passes(self):
        resp = _make_clean_response()
        resp["content"] = None
        assert _check_content_clean(resp) == []

    def test_leaked_think_tag(self):
        resp = _make_clean_response()
        resp["content"] = "Some text <think>internal thought</think> more text"
        failures = _check_content_clean(resp)
        assert len(failures) >= 1
        checks = [f.check for f in failures]
        assert "content_clean" in checks

    def test_leaked_think_open_only(self):
        resp = _make_clean_response()
        resp["content"] = "Text before <think> reasoning here"
        failures = _check_content_clean(resp)
        assert any(f.check == "content_clean" for f in failures)

    def test_leaked_think_close_only(self):
        resp = _make_clean_response()
        resp["content"] = "Result text</think>"
        failures = _check_content_clean(resp)
        assert any(f.check == "content_clean" for f in failures)

    def test_leaked_channel_token(self):
        resp = _make_clean_response()
        resp["content"] = "Text <|channel>thought\nreasoning<channel|> more"
        failures = _check_content_clean(resp)
        assert any(f.check == "content_clean" for f in failures)

    def test_leaked_bare_channel_token(self):
        resp = _make_clean_response()
        resp["content"] = "Text <channel|> remaining"
        failures = _check_content_clean(resp)
        assert any(f.check == "content_clean" for f in failures)

    def test_trailing_whitespace(self):
        resp = _make_clean_response()
        resp["content"] = "Some text   \n"
        failures = _check_content_clean(resp)
        assert any(f.message == "trailing whitespace in content" for f in failures)

    def test_no_trailing_whitespace(self):
        resp = _make_clean_response()
        resp["content"] = "Clean text"
        assert _check_content_clean(resp) == []


class TestToolCalls:
    """Test _check_tool_calls catches malformed tool calls."""

    def test_clean_tool_calls_pass(self):
        resp = _make_tool_call_response()
        assert _check_tool_calls(resp) == []

    def test_no_tool_calls_pass(self):
        resp = _make_clean_response()
        assert _check_tool_calls(resp) == []

    def test_dict_arguments_caught(self):
        resp = _make_tool_call_response()
        resp["tool_calls"][0]["function"]["arguments"] = {"cmd": "pwd"}
        failures = _check_tool_calls(resp)
        assert any(f.check == "tool_call_arguments" for f in failures)

    def test_missing_id_caught(self):
        resp = _make_tool_call_response()
        resp["tool_calls"][0]["id"] = ""
        failures = _check_tool_calls(resp)
        assert any(f.check == "tool_call_id" for f in failures)

    def test_none_id_caught(self):
        resp = _make_tool_call_response()
        del resp["tool_calls"][0]["id"]
        failures = _check_tool_calls(resp)
        assert any(f.check == "tool_call_id" for f in failures)

    def test_invalid_json_arguments_caught(self):
        resp = _make_tool_call_response()
        resp["tool_calls"][0]["function"]["arguments"] = "{broken json"
        failures = _check_tool_calls(resp)
        assert any(f.check == "tool_call_arguments" for f in failures)

    def test_non_dict_tool_call_caught(self):
        resp = _make_tool_call_response()
        resp["tool_calls"] = ["not a dict"]
        failures = _check_tool_calls(resp)
        assert any(f.check == "tool_call_format" for f in failures)


class TestContentNoneConsistency:
    """Test _check_content_none_consistency catches whitespace-only content."""

    def test_empty_string_passes(self):
        resp = _make_tool_call_response()
        # "" is a valid empty state — not a failure
        assert _check_content_none_consistency(resp) == []

    def test_none_passes(self):
        resp = _make_tool_call_response()
        resp["content"] = None
        assert _check_content_none_consistency(resp) == []

    def test_whitespace_only_caught(self):
        resp = _make_tool_call_response()
        resp["content"] = "   \n  "
        failures = _check_content_none_consistency(resp)
        assert any(f.check == "content_empty" for f in failures)

    def test_real_content_passes(self):
        resp = _make_clean_response()
        assert _check_content_none_consistency(resp) == []


class TestFinishReason:
    """Test _check_finish_reason validates canonical set."""

    @pytest.mark.parametrize("reason", sorted(CANONICAL_FINISH_REASONS))
    def test_canonical_reasons_pass(self, reason):
        resp = {"finish_reason": reason}
        assert _check_finish_reason(resp) == []

    def test_non_canonical_reason_caught(self):
        resp = {"finish_reason": "end_turn"}
        failures = _check_finish_reason(resp)
        assert any(f.check == "finish_reason" for f in failures)

    def test_none_reason_caught(self):
        resp = {"finish_reason": None}
        failures = _check_finish_reason(resp)
        assert any(f.check == "finish_reason" for f in failures)


# ──────────────────────────────────────────────
# 3. Server config validation — #53
# ──────────────────────────────────────────────


class TestVerifyServerConfig:
    """Test verify_server_config catches invalid [server] fields."""

    def _write_profile_toml(self, tmp_path, server_overrides: dict) -> Path:
        """Write a minimal profile.toml with custom [server] section."""
        server = {
            "model_path": "~/models/test.gguf",
            "ctx_size": 40960,
            "flash_attn": True,
            "cache_type_k": "q8_0",
            "cache_type_v": "q8_0",
            "jinja": True,
            "reasoning_mode": "none",
            "reasoning_disable_flag": "",
            "extra_flags": "",
        }
        server.update(server_overrides)

        lines = [
            '[profile]',
            'format_version = 1',
            'canonical_version = "openai-v1"',
            'name = "test"',
            'family = "test"',
            'quant = ""',
            'inherits = "_base"',
            '',
            '[model]',
            'context_size = 40960',
            'chat_template = "chatml"',
            'supports_tool_calls = true',
            'supports_system_role = true',
            '',
            '[server]',
        ]
        for k, v in server.items():
            if isinstance(v, bool):
                lines.append(f"{k} = {'true' if v else 'false'}")
            elif isinstance(v, int):
                lines.append(f"{k} = {v}")
            else:
                lines.append(f'{k} = "{v}"')

        profile_dir = tmp_path / "test-profile"
        profile_dir.mkdir(parents=True, exist_ok=True)
        (profile_dir / "profile.toml").write_text("\n".join(lines) + "\n")
        return profile_dir

    def test_valid_server_config_passes(self, tmp_path):
        profile_dir = self._write_profile_toml(tmp_path, {})
        failures = verify_server_config(profile_dir)
        assert failures == []

    def test_empty_model_path_fails(self, tmp_path):
        profile_dir = self._write_profile_toml(tmp_path, {"model_path": ""})
        failures = verify_server_config(profile_dir)
        assert any(f.check == "server_model_path" for f in failures)

    def test_missing_model_path_fails(self, tmp_path):
        """profile.toml without model_path key at all should fail."""
        profile_dir = tmp_path / "no-model-path"
        profile_dir.mkdir(parents=True, exist_ok=True)
        (profile_dir / "profile.toml").write_text(
            '[profile]\nname = "test"\n[server]\nctx_size = 8192\n'
        )
        failures = verify_server_config(profile_dir)
        assert any(f.check == "server_model_path" for f in failures)

    def test_ctx_size_zero_fails(self, tmp_path):
        profile_dir = self._write_profile_toml(tmp_path, {"ctx_size": 0})
        failures = verify_server_config(profile_dir)
        assert any(f.check == "server_ctx_size" for f in failures)

    def test_ctx_size_positive_passes(self, tmp_path):
        profile_dir = self._write_profile_toml(tmp_path, {"ctx_size": 40960})
        failures = verify_server_config(profile_dir)
        ctx_failures = [f for f in failures if f.check == "server_ctx_size"]
        assert ctx_failures == []

    def test_reasoning_think_with_flag_passes(self, tmp_path):
        profile_dir = self._write_profile_toml(tmp_path, {
            "reasoning_mode": "think",
            "reasoning_disable_flag": "--reasoning-budget 0",
        })
        failures = verify_server_config(profile_dir)
        reasoning_failures = [f for f in failures if f.check == "server_reasoning"]
        assert reasoning_failures == []

    def test_reasoning_think_without_flag_fails(self, tmp_path):
        profile_dir = self._write_profile_toml(tmp_path, {
            "reasoning_mode": "think",
            "reasoning_disable_flag": "",
        })
        failures = verify_server_config(profile_dir)
        assert any(f.check == "server_reasoning" for f in failures)

    def test_reasoning_channel_without_flag_fails(self, tmp_path):
        profile_dir = self._write_profile_toml(tmp_path, {
            "reasoning_mode": "channel",
            "reasoning_disable_flag": "",
        })
        failures = verify_server_config(profile_dir)
        assert any(f.check == "server_reasoning" for f in failures)

    def test_reasoning_none_with_empty_flag_passes(self, tmp_path):
        profile_dir = self._write_profile_toml(tmp_path, {
            "reasoning_mode": "none",
            "reasoning_disable_flag": "",
        })
        failures = verify_server_config(profile_dir)
        reasoning_failures = [f for f in failures if f.check == "server_reasoning"]
        assert reasoning_failures == []

    def test_complete_profiles_pass(self):
        """Profiles with populated [server] fields pass validation."""
        profile_dir = PROFILES_DIR / "qwen3-8b-q4"
        if not (profile_dir / "profile.toml").is_file():
            pytest.skip("qwen3-8b-q4 profile not found")
        failures = verify_server_config(profile_dir)
        assert failures == [], (
            "qwen3-8b-q4 server config failures: "
            + "; ".join(f"[{f.check}] {f.message}" for f in failures)
        )

    def test_real_profiles_have_valid_model_path(self):
        """Active profiles have non-empty model_path after pipeline-hardening."""
        for model_name in ["qwen3.5-9b", "glm-4-flash"]:
            profile_dir = PROFILES_DIR / model_name
            if not (profile_dir / "profile.toml").is_file():
                continue
            failures = verify_server_config(profile_dir)
            checks = {f.check for f in failures}
            assert "server_model_path" not in checks, (
                f"{model_name}: model_path should be populated"
            )

    def test_server_failures_in_verify_profile(self, tmp_path):
        """verify_profile includes server config failures in result."""
        import shutil

        profiles_tmp = tmp_path / "profiles"
        profiles_tmp.mkdir()
        shutil.copytree(PROFILES_DIR / "_base", profiles_tmp / "_base")

        # Create a profile with empty model_path
        profile_dir = profiles_tmp / "bad-server"
        profile_dir.mkdir()
        (profile_dir / "profile.toml").write_text(
            '[profile]\nformat_version = 1\ncanonical_version = "openai-v1"\n'
            'name = "bad-server"\nfamily = "test"\nquant = ""\ninherits = "_base"\n\n'
            '[model]\ncontext_size = 40960\nchat_template = "chatml"\n'
            'supports_tool_calls = true\nsupports_system_role = true\n\n'
            '[tokens]\nmethod = "chars_div_4"\ntokenizer = ""\n\n'
            '[capacity]\npreamble = ""\nmax_tools = 6\nsimplify_schemas = false\n\n'
            '[normalize]\nrules = ["rules.toml"]\nmodules = []\n\n'
            '[denormalize]\nrules = ["rules.toml"]\nmodules = []\n\n'
            '[server]\nmodel_path = ""\nctx_size = 0\n'
            'reasoning_mode = "think"\nreasoning_disable_flag = ""\n'
        )
        # Need normalize/denormalize dirs for profile_loader
        norm_dir = profile_dir / "normalize"
        norm_dir.mkdir()
        (norm_dir / "rules.toml").write_text("# empty\n")
        denorm_dir = profile_dir / "denormalize"
        denorm_dir.mkdir()
        (denorm_dir / "rules.toml").write_text('[system_prompt]\nstrategy = "native"\n')

        # Write minimal samples
        samples_dir = profile_dir / "_samples"
        samples_dir.mkdir()
        import json
        samples = [{"scenario_id": "t1", "response": {"content": "hi", "tool_calls": None, "finish_reason": "stop"}}]
        (samples_dir / "_all_results.json").write_text(json.dumps(samples))

        result = verify_profile(profile_dir, samples_dir / "_all_results.json")
        assert not result.all_passed
        # Should have server-level failures
        assert result.server_failures
        checks = {f.check for f in result.server_failures}
        assert "server_model_path" in checks
        assert "server_ctx_size" in checks
        assert "server_reasoning" in checks


# ──────────────────────────────────────────────
# 4. Integration — verify_scenario with dirty data
# ──────────────────────────────────────────────


class TestVerifyScenario:
    """Test verify_scenario end-to-end with a real profile and injected dirty data."""

    @pytest.fixture
    def profile(self):
        """Load _base profile for testing."""
        return load_profile("_base", PROFILES_DIR)

    def test_clean_scenario_passes(self, profile):
        scenario = {
            "scenario_id": "test_clean",
            "response": _make_clean_response(),
        }
        result = verify_scenario(profile, scenario)
        assert result.passed
        assert result.failures == []

    def test_leaked_think_after_normalize(self, profile):
        """Inject a <think> block that's NOT stripped by normalize (unclosed)."""
        scenario = {
            "scenario_id": "test_leaked_think",
            "response": {
                "content": "Result text<think>should be stripped",
                "tool_calls": None,
                "finish_reason": "stop",
                "usage": {"prompt_tokens": 10, "completion_tokens": 5},
            },
        }
        # _base strips <think>.*?</think> — but unclosed <think> leaks through
        result = verify_scenario(profile, scenario)
        assert not result.passed
        assert any(f.check == "content_clean" for f in result.failures)

    def test_dict_arguments_after_normalize(self, profile):
        """Dict arguments are not fixed by _base normalize (no module wired)."""
        scenario = {
            "scenario_id": "test_dict_args",
            "response": {
                "content": "",
                "tool_calls": [
                    {
                        "id": "call_0_0",
                        "type": "function",
                        "function": {
                            "name": "bash",
                            "arguments": {"cmd": "ls"},
                        },
                    }
                ],
                "finish_reason": "tool_calls",
                "usage": {"prompt_tokens": 10, "completion_tokens": 5},
            },
        }
        result = verify_scenario(profile, scenario)
        assert not result.passed
        assert any(f.check == "tool_call_arguments" for f in result.failures)

    def test_missing_tool_call_id_after_normalize(self, profile):
        """Missing tool_call id — _base fix_tool_call rule should generate one."""
        scenario = {
            "scenario_id": "test_missing_id",
            "response": {
                "content": "",
                "tool_calls": [
                    {
                        "id": "",
                        "type": "function",
                        "function": {
                            "name": "bash",
                            "arguments": '{"cmd": "ls"}',
                        },
                    }
                ],
                "finish_reason": "tool_calls",
                "usage": {"prompt_tokens": 10, "completion_tokens": 5},
            },
        }
        result = verify_scenario(profile, scenario)
        # _base has no fix_tool_call rule in rules.toml, so empty id passes through
        assert not result.passed
        assert any(f.check == "tool_call_id" for f in result.failures)

    def test_channel_tokens_after_normalize(self, profile):
        """Channel tokens are not stripped by _base — only gemma profile strips them."""
        scenario = {
            "scenario_id": "test_channel",
            "response": {
                "content": "Text <|channel>thought\nreasoning<channel|> more text",
                "tool_calls": None,
                "finish_reason": "stop",
                "usage": {"prompt_tokens": 10, "completion_tokens": 5},
            },
        }
        result = verify_scenario(profile, scenario)
        assert not result.passed
        assert any(f.check == "content_clean" for f in result.failures)


# ──────────────────────────────────────────────
# 4. Report formatting
# ──────────────────────────────────────────────


class TestFormatReport:
    def test_pass_report(self):
        result = VerificationResult(
            profile_name="test",
            total=3,
            passed=3,
            failed=0,
            scenarios=[
                ScenarioResult("s1", passed=True),
                ScenarioResult("s2", passed=True),
                ScenarioResult("s3", passed=True),
            ],
        )
        report = format_report(result)
        assert "[PASS]" in report
        assert "3/3 passed" in report

    def test_fail_report(self):
        result = VerificationResult(
            profile_name="test",
            total=2,
            passed=1,
            failed=1,
            scenarios=[
                ScenarioResult("s1", passed=True),
                ScenarioResult(
                    "s2",
                    passed=False,
                    failures=[
                        CheckFailure(
                            check="content_clean",
                            message="leaked <think> tag",
                            expected="no special tokens",
                            actual="'...<think>...'",
                        )
                    ],
                ),
            ],
        )
        report = format_report(result)
        assert "[FAIL]" in report
        assert "1/2 passed" in report
        assert "s2:" in report
        assert "leaked <think> tag" in report
