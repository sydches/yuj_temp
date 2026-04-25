"""Quirk detectors + generated code-module templates.

Each ``_detect_*`` function takes a list of scenario sample dicts and returns
a :class:`QuirkResult` or ``None``. Detectors are pure functions; they have
no side effects and never mutate the samples they inspect.

``ALL_DETECTORS`` is the ordered list that :class:`ProfileAnalyzer.analyze`
iterates over. ``_observe_tool_support`` and ``_observe_system_support`` are
capability observations (also consumed by ``ProfileAnalyzer`` when it builds
``[model]`` fields in the profile TOML) that live here because they have the
same "inspect samples" shape as a detector.

The ``_CODE_*`` constants are Python source strings that get written to disk
as normalize code modules when a detector fires. They live in this file
because each is paired with exactly one detector.
"""
from __future__ import annotations

import json
import re

from .types import QuirkResult


_CODE_FIX_ARGUMENTS = '''\
"""Fix arguments-as-dict quirk (llama-server bug #20198)."""
import json


def apply(response: dict) -> dict:
    for tc in response.get("tool_calls", []):
        func = tc.get("function", {})
        args = func.get("arguments")
        if isinstance(args, dict):
            func["arguments"] = json.dumps(args)
    return response
'''


_CODE_EXTRACT_TOOL_CALLS = '''\
"""Extract tool calls embedded as JSON text in content."""
import json
import re


def apply(response: dict) -> dict:
    if response.get("tool_calls"):
        return response
    content = response.get("content") or ""
    if \'{"name"\' not in content:
        return response
    try:
        match = re.search(r\'\\{[^{}]*"name"[^{}]*"arguments"[^{}]*\\}\', content)
        if match:
            tc = json.loads(match.group())
            response["tool_calls"] = [{
                "id": "call_extracted_0",
                "type": "function",
                "function": {
                    "name": tc["name"],
                    "arguments": json.dumps(tc.get("arguments", {}))
                            if isinstance(tc.get("arguments"), dict)
                            else tc.get("arguments", "{}"),
                },
            }]
            response["finish_reason"] = "tool_calls"
            response["content"] = content[:match.start()].strip() or None
    except (json.JSONDecodeError, KeyError):
        pass
    return response
'''


def _detect_thinking_blocks(samples: list[dict]) -> QuirkResult | None:
    affected = []
    fixture_cases = []
    for s in samples:
        content = s.get("response", {}).get("content") or ""
        if "<think>" in content and "</think>" in content:
            affected.append(s["scenario_id"])
            cleaned = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL).strip()
            fixture_cases.append({
                "input": {"content": content, "tool_calls": s["response"].get("tool_calls", []),
                          "finish_reason": s["response"].get("finish_reason", "stop")},
                "expected": {"content": cleaned or None, "tool_calls": s["response"].get("tool_calls", []),
                             "finish_reason": s["response"].get("finish_reason", "stop")},
            })
    if not affected:
        return None
    return QuirkResult(
        name="thinking_blocks",
        description=f"<think> blocks in content ({len(affected)} scenarios). Handled by _base strip rule.",
        affected_scenarios=affected,
        fixture={"description": "Thinking block stripping (confirmed _base handles)", "cases": fixture_cases[:3]},
    )


def _detect_arguments_as_dict(samples: list[dict]) -> QuirkResult | None:
    affected = []
    fixture_cases = []
    for s in samples:
        for tc in s.get("response", {}).get("tool_calls", []):
            args = tc.get("function", {}).get("arguments")
            if isinstance(args, dict):
                affected.append(s["scenario_id"])
                input_tc = [dict(t) for t in s["response"]["tool_calls"]]
                expected_tc = []
                for t in s["response"]["tool_calls"]:
                    t2 = dict(t)
                    t2["function"] = dict(t2["function"])
                    a = t2["function"]["arguments"]
                    if isinstance(a, dict):
                        t2["function"]["arguments"] = json.dumps(a)
                    expected_tc.append(t2)
                fixture_cases.append({
                    "input": {"content": s["response"].get("content"), "tool_calls": input_tc,
                              "finish_reason": s["response"].get("finish_reason", "stop")},
                    "expected": {"content": s["response"].get("content"), "tool_calls": expected_tc,
                                 "finish_reason": s["response"].get("finish_reason", "stop")},
                })
                break
    if not affected:
        return None
    return QuirkResult(
        name="arguments_as_dict",
        description=f"Tool call arguments returned as dict instead of JSON string ({len(affected)} scenarios)",
        affected_scenarios=affected,
        code_module=_CODE_FIX_ARGUMENTS,
        module_filename="fix_arguments.py",
        fixture={"description": "Arguments dict → JSON string conversion", "cases": fixture_cases[:3]},
    )


def _detect_missing_tool_call_id(samples: list[dict]) -> QuirkResult | None:
    affected = []
    for s in samples:
        for tc in s.get("response", {}).get("tool_calls", []):
            if not tc.get("id"):
                affected.append(s["scenario_id"])
                break
    if not affected:
        return None
    return QuirkResult(
        name="missing_tool_call_id",
        description=f"Empty or missing tool_call_id ({len(affected)} scenarios)",
        affected_scenarios=affected,
        rule_toml='[[fix_tool_call]]\nguard = { finish_reason = ["tool_calls", "tool"] }\nwhen = "id_missing"\nstrategy = "generate"\n',
        fixture={"description": "Generate tool_call_id when missing", "cases": []},
    )


def _detect_wrong_finish_reason(samples: list[dict]) -> QuirkResult | None:
    canonical = {"stop", "tool_calls", "length"}
    non_standard: dict[str, list[str]] = {}
    for s in samples:
        fr = s.get("response", {}).get("finish_reason", "")
        if fr and fr not in canonical:
            non_standard.setdefault(fr, []).append(s["scenario_id"])
    if not non_standard:
        return None
    rules = []
    for from_val, scenarios in non_standard.items():
        to_val = "tool_calls" if "tool" in from_val else "stop"
        rules.append(f'[[map_finish_reason]]\nfrom = "{from_val}"\nto = "{to_val}"\n')
    return QuirkResult(
        name="wrong_finish_reason",
        description=f"Non-standard finish_reason values: {list(non_standard.keys())}",
        affected_scenarios=[sid for sids in non_standard.values() for sid in sids],
        rule_toml="\n".join(rules),
        fixture={"description": "finish_reason mapping", "cases": []},
    )


def _detect_tool_calls_in_content(samples: list[dict]) -> QuirkResult | None:
    affected = []
    for s in samples:
        content = s.get("response", {}).get("content") or ""
        tool_calls = s.get("response", {}).get("tool_calls", [])
        if not tool_calls and '{"name"' in content and '"arguments"' in content:
            affected.append(s["scenario_id"])
    if not affected:
        return None
    return QuirkResult(
        name="tool_calls_in_content",
        description=f"Tool calls embedded as JSON in content field ({len(affected)} scenarios)",
        affected_scenarios=affected,
        code_module=_CODE_EXTRACT_TOOL_CALLS,
        module_filename="extract_tool_calls.py",
        fixture={"description": "Extract tool calls from content", "cases": []},
    )


def _detect_empty_content_on_tool_calls(samples: list[dict]) -> QuirkResult | None:
    affected = []
    for s in samples:
        content = s.get("response", {}).get("content")
        tool_calls = s.get("response", {}).get("tool_calls", [])
        if content == "" and tool_calls:
            affected.append(s["scenario_id"])
    if not affected:
        return None
    return QuirkResult(
        name="empty_content_on_tool_calls",
        description=f"Empty string content (not None) when making tool calls ({len(affected)} scenarios). Handled by _base.",
        affected_scenarios=affected,
        fixture={"description": "Empty content → None on tool calls (confirmed _base handles)", "cases": []},
    )


def _detect_trailing_whitespace(samples: list[dict]) -> QuirkResult | None:
    affected = []
    for s in samples:
        content = s.get("response", {}).get("content") or ""
        if content and content != content.rstrip():
            affected.append(s["scenario_id"])
    if not affected:
        return None
    return QuirkResult(
        name="trailing_whitespace",
        description=f"Trailing whitespace/newlines in content ({len(affected)} scenarios). Handled by _base.",
        affected_scenarios=affected,
    )


def _detect_system_prompt_support(samples: list[dict]) -> QuirkResult | None:
    with_system = [s for s in samples
                   if any(m.get("role") == "system" for m in s.get("request", {}).get("messages", []))]
    if not with_system:
        return None
    errors = [s for s in with_system if s.get("error")]
    empty = [s for s in with_system
             if not s.get("error")
             and not s.get("response", {}).get("content")
             and not s.get("response", {}).get("tool_calls")]
    if errors or len(empty) > len(with_system) * 0.5:
        return QuirkResult(
            name="no_system_role",
            description=f"System role not supported ({len(errors)} errors, {len(empty)} empty out of {len(with_system)} scenarios)",
            affected_scenarios=[s["scenario_id"] for s in errors + empty],
            rule_toml='[system_prompt]\nstrategy = "fold_into_user"\n',
        )
    return None


ALL_DETECTORS = [
    _detect_thinking_blocks,
    _detect_arguments_as_dict,
    _detect_missing_tool_call_id,
    _detect_wrong_finish_reason,
    _detect_tool_calls_in_content,
    _detect_empty_content_on_tool_calls,
    _detect_trailing_whitespace,
    _detect_system_prompt_support,
]


def _observe_tool_support(samples: list[dict]) -> bool:
    for s in samples:
        tcs = s.get("response", {}).get("tool_calls", [])
        fr = s.get("response", {}).get("finish_reason", "")
        if tcs and fr in ("tool_calls", "tool"):
            return True
    return False


def _observe_system_support(samples: list[dict]) -> bool:
    with_sys = [s for s in samples
                if any(m.get("role") == "system" for m in s.get("request", {}).get("messages", []))]
    if not with_sys:
        return True  # no data, assume supported
    errors = sum(1 for s in with_sys if s.get("error"))
    return errors < len(with_sys) * 0.5


__all__ = [
    "ALL_DETECTORS",
    "_CODE_FIX_ARGUMENTS",
    "_CODE_EXTRACT_TOOL_CALLS",
    "_detect_thinking_blocks",
    "_detect_arguments_as_dict",
    "_detect_missing_tool_call_id",
    "_detect_wrong_finish_reason",
    "_detect_tool_calls_in_content",
    "_detect_empty_content_on_tool_calls",
    "_detect_trailing_whitespace",
    "_detect_system_prompt_support",
    "_observe_tool_support",
    "_observe_system_support",
]
