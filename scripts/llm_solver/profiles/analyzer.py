"""ProfileAnalyzer — deterministic profile generation from scenario samples.

No LLM call. Reads scenario samples + server metadata + HuggingFace config,
detects quirks via pattern matching, and emits a full profile directory
(rules.toml, normalize/ modules, fixtures, denormalize/ rules).

This module is a thin facade over ``_analyzer/`` which holds the implementation
split into ``types`` (dataclasses), ``detectors`` (quirk detection + capability
observation), ``helpers`` (chat-template classification, family derivation,
TOML-kv serialization), and ``core`` (``ProfileAnalyzer`` class). Importers
should use this facade, not reach into ``_analyzer`` directly.
"""
from ._analyzer.core import ProfileAnalyzer
from ._analyzer.detectors import (
    ALL_DETECTORS,
    _detect_arguments_as_dict,
    _detect_empty_content_on_tool_calls,
    _detect_missing_tool_call_id,
    _detect_system_prompt_support,
    _detect_thinking_blocks,
    _detect_tool_calls_in_content,
    _detect_trailing_whitespace,
    _detect_wrong_finish_reason,
    _observe_system_support,
    _observe_tool_support,
)
from ._analyzer.helpers import _classify_chat_template, _derive_family
from ._analyzer.types import QuirkResult

__all__ = [
    "ProfileAnalyzer",
    "QuirkResult",
    "ALL_DETECTORS",
    "_detect_thinking_blocks",
    "_detect_arguments_as_dict",
    "_detect_missing_tool_call_id",
    "_detect_wrong_finish_reason",
    "_detect_tool_calls_in_content",
    "_detect_empty_content_on_tool_calls",
    "_detect_trailing_whitespace",
    "_detect_system_prompt_support",
    "_classify_chat_template",
    "_derive_family",
    "_observe_tool_support",
    "_observe_system_support",
]
