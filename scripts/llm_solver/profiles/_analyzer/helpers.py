"""Small helpers used while assembling a profile TOML.

Chat-template classification, model-family derivation, passthrough-fixture
selection, and TOML-key formatting. All pure functions.
"""
from __future__ import annotations

import json
import re


def _classify_chat_template(raw: str) -> str:
    if not raw:
        return "chatml"
    t = raw.lower()
    if "im_start" in t or "im_end" in t:
        return "chatml"
    if "[inst]" in t or "[/inst]" in t:
        return "llama"
    if "gmask" in t or "sop" in t:
        return "glm"
    if "<start_of_turn>" in t or "<|turn>" in t:
        return "gemma"
    return "chatml"


def _derive_family(model_name: str) -> str:
    parts = model_name.split("-")
    if len(parts) >= 2 and re.match(r"^\d+[bB]$", parts[1]):
        return parts[0]
    return "-".join(parts[:2]) if len(parts) >= 2 else parts[0]


def _find_passthrough(samples: list[dict], quirk_scenarios: set[str]) -> dict | None:
    """Find the cleanest sample for the passthrough fixture."""
    for s in samples:
        if s["scenario_id"] in quirk_scenarios:
            continue
        if s.get("error"):
            continue
        content = s.get("response", {}).get("content")
        if content and content.strip():
            return s
    return None


def _toml_kv(key: str, value) -> str:
    """Serialize one (key, value) pair as a TOML scalar assignment line."""
    if isinstance(value, bool):
        return f"{key} = {'true' if value else 'false'}"
    if isinstance(value, int):
        return f"{key} = {value}"
    if isinstance(value, float):
        return f"{key} = {value}"
    if isinstance(value, str):
        escaped = value.replace("\\", "\\\\").replace('"', '\\"')
        if len(escaped) > 300:
            escaped = escaped[:297] + "..."
        return f'{key} = "{escaped}"'
    if isinstance(value, (list, dict)):
        s = json.dumps(value, separators=(",", ":"))
        if len(s) > 300:
            s = s[:297] + "..."
        return f"{key} = '{s}'"
    if value is None:
        return f'{key} = ""'
    return f'{key} = "{value}"'


__all__ = [
    "_classify_chat_template",
    "_derive_family",
    "_find_passthrough",
    "_toml_kv",
]
