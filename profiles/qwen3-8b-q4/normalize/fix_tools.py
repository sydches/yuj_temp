"""Fix tool calls for Qwen3-8B-Q4 quirks.

Handles:
- arguments-as-dict (llama-server #20198): arguments field is a dict instead of JSON string
- Missing tool_call_id: generate deterministic IDs when missing
"""
import json
import re


def apply(response: dict) -> dict:
    """Fix tool call quirks in Qwen3 output."""
    tool_calls = response.get("tool_calls")
    if not tool_calls:
        return response

    fixed = []
    for i, tc in enumerate(tool_calls):
        if not isinstance(tc, dict):
            fixed.append(tc)
            continue

        tc = dict(tc)  # shallow copy

        # Fix arguments-as-dict (llama-server bug #20198)
        func = tc.get("function", {})
        if isinstance(func, dict):
            args = func.get("arguments")
            if isinstance(args, dict):
                func = dict(func)
                func["arguments"] = json.dumps(args)
                tc["function"] = func

        # Generate missing tool_call_id
        if not tc.get("id"):
            tc["id"] = f"call_0_{i}"

        fixed.append(tc)

    response["tool_calls"] = fixed
    return response
