# Profile Generation Prompt

You are generating a **model profile** — the adapter artifact that normalizes a specific model's output into canonical format, and denormalizes canonical messages into the model's preferred input format.

## Your task

Analyze the behavioral samples below and produce a complete profile directory.

---

## Canonical format

The harness expects all model output in OpenAI chat completion format:

**TurnResult** (what normalize produces):
```
content: str | None          — text response, None if tool-only
tool_calls: list[ToolCall]   — structured tool invocations
finish_reason: str           — "stop" | "tool_calls" | "length"
usage: Usage                 — prompt_tokens, completion_tokens
```

**ToolCall structure**:
```json
{
  "id": "call_0_0",
  "type": "function",
  "function": {
    "name": "bash",
    "arguments": "{\"cmd\": \"ls\"}"
  }
}
```

Key requirements:
- `tool_calls[].function.arguments` must be a JSON **string**, not a dict
- `tool_calls[].id` must be a non-empty string
- `finish_reason` must be one of: "stop", "tool_calls", "length"
- `content` should be clean text (no thinking blocks, no trailing whitespace)

**Canonical message format** (what denormalize consumes):
```json
{"role": "system", "content": "..."}
{"role": "user", "content": "..."}
{"role": "assistant", "content": "...", "tool_calls": [...]}
{"role": "tool", "tool_call_id": "...", "content": "..."}
```

---

## Quirk categories to check

Analyze each sample for these known quirk patterns:

### Normalize (output cleanup)
1. **Thinking blocks**: `<think>...</think>` or similar wrappers in content
2. **Arguments as dict**: `arguments` field is a dict instead of JSON string (llama-server bug #20198)
3. **Missing tool_call_id**: `id` field is null, empty, or missing
4. **Wrong finish_reason**: model returns "tool" instead of "tool_calls", or "end" instead of "stop"
5. **Tool calls in content**: model puts tool invocations as JSON text in `content` instead of `tool_calls` field
6. **Trailing whitespace/newlines**: extra whitespace in content
7. **Empty content on tool calls**: content is empty string instead of null when making tool calls
8. **Duplicate tool_call_ids**: same id used for multiple tool calls in one response

### Denormalize (input formatting)
1. **System prompt handling**: model doesn't support system role (needs folding into user message)
2. **Tool schema format**: model needs simplified schemas
3. **Chat template quirks**: model needs special message framing

---

## Decision logic: rules vs code modules

Use **rules** (rules.toml) when the fix is:
- A regex pattern replace (strip thinking blocks, trailing whitespace)
- A value mapping (finish_reason "tool" → "tool_calls")
- A simple conditional (generate ID when missing)

Use **code modules** (.py files) when the fix requires:
- Iterating over nested structures (fixing each tool_call's arguments type)
- Conditional logic beyond simple guards
- JSON parsing/serialization
- Multiple fields interacting

**Rules DSL ceiling**: no cross-field mutation, no state across turns, no inter-rule dependencies, no computation beyond regex. If tempted to extend the DSL, write a .py module.

---

## Profile directory structure

```
profiles/<model_name>/
├── profile.toml               # metadata + inheritance + pipeline config
├── normalize/
│   ├── rules.toml              # declarative transforms (regex, mappings)
│   ├── <module>.py             # code modules (only if needed)
│   └── fixtures/
│       ├── <quirk_name>.json   # input/output pairs from observed behavior
│       └── ...
└── denormalize/
    ├── rules.toml              # system prompt strategy, etc.
    ├── <module>.py             # code modules (only if needed)
    └── fixtures/
        └── ...
```

---

## profile.toml schema

```toml
[profile]
format_version = 1
canonical_version = "openai-v1"
name = "<model_name>"
family = "<family>"              # e.g. "qwen3", "devstral", "glm"
quant = "<quant>"                # e.g. "Q4_K_M", "" if unknown
inherits = "_base"

[model]
context_size = 40960             # PLACEHOLDER — will be overridden by server metadata
chat_template = "chatml"         # PLACEHOLDER — will be overridden by server metadata
supports_tool_calls = true       # observe from samples: false if model never uses tool_calls field
supports_system_role = true      # observe from samples: false if system role causes errors

[tokens]
method = "chars_div_4"
tokenizer = ""

[capacity]
preamble = ""
max_tools = 6
simplify_schemas = false

[normalize]
rules = ["rules.toml"]
modules = []                     # add .py filenames if code modules needed

[denormalize]
rules = ["rules.toml"]
modules = []

[provenance]
generated = "<today>"
llama_server = ""
model_file = ""
model_sha256 = ""
process_version = "1"
```

---

## rules.toml format

### Normalize rules

```toml
# Strip patterns from content
[[strip]]
name = "<descriptive_name>"
pattern = '<regex_pattern>'      # DOTALL mode
target = "content"
replace = ""

# Map non-standard finish_reason values
[[map_finish_reason]]
from = "<model_value>"
to = "<canonical_value>"

# Fix missing tool_call_id
[[fix_tool_call]]
guard = { finish_reason = ["tool_calls", "tool"] }
when = "id_missing"
strategy = "generate"
```

### Denormalize rules

```toml
[system_prompt]
strategy = "native"              # "native" | "fold_into_user" | "prefix_user"
```

---

## Code module contract

```python
# <module_name>.py
import json
import re

def apply(response: dict) -> dict:
    """Transform raw API response dict.

    Args:
        response: dict with keys: content, tool_calls, finish_reason
    Returns:
        Modified response dict
    """
    # ... implementation ...
    return response
```

Constraints:
- Only `import json, re, typing, collections, dataclasses` (stdlib only)
- No filesystem access, network, subprocess, eval, exec
- Pure function: dict in → dict out, no side effects
- Must be importable without instantiation

---

## Fixture format

Each fixture file tests the full pipeline (rules + code modules together):

```json
{
  "description": "Human-readable description of what this tests",
  "cases": [
    {
      "input": {
        "content": "...",
        "tool_calls": [...],
        "finish_reason": "..."
      },
      "expected": {
        "content": "...",
        "tool_calls": [...],
        "finish_reason": "..."
      }
    }
  ]
}
```

Guidelines for writing fixtures:
- Use **real data from the behavioral samples** — not invented examples
- Each fixture file covers one quirk or behavior pattern
- Include both the "quirky" case and a "clean" case (to verify no-op on good input)
- `input` is what the model actually produces; `expected` is canonical format
- For normalize fixtures: `input`/`expected` are response dicts (content, tool_calls, finish_reason)
- For denormalize fixtures: `input`/`expected` are message lists

---

## Behavioral samples

The following are raw API responses captured from the target model across standard scenarios.

{{SAMPLES}}

---

## Instructions

1. **Analyze each sample** for quirks listed above. Note which patterns occur.
2. **Decide** for each quirk: rule or code module?
3. **Generate the complete profile directory**. Output as a structured list of files:

For each file, output:

```
=== FILE: <relative_path> ===
<file contents>
=== END FILE ===
```

**CRITICAL FORMAT RULES:**
- Output file contents as RAW text between the `=== FILE ===` markers.
- Do NOT wrap file contents in markdown code fences (no ` ```toml `, no ` ```python `, no ` ```json `).
- The `=== FILE ===` / `=== END FILE ===` markers ARE the delimiters. Code fences inside them will break parsing.
- TOML files: start directly with `[section]`, not with ` ```toml `.
- Python files: start directly with `def` or `import`, not with ` ```python `.
- JSON files: start directly with `{`, not with ` ```json `.

Files to generate:
- `profile.toml`
- `normalize/rules.toml`
- `normalize/<module>.py` (only if code modules needed)
- `normalize/fixtures/<quirk>.json` (one per observed quirk)
- `denormalize/rules.toml`
- `denormalize/fixtures/<scenario>.json` (if denormalize transforms needed)

4. **Use real data from the samples** in fixtures. Do not invent examples.
5. **Inherit from _base**. Only add rules/modules for behavior that differs from _base defaults.
   _base already handles: trailing whitespace stripping, empty content stripping, native system prompt pass-through.
6. **If no quirks are observed**, generate a minimal profile with just `profile.toml` (inherits _base, no extra rules/modules).
7. **Always include a passthrough fixture** in `normalize/fixtures/passthrough.json` — take a clean response from the samples and verify normalize produces identical output. This catches false transforms.
8. **Fill provenance fields** in `profile.toml`: set `generated` to today's date, `process_version = "1"`. Leave `llama_server`, `model_file`, `model_sha256` empty if unknown.

---

## Model info

- Model name: {{MODEL_NAME}}
- Model family: {{MODEL_FAMILY}}
- Quantization: {{MODEL_QUANT}}
