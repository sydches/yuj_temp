## Model Profiles

A model profile is the complete adapter artifact for a model family/quant combination. It contains both halves of the normalize/denormalize sandwich — how to clean the model's output into canonical form, and how to format canonical messages for the model's preferred input.

The harness loads one profile per session. The harness never sees model-specific formats.

---

## Separation of concerns

See `docs/separation_of_concerns.md` for the authoritative treatment of all six layers, diagnostic discipline, and anti-patterns. Profiles are one of those layers — they own everything that differs between models (format adaptation + behavioral suffix) and nothing else.

---

## Automated scenario review — shelved

The step where scenario output would be sent to an LLM (Haiku) for automated review is **shelved**. Our understanding of what constitutes correct vs incorrect scenario behavior is not mature enough to encode as an automated grading prompt.

**Current approach:** Manual review of scenario outputs (`profiles/<model>/_samples/`) to build judgment criteria. The automation follows when the manual process produces stable, repeatable heuristics — not before.

This is deliberate: automating review prematurely encodes wrong assumptions into the pipeline. The profiling process (below) describes what exists today; automated review is a future addition.

---

## Canonical format

OpenAI chat completion format — the de facto standard for local inference servers (llama-server, vLLM, Ollama, LM Studio, SGLang). This is what the harness works with internally. All profiles normalize to and denormalize from this format.

**Canonical output type** (`TurnResult`):

```python
@dataclass(frozen=True)
class TurnResult:
    content: str | None              # text response, None if tool-only
    tool_calls: list[ToolCall]       # structured tool invocations
    finish_reason: str               # "stop" | "tool_calls" | "length"
    usage: Usage                     # prompt_tokens, completion_tokens

@dataclass(frozen=True)
class Usage:
    prompt_tokens: int
    completion_tokens: int

class ToolCall(NamedTuple):          # already exists
    id: str
    name: str
    arguments: dict
```

**Canonical message format** (what the harness builds, what the context manager stores):

```python
# Standard OpenAI chat message dicts
{"role": "system", "content": "..."}
{"role": "user", "content": "..."}
{"role": "assistant", "content": "...", "tool_calls": [...]}
{"role": "tool", "tool_call_id": "...", "content": "..."}
```

The normalize pipeline produces `TurnResult` from raw API responses.
The denormalize pipeline consumes canonical message dicts and produces wire-format dicts.
Both may be identity transforms for well-behaved models.

---

## Failure policy

When the normalize pipeline encounters output that doesn't match any rule or code module:

**Best-effort passthrough + structured warning.** Apply what matches, pass the rest through unchanged, emit a structured log entry:

```
WARN UNHANDLED_PATTERN profile=qwen3-8b-q4 half=normalize field=content snippet="[first 100 chars]"
```

Not an error (too fragile for stochastic model output). Not silent (defeats the purpose of normalization). Over time, unhandled warnings in production logs feed back into the profiling process — they become new fixtures and rules. The profile improves from observed failures.

---

## Observability

Four tap points in the pipeline, logged at DEBUG level (visible with `--verbose`):

| Tap | What | When |
|-----|------|------|
| `DENORM_IN` | Canonical messages as harness sent them | Before denormalize |
| `DENORM_OUT` | Wire-format messages after denormalization | After denormalize, before HTTP |
| `NORM_IN` | Raw model response | After HTTP, before normalize |
| `NORM_OUT` | Canonical TurnResult as harness receives | After normalize |

Per-rule/module logging: which rules fired, which modules ran, what changed. All DEBUG. Zero overhead when not debugging.

---

## Rules DSL ceiling

The rules engine deliberately cannot:

- **Read-only cross-field guards only** — a rule may declare a `guard` that checks other fields as a gate condition, but the transform itself applies to a single `target` field. Guards are read-only; they never modify the guarded field. This handles the most common pattern (e.g., only extract tool calls when `finish_reason` signals tool use) without opening the DSL to arbitrary cross-field mutation.
- **No state across turns** — rules see one response/message, not the conversation
- **No inter-rule dependencies** — rules are independent; order is declaration order only
- **No computation** — no arithmetic, no string formatting beyond regex replace

Anything beyond these limits → code module. This boundary is permanent. If you're tempted to extend the DSL, write a `.py` file instead.

---

## Directory structure

```
profiles/
├── _base/                         # shared defaults — all profiles inherit this
│   ├── profile.toml
│   ├── normalize/
│   │   └── rules.toml             # common output cleanup (empty content, whitespace)
│   └── denormalize/
│       └── rules.toml             # common pass-through (standard OpenAI chat format)
├── qwen3-8b-q4/
│   ├── profile.toml               # metadata + inheritance declaration
│   ├── normalize/
│   │   ├── rules.toml             # declarative: regex patterns, strip rules, mappings
│   │   ├── fix_tools.py           # code: complex repairs not expressible as regex
│   │   └── fixtures/
│   │       ├── thinking_block.json
│   │       ├── malformed_args.json
│   │       └── missing_tool_id.json
│   └── denormalize/
│       ├── rules.toml             # declarative: system prompt strategy, schema rules
│       ├── encode.py              # code: chat template, quant-specific framing
│       └── fixtures/
│           ├── system_prompt.json
│           └── tool_schemas.json
├── qwen3.5-9b/
│   └── ...
└── glm-4-flash/
    └── ...
```

---

## profile.toml

```toml
[profile]
format_version = 1                 # profile schema version; loader refuses unknown versions
canonical_version = "openai-v1"    # canonical message format this profile targets
name = "qwen3-8b-q4"
family = "qwen3"
quant = "Q4_K_M"
inherits = "_base"                 # load _base first, overlay this profile

[model]
context_size = 40960
chat_template = "chatml"           # or "llama", "raw", etc.
supports_tool_calls = true
supports_system_role = true

[tokens]
# Token estimation for context management.
# The context manager calls this profile's estimator to decide when to prune.
method = "chars_div_4"             # "chars_div_4" (default heuristic) | "tokenizer"
tokenizer = ""                     # path to tokenizer model, required if method = "tokenizer"
# chars_div_4 is crude but universal. Profiles with access to the model's
# tokenizer (sentencepiece, tiktoken) can be exact. The interface is the same:
# estimate_tokens(messages) -> int
#
# Estimation error consequences (context manager must handle both):
#   Under-estimate → context overflow: model receives more tokens than window allows,
#       server returns truncated/error response. Silent unless harness checks.
#   Over-estimate → premature pruning: harness discards useful context too early,
#       model loses coherence. Invisible to user.
# chars_div_4 diverges ~2-3x for CJK text, code-heavy contexts (short tokens),
# and emoji. For non-English or code-dominant use cases, configure a tokenizer.

[capacity]
# Quant-specific capacity knobs. These address capability limits (weaker quants
# need simpler inputs), not format differences. A Q4 and Q8 of the same model
# produce the same JSON structure — Q4 just fills it less reliably.
preamble = ""                      # prepend explicit instruction framing for weaker quants
max_tools = 6                      # drop tools beyond this count (least-used first)
simplify_schemas = false           # true = strip descriptions, reduce to name + params only

[normalize]
# Which rules files to load, in order. Inherited rules run first.
rules = ["rules.toml"]
# Which code modules to load, in order. Each must expose apply(raw_output) -> raw_output.
modules = []                       # added per model when code modules needed

[denormalize]
rules = ["rules.toml"]
modules = []                       # added per model when code modules needed

[provenance]
generated = "2026-04-03"           # when this profile was created/last regenerated
llama_server = "b5291"             # llama-server build hash or version
model_file = "qwen3-8b-q4_k_m.gguf"
model_sha256 = "a1b2c3d4e5f6"     # first 12 chars of GGUF file hash
process_version = "1"              # profiling process version that generated this
# Advisory, not enforced. Check when behavior drifts.
```

---

## Rules format (rules.toml)

Declarative transforms applied in order. Handles the 80% case.

```toml
# normalize/rules.toml

[[strip]]
name = "thinking_blocks"
pattern = '<think>.*?</think>'     # regex, DOTALL
target = "content"                 # which field to apply to
replace = ""

[[strip]]
name = "trailing_whitespace"
pattern = '\s+$'
target = "content"
replace = ""

[[map_finish_reason]]
from = "tool"
to = "tool_calls"

[[extract_tool_calls]]
# Only attempt tool call extraction when finish_reason signals tool use.
# Guard is read-only: checks the field, never modifies it.
guard = { finish_reason = ["tool_calls", "tool"] }
target = "tool_calls"
source = "content"                 # extract from content field if tool_calls empty

[[fix_tool_call]]
# When tool_call_id is missing, generate from turn index
guard = { finish_reason = ["tool_calls", "tool"] }
when = "id_missing"
strategy = "generate"              # "generate" | "error"
```

```toml
# denormalize/rules.toml

[system_prompt]
strategy = "native"                # "native" | "fold_into_user" | "prefix_user"

# Capacity knobs (preamble, max_tools, simplify_schemas) live in
# profile.toml [capacity], not here. They address model capability
# limits, not wire-format transforms.
```

---

## Code modules

When a transform can't be expressed declaratively. Each module exposes one function:

```python
# normalize/fix_tools.py

def apply(response: dict) -> dict:
    """Fix tool calls embedded in content instead of tool_calls field.

    Some models (especially at lower quant) return tool invocations as
    JSON in the content string instead of the structured tool_calls field.
    This extracts and restructures them.
    """
    # ... implementation ...
    return response
```

Contract:
- Input/output: raw API response dict (normalize) or raw message list (denormalize)
- Must be pure — no side effects, no network, no filesystem access
- Must be importable and callable without instantiation
- Must have fixtures in the sibling `fixtures/` directory

---

## Fixtures

JSON files with input/output pairs. The fixture runner is the quality gate.

```json
// normalize/fixtures/thinking_block.json
{
  "description": "Strip <think> blocks from content",
  "cases": [
    {
      "input": {
        "content": "<think>Let me analyze this...</think>The file has a bug on line 12.",
        "tool_calls": [],
        "finish_reason": "stop"
      },
      "expected": {
        "content": "The file has a bug on line 12.",
        "tool_calls": [],
        "finish_reason": "stop"
      }
    }
  ]
}
```

Fixtures test the full pipeline (rules + code), not individual transforms. A profile is valid iff all its fixtures pass.

---

## Inheritance

`_base` provides defaults. A profile declares `inherits = "_base"` (or another profile).

Loading order:
1. Load inherited profile's rules and modules
2. Load this profile's rules and modules
3. Rules: inherited rules run first, then this profile's rules
4. Modules: inherited modules run first, then this profile's modules
5. profile.toml values: this profile overrides inherited values

A well-behaved model (standard OpenAI format, no quirks) needs only `profile.toml` — everything else inherited from `_base`.

---

## Profile fallback resolution

The loader resolves a model name to a profile directory via fallback chain:

```
exact name → family → _base

"qwen3-8b-q5" → profiles/qwen3-8b-q5/     (exact match)
             → profiles/qwen3/             (family match, from profile.toml family field)
             → profiles/_base/             (universal fallback)
```

First match wins. Fallback is logged: `INFO: No profile for 'qwen3-8b-q5', falling back to family 'qwen3'`. This matters for open-source adoption — users shouldn't need an exact-match profile to get started. A family-level profile covers most quants; `_base` covers well-behaved models with no quirks.

---

## Runtime loading

```
config.toml: model.name = "qwen3-8b-q4"
    ↓
Server layer: resolve profile directory: profiles/qwen3-8b-q4/
    ↓
Load profile.toml → resolve inheritance chain
    ↓
Build normalize pipeline: [_base rules] → [profile rules] → [_base modules] → [profile modules]
Build denormalize pipeline: same pattern
    ↓
Server.chat():
    messages → denormalize pipeline → wire format → HTTP → raw response → normalize pipeline → canonical types
```

The harness calls `server.chat(canonical_messages)` and receives `TurnResult` (canonical). It never knows a profile exists.

---

## Profiling process

The process that generates profiles. This is the core contribution — not the profiles themselves.

**Input:**
- Behavioral samples from the model (raw API responses across tool-calling scenarios)
- Canonical format spec (what the harness expects)
- Profile directory template (the structure above)

**Method** (`scripts/llm_solver/profiles/generate.py`):

1. Start llama-server with the model
2. `generate.py` queries server metadata (`/props`, `/slots`, `/v1/models`)
3. `generate.py` fetches HuggingFace `config.json` for the model
4. Run scenario suite against the model → collect samples
5. Factual fields (context_size, chat_template, architecture) set from metadata
6. Quirk detection from samples → normalize/denormalize rules and fixtures
7. Complete profile directory written with `_metadata/` raw files

```bash
python3 -m scripts.llm_solver.profiles.generate http://localhost:8080/v1 <model-name> \
  --family <family> --quant <quant>
```

**What's factual** (from server + HuggingFace, never guessed):
- `context_size` — from HF `max_position_embeddings` or server `n_ctx_train`
- `chat_template` — classified from server's raw Jinja template string
- `model_file` — from server `/props`
- `[metadata.hf_config]` — complete HuggingFace config.json, every field
- `[metadata.server]` — complete server default_generation_settings
- `_metadata/chat_template.jinja` — raw Jinja template (3-12KB)
- `_metadata/*.json` — all raw server and HuggingFace responses

**What's observed** (from scenario samples, deterministic):
- `supports_tool_calls` — did any sample use the tool_calls field?
- `supports_system_role` — did system-role scenarios work?
- Quirk rules — pattern matching on known categories (see `docs/profile_analyzer_spec.md`)

**Runtime context resolution**: The profile stores the model-native max context. At runtime, the harness queries the server's `n_ctx` (hardware-constrained) and uses `min(profile, server)`. No manual VRAM calculation needed.

See also: `docs/profile_analyzer_spec.md` for the deterministic quirk detection design.

---

## Denormalization feedback loop

The closed loop for improving behavioral denormalization rules:

| Step | Tool | Input | Output |
|------|------|-------|--------|
| 1. Run | `run_multi.sh` / `run_experiment.sh` | task files | run dirs with `.trace.jsonl` |
| 2. Discover | `denorm_discover.py` | run dir(s) | ranked anti-pattern findings + copy-pasteable rule recommendations |
| 3. Write rules | human | findings | new/edited `behavioral.py` in model's profile |
| 4. Audit | `denorm_audit.py` | run dir + profile | compliance report: rules followed vs violated |
| 5. Iterate | → step 1 | — | — |

This mirrors the normalization side: generate profile → run 47 scenarios → verify → refine.

Both loops are driven by evidence from observed traces, not assumption.

---

## Analysis tooling

`scripts/llm_solver/analysis/` — four read-only tools that consume `.trace.jsonl` files.

| Tool | Purpose | CLI |
|------|---------|-----|
| `coherence.py` (510 lines) | Detect thrashing: repeated calls, write→overwrite, never-read-tests, session regression | `python3 -m scripts.llm_solver.analysis.coherence <run_dir>` |
| `compare.py` (330 lines) | Diff results across runs — model×mode×task tables | `python3 -m scripts.llm_solver.analysis.compare <run_dir1> <run_dir2> [...]` |
| `denorm_audit.py` (513 lines) | Verify existing `behavioral.py` rules are being followed | `python3 -m scripts.llm_solver.analysis.denorm_audit --profile <model_dir> <run_dir>` |
| `denorm_discover.py` (734 lines) | Scan traces for anti-patterns; recommend new behavioral rules | `python3 -m scripts.llm_solver.analysis.denorm_discover <run_dir> [...]` |

`denorm_discover.py` has 10 pattern detectors across 4 categories: verification, efficiency, navigation, cross-session. It requires no existing rules — it discovers what rules to write.

`denorm_audit.py` reads `behavioral.py`'s `_BEHAVIORAL_SUFFIX` block to extract declared intent, then checks traces for compliance. Produces JSON + human-readable summary.

Typical workflow after a run:

```bash
# Find what behavioral rules to add
python3 -m scripts.llm_solver.analysis.denorm_discover results/with_yuj/<run_dir>

# Check coherence issues (thrashing, loops)
python3 -m scripts.llm_solver.analysis.coherence results/with_yuj/<run_dir>

# After adding rules to profiles/<model>/denormalize/behavioral.py, verify compliance
python3 -m scripts.llm_solver.analysis.denorm_audit --profile profiles/<model> results/with_yuj/<run_dir>

# Compare two runs (before/after rule change)
python3 -m scripts.llm_solver.analysis.compare results/before/<run_dir> results/after/<run_dir>
```

---

## Security boundary

Profiles contain executable Python code. Trust model:

- **First-party profiles** (shipped with repo): trusted, reviewed like any code
- **Community profiles** (contributed via PR): must pass fixture runner + security validation before merge
- **User-local profiles** (config.local.toml override): user's responsibility, like any local code

Security validation for contributed profiles:
- No imports beyond stdlib (re, json, typing)
- No filesystem access, network access, subprocess, eval, exec
- No side effects — pure input→output transforms
- Static analysis gate in CI (AST walk checking for banned nodes)
- Fixture coverage: every code path must have a fixture exercising it

The `apply()` contract is deliberately restrictive: dict in, dict out, no context. This makes sandboxing trivial and auditing feasible.

---

## What a minimal profile looks like

A model that behaves well (standard OpenAI format, no quirks):

```
profiles/well-behaved-model/
└── profile.toml          # just metadata + inherits = "_base"
```

3 lines of TOML. Everything inherited. Fixtures from `_base` pass. Done.

---

## What a complex profile looks like

A model with thinking blocks, broken tool IDs, and needs system prompt folded:

```
profiles/quirky-model-q4/
├── profile.toml
├── normalize/
│   ├── rules.toml              # strip thinking, fix finish_reason mapping
│   ├── fix_tools.py            # restructure tool calls from content field
│   └── fixtures/
│       ├── thinking.json
│       ├── tool_restructure.json
│       └── finish_reason.json
└── denormalize/
    ├── rules.toml              # fold system prompt into user message
    └── fixtures/
        ├── system_fold.json
        └── schema_simplify.json
```

No code in denormalize — rules suffice. Code in normalize for one complex case. 8 fixture files covering all adaptations.
