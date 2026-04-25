# Knob catalog

**Single source of truth:** [`configs/knobs.toml`](../configs/knobs.toml).

Every tunable in the harness — 160+ knobs across guardrails, output
hygiene, context strategies, tool dispatch, post-edit checks, injections,
server, model, prompts — is cataloged in one machine-readable file with
tags, descriptions, blast-radius, and allowed mutation scope.

## What's in the catalog

Each knob entry:

```toml
[knob."loop.loop_detect_enabled"]
type = "bool"
default = false
description = "Structural hash-based loop detector beyond duplicate_guard. Off by default (experimental)."
tags = ["guardrail", "toggle", "experimental"]
blast_radius = "low"
mode = "both"
```

**Tags** group knobs by concern (`safety`, `guardrail`, `context`,
`output-hygiene`, `bash-quirks`, `edit`, `model`, `server`, `post-edit`,
`injections`, `prompt-text`, `timing`, `toggle`, `experimental`,
`observability`, `tools`, `budget`, `reproducibility`).

**`blast_radius`** is the change-size signal:
- `low` — safe to flip, reversible, tiny scope
- `medium` — review the diff
- `high` — changes the independent variable / breaks reproducibility /
  has broad side-effects. Skills must require typed confirmation.

**`mode`** — `measurement`, `assistant`, or `both`. Measurement-mode
runs refuse knobs not marked `both` or `measurement`.

## Presets

Curated bundles in `[preset.*]` — use these before hand-assembling
individual knobs:

- `quiet` — minimum verbosity, aggressive truncation
- `paranoid` — every guardrail on (= `configs/full.toml`)
- `permissive` — guardrails off, correctness layer on (= `configs/substantive.toml`)
- `fast` — short sessions, low turn budget
- `debug` — loose budgets, verbose observability (assistant-only)
- `reproducible` — lock independent variables for measurement runs
- `post-edit-python` — enable py_compile rollback on edits

## CLI

```bash
python3 scripts/knob.py list                               # list all knobs
python3 scripts/knob.py list --tag guardrail               # filter by tag
python3 scripts/knob.py list --blast-radius high           # by blast
python3 scripts/knob.py list --mode measurement            # by mode
python3 scripts/knob.py list --pending                     # curation_pending knobs only
python3 scripts/knob.py describe loop.rumination_enabled   # full detail
python3 scripts/knob.py search truncate                    # substring search
python3 scripts/knob.py tags                               # tags with counts
python3 scripts/knob.py presets                            # list presets
python3 scripts/knob.py preset quiet                       # show preset contents
python3 scripts/knob.py diff-vs-default config.local.toml  # show overlay deltas
```

## Generator and validator

```bash
python3 scripts/build_knob_catalog.py build      # merge skeleton for new config.toml keys
python3 scripts/build_knob_catalog.py validate   # CI check — fails on undocumented keys
```

The `build` command walks `config.toml` and the overlay surface, adds
skeleton entries for new keys (marked `curation_pending = true`), and
preserves hand-authored `[preset.*]` and `[meta]` sections verbatim.

The `validate` command is the CI contract: every key in `config.toml`
must have a `[knob."..."]` entry in `configs/knobs.toml`. Any key
added to config.toml without a catalog entry fails CI.

## Scope boundaries (what the catalog represents)

The catalog covers the **loop-tunable surface** — keys read by the
harness loop, guardrails, state writer, context strategies, and tool
dispatch.

It does **not** catalog (these are research-grade artifacts with their
own ownership):

| Surface | Owner | Why not in catalog |
|---|---|---|
| `profiles/<model>/*` | Model quirks absorber | Per-model normalize/denormalize + behavioral — changing these corrupts arm identity |
| `bash_quirks/rewrites.toml` | Bash quirks absorber | Universal shell rewrites — correctness layer |
| `language_quirks/*.toml` | Language quirks absorber | Per-runner output semantics — correctness layer |
| `docs/commandments*.md` | Protocol | READ-ONLY independent variable |
| `config/prompt_*.toml` | Server-side prompt normalize | Wire-format concern |

These live in their own directories with their own READMEs. The skill
never edits them.

## The intent-translator skill

[`.claude/skills/harness-tune/SKILL.md`](../.claude/skills/harness-tune/SKILL.md)
reads the catalog and translates plain-English intent into concrete
diffs on `config.local.toml`. Never silently applies; always shows the
diff; requires typed confirmation for `blast_radius=high` changes.

Invoke with `/harness-tune` (or similar phrasing — "make it less
chatty", "turn on loop detection", etc.).

## Adding a new knob (workflow)

1. Add the key to `config.toml` with a `# description above the key`.
2. Run `python3 scripts/build_knob_catalog.py build` to add the
   skeleton entry.
3. Open `configs/knobs.toml`, find the new entry, and fill in
   `description`, `tags`, `blast_radius`, `mode`, optionally `range`
   / `values` / `implies` / `affects`. Remove `curation_pending = true`.
4. If the knob participates in a preset, update `[preset.*]` blocks.
5. Run `python3 scripts/build_knob_catalog.py validate` to confirm.
6. Commit both `config.toml` and `configs/knobs.toml` in the same
   change.

## Relationship to other config files

```
config.toml                  ← checked-in defaults (authoritative for knob set)
  ↓
configs/knobs.toml           ← source of truth for metadata (this catalog)
  ↓
config.local.toml            ← user-local overrides (skill-writable)
  ↓
configs/<preset>.toml        ← named overlay bundles
configs/toggles/<knob>.*.toml ← atomic on/off overlays
  ↓
CLI flags                    ← highest priority
```
