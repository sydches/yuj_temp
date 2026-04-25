# Mechanical State Writer

## Purpose

The harness writes `.solver/state.json` mechanically from `.trace.jsonl`.
The model never maintains state files itself.

This keeps one source of truth:

- `.trace.jsonl` is authoritative
- `.solver/state.json` is a projection
- same trace means same state, always

## Activation

The writer is active only when `.solver/state.json` already exists at task or
session start.

That rule matters because it lets the same engine serve two cases cleanly:

- stateless runs: no `.solver/state.json`, so the writer stays off
- stateful runs: seed the file first, and the writer keeps it updated

Assistant-mode sessions seed the file inside the session artifact directory,
not inside the repo being edited.

## Projection Contract

The projection is content-blind by design. It reads only harness-written
markers:

- `ERROR:` wrappers from tool failures
- `[exit code: N]` suffixes from `bash`
- `[harness gate]` prefixes for blocked calls

It does **not** parse task-specific output formats inside the live loop.

## Sections

`state.json` contains:

- `state.current_attempt`
- `state.last_verify`
- `state.next_action`
- `trace`
- `gates`
- `evidence`
- `inference`

Today:

- `trace` is populated from every `tool_call`
- `evidence` is populated from executed `bash` calls
- `gates` stays empty
- `inference` stays empty
- `next_action` stays empty

That emptiness is deliberate. The writer only fills fields with a mechanical,
content-blind population rule.

## Runtime Behavior

Writes happen:

- after every `tool_call`
- after every `session_start`
- after every `session_end`

Each write is atomic:

1. render the new JSON to a temp file
2. rename it into place

If the process dies mid-write, the previous file remains intact.

## Why This Exists

Two failure modes disappear when the harness owns state:

1. models forgetting to keep the file updated
2. drift between model-written and harness-observed history

The model should reason over state, not author the bookkeeping substrate.

## Related Files

- `scripts/llm_solver/harness/state_writer.py`
- `scripts/llm_solver/harness/loop.py`
- `tests/test_state_writer.py`
