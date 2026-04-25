# Tool descriptions

Each subdirectory is a named description **mode**. Each `.txt` file inside is
the verbatim `description` string for one tool, keyed by filename (e.g.
`bash.txt` → the `bash` tool's description).

Tool parameter schemas live in `../tool_schemas.toml` and are shared across all
modes. A mode only controls the prose the model sees; it never changes tool
names or argument shapes.

## Current modes

- `minimal/` — harness-native one-liners with embedded guardrails (~185 tok).
  The default. Each description is a single line: what the tool does plus the
  anti-patterns to avoid (e.g. "don't use bash for file search — use glob").
  Guardrails are inline because they are load-bearing for weak models that
  otherwise reach for `find`/`grep`/`cat` via bash.
- `opencode/` — opencode `packages/opencode/src/tool/*.txt` lifted verbatim
  (~6k tok). Mechanical fixups applied: `${os}`/`${shell}`/`${maxLines}` etc.
  replaced with concrete values; `oldString`/`newString` → `old_str`/`new_str`;
  tool names lowercased. References to tools we don't ship (`todowrite`,
  `task`, `webfetch`, `multiedit`, `workdir`, `replaceAll`) are left intact.

## Selecting a mode

Set `[experiment] tool_desc = "<mode>"` in `config.toml` or pass
`--tool-desc <mode>` on the CLI. The loader is
`scripts/llm_solver/harness/schemas.py::get_tool_schemas`.

## Adding a mode

Create a new directory with one `.txt` file per tool name in
`tool_schemas.toml`. The loader verifies every tool has a corresponding file
and fails fast with a clear error if any are missing.
