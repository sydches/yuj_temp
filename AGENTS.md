# AGENTS.md

For extraction or public-repo work, read these first:

1. `docs/harness_spec.md`
2. `docs/assistant_spec.md`
3. `docs/coupling_spec.md`
4. `docs/public_repo_spec.md`

Rules:

- Copy by whitelist only.
- Preserve internal paths in phase 1.
- Keep `bwrap`; leave FeatureBench/eval/DOE behind.
- Assistant-mode artifacts live under `.llm_assist/`, not inside the repo being edited.
- Do not add a TypeScript wrapper, web-first shell, memory, subagents, or dashboard in v1.
- If a required file is not named in `docs/public_repo_spec.md`, stop and update the spec before copying anything else.
