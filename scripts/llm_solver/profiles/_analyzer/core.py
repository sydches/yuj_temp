"""ProfileAnalyzer — orchestrates detectors and emits a profile directory.

The class owns three responsibilities:

1. Run every detector in ``ALL_DETECTORS`` over the scenario samples and
   collect :class:`QuirkResult` entries.
2. Assemble the profile TOML (``profile.toml``), normalize rules, denormalize
   rules, and per-quirk fixtures from those results plus the server metadata
   and HuggingFace config.
3. Write the whole profile directory to disk.

Every sub-step is a small method so tests can exercise them in isolation.
"""
from __future__ import annotations

import json
from datetime import date
from pathlib import Path

from .._provenance import read_existing_provenance
from .detectors import (
    ALL_DETECTORS,
    _observe_system_support,
    _observe_tool_support,
)
from .helpers import (
    _classify_chat_template,
    _derive_family,
    _find_passthrough,
    _toml_kv,
)
from .types import QuirkResult


class ProfileAnalyzer:
    def __init__(
        self,
        samples: list[dict],
        model_name: str,
        server_meta: dict | None = None,
        hf_config: dict | None = None,
        hf_model_id: str = "",
        quant: str = "",
        existing_provenance: dict | None = None,
        existing_ctx_size: int | None = None,
    ):
        self.samples = samples
        self.model_name = model_name
        self.server_meta = server_meta or {}
        self.hf_config = hf_config or {}
        self.hf_model_id = hf_model_id
        self.quant = quant
        self.existing_ctx_size = existing_ctx_size
        self.family = _derive_family(model_name)
        self.existing_provenance = existing_provenance or {}

    # ── Analysis ──────────────────────────────────────────────────

    def analyze(self) -> list[QuirkResult]:
        results: list[QuirkResult] = []
        for detector in ALL_DETECTORS:
            r = detector(self.samples)
            if r:
                results.append(r)
        return results

    def _detect_reasoning_mode(self) -> tuple[str, str]:
        """Detect reasoning style from samples + server metadata.

        Returns ``(mode, disable_flag)`` where mode is one of:

        - ``"think"``   — ``<think>...</think>`` tags, disable with ``--reasoning-budget 0``
        - ``"channel"`` — ``<|channel>thought...<channel|>`` blocks, disable with ``--reasoning off``
        - ``"none"``    — no reasoning detected
        """
        for s in self.samples:
            content = s.get("response", {}).get("content") or ""
            if "<|channel>" in content or "<channel|>" in content:
                return "channel", "--reasoning off"
        for s in self.samples:
            content = s.get("response", {}).get("content") or ""
            if "<think>" in content and "</think>" in content:
                return "think", "--reasoning-budget 0"
        # Samples may have been collected with reasoning suppressed. Fall back
        # to scanning the chat template.
        chat_tmpl = self.server_meta.get("chat_template_raw", "")
        if "channel" in chat_tmpl and "thought" in chat_tmpl:
            return "channel", "--reasoning off"
        if "enable_thinking" in chat_tmpl or "<think>" in chat_tmpl:
            return "think", "--reasoning-budget 0"
        return "none", ""

    # ── TOML / rule generation ────────────────────────────────────

    def build_profile_toml(self, quirks: list[QuirkResult]) -> str:
        lines: list[str] = []

        # [profile]
        lines.append("[profile]")
        lines.append("format_version = 1")
        lines.append('canonical_version = "openai-v1"')
        lines.append(f'name = "{self.model_name}"')
        lines.append(f'family = "{self.family}"')
        lines.append(f'quant = "{self.quant}"')
        lines.append('inherits = "_base"')
        lines.append("")

        # [model] — from metadata
        lines.append("[model]")

        # context_size: HF config > server n_ctx_train > server n_ctx
        ctx = None
        text_cfg = self.hf_config.get("text_config", {})
        merged_cfg = {**self.hf_config, **text_cfg}
        for key in ("max_position_embeddings", "seq_length", "n_positions"):
            val = merged_cfg.get(key)
            if val and isinstance(val, int):
                ctx = val
                break
        if not ctx:
            ctx = self.server_meta.get("n_ctx_train") or self.server_meta.get("n_ctx") or 40960
        lines.append(f"context_size = {ctx}")

        # chat_template: from server
        raw_tmpl = self.server_meta.get("chat_template_raw", "")
        tmpl = _classify_chat_template(raw_tmpl)
        lines.append(f'chat_template = "{tmpl}"')

        # supports_tool_calls / supports_system_role: from samples
        lines.append(f"supports_tool_calls = {'true' if _observe_tool_support(self.samples) else 'false'}")
        lines.append(f"supports_system_role = {'true' if _observe_system_support(self.samples) else 'false'}")
        lines.append("")

        # [tokens]
        lines.append("[tokens]")
        lines.append('method = "chars_div_4"')
        lines.append('tokenizer = ""')
        lines.append("")

        # [capacity]
        lines.append("[capacity]")
        lines.append('preamble = ""')
        lines.append("max_tools = 6")
        lines.append("simplify_schemas = false")
        lines.append("")

        # [normalize]
        norm_modules = [q.module_filename for q in quirks if q.module_filename]
        lines.append("[normalize]")
        lines.append('rules = ["rules.toml"]')
        lines.append(f'modules = {json.dumps(norm_modules)}')
        lines.append("")

        # [denormalize]
        lines.append("[denormalize]")
        lines.append('rules = ["rules.toml"]')
        lines.append('modules = []')
        lines.append("")

        # [server] — launch config
        lines.append("[server]")
        model_file = self.server_meta.get("model_file", "")
        if not model_file:
            model_file = self.existing_provenance.get("model_file", "")
        if model_file:
            model_file = Path(model_file).name  # normalize to basename
            model_path = f"~/models/{model_file}"
        else:
            model_path = ""
        lines.append(f'model_path = "{model_path}"')
        # ctx_size = min(VRAM-viable, model-native max).
        # model-native max = ctx (from [model].context_size).
        # VRAM-viable = existing_ctx_size if re-profiling, else model-native.
        ctx_size = ctx
        if self.existing_ctx_size and self.existing_ctx_size > 0:
            ctx_size = min(self.existing_ctx_size, ctx)
        lines.append(f"ctx_size = {ctx_size}")
        lines.append("flash_attn = true")
        lines.append('cache_type_k = "q8_0"')
        lines.append('cache_type_v = "q8_0"')
        lines.append("jinja = true")
        reasoning_mode, reasoning_flag = self._detect_reasoning_mode()
        lines.append(f'reasoning_mode = "{reasoning_mode}"')
        lines.append(f'reasoning_disable_flag = "{reasoning_flag}"')
        lines.append('extra_flags = ""')
        lines.append("")

        # [provenance]
        lines.append("[provenance]")
        lines.append(f'generated = "{date.today().isoformat()}"')
        llama_server = self.existing_provenance.get("llama_server", reasoning_flag)
        lines.append(f'llama_server = "{llama_server}"')
        lines.append(f'model_file = "{model_file}"')
        model_sha256 = self.existing_provenance.get("model_sha256", "")
        lines.append(f'model_sha256 = "{model_sha256}"')
        lines.append('process_version = "2"')
        lines.append('self_profiled = false')

        # Quirk summary
        base_handled = [q.name for q in quirks if "Handled by _base" in q.description]
        extra = [q.name for q in quirks if q.name not in base_handled]
        summary = f"{len(self.samples)} scenarios analyzed, {len(quirks)} quirks detected"
        if base_handled:
            summary += f", {len(base_handled)} handled by _base ({', '.join(base_handled)})"
        if extra:
            summary += f", {len(extra)} model-specific ({', '.join(extra)})"
        lines.append(f'scenario_results = "{summary}"')
        lines.append("")

        # [metadata.server] — complete server metadata
        if self.server_meta:
            lines.append("[metadata.server]")
            for k, v in sorted(self.server_meta.items()):
                if k == "chat_template_raw":
                    continue  # too long for TOML
                lines.append(_toml_kv(k, v))
            lines.append("")

        # [metadata.hf_config] — complete HuggingFace config
        if self.hf_config:
            lines.append("[metadata]")
            lines.append(f'hf_model_id = "{self.hf_model_id}"')
            lines.append("")
            lines.append("[metadata.hf_config]")
            for k, v in sorted(self.hf_config.items()):
                lines.append(_toml_kv(k, v))
            lines.append("")

        return "\n".join(lines) + "\n"

    def build_normalize_rules(self, quirks: list[QuirkResult]) -> str:
        lines = ["# Normalize rules — auto-generated by ProfileAnalyzer"]
        lines.append(f"# {self.model_name}, {date.today().isoformat()}")
        lines.append(f"# Quirks detected: {[q.name for q in quirks]}")
        lines.append("")
        has_rules = False
        for q in quirks:
            if q.rule_toml and "denormalize" not in q.name and "system" not in q.name:
                lines.append(f"# {q.name}: {q.description}")
                lines.append(q.rule_toml)
                has_rules = True
        if not has_rules:
            lines.append("# No model-specific normalize rules needed beyond _base.")
        return "\n".join(lines) + "\n"

    def build_denormalize_rules(self, quirks: list[QuirkResult]) -> str:
        lines = ["# Denormalize rules — auto-generated by ProfileAnalyzer"]
        lines.append(f"# {self.model_name}, {date.today().isoformat()}")
        lines.append("")
        sys_quirk = next((q for q in quirks if q.name == "no_system_role"), None)
        if sys_quirk:
            lines.append(sys_quirk.rule_toml)
        else:
            lines.append('[system_prompt]')
            lines.append('strategy = "native"')
        return "\n".join(lines) + "\n"

    def build_fixtures(self, quirks: list[QuirkResult]) -> dict[str, str]:
        """Return ``{relative_path: json_content}`` for all fixtures to emit."""
        fixtures: dict[str, str] = {}

        # Passthrough fixture — the cleanest non-quirky sample.
        quirk_scenarios: set[str] = set()
        for q in quirks:
            quirk_scenarios.update(q.affected_scenarios)
        clean = _find_passthrough(self.samples, quirk_scenarios)
        if clean:
            resp = clean["response"]
            passthrough = {
                "description": "Clean response passes through normalize unchanged",
                "cases": [{
                    "input": {"content": resp.get("content"), "tool_calls": resp.get("tool_calls", []),
                              "finish_reason": resp.get("finish_reason", "stop")},
                    "expected": {"content": resp.get("content"), "tool_calls": resp.get("tool_calls", []),
                                 "finish_reason": resp.get("finish_reason", "stop")},
                }],
            }
            fixtures["normalize/fixtures/passthrough.json"] = json.dumps(passthrough, indent=2) + "\n"

        # Per-quirk fixtures
        for q in quirks:
            if q.fixture and q.fixture.get("cases"):
                path = f"normalize/fixtures/{q.name}.json"
                fixtures[path] = json.dumps(q.fixture, indent=2) + "\n"

        return fixtures

    # ── Disk write ────────────────────────────────────────────────

    def write_profile(self, output_dir: Path) -> list[QuirkResult]:
        """Analyze and write the complete profile directory. Returns quirks found."""
        quirks = self.analyze()

        # Merge existing provenance from disk if not provided via constructor.
        if not self.existing_provenance:
            self.existing_provenance = read_existing_provenance(output_dir)

        output_dir.mkdir(parents=True, exist_ok=True)

        # profile.toml
        (output_dir / "profile.toml").write_text(self.build_profile_toml(quirks))

        # normalize/
        norm_dir = output_dir / "normalize"
        norm_dir.mkdir(exist_ok=True)
        (norm_dir / "rules.toml").write_text(self.build_normalize_rules(quirks))

        # Code modules
        for q in quirks:
            if q.code_module and q.module_filename:
                (norm_dir / q.module_filename).write_text(q.code_module)

        # Fixtures
        fixtures = self.build_fixtures(quirks)
        for rel_path, content in fixtures.items():
            full = output_dir / rel_path
            full.parent.mkdir(parents=True, exist_ok=True)
            full.write_text(content)

        # denormalize/
        denorm_dir = output_dir / "denormalize"
        denorm_dir.mkdir(exist_ok=True)
        (denorm_dir / "rules.toml").write_text(self.build_denormalize_rules(quirks))

        return quirks


__all__ = ["ProfileAnalyzer"]
