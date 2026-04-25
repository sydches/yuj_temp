"""Profile generator — metadata + scenarios → complete profile directory.

Collects factual model data from two authoritative sources (server metadata
and HuggingFace config.json), runs scenario samples for quirk detection,
and uses ProfileAnalyzer for deterministic profile generation (no LLM).

Usage:
    python -m scripts.llm_solver.profiles.generate <model_endpoint> <model_name> [options]

Options:
    --output <dir>              Where to write the profile (default: profiles/<model_name>/)
    --samples-dir <dir>         Use pre-collected samples instead of running scenarios
    --family <name>             Model family (default: derived from model name)
    --quant <name>              Quantization (default: "")

Flow:
  0. Query server metadata (/props, /slots, /v1/models) → factual fields
  0b. Fetch HuggingFace config.json → model architecture, context size
  1. Run scenario suite against target model → collect samples
  2. ProfileAnalyzer: detect quirks, assemble profile, write files
  3. Save raw metadata to _metadata/ directory
  4. Run fixture + security validation
  5. Report

Metadata saved to _metadata/:
  server_props.json       — complete /props response
  server_slots.json       — complete /slots response
  server_models.json      — complete /v1/models response
  hf_config.json          — complete HuggingFace config.json
  hf_text_config.json     — nested text_config (multimodal models)
  chat_template.jinja     — raw Jinja chat template (3-12KB)

See also:
  docs/model_profiles.md          — profile system design
  docs/profile_analyzer_spec.md   — deterministic quirk detection design
"""
import argparse
import json
import logging
import re
import sys
from pathlib import Path

import requests

from .analyzer import ProfileAnalyzer
from .refiner import refine_rules
from .verify import verify_profile, format_report

log = logging.getLogger(__name__)


def _query_server_metadata(endpoint: str) -> dict:
    """Query llama-server for model metadata.

    Tries /props first (richest), falls back to /slots, then /v1/models.
    Returns dict with available fields: n_ctx, n_ctx_train, model_file,
    chat_template, etc.  Missing fields are absent, not None.
    """
    base = endpoint.rstrip("/v1").rstrip("/")
    metadata = {}

    # /props — server properties (richest endpoint)
    try:
        resp = requests.get(f"{base}/props", timeout=5)
        if resp.ok:
            data = resp.json()
            metadata["_raw_props"] = data  # store complete response
            if "default_generation_settings" in data:
                dgs = data["default_generation_settings"]
                if "n_ctx" in dgs:
                    metadata["n_ctx"] = dgs["n_ctx"]
                if "model" in dgs:
                    metadata["model_file"] = dgs["model"]
            if "chat_template" in data:
                metadata["chat_template_raw"] = data["chat_template"]
    except Exception as e:
        log.debug("/props failed: %s", e)

    # /slots — slot info
    try:
        resp = requests.get(f"{base}/slots", timeout=5)
        if resp.ok:
            data = resp.json()
            metadata["_raw_slots"] = data
            if data and isinstance(data, list):
                slot = data[0]
                if "n_ctx" in slot:
                    metadata.setdefault("n_ctx", slot["n_ctx"])
    except Exception as e:
        log.debug("/slots failed: %s", e)

    # /v1/models
    try:
        resp = requests.get(f"{base}/v1/models", timeout=5)
        if resp.ok:
            data = resp.json()
            metadata["_raw_models"] = data
            if data.get("data"):
                model_info = data["data"][0]
                metadata["model_id"] = model_info.get("id", "")
                if "meta" in model_info:
                    meta = model_info["meta"]
                    if "n_ctx_train" in meta:
                        metadata["n_ctx_train"] = meta["n_ctx_train"]
    except Exception as e:
        log.debug("/v1/models failed: %s", e)

    return metadata



def _search_model_card(model_name: str, family: str) -> dict:
    """Search HuggingFace for model config.json as metadata fallback.

    Fetches config.json directly from candidate HuggingFace repos.
    Returns dict with fields found (may be empty).
    """
    metadata = {}

    # Build candidate HF repo IDs from model name
    clean = re.sub(r"[-_](q\d|iq\d|fp\d|gguf|awq|Q\d).*$", "", model_name, flags=re.IGNORECASE)
    clean = clean.replace("-", " ").replace("_", " ").strip()

    # Search HF API for matching models
    try:
        resp = requests.get(
            f"https://huggingface.co/api/models?search={clean}&limit=10",
            timeout=10,
        )
        if not resp.ok:
            return metadata
        all_candidates = [m["modelId"] for m in resp.json() if m.get("modelId")]
        # Prefer non-VL/non-multimodal variants, and original org repos
        def _rank(mid: str) -> int:
            mid_lower = mid.lower()
            score = 0
            if "-vl" in mid_lower or "vision" in mid_lower:
                score += 10
            if "gguf" in mid_lower or "awq" in mid_lower or "gptq" in mid_lower:
                score += 5
            # Prefer shorter model IDs (less likely to be derivative)
            score += len(mid) // 20
            return score
        candidates = sorted(all_candidates, key=_rank)
    except Exception as e:
        log.debug("HuggingFace search failed: %s", e)
        return metadata

    # Fetch config.json from each candidate
    for model_id in candidates:
        try:
            resp = requests.get(
                f"https://huggingface.co/{model_id}/resolve/main/config.json",
                timeout=10,
            )
            if not resp.ok:
                continue
            config = resp.json()

            # Some models nest text config (multimodal models like gemma-4)
            text_config = config.get("text_config", {})
            # Merge: text_config fields are more specific, top-level is fallback
            merged = {**config, **text_config}

            metadata["hf_model_id"] = model_id
            metadata["hf_config"] = merged  # store the entire config

            # Extract max_context for the injection logic
            for key in ("max_position_embeddings", "seq_length", "n_positions", "max_length"):
                val = merged.get(key)
                if val and isinstance(val, int) and val > 0:
                    metadata["max_context_web"] = val
                    metadata["max_context_source"] = f"huggingface:{model_id}:{key}"
                    break

            if metadata.get("max_context_web"):
                break  # found primary target

        except Exception as e:
            log.debug("config.json fetch failed for %s: %s", model_id, e)

    return metadata



def _save_metadata_files(output_dir: Path, server_meta: dict, web_meta: dict) -> None:
    """Save raw metadata files to _metadata/ directory."""
    meta_dir = output_dir / "_metadata"
    meta_dir.mkdir(parents=True, exist_ok=True)

    # Server metadata
    if server_meta.get("_raw_props"):
        (meta_dir / "server_props.json").write_text(
            json.dumps(server_meta["_raw_props"], indent=2) + "\n"
        )
    if server_meta.get("_raw_slots"):
        (meta_dir / "server_slots.json").write_text(
            json.dumps(server_meta["_raw_slots"], indent=2) + "\n"
        )
    if server_meta.get("_raw_models"):
        (meta_dir / "server_models.json").write_text(
            json.dumps(server_meta["_raw_models"], indent=2) + "\n"
        )

    # Chat template as separate file (too large for TOML inline)
    if server_meta.get("chat_template_raw"):
        (meta_dir / "chat_template.jinja").write_text(server_meta["chat_template_raw"])

    # HuggingFace config
    hf_config = web_meta.get("hf_config", {})
    if hf_config:
        (meta_dir / "hf_config.json").write_text(
            json.dumps(hf_config, indent=2) + "\n"
        )
        # Save nested text_config separately if present
        text_config = hf_config.get("text_config")
        if isinstance(text_config, dict):
            (meta_dir / "hf_text_config.json").write_text(
                json.dumps(text_config, indent=2) + "\n"
            )




_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
_PROFILES_DIR = _PROJECT_ROOT / "profiles"


from ._provenance import read_existing_provenance as _read_existing_provenance  # noqa: E402




def _run_fixture_validation(profile_name: str) -> tuple[bool, str]:
    """Run the fixture validator. Returns (passed, output_text)."""
    import subprocess

    result = subprocess.run(
        [sys.executable, "-m", "profiles.validate", profile_name],
        capture_output=True,
        text=True,
        cwd=str(_PROJECT_ROOT),
    )
    output = result.stdout + result.stderr
    passed = result.returncode == 0
    return passed, output


def _run_security_validation(profile_dir: Path) -> tuple[bool, str]:
    """Run the security validator. Returns (passed, output_text)."""
    from ..server.security import validate_profile

    violations = validate_profile(profile_dir)
    if violations:
        return False, "\n".join(violations)
    return True, "No security violations"



def _load_samples(samples_dir: Path) -> list[dict]:
    """Load scenario samples from _all_results.json or individual files."""
    combined = samples_dir / "_all_results.json"
    if combined.is_file():
        with open(combined) as f:
            return json.load(f)
    results = []
    for path in sorted(samples_dir.glob("*.json")):
        if path.name.startswith("_"):
            continue
        with open(path) as f:
            results.append(json.load(f))
    if not results:
        raise FileNotFoundError(f"No sample files found in {samples_dir}")
    return results


def _ensure_samples_path(samples_dir: Path, samples: list[dict]) -> Path:
    """Return path to _all_results.json, creating it if only individual files exist."""
    path = samples_dir / "_all_results.json"
    if not path.is_file():
        path.write_text(json.dumps(samples, indent=2))
    return path


def generate(
    model_endpoint: str,
    model_name: str,
    output_dir: Path | None = None,
    samples_dir: Path | None = None,
    family: str | None = None,
    quant: str = "",
    max_iterations: int = 5,
) -> bool:
    """Generate a profile for the given model. Returns True on success."""
    output_dir = output_dir or (_PROFILES_DIR / model_name)

    print(f"Generating profile for: {model_name}")
    print(f"  Output: {output_dir}")
    print()

    # Step 0: Query factual model metadata
    print("Querying server metadata...")
    server_metadata = _query_server_metadata(model_endpoint)
    if server_metadata:
        for k, v in server_metadata.items():
            if k == "chat_template_raw":
                print(f"  {k}: {str(v)[:80]}...")
            elif not k.startswith("_raw"):
                print(f"  {k}: {v}")
    else:
        print("  WARNING: no metadata retrieved from server")

    print("Searching web for model card...")
    web_metadata = _search_model_card(model_name, family or "")
    hf_config = web_metadata.get("hf_config", {})
    hf_model_id = web_metadata.get("hf_model_id", "")
    if web_metadata:
        print(f"  Source: {hf_model_id or '?'}")
        print(f"  Config fields: {len(hf_config)}")
    else:
        print("  No model card found via web search")
    print()

    # Step 1: Collect samples (or use pre-collected)
    if samples_dir and samples_dir.is_dir():
        print(f"Using pre-collected samples from: {samples_dir}")
    else:
        from .run_scenarios import run_all

        samples_dir = output_dir / "_samples"
        print("Running scenario suite...")
        results = run_all(
            endpoint=model_endpoint,
            model=model_name,
            output_dir=samples_dir,
        )
        errors = sum(1 for r in results if r["error"])
        if errors == len(results):
            print("ERROR: All scenarios failed. Is the model endpoint running?")
            return False
        print()

    # Step 2: Load samples and run ProfileAnalyzer
    samples = _load_samples(samples_dir)
    print(f"Loaded {len(samples)} samples")

    existing_provenance = _read_existing_provenance(output_dir)
    if existing_provenance:
        preserved = [k for k in ("model_file", "model_sha256", "llama_server")
                     if existing_provenance.get(k)]
        if preserved:
            print(f"  Preserving from existing profile: {', '.join(preserved)}")

    analyzer = ProfileAnalyzer(
        samples=samples,
        model_name=model_name,
        server_meta=server_metadata,
        hf_config=hf_config,
        hf_model_id=hf_model_id,
        quant=quant,
        existing_provenance=existing_provenance,
    )
    if family:
        analyzer.family = family

    quirks = analyzer.write_profile(output_dir)
    quirk_names = [q.name for q in quirks]
    print(f"  Quirks detected: {quirk_names}")
    print(f"  Profile written to: {output_dir}")

    # Step 3: Save raw metadata files
    _save_metadata_files(output_dir, server_metadata, web_metadata)
    print("  Saved raw metadata to _metadata/")

    # Step 4: Post-normalization verification loop
    samples_path = _ensure_samples_path(samples_dir, samples)
    print()
    print("Post-normalization verification loop:")
    all_passed = False
    last_result = None
    for iteration in range(1, max_iterations + 1):
        result = verify_profile(output_dir, samples_path)
        last_result = result
        if result.all_passed:
            all_passed = True
            print(f"  Iteration {iteration}: all {result.total} scenarios pass")
            break

        failed_ids = [s.scenario_id for s in result.scenarios if not s.passed]
        print(f"  Iteration {iteration}: {result.failed}/{result.total} failed — {failed_ids}")
        for s in result.scenarios:
            if not s.passed:
                for f in s.failures:
                    print(f"    {s.scenario_id}: [{f.check}] {f.message}")

        refinement = refine_rules(result, output_dir)
        if not refinement.changed:
            print("  Refiner made no actionable changes — remaining failures unfixable")
            break

        for change in refinement.changes:
            if change.action != "report":
                print(f"    {change.action}: {change.rule_name} — {change.reason}")

    if not all_passed:
        n_failed = last_result.failed if last_result else 0
        print(f"\nProfile incomplete: {n_failed} scenarios still failing after verification loop")
        if last_result:
            print(format_report(last_result))
        return False

    print(f"Profile complete: all {last_result.total} scenarios pass post-normalization verification")

    # Step 5: Fixture validation
    print("  Running fixture validation...")
    fixture_ok, fixture_output = _run_fixture_validation(model_name)
    print(fixture_output)
    if not fixture_ok:
        print("  Fixtures FAILED")
        return False
    print("  Fixtures PASSED")

    # Step 6: Security validation
    print("  Running security validation...")
    security_ok, security_output = _run_security_validation(output_dir)
    print(f"  {security_output}")
    if not security_ok:
        print("  SECURITY VALIDATION FAILED — profile rejected")
        return False

    print()
    print(f"Profile generated successfully: {output_dir}")
    return True


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Generate a model profile via deterministic ProfileAnalyzer"
    )
    parser.add_argument(
        "model_endpoint",
        help="OpenAI-compatible API endpoint for the target model (e.g. http://localhost:8080/v1)",
    )
    parser.add_argument(
        "model_name",
        help="Model name (e.g. qwen3-8b-q4)",
    )
    parser.add_argument(
        "--output", type=Path,
        help="Output directory (default: profiles/<model_name>/)",
    )
    parser.add_argument(
        "--samples-dir", type=Path,
        help="Use pre-collected samples instead of running scenarios",
    )
    parser.add_argument(
        "--family",
        help="Model family (default: derived from model name)",
    )
    parser.add_argument(
        "--quant", default="",
        help="Quantization (default: empty)",
    )
    parser.add_argument(
        "--max-iterations", type=int, default=5,
        help="Max verify→refine iterations (default: 5)",
    )
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args(argv)

    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="[%(asctime)s] %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    ok = generate(
        model_endpoint=args.model_endpoint,
        model_name=args.model_name,
        output_dir=args.output,
        samples_dir=args.samples_dir,
        family=args.family,
        quant=args.quant,
        max_iterations=args.max_iterations,
    )
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
