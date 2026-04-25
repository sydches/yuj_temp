"""Pipeline integration — system prompt, checkpoint, task enumeration, provenance."""
import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

from .._shared.checkpoints import collect_pending as _collect_pending
from .._shared.paths import expand_user_path
from ..config import Config, dump_config

# Re-export for back-compat with ``from llm_solver.harness.solver import collect_pending``.
collect_pending = _collect_pending


def build_system_prompt(header: str, system_prompt_file: Path | None = None) -> str:
    """Assemble system prompt: optional file content + header.

    header: the harness header text (wired from cfg.system_header).
    system_prompt_file: if provided, its content is prepended to the header.
    The harness does not interpret the file — it could be any protocol.
    """
    if system_prompt_file is None:
        return header
    if not system_prompt_file.is_file():
        raise FileNotFoundError(f"System prompt file not found: {system_prompt_file}")
    return system_prompt_file.read_text().rstrip() + "\n\n" + header


def write_checkpoint(repo_dir: Path, model: str, status: str) -> None:
    """Write checkpoint.json for task/session status tracking."""
    checkpoint = {
        "status": status,
        "model": model,
        "solver": "llm_solver",
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    (repo_dir / "checkpoint.json").write_text(json.dumps(checkpoint, indent=2) + "\n")


def collect_provenance(cfg: Config, profile_path: Path | None = None) -> dict:
    """Gather reproducibility metadata for a run.

    Includes the full resolved Config so a run's exact parameters can be
    reconstructed from ``metrics.json`` after the fact (no reliance on the
    current ``config.toml`` which may have since changed).
    """
    prov: dict = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "model": cfg.model,
        "config": dump_config(cfg),
        "pretest_enabled": True,
    }

    # Harness git commit
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True, text=True, timeout=5,
        )
        if result.returncode == 0:
            prov["harness_git_commit"] = result.stdout.strip()
    except (subprocess.TimeoutExpired, FileNotFoundError):
        pass

    # llama.cpp version — path from config, not hardcoded
    llama_bin = expand_user_path(cfg.llama_server_bin)
    if llama_bin.exists():
        try:
            result = subprocess.run(
                [str(llama_bin), "--version"],
                capture_output=True, text=True, timeout=5,
            )
            if result.returncode == 0:
                prov["llama_cpp_version"] = result.stdout.strip()
        except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
            pass

    # Profile TOML hash
    if profile_path is not None and profile_path.is_file():
        prov["profile_toml_sha256"] = hashlib.sha256(
            profile_path.read_bytes()
        ).hexdigest()

    return prov


def write_run_metrics(repo_dir: Path, metrics: dict, provenance: dict) -> None:
    """Write metrics.json with cost/efficiency metrics and provenance."""
    data = {"metrics": metrics, "provenance": provenance}
    (repo_dir / "metrics.json").write_text(json.dumps(data, indent=2) + "\n")

