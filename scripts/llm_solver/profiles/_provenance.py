"""Profile provenance helpers — read the ``[provenance]`` block off disk.

Both ``generate.py`` and ``_analyzer/core.py`` need to merge a run's
existing provenance before writing a new profile TOML. Centralizing the
read-and-extract logic here prevents drift and fixes a subtle bug: the
previous inline implementations called ``tomllib.loads(path.read_text())``
which decodes with the platform locale. The TOML spec requires UTF-8, so
this module uses ``load_toml(path)`` which opens in binary mode.
"""
from __future__ import annotations

from pathlib import Path

from .._shared.toml_compat import load_toml


def read_existing_provenance(profile_dir: Path) -> dict:
    """Return the ``[provenance]`` block from ``<profile_dir>/profile.toml``.

    Empty dict if the file does not exist yet. Raises on parse errors — a
    malformed profile TOML is a bug, not a fall-through condition.
    """
    toml_path = profile_dir / "profile.toml"
    if not toml_path.is_file():
        return {}
    parsed = load_toml(toml_path)
    return parsed.get("provenance", {})


__all__ = ["read_existing_provenance"]
