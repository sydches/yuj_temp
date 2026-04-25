"""Context strategies — swappable conversation layouts.

Each strategy decides which prior turns the model sees on the next call.
All implement the ``ContextManager`` protocol defined in
``harness/context.py``.

Discovery contract:
- Built-in ``full`` is always mapped to ``FullTranscript``.
- Any sibling module can register a mode by exporting:
    CONTEXT_MODE = "<name>"
    CONTEXT_CLASS = <ContextManager subclass>

Adding a new strategy can therefore be file-drop only (no edits here),
as long as the module follows that contract.
"""
from __future__ import annotations

import importlib
import pkgutil

from ..context import FullTranscript


def _discover_context_modes() -> dict[str, type]:
    """Discover mode->class pairs from strategy modules in this package."""
    discovered: dict[str, type] = {}
    prefix = __name__ + "."
    for mod_info in pkgutil.iter_modules(__path__):  # type: ignore[name-defined]
        if mod_info.name.startswith("_"):
            continue
        mod = importlib.import_module(prefix + mod_info.name)
        mode = getattr(mod, "CONTEXT_MODE", None)
        klass = getattr(mod, "CONTEXT_CLASS", None)
        if isinstance(mode, str) and mode and klass is not None:
            discovered[mode] = klass
    return discovered


_discovered = _discover_context_modes()
_MODE_TO_CLASS = {"full": FullTranscript}
for _name in (
    "compact",
    "concise",
    "slot",
    "yuj",
    "yconcise",
    "yslot",
    "stateful",
    "compound",
    "focused_compound",
    "compound_selective",
):
    if _name in _discovered:
        _MODE_TO_CLASS[_name] = _discovered.pop(_name)
for _name in sorted(_discovered):
    _MODE_TO_CLASS[_name] = _discovered[_name]

# Re-export commonly used classes for import stability.
CompactTranscript = _MODE_TO_CLASS["compact"]
ConciseTranscript = _MODE_TO_CLASS["concise"]
SlotTranscript = _MODE_TO_CLASS["slot"]
YujTranscript = _MODE_TO_CLASS["yuj"]
YconciseContext = _MODE_TO_CLASS["yconcise"]
YslotContext = _MODE_TO_CLASS["yslot"]
SolverStateContext = _MODE_TO_CLASS["stateful"]
CompoundContext = _MODE_TO_CLASS["compound"]
FocusedCompoundContext = _MODE_TO_CLASS["focused_compound"]
CompoundSelectiveContext = _MODE_TO_CLASS["compound_selective"]


def list_context_modes() -> tuple[str, ...]:
    """Return CLI mode names in stable order."""
    return tuple(_MODE_TO_CLASS.keys())


def resolve_context_class(mode: str):
    """Resolve a context mode name to its ContextManager class."""
    try:
        return _MODE_TO_CLASS[mode]
    except KeyError as exc:
        raise ValueError(
            f"Unknown context mode '{mode}'. Available: {list_context_modes()}"
        ) from exc


__all__ = [
    "list_context_modes",
    "resolve_context_class",
    "CompactTranscript",
    "ConciseTranscript",
    "CompoundContext",
    "CompoundSelectiveContext",
    "FocusedCompoundContext",
    "FullTranscript",
    "SolverStateContext",
    "SlotTranscript",
    "YconciseContext",
    "YslotContext",
    "YujTranscript",
]
