"""Knob catalog query CLI.

Thin tool for humans and skills to explore `configs/knobs.toml`.

Subcommands:
  list                         — list all knobs (one per line: name, type, default)
  list --tag <tag>             — filter by tag
  list --blast-radius <level>  — filter by blast_radius
  list --mode <mode>           — filter by mode (measurement | assistant | both)
  describe <name>              — full detail for one knob
  search <text>                — substring search over names + descriptions
  tags                         — list known tags with counts
  presets                      — list curated presets with descriptions
  preset <name>                — show preset contents
  diff-vs-default <file>       — show how a config.local.toml differs from defaults
"""
from __future__ import annotations

import argparse
import pathlib
import sys

from scripts.llm_solver._shared.toml_compat import tomllib


ROOT = pathlib.Path(__file__).resolve().parent.parent
CATALOG = ROOT / "configs" / "knobs.toml"


def load() -> dict:
    if not CATALOG.exists():
        print(f"error: {CATALOG} not found. Run: python3 scripts/build_knob_catalog.py build", file=sys.stderr)
        sys.exit(2)
    return tomllib.loads(CATALOG.read_text())


def cmd_list(args) -> int:
    cat = load()
    knobs = cat.get("knob", {})
    rows = []
    for name, entry in knobs.items():
        if args.tag and args.tag not in entry.get("tags", []):
            continue
        if args.blast_radius and entry.get("blast_radius") != args.blast_radius:
            continue
        if args.mode and entry.get("mode") != args.mode:
            continue
        if args.pending and not entry.get("curation_pending"):
            continue
        rows.append((name, entry))

    if not rows:
        print("(no knobs match filter)")
        return 0

    for name, e in sorted(rows):
        t = e.get("type", "?")
        d = e.get("default", "?")
        br = e.get("blast_radius", "?")
        tags = ",".join(e.get("tags", []))
        pending = " [curation_pending]" if e.get("curation_pending") else ""
        print(f"{name}  [{t}]  default={d!r}  blast={br}  tags=[{tags}]{pending}")
    print(f"\n{len(rows)} knob(s)")
    return 0


def cmd_describe(args) -> int:
    cat = load()
    knobs = cat.get("knob", {})
    entry = knobs.get(args.name)
    if not entry:
        # Try prefix search to help the user
        matches = [n for n in knobs if args.name in n]
        print(f"error: knob '{args.name}' not found.", file=sys.stderr)
        if matches:
            print(f"  close matches: {', '.join(matches[:5])}", file=sys.stderr)
        return 1
    print(f"[{args.name}]")
    for k in ("type", "default", "description", "blast_radius", "mode",
              "tags", "range", "values", "implies", "affects",
              "curation_pending"):
        if k in entry:
            print(f"  {k} = {entry[k]!r}")
    return 0


def cmd_search(args) -> int:
    cat = load()
    knobs = cat.get("knob", {})
    q = args.query.lower()
    rows = []
    for name, entry in knobs.items():
        desc = entry.get("description", "").lower()
        if q in name.lower() or q in desc:
            rows.append((name, entry))
    for name, e in sorted(rows):
        desc = e.get("description", "")
        print(f"{name}\n    {desc}")
    print(f"\n{len(rows)} match(es)")
    return 0


def cmd_tags(args) -> int:
    cat = load()
    counts: dict[str, int] = {}
    for entry in cat.get("knob", {}).values():
        for t in entry.get("tags", []):
            counts[t] = counts.get(t, 0) + 1
    for tag, n in sorted(counts.items(), key=lambda kv: (-kv[1], kv[0])):
        print(f"{n:5d}  {tag}")
    return 0


def cmd_presets(args) -> int:
    cat = load()
    presets = cat.get("preset", {})
    for name, entry in sorted(presets.items()):
        desc = entry.get("description", "")
        mode = entry.get("mode", "both")
        tags = ",".join(entry.get("tags", []))
        equiv = entry.get("equivalent_to", "")
        line = f"{name}  [{mode}]  tags=[{tags}]"
        if equiv:
            line += f"  equiv={equiv}"
        print(line)
        print(f"    {desc}")
    return 0


def cmd_preset(args) -> int:
    cat = load()
    preset = cat.get("preset", {}).get(args.name)
    if not preset:
        print(f"error: preset '{args.name}' not found.", file=sys.stderr)
        print("  available: " + ", ".join(sorted(cat.get("preset", {}).keys())), file=sys.stderr)
        return 1
    print(f"[preset.{args.name}]")
    print(f"  description = {preset.get('description')!r}")
    print(f"  mode = {preset.get('mode')!r}")
    print(f"  tags = {preset.get('tags')!r}")
    applies = preset.get("applies", {})
    print(f"  applies ({len(applies)} key(s)):")
    for k, v in applies.items():
        print(f"    {k} = {v!r}")
    if "equivalent_to" in preset:
        print(f"  equivalent_to = {preset['equivalent_to']!r}")
    return 0


def cmd_diff_vs_default(args) -> int:
    cat = load()
    knobs = cat.get("knob", {})
    if not pathlib.Path(args.path).exists():
        print(f"error: {args.path} not found", file=sys.stderr)
        return 2
    overlay = tomllib.loads(pathlib.Path(args.path).read_text())

    def walk(d, prefix=""):
        out: dict[str, object] = {}
        for k, v in d.items():
            p = f"{prefix}.{k}" if prefix else k
            if isinstance(v, dict):
                out.update(walk(v, p))
            else:
                out[p] = v
        return out

    flat = walk(overlay)
    diffs = []
    unknown = []
    for key, val in flat.items():
        entry = knobs.get(key)
        if entry is None:
            unknown.append(key)
            continue
        default = entry.get("default")
        if val != default:
            diffs.append((key, default, val, entry.get("blast_radius", "?")))

    if diffs:
        print(f"{len(diffs)} knob(s) differ from default:")
        for key, default, val, br in diffs:
            print(f"  {key}  [{br}]  {default!r} -> {val!r}")
    if unknown:
        print(f"\n{len(unknown)} unknown key(s) in {args.path}:")
        for k in unknown:
            print(f"  {k}")
    if not diffs and not unknown:
        print("overlay matches defaults")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p_list = sub.add_parser("list", help="list knobs, with optional filters")
    p_list.add_argument("--tag")
    p_list.add_argument("--blast-radius", choices=["low", "medium", "high"])
    p_list.add_argument("--mode", choices=["measurement", "assistant", "both"])
    p_list.add_argument("--pending", action="store_true", help="only show curation_pending knobs")
    p_list.set_defaults(func=cmd_list)

    p_desc = sub.add_parser("describe", help="describe a single knob")
    p_desc.add_argument("name")
    p_desc.set_defaults(func=cmd_describe)

    p_search = sub.add_parser("search", help="substring search over knob names + descriptions")
    p_search.add_argument("query")
    p_search.set_defaults(func=cmd_search)

    p_tags = sub.add_parser("tags", help="list tags with counts")
    p_tags.set_defaults(func=cmd_tags)

    p_presets = sub.add_parser("presets", help="list curated presets")
    p_presets.set_defaults(func=cmd_presets)

    p_preset = sub.add_parser("preset", help="show one preset")
    p_preset.add_argument("name")
    p_preset.set_defaults(func=cmd_preset)

    p_diff = sub.add_parser("diff-vs-default", help="diff a config file against defaults")
    p_diff.add_argument("path", help="path to a config.local.toml or overlay")
    p_diff.set_defaults(func=cmd_diff_vs_default)

    args = ap.parse_args()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
