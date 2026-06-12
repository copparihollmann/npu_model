#!/usr/bin/env python3
"""
load.py — read the spec machine YAML into a typed node graph, register dialects,
and validate it against core.py.

Usage:
    python spec/schema/load.py            # validate; exit 0 = ok
    python spec/schema/load.py --ssa vpu  # print SSA def-use edges for a family (smoke test)

Files:
    spec/machine/*.yaml   each is a list of node dicts, or {nodes: [...]} (+ optional meta).
    spec/dialects/*.yaml   {dialect: <name>, kinds: [ {KindSpec-as-data}, ... ]}
    spec/manifest.yaml     may list `dialects: [npu, ...]` to load (else all in dialects/).

Stdlib + PyYAML only (keeps the bundle portable).
"""
from __future__ import annotations
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import core  # noqa: E402

try:
    import yaml
except ImportError:
    print("FAIL: PyYAML not installed"); sys.exit(1)

SPEC = Path(__file__).resolve().parent.parent          # .../spec
MACHINE, DIALECTS = SPEC / "machine", SPEC / "dialects"


def _load_dialects(registry: dict) -> list[str]:
    """Register kinds declared by dialect files. Returns dialect names loaded."""
    names: list[str] = []
    man = SPEC / "manifest.yaml"
    wanted = None
    if man.exists():
        m = yaml.safe_load(man.read_text()) or {}
        wanted = m.get("dialects")
    files = sorted(DIALECTS.glob("*.yaml")) if DIALECTS.exists() else []
    for f in files:
        d = yaml.safe_load(f.read_text()) or {}
        name = d.get("dialect") or f.stem
        if wanted is not None and name not in wanted:
            continue
        for kd in d.get("kinds", []) or []:
            core.register_kind(registry, core.KindSpec(
                name=kd["name"], id_prefix=kd.get("id_prefix", name),
                required_attrs=tuple(kd.get("required_attrs", [])),
                optional_attrs=tuple(kd.get("optional_attrs", [])),
                ref_roles={r: (tuple(v) if isinstance(v, list) else v)
                           for r, v in (kd.get("ref_roles") or {}).items()},
                has_operands=bool(kd.get("has_operands", False)),
                has_effect=bool(kd.get("has_effect", False)),
                dialect=name,
            ))
        names.append(name)
    return names


def load() -> tuple[list[core.Node], dict, list[str]]:
    """Return (nodes, registry, errors)."""
    registry = core.fresh_registry()
    dialects = _load_dialects(registry)
    nodes: list[core.Node] = []
    errs: list[str] = []
    if not MACHINE.exists():
        return nodes, registry, [f"missing {MACHINE}"]
    for f in sorted(MACHINE.glob("*.yaml")):
        doc = yaml.safe_load(f.read_text())
        if isinstance(doc, dict) and "nodes" in doc:
            items = doc["nodes"]
        elif isinstance(doc, list):
            items = doc
        else:
            errs.append(f"{f.name}: expected a list of nodes or {{nodes: [...]}}"); continue
        for it in items:
            if not isinstance(it, dict):
                errs.append(f"{f.name}: non-mapping node entry"); continue
            nodes.append(core.Node.from_dict(it, src_file=f.name))
    errs += core.validate_graph(nodes, registry)
    return nodes, registry, errs


def ssa_edges(nodes: list[core.Node], family: str) -> list[str]:
    """Smoke test: print def-use edges for op nodes in a family (proves lowering works)."""
    out: list[str] = []
    for n in nodes:
        if n.kind != "op" or n.attrs.get("family") != family:
            continue
        uses = [o["name"] for o in n.operands if o.get("direction") in ("use", "inout")]
        defs = [o["name"] for o in n.operands if o.get("direction") in ("def", "inout")]
        out.append(f"{n.id}:  uses={uses}  defs={defs}")
        for e in n.effect:
            out.append(f"    {e.get('def')} <- {e.get('expr')}")
    return out


def main() -> int:
    nodes, registry, errs = load()
    if "--ssa" in sys.argv:
        fam = sys.argv[sys.argv.index("--ssa") + 1]
        print(f"== SSA def-use for family '{fam}' ==")
        for line in ssa_edges(nodes, fam) or ["  (no op nodes in that family yet)"]:
            print("  " + line)
    kinds = sorted({n.kind for n in nodes})
    dialects = sorted({k.dialect for k in registry.values()} - {"core"})
    print(f"== load == {len(nodes)} nodes, kinds={kinds}, dialects={dialects or ['(none)']}")
    if errs:
        print(f"== {len(errs)} ERROR(S) ==")
        for e in errs[:200]:
            print("  FAIL:", e)
        return 1
    print("== OK (graph valid) ==")
    return 0


if __name__ == "__main__":
    sys.exit(main())
