#!/usr/bin/env python3
"""
validate.py — standalone validator for the MXU slice.

The slice is self-contained DATA; it reuses the SHARED core meta-model
(spec/schema/core.py) so its type system is identical to the full bundle's —
only the node set is restricted to the MXU closure. This loads this slice's
machine/*.yaml + dialects/*.yaml and validates the node graph.

    python spec/slices/mxu/validate.py             # exit 0 = slice graph valid
    python spec/slices/mxu/validate.py --ssa mxu_matmul   # SSA def-use smoke test

Stdlib + PyYAML only.
"""
from __future__ import annotations
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
SCHEMA = HERE.parent.parent / "schema"        # spec/schema (shared core meta-model)
sys.path.insert(0, str(SCHEMA))
import core  # noqa: E402

try:
    import yaml
except ImportError:
    print("FAIL: PyYAML not installed"); sys.exit(1)

MACHINE, DIALECTS = HERE / "machine", HERE / "dialects"


def _load_dialects(registry: dict) -> list[str]:
    names: list[str] = []
    man = HERE / "manifest.yaml"
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
    registry = core.fresh_registry()
    _load_dialects(registry)
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
        for line in ssa_edges(nodes, fam) or ["  (no op nodes in that family)"]:
            print("  " + line)
    kinds = sorted({n.kind for n in nodes})
    dialects = sorted({k.dialect for k in registry.values()} - {"core"})
    print(f"== MXU slice == {len(nodes)} nodes, kinds={kinds}, dialects={dialects or ['(none)']}")
    if errs:
        print(f"== {len(errs)} ERROR(S) ==")
        for e in errs[:200]:
            print("  FAIL:", e)
        return 1
    print("== OK (slice graph valid) ==")
    return 0


if __name__ == "__main__":
    sys.exit(main())
