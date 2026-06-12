#!/usr/bin/env python3
"""
spec/lint.py — export gate for the spec bundle.

  1. validates the typed node graph (schema/load.py: envelope, ids, refs,
     operands/effect, dialects).
  2. checks critical constants agree between machine/ nodes and human/SPEC.md
     (no value drift).
  3. checks the manifest file-set is complete.

Run:  python spec/lint.py     (exit 0 = ok). Stdlib + PyYAML only.
"""
from __future__ import annotations
import sys
from pathlib import Path

SPEC = Path(__file__).resolve().parent
sys.path.insert(0, str(SPEC / "schema"))
import load as loader  # noqa: E402
import yaml  # noqa: E402

FAILS: list[str] = []
def fail(m): FAILS.append(m); print(f"  FAIL: {m}")
def ok(m):   print(f"  ok:   {m}")
def warn(m): print(f"  WARN: {m}")
def norm(s): return str(s).replace("_", "").replace(" ", "").lower()

# 1) graph -------------------------------------------------------------------
print("== node graph ==")
nodes, registry, gerrs = loader.load()
by_id = {n.id: n for n in nodes}
for e in gerrs:
    fail(e)
if not gerrs:
    dia = sorted({k.dialect for k in registry.values()} - {"core"})
    ok(f"{len(nodes)} nodes valid; dialects={dia or ['(none)']}")

# 2) critical-constant agreement machine -> SPEC.md --------------------------
print("== critical-constant agreement (nodes -> human/SPEC.md) ==")
spec_md = (SPEC / "human" / "SPEC.md")
md = norm(spec_md.read_text()) if spec_md.exists() else ""
if not md:
    fail("human/SPEC.md missing")
checks = [
    ("timing.mxu0_matmul", "cycles", 96),
    ("timing.mxu1_matmul", "cycles", 35),
    ("timing.vpu_col_reduction", "cycles", 130),
    ("param.mrf.count", "value", 64),
    ("param.xrf.count", "value", 32),
    ("param.dma.channels", "value", 8),
    ("constraint.vmem.capacity", "value", "1 MiB"),
    ("constraint.imem.capacity", "value", "128 KiB"),
]
for nid, attr, expect in checks:
    n = by_id.get(nid)
    if n is None:
        fail(f"node '{nid}' missing"); continue
    val = n.attrs.get(attr)
    if str(val) != str(expect):
        fail(f"{nid}.{attr} = {val!r}, expected {expect!r}")
    elif norm(val) in md:
        ok(f"{nid}.{attr} = {val} present in SPEC.md")
    else:
        fail(f"{nid}.{attr} = {val} NOT found in SPEC.md (drift)")

# 3) manifest file-set -------------------------------------------------------
print("== manifest file-set ==")
man = yaml.safe_load((SPEC / "manifest.yaml").read_text()) or {}
listed = {a["file"] for a in man.get("artifacts", []) if isinstance(a, dict)}
on_disk = set()
for p in SPEC.rglob("*"):
    if p.is_file() and p.suffix in (".md", ".yaml", ".py") and "assets" not in p.parts:
        on_disk.add(str(p.relative_to(SPEC)))
for f in sorted(listed):
    if not (SPEC / f).exists():
        fail(f"manifest lists missing file: {f}")
for f in sorted(on_disk - listed):
    warn(f"file not in manifest: {f}")
if listed and not (listed - on_disk):
    ok(f"{len(listed)} artifacts listed, all present")

# summary --------------------------------------------------------------------
opens = sorted(n.id for n in nodes if n.status == "open")
print(f"\n== open items ({len(opens)}) =="); [print("  open:", i) for i in opens]
print(f"\n== summary == {len(FAILS)} fail")
sys.exit(1 if FAILS else 0)
