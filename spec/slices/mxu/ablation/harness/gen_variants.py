#!/usr/bin/env python3
"""
gen_variants.py — deterministically generate the ISA variants that are MORE complex than
the 14-op baseline (C3). Two points, separating two complexity axes:

  C5_expanded  — LARGE but REGULAR: bake weight-slot + accumulator INDICES into the opcode
                 (baseline's "fields-in-opcode" philosophy pushed further). Uniform encoding,
                 uniform operand order. Tests whether sheer opcode COUNT hurts one-shot RTL gen
                 even when the pattern is regular/learnable.

  C6_scattered — LARGE and IRREGULAR: the same explosion PLUS encoding diversity — mixed
                 formats, inconsistent operand order, ad-hoc immediates, fused chains
                 (the TPU-LLO-258-op style). Tests whether DIVERSITY/SCATTER hurts, holding
                 capability fixed.

Deterministic (index-driven, no RNG / no Date) so runs are reproducible. Materializes
../variants/{C5_expanded,C6_scattered}/isa.yaml.

    python gen_variants.py        # (re)write the two expanded variant isa.yaml files
"""
from __future__ import annotations
from pathlib import Path

HERE = Path(__file__).resolve().parent
VARIANTS = HERE.parent / "variants"
BASELINE_ISA = HERE.parent.parent / "machine" / "isa.yaml"   # the 14-op baseline (C3)


def _read_baseline() -> list[dict]:
    import copy
    doc = yaml.safe_load(BASELINE_ISA.read_text())
    nodes = doc["nodes"] if isinstance(doc, dict) and "nodes" in doc else doc
    return [copy.deepcopy(n) for n in nodes if n.get("kind") == "op"]

try:
    import yaml
except ImportError:
    raise SystemExit("FAIL: PyYAML not installed")

# operand building blocks (all reference shared substrate states/types) -------
ACT = lambda n="vs1": {"name": n, "type": "type.mrf_tile_fp8", "location": "state.mrf", "direction": "use"}
WT  = lambda n="vs2": {"name": n, "type": "type.weight_tile_fp8", "location": "state.mxu_weight_slots", "direction": "use"}
WTD = lambda n="wd":  {"name": n, "type": "type.weight_tile_fp8", "location": "state.mxu_weight_slots", "direction": "def"}
ACC = lambda n, d:    {"name": n, "type": "type.acc_tile_bf16", "location": "state.mxu_accum_buffers", "direction": d}
MRFW= lambda n="vd":  {"name": n, "type": "type.mrf_tile_bf16", "location": "state.mrf", "direction": "def"}
MRFF= lambda n="vd":  {"name": n, "type": "type.mrf_tile_fp8", "location": "state.mrf", "direction": "def"}
SCALE = lambda n="es1": {"name": n, "type": "type.scale", "location": "state.erf", "direction": "use"}
IMM = lambda n: {"name": n, "type": "type.mod", "direction": "use"}


def op(oid, family, unit, funct7, operands, effect, fmt="VR", extra=None,
       rationale=None, adds_cap=False):
    enc = {"format": fmt, "opcode": 0b1110111, "funct7": funct7, "allocated": "generated"}
    if extra:
        enc.update(extra)
    lat = f"timing.mxu{unit}_matmul" if family.startswith("mxu_matmul") or "matmul" in family else "timing.mxu_push_pop"
    attrs = {"family": family, "exu": "MATRIX", "encoding": enc, "adds_capability": adds_cap}
    if rationale:
        attrs["rationale"] = rationale
    return {"id": oid, "kind": "op", "status": "reference", "attrs": attrs,
            "refs": {"latency_class": lat, "unit": f"unit.mxu{unit}"},
            "operands": operands, "effect": effect}


def gen_c5_expanded() -> list[dict]:
    """Regular explosion: bake w-slot (s) and acc-index (a) into the opcode. ~36 ops."""
    ops, f7 = [], 0
    for u in (0, 1):
        for s in (0, 1):                                   # weight push per slot
            ops.append(op(f"op.c5.vmatpush.weight.mxu{u}.w{s}", "mxu_weight_push", u, f7,
                          [WTD("wd"), ACT("vs1")], [{"def": "wd", "expr": {"op": "push", "args": ["vs1"]}}])); f7 += 1
        for a in (0, 1):                                   # acc push per dtype per acc
            for dt in ("fp8", "bf16"):
                src = ACT("vs1") if dt == "fp8" else {"name": "vs1", "type": "type.mrf_tile_bf16", "location": "state.mrf", "direction": "use"}
                ops.append(op(f"op.c5.vmatpush.acc.{dt}.mxu{u}.acc{a}", "mxu_acc_push", u, f7,
                              [ACC("ad", "def"), src], [{"def": "ad", "expr": {"op": "acc_push", "args": ["vs1"]}}])); f7 += 1
            for dt in ("fp8", "bf16"):                     # acc pop per dtype per acc
                dst = MRFF("vd") if dt == "fp8" else MRFW("vd")
                opnds = [dst, ACC("vs2", "use")] + ([SCALE()] if dt == "fp8" else [])
                args = ["vs2", "es1"] if dt == "fp8" else ["vs2"]
                ops.append(op(f"op.c5.vmatpop.{dt}.acc.mxu{u}.acc{a}", "mxu_acc_pop", u, f7,
                              opnds, [{"def": "vd", "expr": {"op": "acc_pop", "args": args}}])); f7 += 1
            for s in (0, 1):                               # matmul[.acc] per acc per w-slot
                ops.append(op(f"op.c5.vmatmul.mxu{u}.acc{a}.w{s}", "mxu_matmul", u, f7,
                              [ACC("vd", "def"), ACT("vs1"), WT("vs2")],
                              [{"def": "vd", "expr": {"op": "matmul", "args": ["vs1", "vs2"]}}])); f7 += 1
                ops.append(op(f"op.c5.vmatmul.acc.mxu{u}.acc{a}.w{s}", "mxu_matmul", u, f7,
                              [ACC("vd", "inout"), ACT("vs1"), WT("vs2")],
                              [{"def": "vd", "expr": {"op": "add", "args": ["vd", {"op": "matmul", "args": ["vs1", "vs2"]}]}}])); f7 += 1
    for o in ops:   # honest label: C5 is the BLOAT CONTROL — re-encoding, not new capability
        o["attrs"]["adds_capability"] = False
        o["attrs"]["rationale"] = "selector index baked into the opcode (frees operand bits / simpler per-op decode); NO new capability vs baseline"
    return ops


def gen_c6_scattered() -> list[dict]:
    """Irregular explosion: C5 + encoding diversity (mixed formats, shuffled operand order,
    ad-hoc immediates) + fused chains. ~64 ops."""
    base = gen_c5_expanded()
    fmts = ["VR", "VRR", "VRI", "VRX", "CUSTOM"]
    ops, f7 = [], 0
    for i, o in enumerate(base):
        o = {**o, "id": o["id"].replace("op.c5.", "op.c6."),
             "attrs": {**o["attrs"], "encoding": {**o["attrs"]["encoding"]}}}
        o["attrs"]["encoding"]["format"] = fmts[i % len(fmts)]        # mixed formats
        o["attrs"]["encoding"]["funct7"] = f7; f7 += 1
        if i % 3 == 0:                                                # ad-hoc immediate on 1/3
            o["operands"] = o["operands"] + [IMM("aux")]
            o["attrs"]["encoding"]["reserved_zero"] = "vs2[6]=0" if i % 2 else "vd[6:5]=0"
        if i % 4 == 1:                                                # shuffle operand order on 1/4
            o["operands"] = list(reversed(o["operands"]))
        ops.append(o)
    # ad-hoc fused chains (the "diverse CISC sprinkled on top" — these DO add capability,
    # but C6 packages them irregularly; C7_capable adds the same kind REGULARLY).
    for u in (0, 1):
        ops.append(op(f"op.c6.vmatmulpop.bf16.mxu{u}", "fused_matmul_pop", u, f7,
                      [MRFW("vd"), ACT("vs1"), WT("vs2")],
                      [{"def": "vd", "expr": {"op": "matmul_then_pop", "args": ["vs1", "vs2"]}}],
                      fmt="CUSTOM", adds_cap=True,
                      rationale="fused matmul->MRF: skips explicit pop + acc->MRF round-trip (~32cyc/tile)")); f7 += 1
        ops.append(op(f"op.c6.vtmatmul.mxu{u}", "fused_transpose_matmul", u, f7,
                      [ACC("vd", "def"), ACT("vs1"), WT("vs2")],
                      [{"def": "vd", "expr": {"op": "matmul", "args": [{"op": "transpose", "args": ["vs1"]}, "vs2"]}}],
                      fmt="VRR", adds_cap=True,
                      rationale="transpose-fused matmul: A^T@W without an XLU pass (~66cyc + MRF round-trip; attention Q@K^T)")); f7 += 1
        ops.append(op(f"op.c6.vmatmulrq.fp8.mxu{u}", "fused_matmul_requant", u, f7,
                      [MRFF("vd"), ACT("vs1"), WT("vs2"), SCALE()],
                      [{"def": "vd", "expr": {"op": "requant", "args": [{"op": "matmul", "args": ["vs1", "vs2"]}, "es1"]}}],
                      fmt="VRX", adds_cap=True,
                      rationale="matmul + fp8 requant fused: direct fp8 output with scale (saves pop+pack)")); f7 += 1
    return ops


def gen_c7_capable() -> list[dict]:
    """JUSTIFIED complexity: the 14-op baseline + fine-control ops where EVERY added opcode
    buys a concrete capability/perf win (rationale recorded). The 'fair' high-complexity arm:
    same direction as C5/C6 in opcode count, but no gratuitous opcodes."""
    base = _read_baseline()
    for o in base:                       # baseline ops are capability-equivalent (not new)
        o.setdefault("attrs", {})["adds_capability"] = False
        o["attrs"]["rationale"] = "baseline op (capability-equivalent re-packaging point)"
    f7 = 0b1000000
    add = []
    for u in (0, 1):
        add.append(op(f"op.c7.vtmatmul.mxu{u}", "fused_transpose_matmul", u, f7,
                      [ACC("vd", "def"), ACT("vs1"), WT("vs2")],
                      [{"def": "vd", "expr": {"op": "matmul", "args": [{"op": "transpose", "args": ["vs1"]}, "vs2"]}}],
                      adds_cap=True, rationale="A^T@W without an XLU transpose pass: ~66cyc + 1 MRF tile + round-trip saved per tile (attention Q@K^T)")); f7 += 1
        add.append(op(f"op.c7.vmatmulpop.bf16.mxu{u}", "fused_matmul_pop", u, f7,
                      [MRFW("vd"), ACT("vs1"), WT("vs2")],
                      [{"def": "vd", "expr": {"op": "matmul_then_pop", "args": ["vs1", "vs2"]}}],
                      adds_cap=True, rationale="matmul result straight to MRF: skips the explicit pop + acc->MRF round-trip (~32cyc/tile)")); f7 += 1
        add.append(op(f"op.c7.vmatmulrq.fp8.mxu{u}", "fused_matmul_requant", u, f7,
                      [MRFF("vd"), ACT("vs1"), WT("vs2"), SCALE()],
                      [{"def": "vd", "expr": {"op": "requant", "args": [{"op": "matmul", "args": ["vs1", "vs2"]}, "es1"]}}],
                      adds_cap=True, rationale="direct fp8 output with scale: saves a separate pop + pack/requant")); f7 += 1
        add.append(op(f"op.c7.vmatpop.bias.bf16.mxu{u}", "fused_pop_bias", u, f7,
                      [MRFW("vd"), ACC("vs2", "use"), {"name": "vb", "type": "type.mrf_tile_bf16", "location": "state.mrf", "direction": "use"}],
                      [{"def": "vd", "expr": {"op": "add", "args": ["vs2", "vb"]}}],
                      adds_cap=True, rationale="bias-add fused into the drain (C = acc + bias): saves a separate vadd (~66cyc/tile)")); f7 += 1
        add.append(op(f"op.c7.vmatmul.sub16.mxu{u}", "mxu_matmul_subtile", u, f7,
                      [ACC("vd", "def"), ACT("vs1"), WT("vs2")],
                      [{"def": "vd", "expr": {"op": "matmul_subtile", "args": ["vs1", "vs2"]}}],
                      adds_cap=True, rationale="16x16 sub-tile matmul: fine-grained shape control, ~4x less waste on <=16 dims (small attention heads / odd shapes)")); f7 += 1
    return base + add


def write(name: str, nodes: list[dict], header: str) -> None:
    out = VARIANTS / name / "isa.yaml"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(header + "\n" + yaml.safe_dump({"nodes": nodes}, sort_keys=False, default_flow_style=True, width=4096))
    print(f"  wrote {out.relative_to(HERE.parent.parent.parent.parent)}  ({len(nodes)} ops)")


if __name__ == "__main__":
    write("C5_expanded", gen_c5_expanded(),
          "# C5_expanded — GENERATED by harness/gen_variants.py. Large but REGULAR: weight-slot\n"
          "# + accumulator indices baked into the opcode; uniform VR encoding + operand order.")
    write("C6_scattered", gen_c6_scattered(),
          "# C6_scattered — GENERATED by harness/gen_variants.py. Large and IRREGULAR: mixed\n"
          "# encoding formats, shuffled operand order, ad-hoc immediates, fused chains.")
    write("C7_capable", gen_c7_capable(),
          "# C7_capable — GENERATED by harness/gen_variants.py. JUSTIFIED complexity: baseline\n"
          "# + fine-control ops where EVERY added opcode buys a concrete win (see attrs.rationale).")
