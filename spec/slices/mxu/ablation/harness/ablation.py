#!/usr/bin/env python3
"""
ablation.py — drive the MXU ISA-complexity ablation.

Two ablation families (see ../README.md):
  (A) ISA DESIGN complexity  — different instruction sets for the SAME MXU capability:
        C4_cisc_macro (1 op)  <  C1_risc_min (3)  ~  C2_collapsed (4)  <<  C3_baseline (14)
  (B) ISA DESCRIPTION complexity — the SAME ISA rendered at 3 verbosity levels
        (concise / structured / verbose), with token accounting.

Each variant is a swappable op-node set over the SHARED substrate
(../../machine/{types,state,spec}.yaml + ../shared/{types,state}.yaml), validated by the
SAME core meta-model (../../../schema/core.py). Adding the whole ISA later = drop in more
op nodes; the harness is variant-agnostic.

Usage:
    python ablation.py validate     # every variant graph validates (exit 0 = all ok)
    python ablation.py opcodes      # family (A): opcode-surface table
    python ablation.py present      # family (B): verbosity x token-cost table
    python ablation.py run          # full experiment matrix (RTL-gen/eval are hooks)

Stdlib + PyYAML only.
"""
from __future__ import annotations
import math
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent          # .../ablation/harness
ABL = HERE.parent                               # .../ablation
MXU = ABL.parent                                # .../slices/mxu
SPEC = MXU.parent.parent                        # .../spec
sys.path.insert(0, str(SPEC / "schema"))
import core  # noqa: E402

try:
    import yaml
except ImportError:
    print("FAIL: PyYAML not installed"); sys.exit(1)

# Shared substrate loaded for every variant (so refs resolve).
BASE = [MXU / "machine" / "types.yaml", MXU / "machine" / "state.yaml",
        MXU / "machine" / "spec.yaml", ABL / "shared" / "types.yaml",
        ABL / "shared" / "state.yaml"]

# Variants, ordered by opcode surface. isa = the op-node set under test.
# C5/C6 are GENERATED (harness/gen_variants.py) and live ABOVE the baseline in complexity:
#   C5 = large but regular ; C6 = matched size but irregular/scattered.
VARIANTS = [
    ("C4_cisc_macro", ABL / "variants" / "C4_cisc_macro" / "isa.yaml"),
    ("C1_risc_min",   ABL / "variants" / "C1_risc_min"   / "isa.yaml"),
    ("C2_collapsed",  ABL / "variants" / "C2_collapsed"  / "isa.yaml"),
    ("C3_baseline",   MXU / "machine" / "isa.yaml"),  # the full 14-op baseline ISA
    ("C7_capable",    ABL / "variants" / "C7_capable"   / "isa.yaml"),
    ("C5_expanded",   ABL / "variants" / "C5_expanded"   / "isa.yaml"),
    ("C6_scattered",  ABL / "variants" / "C6_scattered"  / "isa.yaml"),
]
STYLES = ["concise", "structured", "verbose"]


def _read_nodes(path: Path) -> list[dict]:
    doc = yaml.safe_load(path.read_text())
    return doc["nodes"] if isinstance(doc, dict) and "nodes" in doc else doc


def load_variant(isa_path: Path):
    """Return (nodes, raw_op_dicts, errors) for one variant over the shared substrate."""
    registry = core.fresh_registry()
    nodes, raw_ops, errs = [], [], []
    for f in BASE + [isa_path]:
        for it in _read_nodes(f):
            nodes.append(core.Node.from_dict(it, src_file=f.name))
            if it.get("kind") == "op" and f == isa_path:
                raw_ops.append(it)
    errs += core.validate_graph(nodes, registry)
    return nodes, raw_ops, errs


# ---- approximate token accounting (no tokenizer dependency) -----------------
def approx_tokens(text: str) -> int:
    """Crude GPT-ish estimate: ~chars/4, floored by word count. Labelled approximate."""
    return max(len(text.split()), math.ceil(len(text) / 4))


# ---- family (B): render one op at a verbosity level -------------------------
def render(op: dict, style: str) -> str:
    a = op.get("attrs", {})
    enc = a.get("encoding", {})
    mn = op["id"].replace("op.", "")
    if style == "concise":
        tag = enc.get("hex") or enc.get("funct7") or a.get("family", "")
        return f"{mn}\t{tag}\t{a.get('sem', a.get('family',''))}"
    if style == "structured":
        return yaml.safe_dump(op, sort_keys=False, default_flow_style=False).strip()
    # verbose
    lines = [f"Instruction: {mn}",
             f"  family: {a.get('family')}   exu: {a.get('exu')}   blocking: {a.get('blocking', False)}",
             f"  semantics: {a.get('sem','(prose)')}",
             f"  encoding: format={enc.get('format')} opcode={enc.get('opcode')} "
             f"funct7={enc.get('funct7')} hex={enc.get('hex','-')} allocated={enc.get('allocated','baseline')}"]
    if enc.get("reserved_zero"):
        lines.append(f"  reserved_zero: {enc['reserved_zero']}")
    if enc.get("selector"):
        lines.append(f"  selector: {enc['selector']}")
    lines.append(f"  latency_class: {op.get('refs',{}).get('latency_class')}")
    for o in op.get("operands", []):
        lines.append(f"  operand {o['name']}: type={o.get('type','-')} "
                     f"location={o.get('location','-')} direction={o['direction']}")
    for e in op.get("effect", []):
        lines.append(f"  effect: {e.get('def')} <- {e.get('expr')}")
    return "\n".join(lines)


def variant_presentation(raw_ops: list[dict], style: str) -> str:
    return "\n\n".join(render(op, style) for op in raw_ops)


# ---- RTL-gen / evaluation hooks (the parts an external model/tool fills) ----
def rtl_generate(prompt: str):
    """HOOK: one-shot LLM RTL generation from a presentation. Wire to your model."""
    raise NotImplementedError  # intentionally a hook


def evaluate(rtl, variant: str):
    """HOOK: functional check vs npu_model oracle + perf overhead vs Layer B (<=10%)."""
    raise NotImplementedError  # intentionally a hook


# ---- commands --------------------------------------------------------------
def cmd_validate() -> int:
    bad = 0
    for name, isa in VARIANTS:
        nodes, raw_ops, errs = load_variant(isa)
        status = "OK" if not errs else f"{len(errs)} ERROR(S)"
        print(f"  {name:16s} {len(nodes):3d} nodes, {len(raw_ops):2d} opcodes  -> {status}")
        for e in errs[:20]:
            print("      FAIL:", e); bad += 1
    print("== all variants valid ==" if not bad else f"== {bad} errors ==")
    return 1 if bad else 0


def cmd_opcodes() -> int:
    print("== Family (A): ISA design complexity — opcode surface ==")
    print(f"  {'variant':16s} {'opcodes':>7s}   note")
    notes = {"C4_cisc_macro": "1 microcoded macro; max per-op semantics",
             "C1_risc_min": "uniform tile load/store + mma over unified regs",
             "C2_collapsed": "unit/dtype/acc folded into a modifier field",
             "C3_baseline": "the real ISA: unit & dtype baked into the opcode",
             "C7_capable": "ABOVE baseline: +fine-control ops, every one JUSTIFIED (see capability)",
             "C5_expanded": "ABOVE baseline: slot+acc indices baked in (large, REGULAR, NO new capability)",
             "C6_scattered": "ABOVE baseline: mixed formats/order + some fused (IRREGULAR)"}
    for name, isa in VARIANTS:
        _, raw_ops, _ = load_variant(isa)
        print(f"  {name:16s} {len(raw_ops):7d}   {notes.get(name,'')}")
    return 0


def cmd_capability() -> int:
    """Does the extra opcode COUNT buy capability, or is it packaging/bloat?
    Separates 'why does this ISA have more opcodes' into justified vs gratuitous."""
    print("== Capability vs packaging — do the extra opcodes earn their keep? ==")
    print(f"  {'variant':16s} {'opcodes':>7s} {'+cap':>5s} {'over_base':>9s} {'cap/added':>9s}  class")
    BASE = 14
    for name, isa in VARIANTS:
        _, raw_ops, _ = load_variant(isa)
        n = len(raw_ops)
        newcap = sum(1 for o in raw_ops if o.get("attrs", {}).get("adds_capability"))
        added = n - BASE
        ratio = (newcap / added) if added > 0 else (0.0 if added == 0 else float("nan"))
        if added <= 0:
            klass = "re-packaging (<= baseline capability)"
        elif ratio >= 0.99:
            klass = "JUSTIFIED (every added op adds capability)"
        elif newcap == 0:
            klass = "BLOAT (more opcodes, no new capability)"
        else:
            klass = "mixed (some justified, some gratuitous)"
        rs = f"{ratio:9.2f}" if added > 0 else f"{'-':>9s}"
        print(f"  {name:16s} {n:7d} {newcap:5d} {added:9d} {rs}  {klass}")
    print("  +cap = opcodes flagged attrs.adds_capability; cap/added = new capability per extra opcode")
    return 0


def cmd_diversity() -> int:
    """Separate SIZE (opcode count) from SCATTER (encoding/operand diversity)."""
    print("== Size vs scatter (the two 'complexity' axes are not the same) ==")
    print(f"  {'variant':16s} {'opcodes':>7s} {'formats':>7s} {'arities':>7s} {'dir_sigs':>8s} {'scatter':>7s}")
    for name, isa in VARIANTS:
        _, raw_ops, _ = load_variant(isa)
        fmts = {o.get("attrs", {}).get("encoding", {}).get("format", "VR") for o in raw_ops}
        arities = {len(o.get("operands", [])) for o in raw_ops}
        dir_sigs = {tuple(x.get("direction") for x in o.get("operands", [])) for o in raw_ops}
        n = max(len(raw_ops), 1)
        # scatter = how many distinct shapes per opcode (1.0 = every op structurally unique)
        scatter = round((len(fmts) + len(arities) + len(dir_sigs)) / (3 * n), 3)
        print(f"  {name:16s} {len(raw_ops):7d} {len(fmts):7d} {len(arities):7d} {len(dir_sigs):8d} {scatter:7.3f}")
    print("  formats/arities/dir_sigs = distinct encoding formats / operand counts / direction signatures")
    print("  scatter ~ structural diversity per opcode; C5 (regular) << C6 (scattered) at similar size")
    return 0


def cmd_present() -> int:
    print("== Family (B): description complexity — approx tokens (whole ISA) ==")
    print(f"  {'variant':16s} {'concise':>9s} {'structured':>11s} {'verbose':>9s}")
    for name, isa in VARIANTS:
        _, raw_ops, _ = load_variant(isa)
        cells = [approx_tokens(variant_presentation(raw_ops, s)) for s in STYLES]
        print(f"  {name:16s} {cells[0]:9d} {cells[1]:11d} {cells[2]:9d}")
    print("  (token estimate ~ max(words, chars/4); swap in a real tokenizer in approx_tokens)")
    return 0


def cmd_run() -> int:
    print("== Experiment matrix: variant x style ==")
    print(f"  {'variant':16s} {'style':12s} {'~tokens':>8s} {'rtl':>6s} {'funcOK':>7s} {'perf<=10%':>9s}")
    for name, isa in VARIANTS:
        _, raw_ops, _ = load_variant(isa)
        for style in STYLES:
            prompt = variant_presentation(raw_ops, style)
            toks = approx_tokens(prompt)
            try:
                rtl = rtl_generate(prompt); gen = "ok"
            except NotImplementedError:
                rtl, gen = None, "TODO"
            try:
                res = evaluate(rtl, name); fok, perf = res["functional"], res["perf_ok"]
            except (NotImplementedError, TypeError):
                fok = perf = "TODO"
            print(f"  {name:16s} {style:12s} {toks:8d} {gen:>6s} {str(fok):>7s} {str(perf):>9s}")
    print("  wire rtl_generate()/evaluate() to your model + npu_model oracle to fill the TODO columns")
    return 0


CMDS = {"validate": cmd_validate, "opcodes": cmd_opcodes, "capability": cmd_capability,
        "diversity": cmd_diversity, "present": cmd_present, "run": cmd_run}


def main() -> int:
    cmd = sys.argv[1] if len(sys.argv) > 1 else "validate"
    if cmd not in CMDS:
        print(f"usage: python ablation.py [{'|'.join(CMDS)}]"); return 2
    return CMDS[cmd]()


if __name__ == "__main__":
    sys.exit(main())
