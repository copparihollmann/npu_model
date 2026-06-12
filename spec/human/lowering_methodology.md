# Lowering methodology — the human→machine→IR contract

Formal companion to `../LOWERING.md` (the narrative). This file is the precise, lint-tracked contract for
the two lowerings. It depends on `META.md` (the node envelope, the 18 core kinds, the effect grammar),
`ir_mapping.md` (the machine→IR mapping), and `ir_requirements.md` (what an IR must abstract). It adds no
machine nodes; it defines the *procedure* and its *verification gates*.

## A. The structured way to define the spec

Every machine entry is a node `{id, kind, status, provenance, doc, attrs, refs, ext, operands, effect}`
(see `../META.md`). Authoring is: pick the kind, fill its required attrs, wire refs by-id, set `status`.
The per-kind lowering targets — what prose element each kind captures and the gate that checks it:

| Prose element in `SPEC.md` | machine kind | required attrs | verified by |
|---|---|---|---|
| a data type / tile / operand class | `type` | `category` | graph validity |
| a fixed scalar parameter | `param` | `value` | lint critical-constant (if listed) |
| an execution unit | `unit` | `role` | refs resolve (visible_state, latency_class) |
| a register file / memory / flag | `state` | `category, visibility` | refs resolve (element_type, per_unit) |
| an instruction / operation | `op` | `family` (+ `operands`, `effect`) | `--ssa` def-use extraction |
| a stated latency | `timing_class` | — (`cycles`/`formula`) | lint critical-constant (if listed) |
| a data-movement path | `edge` | — (refs src/dst/via) | refs resolve |
| a hard limit / functional rule / target | `constraint` | `kind_of` | lint critical-constant (if listed) |
| a design-space parameter | `knob` | — (`choices`/`range`, refs.varies) | refs resolve |
| a verification obligation | `test_intent` / `coverage_goal` | `test_type` / `text` | refs resolve |
| a provenance statement | `claim` / `source` | `text` / `type,trust` | refs resolve |
| an open question / axis / artifact | `decision` / `axis` / `artifact` | `question` / `title` / `name,owner` | — |

Accelerator-specific concepts are added as **dialect** kinds (`dialects/npu.yaml`: `kernel`, `mxu_profile`),
never by changing the core. A node’s `status` records maturity (`normative` is the contract; `reference`,
`assumption`, `open` are not yet binding).

## B. Lowering #1 — human → machine (S1 → S2)

### B.1 Deterministic part (the contract)
Map each prose element to its kind per the table in §A, preserving every value exactly. This part is
**not** creative. It is gated by three checks, in order:

1. **`python spec/schema/load.py`** — graph validity: ids unique + prefixed by kind; every `refs` id
   resolves; operand `type`/`location` resolve to `type`/`state` nodes; `effect.def` is a def/inout operand;
   effect operand names are declared.
2. **`python spec/lint.py`** — human↔machine agreement: each listed critical constant
   (`timing.mxu0_matmul=96`, `timing.mxu1_matmul=35`, `timing.vpu_col_reduction=130`, `param.mrf.count=64`,
   `param.xrf.count=32`, `param.dma.channels=8`, `constraint.vmem.capacity="1 MiB"`,
   `constraint.imem.capacity="128 KiB"`) must have the right value in the node AND appear in `SPEC.md`
   (normalized match). Extend this list when a new value becomes load-bearing.
3. **Oracle cross-check vs `npu_model`** — for any semantics/latency/encoding claim, reconcile against the
   executable model (and `npu_spec/`). This gate is what catches real defects, not just drift; it has
   already found the `dma.config`/`dma.wait` funct7 collision, PC byte-vs-index, and the ERF scale
   discrepancy. Discrepancies are recorded as `claim`/`constraint` nodes with `status: open`, never silently
   resolved.

### B.2 Creative surface (b): lowering-method freedom
Freedom in *how* prose becomes nodes — encoding and framing choices, with values pinned. LLM assistance is
appropriate here precisely because §B.1's three gates make a proposal checkable. The canonical pattern is
**layered framing**: when a unit admits both a clean datapath contract and a more detailed oracle behavior,
author the datapath layer `status: normative` and the oracle layer `status: reference` (e.g. the MXU
register-file framing). Same values, two legitimate encodings; the method chooses, the gates verify.

## C. Lowering #2 — machine → IR (S2 → S4)

### C.1 Mechanical part
Per `ir_mapping.md`: walk nodes, emit IR objects, resolve refs to IR handles. Op operands give def-use
(`use`→use edge, `def`/`inout`→def, `inout`→read-modify-write); `effect` exprs give the op-body value DAG;
`type`/`location` place SSA values. `python spec/schema/load.py --ssa <family>` already emits this for op
families today. The invariants in §B.1.1 are exactly what make the walk safe.

### C.2 Required IR layer (next build)
The target IR is multi-dialect refinement: **arch/semantic dialect (this spec) → `mvp-lhwir` structural →
RTL**. `ir_requirements.md` enumerates what the arch dialect must add above `mvp-lhwir`: first-class
op-semantics, numeric semantics (rounding/saturation/accumulation/NaN), a memory model (address spaces,
banking, 32-byte-line conflict policy, consistency), the programmer’s-model/SW interface (CSR/MMIO,
start/done/status, IMEM image, ABI), and **provenance carriage** (every IR node keeps `status`+`source`+
`refs.about`). Each lowering boundary uses assume/guarantee refinement contracts (as `mvp-lhwir` already
does for `assumes`↔`guarantees`).

## D. Creative surface (a): hardware design-space flavors
Distinct from §B.2. Freedom in *what hardware* the contract permits, enumerated as `knob` nodes in
`machine/design_space.yaml` (MXU count/topology, dataflow, VMEM banking, pipeline depth, VPU lanes) and
realized as `mvp-lhwir` `flavor`/`flavor_def` verified transforms (`ScaleMesh`, `InsertPipelineStage`, …).
Each knob declares `choices`/`range`, `refs.varies` (the nodes a transform edits), and `sw_compatible` (is
the flavor transparent to software). The creative space is therefore bounded and machine-readable.

## E. What is never creative
The normative contract: parameter values, ISA op semantics + encodings, latencies, and legality
constraints. These are fixed by §B.1’s three gates. Surfaces (a) and (b) may not alter them.

## F. Run the gates
```
python spec/schema/load.py            # graph validity → "N nodes, OK (graph valid)"
python spec/schema/load.py --ssa mxu_matmul   # machine→IR def-use for a family
python spec/lint.py                   # human↔machine agreement + file-set → "0 fail"
```
