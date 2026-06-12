# From human spec to IR: a verifiable two-level lowering for the Atlas NPU

**Purpose.** This document explains what I want to do and where the work stands. The thesis is that a
hardware specification should be treated as a *lowering pipeline*, not a single document: a human-readable
spec is lowered to a typed machine spec, and the machine spec is the entry point into a compiler IR. Each
lowering is made **explicit and verifiable** where it carries a contract, and is given **bounded creative
surfaces** ("flavors") where design or method freedom is legitimate. The focus here is the two levels
**human → machine → IR**.

This is grounded in the spec bundle already built under `spec/` (a from-scratch architecture spec for the
Atlas NPU subset — a statically-scheduled, FP8/BF16, tile-oriented accelerator for edge VLA / physical-AI
inference). Nothing below is aspirational unless marked "next"; the current state is real and validates.

---

## 1. The pipeline

```
 S0  NL design intent + workload characterization        human/00_intent.md, workloads.md
  │   (human → machine lowering)
 S1  human-readable architecture spec                    human/SPEC.md (§0–§22)
  │
 S2  structured machine-readable spec  ◄── source of truth for values/claims
  │      machine/*.yaml  — a typed node graph (325 nodes)
  │   (machine → IR lowering)
 S4  IR-ready → IR / LHWIR
  └─► generated FROM the IR: functional models · cost models · tests · scoreboards · RTL-gen tasks

 npu_model = side ORACLE (not a pipeline stage): it informed the spec AND is the golden for
             functional-equivalence checks on everything generated downstream.
```

The two lowerings this document is about: **human→machine** (S1→S2) and **machine→IR** (S2→S4). The machine
spec is deliberately the *entry point into the IR* — it is already a typed graph that lowers mechanically
to SSA (Section 4).

---

## 2. What the human spec contains, per module

`human/SPEC.md` is the canonical narrative (§0–§22) plus annexes (`numerics.md`, `sw_interface.md`,
`ir_requirements.md`, `ir_mapping.md`) and the front matter (`00_intent.md`, `workloads.md`). Coverage by
module:

| Module | Where (SPEC.md §§) | Content kinds | Representative pin | Maturity / open |
|---|---|---|---|---|
| **MXU** (matrix) | §6 semantics, §9 scheduling+bank-conflict, §4 state, §10 component | op-semantics, latencies, state, legality constraints | `vmatmul.acc.mxu0: vd ← add(vd, matmul(vs1,vs2))`; weight-stationary FP8×FP8→BF16; **96 cyc** (mxu0) / 35 (mxu1) | normative; 1 open: over-issue *stall vs error* |
| **VPU** (vector) | §6 (binary/unary/reduction/convert/vli — ~29 ops), §7 numerics | op-semantics, latencies, numeric rules | `vredsum.bf16: vd ← redsum_col(vs1)`, **130 cyc** col-reduction; transcendentals (exp/sin/tanh/…) | normative |
| **LSU** (load/store) | §5 alignment, §6 (lb…sw, vload/vstore) | op-semantics, blocking, alignment | `lw: rd ← load(mem, rs1, imm)`, blocking, align=4 | normative |
| **DMA** | §5 per-channel model, §6 (config/load/store/wait) | op-semantics, channel model, blocking | 8 channels; `dma.wait.chN` is one of only two blocking primitives | normative; encoding reconciled vs `npu_spec` |
| **scalar / control** | §6 (RV32-style ALU, branches, CSR), §5 host lifecycle | op-semantics (prose for control flow), 2 delay slots, control plane | RV32I-ish ALU + `beq/jal` with **2 delay slots**; `delay imm` is the other blocking primitive | normative; control flow stays prose (Section 4) |
| **memory / register files** | §4 state, §8 memory model | state objects, banking, conflict policy | MRF 64×1024 B; VMEM **1 MiB / 8 banks**, conflict at **32-byte-LINE** granularity; DRAM 16 GiB | normative; 3 ratification items (VMEM bus width, DRAM aperture, scalar width) |
| **ISA (whole)** | §6 (131 ops, fixed 32-bit), encoding | encoding tables, semantics | 131-instruction RV32-style tensor ISA; recommended bit-layout is normative-values / implementer-defined-layout (the model's packers are buggy and flagged) | normative values; bit-layout open |

The rest of the section map: §0 document-set/source-of-truth, §1 overview, §2 provenance, §3 workloads,
§7 data types & numerics, §11 interfaces/protocols, §12–13 constraints & QoI targets, §14 verification
obligations, §15 error handling, §16 design-space parameters, §17 mapping to LHWIR, §18 the structured
machine spec, §19 claim schema, §20 out-of-scope, §21 downstream artifacts + decision register, §22
cross-cutting concerns. §0 also defines reading bundles; an "NPU-required coverage map" and a review
checklist close the file.

The character of the human spec is therefore **mostly normative contract** (semantics, latencies, state,
legality) with a small set of explicitly-marked **open ratification items** — none of which need new
semantics, only an architect's decision.

---

## 3. The machine spec (S2): a structured way to define the spec

`machine/*.yaml` is one **typed node graph** — the structured form the human spec lowers into. The point of
the structure is that the spec can (a) *validate*, (b) *lower mechanically* to an IR, and (c) *grow by adding
data*, not bespoke prose.

- **One node envelope** for every entry: `{id (namespaced), kind, status, provenance, doc, attrs, refs, ext,
  operands, effect}`. `status ∈ {normative, reference, assumption, open, rejected}` makes maturity
  first-class.
- **A fixed core vocabulary of 18 kinds** (`type param unit state port channel op timing_class edge
  constraint knob test_intent coverage_goal claim source decision axis artifact`) defined in
  `schema/core.py`. The IR depends on this set, so it is versioned and closed.
- **By-id references** are the graph edges (e.g. `op.refs.latency_class → timing_class`, `operand.location →
  state`); every id must resolve or it is an error.
- **Op nodes carry typed `operands` (`direction: def|use|inout`) and a structured `effect`** (a small AST,
  `{def, expr}` with `{op, args}` / `{lit}`). This is what makes def-use mechanical.
- **Extensibility is via registered dialects, never by editing the core.** `dialects/npu.yaml` adds
  `kernel` and `mxu_profile` kinds; a designer extends a dialect.

Today the graph is **325 nodes, 0 errors** (`python spec/schema/load.py`). It is hand-authored; consistency
with the human prose is enforced, not generated (Section 4).

---

## 4. Lowering #2 made explicit: machine → IR (largely mechanical today)

Because every node has a stable id, a kind, resolved refs, and (for ops) typed operands + an effect AST,
lowering to an SSA-style IR is a **mechanical walk**. The contract is `human/ir_mapping.md`; the mapping is
deterministic:

| machine node | IR object |
|---|---|
| `type` | IR type · `state` | IR location · `op` | IR operation |
| operand `use` | SSA use edge · operand `def`/`inout` | SSA def (inout = read-modify-write) |
| `effect {def, expr}` | IR op body (value DAG) · `edge` | dataflow movement · `timing_class` | latency attr |
| `param`/`constraint` | attribute/contract · `knob` | transform-target · `claim` | provenance annotation |

This is not a paper contract: `python spec/schema/load.py --ssa mxu_matmul` emits real def-use **from the
nodes** —

```
op.vmatmul.mxu0:      uses=['vs1','vs2']        defs=['vd']
    vd <- {op: matmul, args: ['vs1','vs2']}
op.vmatmul.acc.mxu0:  uses=['vd','vs1','vs2']   defs=['vd']      # vd inout = accumulate
    vd <- {op: add, args: ['vd', {op: matmul, args: ['vs1','vs2']}]}
```

So **the machine spec is the IR entry point** for the op graph and its def-use today. What is *not* yet
mechanical, and is the next build, is the **arch/semantic dialect layer** the IR needs above the existing
structural IR `mvp-lhwir` (`/scratch2/agustin/mvp-lhwir`, an xDSL structural/microarch + verification IR).
Per `ir_requirements.md`, mvp-lhwir is strong on modules/ports/edges/FSM/schedule/protocol/constraints/
flavors but is **missing**: first-class op-semantics, numeric semantics (rounding/saturation/accumulation),
a memory model (address spaces, banking, consistency), the programmer's-model/SW interface, and provenance
carriage. The target is MLIR-style refinement: *arch/semantic dialect (this spec) → mvp-lhwir structural →
RTL*, with assume/guarantee refinement contracts at each boundary.

---

## 5. Lowering #1 made explicit: human → machine (the open frontier)

This is where the real methodological work is. Today both `human/` and `machine/` are **hand-authored**;
`lint.py` keeps them in agreement (8 critical-constant checks, e.g. `timing.mxu0_matmul=96`,
`param.mrf.count=64`, `constraint.vmem.capacity="1 MiB"`, plus graph validity and the manifest file-set).
The proposal is to make this lowering an explicit, verifiable procedure with three gates and two clearly
delimited creative surfaces.

**Deterministic targets (must be exact).** Each prose element maps to a specific node kind: a stated
latency → `timing_class`, a register file → `state`, an instruction → `op` (with operands + effect), a hard
limit → `constraint`, a parameter → `param`. The values are a contract and are verified by three gates:
1. `schema/load.py` — graph validity (ids unique, refs resolve, operands/effect well-formed).
2. `lint.py` — critical-constant agreement between `machine/*.yaml` and `human/SPEC.md`.
3. **Cross-check against the `npu_model` executable oracle** — the adversarial-agent reconciliation that
   already caught real defects (the `dma.config`/`dma.wait` funct7 collision, PC byte-vs-index, ERF scale
   semantics). This is the gate that makes the lowering *trustworthy*, not just *consistent*.

**Where LLMs fit.** The prose→graph extraction and the candidate-encoding proposals are well-suited to LLM
assistance, *because* the three gates above turn a fuzzy task into a checkable one: the model proposes nodes,
the verifiers reject anything that doesn't validate, doesn't agree with the prose, or doesn't match the
oracle.

---

## 6. The two creative surfaces ("flavors"), explicitly bounded

Creativity is allowed, but only in two named places, each with a defined boundary. Everything else — the
normative contract (constants, ISA semantics, latencies, legality) — is **never** creative; lint + oracle
verify it.

**(a) Hardware design-space flavors.** Legitimate freedom in *what hardware* the contract permits. These are
the `machine/design_space.yaml` `knob` nodes (MXU count/topology, dataflow, VMEM banking, pipeline depth,
VPU lanes) plus `mvp-lhwir`'s `flavor`/`flavor_def` **verified transforms** (e.g. `ScaleMesh`,
`InsertPipelineStage`). A knob declares its `choices`/`range`, what it `varies` (by-id), and whether it is
`sw_compatible` — so the creative space is enumerated and the SW contract tells you which flavors are
transparent to software.

**(b) Lowering-method freedom.** Legitimate freedom in *how* prose becomes the typed graph — the encoding
and framing choices, not the values. The worked example is the MXU register-file framing, captured as two
layers: a **normative** datapath contract (MRF is the only lower-in/out path; weight-stationary; accumulate
is inout) versus a **reference** oracle behavior (the bank-conflict complexity). Same pinned values, two
legitimate framings; the method picks one and marks the other `reference`.

Keeping these two surfaces *distinct and bounded* is the proposal's core: it lets the spec be exact where it
is a contract and open where design or method genuinely has freedom, without the two leaking into each other.

---

## 7. Path of action

**Done.** The S0→S2 scaffold; the typed node graph + schema (`core.py`/`load.py`, 18 kinds + `npu` dialect);
the three-gate consistency (`lint.py` 0 fail, oracle reconciliation); the mechanical machine→IR contract
(`ir_mapping.md`) with a real `--ssa` extraction; and a self-contained **MXU slice**
(`slices/mxu/`, validates standalone with `--ssa mxu_matmul`) as the first end-to-end driving subdesign.

**Next.** (1) Turn human→machine into a tooled, LLM-assisted-but-gated distillation procedure (Section 5).
(2) Build the arch/semantic IR dialect above `mvp-lhwir` (op-semantics, numerics, memory model, SW
interface, provenance), so machine→IR is mechanical end-to-end (Section 4). (3) Extend from the MXU slice to
the other modules. (4) Close the open ratification items (scalar width, VMEM bus, DRAM aperture, host
transport, over-issue stall-vs-error).

**Why it matters.** This makes the spec a first-class, verifiable input to a compiler/cost-model/RTL-gen
pipeline rather than a PDF — the architecture-as-graph that the MXU-first end-to-end direction and the
ISA-vs-RTL-generation research need.

The formal version of the two lowering contracts lives in `human/lowering_methodology.md`.
