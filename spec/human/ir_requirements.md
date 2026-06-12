# What the IR Must Abstract and Represent (derived from this spec)

> **Purpose.** This document does **not** define or implement an IR. It states, for
> each part of the architecture spec, **what an IR must be able to represent** to
> host this design — so that the spec can eventually be lowered mechanically into
> whatever IR we choose. It uses `mvp-lhwir` (github.com/copparihollmann/mvp-lhwir)
> as a concrete reference point and marks, honestly, where that prototype already
> covers a concept (**has**), partially covers it (**partial**), or does not
> (**missing**). The IR will change; these are *requirements*, not a design.

## 1. Abstraction-level relationship

This spec is an **architectural contract** (the programmer-/compiler-visible
"what"). `mvp-lhwir` is a **structural + microarchitecture + verification IR** (the
"how it is built and checked": modules, ports, edges, FSM, schedule, pipeline,
assume/guarantee contracts, constraints, flavors, and an executable spec-runner
with an L0–L11 verification ladder). They are **different levels of the Y-chart**
(Gajski–Kuhn: behavioral vs structural), not competitors.

The IR we eventually want should therefore be **multi-level** (MLIR-style dialect
layering), with **refinement/lowering** between levels and **provenance carried on
every node** so traceability survives lowering:

```
 Arch / semantic level   ← what THIS spec defines (ISA op-semantics, numerics,
   (new dialect(s))         memory model, programmer's model, workloads, schedule contract)
        │  lower (refine)
 Structural / microarch  ← what mvp-lhwir already prototypes (modules, ports,
   (≈ mvp-lhwir)            edges, fsm, schedule, pipeline, contracts, constraints, flavors)
        │  lower
 RTL                     ← out of scope of the spec
```

## 2. Concept → IR-representation requirements

| Spec concept (section) | What the IR MUST be able to abstract/represent | mvp-lhwir today |
|---|---|---|
| **Operation semantics — ISA** (§6, `isa.yaml`) | First-class **op-semantics** objects: operands + typed operand classes, preconditions/postconditions, state-transfer function, side effects, completion/latency class, legal/illegal cases. Not just an opcode list — the *semantics* must be representable (and executable). | **missing** — has `fsm_state.drives`/`bind_*` for control, but no instruction/op-semantics node |
| **Data types & numerics** (§7, `numerics.md`) | Typed numeric attributes: dtype (fp8_e4m3/bf16/…), bit-layout, **rounding mode, saturation, NaN/Inf, accumulation precision, conversion/quant rules**, and a numeric-semantics function. | **partial** — only bit-widths on ports (`type_name="sint8"`); no rounding/saturation/accumulation semantics |
| **Architectural state** (§4, `state.yaml`) | Aggregate state types: **register files** (count×shape), **memories** (capacity, banking, granularity), **architectural buffers**, **flags/sync flags**, CSRs — with reset semantics and visibility (software-visible vs internal). | **partial** — has `state`, `counter`; no register-file/memory/bank aggregate types |
| **Memory model** (§8) | **Address spaces**, address composition, banking + **conflict policy** (the 32-byte-line exclusion), alignment/granularity, **ordering/consistency rules**, DMA movement + completion/sync semantics. A memory-consistency contract, not just wires. | **missing** — `data_edge.latency` models movement timing, but no address space / consistency / bank-conflict model |
| **Execution & scheduling** (§9) | **Static schedule** with latency classes, **single-issue occupancy contract** (over-issue = illegal), delay slots, barriers/fences (`dma.wait`), determinism. Distinguish functional ordering vs architectural timing. | **has (good fit)** — `schedule`/`phase`/`counter`/`pipeline`/`stream_skew`; needs the occupancy/latency-class contract added |
| **Dataflow & tensor movement** (§9.2) | A **movement graph** (edge = which op moves data across which hub) **plus tensor/tile layout types** (32×32 tile, BF16 register-pair split, tiled tensor layout) and token semantics. | **partial** — `data_edge`/`result_edge`/`control_fanout`/`drain_order`/`stream_skew` cover movement+timing; **no tensor/tile layout type** |
| **Components** (§10) | **module/unit/instance** hierarchy with architectural responsibility and visible state; topology as a *constraint*, not fixed structure. | **has** — `module_def`/`instance`/`param` |
| **Interfaces & protocols** (§11) | **ports/channels**, valid/ready or transaction protocol, backpressure, reset, **assume/guarantee contracts**, clock/reset domains. | **has (strong)** — `port`/`protocol`/`assumes`/`guarantees`/`instance_assumes` |
| **Constraints / PPA / perf** (§12, §13) | Numeric **constraint** objects (hard vs target vs assumption), rooflines, PPA budgets, with comparators and metric paths. | **has (strong)** — `constraint`/`constraint_def` |
| **Design-space knobs** (§16, `design_space.yaml`) | **parameter / transform-target** objects + a set of **verified transforms** that mutate the IR (scale mesh, change dataflow, bank memory, pipeline depth), each tagged arch/uarch/sw-compat. | **has (strong)** — `flavor`/`flavor_def` + transform primitives/passes |
| **Verification obligations** (§14, `test_obligations.yaml`) | **test-intent / coverage-goal** objects, protocol assertions, equivalence modes, oracle references. | **has (strong)** — `protocol`/`invariant`/`scenario_hint` + the L0–L11 ladder |
| **Programmer's model / SW interface** (§5, `sw_interface.md`) | MMIO/CSR **register map**, start/done/status, interrupt-vs-poll, IMEM image, ABI — a software-visible control surface object. | **missing** — no register-map / control-surface abstraction |
| **Provenance & claims** (`provenance.yaml`) | **Every IR node should carry** `id / status (normative|reference|assumption|open) / source / owner` so traceability and the normative-vs-reference distinction survive lowering. | **missing** — IR nodes have no provenance/status metadata |

## 3. Cross-cutting requirements the IR must satisfy

1. **Executable semantics (golden).** The IR must be *runnable* as a cycle/functional golden — like `npu_model` (this spec's S3 oracle) and like `mvp-lhwir`'s spec-runner. The op-semantics and memory model must be evaluatable, not just drawn.
2. **Parametric / symbolic values.** Shapes, latencies, phase cycles must support symbolic expressions (e.g., `cycles_expr="K+M+N-2"`) — `mvp-lhwir` already does this.
3. **Provenance + status on every node** (see §2 last row) — carry the claim ledger into the IR.
4. **Refinement contracts between levels.** Lowering from arch→structural must be checkable (assume/guarantee at each boundary), as `mvp-lhwir`'s `instance_assumes ↔ guarantees` (L2) demonstrates — extended upward to "this structural module *implements* this op-semantics."
5. **Multi-view consistency.** A single design projected into views (dataflow, schedule, FSM, contract, protocol, …) that must stay cross-consistent — `mvp-lhwir`'s 10-view + ladder is a good model; add **semantic** and **numeric** views for the arch level.
6. **Design-space transforms as first-class, verified operations** — `mvp-lhwir`'s transform primitives/passes are the right shape; the arch level adds knobs like dataflow and tile shape.

## 4. Net

`mvp-lhwir` is a strong prototype of the **structural/microarch + verification**
dialect (the bottom of the stack). What this spec adds — and what the IR must grow
to abstract — is the **architectural-semantic layer**: instruction/op semantics,
numeric semantics, the memory-consistency model, the software-visible control
surface, tensor/tile layout types, and provenance metadata. The clean path is a
**layered IR with refinement**, where this spec's structured family (`spec/*.yaml`)
is the source from which the top dialect is generated, and `mvp-lhwir`-style
structural IR is one lowering target below it.
