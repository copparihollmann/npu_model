# Atlas NPU Subset — S0: Natural-Language Design Intent

> **Maturity S0** — informal design goal and use case. This is the top of the
> pipeline; everything else refines from here. Where the repo does not record the
> original rationale, intent is **reconstructed** from the implemented design and
> tagged `assumption`/`open` (the genuine S0 brainstorm is not a repo artifact).

## Pipeline (canonical)

```
 S0  NL design intent (this file) + workload characterization (workloads.md)
  │
 S1  human-readable architecture spec        (SPEC.md)
  │
 S2  structured machine-readable spec         (spec/*.yaml)   ← source of truth for values/claims
  │   (S3 = the spec becomes EXECUTABLE: it carries reference semantics)
  │
 S4  IR-ready  →  IR / LHWIR
  │
  └─► generated FROM the IR: functional models · cost models · tests ·
                             scoreboards · assertions · RTL-generation tasks

   npu_model  ── side ORACLE (not a pipeline stage) ──────────────────────┐
     (a) SOURCE that informed this spec (provenance)                       │
     (b) EVALUATION oracle: golden for functional-equivalence checks ──────┘
         on the IR-generated models and on the RTL
```

The IR is generated from the **spec**, never "after `npu_model`." `npu_model`
**evaluates execution** and was an authoring source — it sits beside the chain.

## What this is

The **Atlas NPU subset** is an accelerator for **edge-deployed VLA / Physical-AI
inference** with low-precision, tile-oriented tensor execution. A small in-order
scalar control core statically sequences a 32×32-tile datapath (two matrix
engines, a vector unit, a transpose unit, a multi-channel DMA) that streams FP8
operands and accumulates in BF16, staging tiles in an on-chip scratchpad backed by
off-chip DRAM. *(reconstructed from `README.md`, `npu_spec/01_introduction`; status: reference)*

## Why (design intent)

- **Target the math that dominates VLA/Physical-AI inference** — GEMM and attention,
  plus the surrounding norm/softmax/activation/elementwise glue (see `workloads.md`). *(reference)*
- **Trade precision for throughput** — FP8 operands, BF16 accumulation — the standard
  edge-inference DSA lever. *(reference)*
- **Be deterministic and cheap to control** — static scheduling, no dynamic
  dependency tracking, so timing is predictable and the control hardware is small;
  the compiler/programmer owns correctness of overlap and spacing. *(reference)*
- **Keep one narrow async boundary** — DRAM↔VMEM via DMA; everything on-chip is
  deterministic. *(reference)*

## Goals

- Run the workload set in `workloads.md` correctly within the stated numeric tolerances.
- Expose an architecture precise enough to implement from (this spec) and to compare
  RTL-generation approaches against a golden (`npu_model`).
- Be iterable: NL → spec → structured spec → IR, with provenance preserved.

## Non-goals

- General-purpose CPU replacement.
- Training (forward-inference only).
- Dynamic out-of-order execution / dynamic dependency scoreboarding.
- A fixed RTL microarchitecture (the spec constrains, it does not implement).

## Target integration & software stack

- **Integration:** a memory-mapped inference accelerator; host loads an IMEM image,
  runs to halt, polls for completion. The concrete host transport (MMIO/CSR block,
  interrupts) is an **open decision** — see `sw_interface.md`. *(open)*
- **Software stack:** static assembly programs today (in-repo assembler); a compiler/
  scheduler is the eventual front-end (out of scope here; `downstream.yaml` artifact G). *(reference)*
- **Implementation target:** reference RTL + the executable model (`npu_model`).

## Intent gaps (not in the repo → tagged; not blockers)

| Gap | Status | Disposition |
|-----|--------|-------------|
| Original design rationale / decision history (true S0 brainstorm) | assumption | reconstructed from the implemented design |
| Product targets (FPS/latency, power & energy budget, edge platform) | open | to be set with a product owner |
| Full-model op-frequency profile (how kernels compose a whole VLA/Gemma run) | open | `workloads.md` uses representative layers, not full-model weighting |
| Batch-size / sequence-length distributions | assumption | programs exercise small / batch-1; broaden when known |
