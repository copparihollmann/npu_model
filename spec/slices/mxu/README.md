# MXU slice — first e2e driving subdesign

A **self-contained, validating** slice of the Atlas NPU spec restricted to the
**Matrix Execution Unit** and its lower-in/out interface (MRF). Carved from the full
bundle (`../../`) to be the first subdesign we drive end to end:
**spec → IR → cost model → RTL**, measured against the oracle (`npu_model`).

## Start here

1. `human/MXU.md` — the narrative. Reads the register file in **two layers**:
   **Layer A** = a simple datapath with registers we lower in/out (the contract);
   **Layer B** = `npu_model`'s banked register file (the oracle we're graded against).
2. `programs/gemm_32x32x32.s` — the minimal driver (one 32×32×32 tile, 160 cycles).
3. `machine/*.yaml` — the typed node graph (MXU closure only).

## Validate

```
python spec/slices/mxu/validate.py                  # exit 0 = slice graph valid (66 nodes)
python spec/slices/mxu/validate.py --ssa mxu_matmul # mechanical def-use extraction
```

The validator reuses the **shared** core meta-model (`../../schema/core.py`) — only the
*data* is sliced, so the type system is identical to the full bundle's.

## What's here

```
human/MXU.md              datapath contract + oracle layer + driver walkthrough
machine/types.yaml        9 type nodes  (fp8_e4m3, bf16, fp16, e8m0, tiles, scale)
machine/state.yaml        4 state nodes (mrf, weight slots, accumulators, erf)
machine/spec.yaml         params + 3 timing classes + 2 units + 4 dataflow edges
machine/isa.yaml          14 op nodes  (push/matmul/pop, operands+effect → SSA)
machine/constraints.yaml  Layer A (normative) + Layer B (reference/oracle)
machine/kernels.yaml      kernel.gemm + mxu_profile (npu dialect)
dialects/npu.yaml         kernel + mxu_profile kinds
programs/gemm_32x32x32.s  the e2e driving program
manifest.yaml             slice index, lineage, register-file framing, node closure
```

## Scope

In: the MXU (mxu0 systolic / mxu1 inner-product), weight/accumulator registers, MRF as
the in/out boundary, FP8×FP8→BF16 numerics, the 14 ops, and the conflict-discipline the
oracle enforces. Out: VMEM/DMA, scalar core, VPU/XLU/LSU, control flow — those stay in
the full bundle (`spec/`). The MXU never reads VMEM or DRAM directly, so this boundary is
architecturally clean.
