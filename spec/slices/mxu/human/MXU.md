# MXU — the first end-to-end driving subdesign

This slice isolates **everything (human + machine) about the Matrix Execution Unit**
into one self-contained, validating bundle. It exists to be the **first e2e driving
example** for the flow: a small, well-bounded subdesign we can take from
spec → IR → cost model → RTL and measure against the oracle (`npu_model`).

The register-file design is presented in **two layers** (deliberately):

- **Layer A — the simple datapath contract** (`status: normative`). A clean register
  model with registers we lower *in* and *out*. This is what the IR and a generated
  design should reason about.
- **Layer B — the oracle behavior** (`status: reference`). The actual banking /
  conflict machinery `npu_model` implements — the "annoyingly complex register file."
  We don't *design to* it, but every schedule we emit is *graded against* it, so it is
  recorded faithfully (`machine/constraints.yaml`).

---

## 1. Layer A — the simple datapath

The MXU is a **weight-stationary** tile engine: `FP8(e4m3) × FP8(e4m3) → BF16`
(internal product/accumulate precision ≥ fp16). There are two instances:

| Unit | Fabric | matmul latency | push/pop latency |
|------|--------|---------------:|-----------------:|
| `unit.mxu0` | systolic array | **96 cyc** (`timing.mxu0_matmul`) | 32 cyc |
| `unit.mxu1` | inner-product tree | **35 cyc** (`timing.mxu1_matmul`) | 32 cyc |

Three register files are the whole architectural surface (`machine/state.yaml`):

```
        lower in (push)                compute                 lower out (pop)
 MRF ───────────────────────►  weight slots  ─┐
 (m0..m63, 32x32 tiles)         (w0,w1 / MXU)  ├─► vmatmul ─► accumulator ──► MRF
 MRF ──(activation, read)──────────────────────┘            (acc0,acc1 / MXU)
```

- **`state.mrf`** — shared 32×32 tile staging RF (m0..m63). The **only** path in/out of
  the MXU: activations and weights are *read from* MRF; results are *written back to*
  MRF. The MXU never touches VMEM or DRAM.
- **`state.mxu_weight_slots`** — MXU-local weight registers (2/MXU). Loaded by
  `vmatpush.weight`, then **resident**: reused by every `vmatmul` until overwritten.
- **`state.mxu_accum_buffers`** — MXU-local BF16 accumulators (2/MXU). Target of
  `vmatmul[.acc]`; seeded by `vmatpush.acc`; drained by `vmatpop`.

### The 14 ops (4 families)

All are VR-format, opcode `0b1110111`, `funct7` enumerating the op (low bit = MXU id).
Each is **blocking**: it holds its MXU issue port for its full latency class.

| Family | Ops | Effect (SSA) |
|--------|-----|--------------|
| `mxu_weight_push` | `vmatpush.weight.mxu{0,1}` | `wd ← push(vs1)` — MRF tile → weight slot |
| `mxu_acc_push` | `vmatpush.acc.{fp8,bf16}.mxu{0,1}` | `ad ← dequantize_bf16(vs1)` / `copy(vs1)` — seed accumulator |
| `mxu_matmul` | `vmatmul[.acc].mxu{0,1}` | `vd ← matmul(vs1,vs2)` / `vd ← add(vd, matmul(vs1,vs2))` |
| `mxu_acc_pop` | `vmatpop.{fp8,bf16}.acc.mxu{0,1}` | `vd ← quantize_fp8_trunc(vs2,es1)` / `copy(vs2)` — drain to MRF |

`vmatmul.*` **overwrites** the accumulator; `vmatmul.acc.*` **accumulates** into it
(`vd` is `inout` — a read-modify-write, the textbook SSA def-with-prior-use).

**Selector encoding (normative, `npu_spec` §04/§06).** Each MXU has 2 weight slots and 2
accumulators, selected by a single encoded bit: weight slot = `vd` (push) / `vs2[0]`
(matmul); accumulator = `vd[0]` (push/matmul) / `vs2[0]` (pop). The rest of those fields are
reserved-zero (`vd[5:1]=0`, `vs2[5:1]=0`). BF16 tiles name the **low** register of the pair
`{m[r], m[r+1]}`; `r=63` is illegal. Each op's authoritative Verilog semantic is carried in
`attrs.sem`, e.g. `mxu0.acc[vd[0]] = mxu0.acc[vd[0]] + m[vs1] @ mxu0.w[vs2[0]]`.

The def-use you get mechanically from the node graph:

```
$ python spec/slices/mxu/validate.py --ssa mxu_matmul
  op.vmatmul.mxu0:      uses=['vs1','vs2']        defs=['vd']    vd <- matmul(vs1, vs2)
  op.vmatmul.acc.mxu0:  uses=['vd','vs1','vs2']   defs=['vd']    vd <- add(vd, matmul(vs1, vs2))
```

### Numerics

- Operands: FP8 e4m3 (`type.fp8_e4m3`, finite, ±448 max, saturating).
- Accumulate/output: BF16 (`type.bf16`); internal intermediate ≥ fp16 (`type.fp16`).
- `vmatpop.fp8` requantizes BF16 → FP8 using an ERF scale (`type.scale`, `state.erf`).
  **Open:** scale is E8M0 power-of-2 (`npu_spec`) vs raw-uint8 multiply (`npu_model`) —
  `constraint.oracle.scale_semantics`. Ratify before freeze.

---

## 2. Layer B — the oracle behavior (the complex register file)

`npu_model/hardware/bank_conflict.py` + `mxu.py` implement the register file as banked
SRAM with a runtime conflict checker. The contract above abstracts this away; here is
what a schedule is actually graded against (`machine/constraints.yaml`, `status: reference`):

- **MRF = one SRAM bank per register** (`constraint.oracle.mrf_bank_per_register`): m0..m63
  each its own bank; two in-flight ops sharing any register index collide.
- **BF16 register pairing** (`constraint.oracle.bf16_register_pairing`): a 32×32 BF16 MRF
  tile occupies a register **pair** `{r, r+1}`; the conflict set covers both indices.
- **MXU buffers = one bank per MXU id** (`constraint.oracle.mxu_buf_bank_per_mxu`): the
  weight buffer and accumulator buffer each have a single conflict bank per MXU; both
  physical slots of an MXU share it.
- **Acquire / release discipline** (`constraint.oracle.conflict_acquire_release`): an op
  acquires its MRF + weight + acc banks at issue and releases at completion. A concurrent
  access to a held bank raises `BankConflictError` **immediately** — a runtime-detected
  *schedule violation*, never a stall. Software must avoid it statically.
- **Authoritative latencies** (`constraint.oracle.matmul_latency`): from
  `MXU_OP_LATENCIES` — matmul mxu0=96, mxu1=35, push/pop=32. (`default.py`'s `mxu=32` is a
  fallback the per-op dict overrides.) `npu_spec` confirms all of these, *and* that push/pop
  is 32 for **both** the 1024-B FP8 and 2048-B BF16 tiles.
- **Over-issue: stall vs error** (`constraint.oracle.over_issue_stall_vs_error`, **open**):
  `npu_spec` §04 says the frontend **stalls** when the target unit can't accept (hardware
  backpressure); `npu_model` **raises** `RuntimeError` on over-issue to a busy non-DMA unit
  (`idu.py:210`), assuming a correct static schedule. `dma.wait` is the one case the model
  stalls. To ratify before freeze.

> **Reconciliation status vs `npu_spec`:** latencies, `funct7` (77/00–77/13), tile/slot/acc
> sizes, weight-stationary + tensor-register-only accumulator path all **agree**. This slice
> additionally carries the `npu_spec` normative encoding (selectors, reserved-zero, BF16 pair
> legality) and the one open discrepancy above.

**Why two layers:** the professor's direction is a *very simple datapath with registers we
can lower in and out*, reacting to this banked design being hard to start from. Layer A is
that simple target; Layer B keeps us honest against the oracle so the simplification is
*measured*, not *assumed* (the ≤10% abstraction-overhead bar applies here — Layer A must
not cost >10% vs what Layer B admits).

---

## 3. The driving example

`programs/gemm_32x32x32.s` is the minimal e2e MXU program — one 32×32×32 tile,
weight-stationary, with the explicit static `delay`s the model expects:

```
vmatpush.weight.mxu0 w0, m2      # B -> weight slot        (32)
delay 32
vmatmul.mxu0 acc0, m0, w0        # acc0 = A @ B            (96)
delay 96
vmatpop.bf16.acc.mxu0 m6, acc0   # C = acc0 -> MRF         (32)
delay 32
# total 160 cycles
```

A K-tiled accumulate variant (in the same file) shows the `inout` accumulator and the
weight swap across K. Because each `delay` fully covers its op's latency, the schedule is
bank-conflict-free by construction — the simplest possible Layer-A↔Layer-B agreement.

`kernel.gemm` (`machine/kernels.yaml`) is the structured handle for this driver: it
references the three ops it uses and declares MRF as its in/out boundary.

---

## 4. What this slice contains / how to drive it

```
spec/slices/mxu/
  human/MXU.md          ← this file (datapath contract + oracle layer + driver)
  machine/              ← typed node graph (MXU closure only): types, state, spec
                          (params/timing/units/edges), isa (14 ops), constraints
                          (Layer A + Layer B), kernels (gemm + mxu_profile)
  dialects/npu.yaml     ← kernel + mxu_profile kinds (self-contained copy)
  programs/gemm_32x32x32.s
  validate.py           ← standalone validator (reuses ../../schema/core.py)
  manifest.yaml         ← slice index + lineage back to the full bundle
```

Validate / extract SSA:
```
python spec/slices/mxu/validate.py                  # 0 = slice graph valid
python spec/slices/mxu/validate.py --ssa mxu_matmul # def-use edges
```

**Boundary (intentionally out of scope):** VMEM/DMA staging, the scalar core, the VPU/XLU,
control flow. Those live in the full bundle (`spec/`). This slice is the MXU and exactly
its lower-in/out interface (MRF), nothing more — so it can be the focused first target.
