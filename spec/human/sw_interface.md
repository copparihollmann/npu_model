# Atlas NPU Subset — Software / Host Interface

> Companion to `SPEC.md` (architecture). This document defines the **software-
> visible interface** at the fidelity the executable model actually provides, and
> explicitly lists the **decisions** that must be made to complete a host driver.
> Everything marked *(model)* is verified against `npu_model`; everything marked
> *(decision)* is required but not yet defined in the ISA.

## 1. Execution model *(model)*

- A program is a sequence of **32-bit instructions** stored in IMEM. The PC is a
  **byte address** starting at 0; `npc = pc + 4`; the instruction fetched is
  `IMEM[pc/4]` (`program.get_instruction`).
- Execution is **single-issue, in-order, statically scheduled**. The hardware does
  no dependency tracking; the program (or compiler) owns timing correctness — see
  `SPEC.md §9` (occupancy rule, `delay`, `dma.wait`) and `§8` (bank conflicts).
- A program **terminates** when it executes `ecall`/`ebreak` (sets `halted`) or
  when the PC runs past the last instruction (`pc ≥ len·4`). *(model)*

## 2. Memory addressing *(model)*

All regions are addressed **from 0** (the `0x2000_0000`-style system-map bases in
`npu_spec` are a future SoC convention, not applied by the model — see
`SPEC.md §5`):

| Region | How software addresses it |
|--------|---------------------------|
| VMEM (1 MiB) | 0-based byte offset; `vload/vstore` use `x[rs1] + (imm<<5)`, 32 B aligned; scalar `lb/sb…` use `x[rs1] + sext(imm)` |
| DRAM (16 GiB) | effective byte addr = `(dma_base<<32) \| offset`; `dma_base` set by `dma.config.chN`; initially 0 |
| IMEM (128 KiB) | host-loaded image; PC indexes it |

Program inputs are staged into DRAM before run (the model's `memory_regions`
load to `write_dram(offset, data)` at 0-based offsets). Typical flow: host stages
tensors in DRAM → program `dma.load`s them into VMEM → `vload` into MRF → compute
→ `vmatpop`/`vstore` to VMEM → `dma.store` to DRAM → host reads DRAM.

## 3. Instruction image / assembler *(model)*

- The IMEM image is the array of per-instruction 32-bit words. **Caveat:** the
  model's bit-packing (`to_bytecode`) is buggy/unvalidated and the simulator is
  mnemonic-driven (`SPEC.md §6`). The **field values** are authoritative; an
  assembler/decoder MUST adopt a clean bit layout (recommended table in §6) and
  resolve the `dma.config`/`dma.wait` encoding overlap.
- Assembler conveniences (`util/converter.py`): pseudo-ops **`li`** (→ `lui`+`addi`
  or a single `addi`) and **`nop`** (→ `addi x0,x0,0`); **labels** end in `:` and
  resolve to a byte displacement `(label_index − pc_index)·4`. Registers are
  written `x0..x31`, `m0..m63`, `e0..e31`, `w0/w1`, `acc0/acc1`.

## 4. Register/ABI conventions *(model + convention)*

- `x0` is hardwired 0 (writes dropped) — **enforced**. *(model)*
- A full **RV32I ABI is specified** in `npu_spec/06 §1` *(reference)*: `x0=zero,
  x1=ra, x2=sp (callee), x3=gp, x4=tp, x5-7/28-31=t0-t6 (caller), x8-9/18-27=s0-s11
  (callee), x10-17=a0-a7 (caller)`; and **all tensor/scale/MXU state**
  (`m0-m63, e0-e31, mxu*.w*, mxu*.acc*, dma.base`) is **Caller-saved**. The hardware
  imposes none of this — it is the software convention a toolchain should follow.

## 5. DMA programming sequence *(model)*

Per channel `chN` (N∈0..7):
1. `dma.config.chN x_base` — sets the shared `dma_base` (upper 32 DRAM addr bits).
2. `dma.load.chN  x_vmem, x_dram, x_len` — DRAM→VMEM, `x_len` bytes (32 B aligned).
   `dma.store.chN x_dram, x_vmem, x_len` — VMEM→DRAM.
   On issue, channel N's completion flag is **set**; on completion it is **cleared**.
3. `dma.wait.chN` — stalls the frontend until channel N's flag clears (data fence).

Rules: ≤1 outstanding transfer per channel; issuing to a busy channel is illegal;
DMA may overlap compute (it does not block issue). *(model)*

## 6. Worked schedule sketch (illustrates the static-schedule discipline)

```
  dma.config.ch0 x_base          # set DRAM base
  dma.load.ch0   x_vmemA, x_dramA, x_len
  dma.wait.ch0                   # fence: A is now in VMEM
  vload   m0, 0(x_vmemA)         # VMEM -> MRF (LSU, 34 cyc, blocking)
  vmatpush.weight.mxu0 w0, m2    # MRF -> weight slot (32 cyc, blocking)
  vmatmul.mxu0   acc0, m0, w0    # FP8xFP8 -> BF16 acc (96 cyc; occupies MXU0)
  delay 96                       # space out: wait MXU0 latency before reading acc
  vmatpop.bf16.acc.mxu0 m4, acc0 # acc -> MRF (32 cyc, blocking)
  vstore  m4, 0(x_vmemC)         # MRF -> VMEM
  dma.store.ch0  x_dramC, x_vmemC, x_len
  dma.wait.ch0
  ecall                          # halt
```
The `delay 96` is mandatory: issuing the dependent `vmatpop` into the still-busy
MXU0 without spacing is a backpressure schedule violation (`SPEC.md §9`, §15).

## 7. Control surface and lifecycle *(model + decisions)*

*(model):* load IMEM → execute from PC 0 → poll `halted`.

*(npu_spec intent, reference):* `npu_spec/03` states the architecture *does* expose a
control plane — **execution enable/halt, execution status / stop-reason, PC visibility,
and per-channel DMA busy visibility**. And `npu_spec/06 §4` maps off-chip memory on a
**serial TileLink interface** with a `PERIPH` region (`0x4000_0000–0x8000_0000`) plus
cacheable DRAM from `0x8000_0000`. So the control-plane and bus *intent* exist; what is
missing is the concrete **register map / encoding** (the 4096 CSRs are still scratch).

**Decisions required for a real driver** (concretize the above intent):

| # | Decision | Recommendation |
|---|----------|----------------|
| D1 | Host transport for IMEM load / control | memory-mapped IMEM aperture + an MMIO control block |
| D2 | START mechanism | a doorbell/START register (vs implicit run-on-reset-deassert) |
| D3 | Completion signaling | architected STATUS.DONE bit; **interrupt** in addition to poll |
| D4 | Error reporting | STATUS.ERROR + an **error-cause** register (bank-conflict / illegal-DMA / backpressure / illegal-insn) |
| D5 | Identity/version | an ID/VERSION register for driver compatibility |
| D6 | Host↔accelerator memory ordering | define whether the host may touch VMEM/DRAM during a run; coherence/ordering w.r.t. host caches |
| D7 | Performance counters | the scratch CSR space could expose cycle/stall/utilization counters for profiling |
| D8 | Off-chip bus binding | concretize the `npu_spec` serial **TileLink** interface (widths, bursts, IDs) — artifact **D** |

Until D1–D8 are decided, the lifecycle above is the architectural contract and the
binding is implementation-defined. These map to `downstream.yaml:ratification_decisions`
and artifact **D** (SoC integration). Note: the `dma.config`/`dma.wait` encoding is
**not** open — `npu_spec` resolves it (distinct `funct7`); the model has a defect (§6).
