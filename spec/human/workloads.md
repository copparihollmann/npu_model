# Atlas NPU Subset — Workload Characterization (S0/S1)

> Companion to `SPEC.md §3`. Grounded in `npu_model/configs/programs/*`,
> `npu_model/workload/{gemma_blocks,smolvla_ops}.py`, `tests/test_programs.py`, and
> `npu_speed_of_light/simple_throughput_model.py`. Workload characterization is part
> of the architecture problem (H&P: design from the workloads). Numbers here are
> verified against those sources; product-level gaps are tagged at the end.

## 1. Kernel inventory (the supported op set)

Source: `npu_model/configs/programs/` (`parameterized_*`, `smolvla_*`, `gemma_*`),
shapes from the `.hex`/program variants; tolerances from `tests/test_programs.py`.

| Kernel | Operation | In→Out dtype | Shapes exercised | Engine(s) | Tol (rtol/atol) |
|--------|-----------|--------------|------------------|-----------|-----------------|
| matmul | `C = A·B`, BF16 acc per K-tile | FP8 → BF16 | 32×32×32, 32×64×32, 64×32×96, 64×64×64 | MXU | 1e-2 |
| batch_matmul | batched GEMM | FP8 → BF16 | 2×32×32×32, 2×64×32×64, 4×32×64×32 | MXU | 1e-2 |
| fused_matmul_bias | GEMM + bias (+cast) | FP8/BF16 | 32×32, 64×32, 64×64 | MXU+VPU | 1e-2 |
| bias_add_cast | matmul result + bias → cast | BF16 | 32×32 | VPU | 1e-1 |
| rms_norm | `x·rsqrt(mean(x²)+eps)` | BF16 | 32×32, 64×32, 96×32 | VPU | 1e-2 |
| softmax | row-stable `exp(x−max)/Σ` | BF16 | 32×32, 64×32, 96×32 | VPU | 1e-2 |
| silu | `x·σ(x)` | BF16 | 32×32, 64×32, 64×64 | VPU | 1e-2 |
| gelu_tanh | tanh-approx GELU | BF16 | 32×32, 64×32, 96×32 | VPU | 1e-2 |
| fused_silu_gate | SiLU(gate)·up | BF16 | 32×32, 32×64, 64×64 | VPU | 1e-2 |
| fused_norm_scale | RMS-norm · scale | BF16 | 32×32, 64×32, 64×64 | VPU | 1e-2 |
| rope_frequency | freq matrix + sin/cos | BF16 | 32×32, 32×64, 64×64 | VPU | 1e-2 |
| reduction_sum | row/col reduce | BF16 | 32×32, 64×32, 96×32 | VPU | 1e-2 |
| requant | BF16 → FP8 (`vpack`) | BF16 → FP8 | 32×32, 64×32, 64×64 | VPU | 1e-2 |
| elementwise add/sub/mul/div | `c = a∘b` (div = `a·recip(b)`) | BF16 | 32×32, 32×64, 64×64 | VPU | 1e-2 |
| fused_attention | online/flash softmax SDPA | BF16 (FP8 in matmul) | q∈{32,64}, k∈{64,96,128} | MXU+VPU | 5e-2 |
| smolvla_attention | model attention cross-check | BF16 | model dims | MXU+VPU | 0.2 |

**Layouts / tiling (hardware-imposed + canonical convention):** 32×32 row-major
tiles; full BF16 tile = register pair (cols 0–15 / 16–31); large tensors stored
**tiled** — `(M_tiles×K_tiles)` contiguous 32×32 blocks, tile `(m,k)` at byte
`(m·K_tiles+k)·1024`, so each DMA is a contiguous 1 KB transfer (see `SPEC.md §9.2`,
`parameterized_matmul.py`). Reference math for each kernel: `npu_model/workload/`.

## 2. Performance-critical kernels & arithmetic intensity

Source: `npu_speed_of_light/simple_throughput_model.py` (`HardwareParams`:
input 1 B/FP8, output 2 B/BF16, MT=NT=KT=32; peak `2·NT·KT = 2048 FLOP/cycle`).
Real model-layer case studies (exact M/N/K from the model):

| Case study | M | N | K | Total FLOPs `2·M·N·K` |
|------------|---|---|---|------------------------|
| GemmaMLP.up_proj | 816 | 16384 | 2048 | 5.47e10 |
| GemmaMLP.down_proj | 816 | 2048 | 16384 | 5.47e10 |
| GemmaAttention.q_proj | 816 | 2048 | 2048 | 6.84e9 |
| SigLIPAttention.self_attn | 256 | 1152 | 1152 | 6.79e8 |
| SigLIPAttention.fc1 | 256 | 1152 | 4304 | 2.54e9 |
| SigLIPAttention.fc2 | 256 | 4304 | 1152 | 2.54e9 |
| GemmaMLP.up_proj (bs1) | 51 | 4096 | 1024 | 4.28e8 |
| GemmaMLP.down_proj (bs1) | 51 | 1024 | 4096 | 4.28e8 |
| GemmaAttention KxQ | 816 | 256 | 816 | 3.41e8 |
| GemmaAttention PxV | 816 | 816 | 256 | 3.41e8 |

The SoL model compares **weight-stationary vs output-stationary** dataflow memory
efficiency per case (output → `reports/dataflow_comparison.png`). Takeaway baked
into the spec: the MXU is **weight-stationary** (`SPEC.md §9.1`); dataflow is a DSE
knob (`design_space.yaml`). GEMM and attention dominate FLOPs → **perf-critical**.

## 3. Application contract (entry points)

- Programs are static assembly loaded into IMEM; inputs staged in DRAM at 0-based
  offsets; results read back from DRAM (see `sw_interface.md`).
- Numeric tolerances above are the **acceptance contract** (also in
  `test_obligations.yaml`); the deterministic-op golden is `npu_model` (bit-exact),
  transcendentals tolerance-bound (see `numerics.md`).

## 4. Characterization gaps (tagged; not blockers)

| Gap | Status | Note |
|-----|--------|------|
| Full-model op-frequency / layer-count weighting | open | SoL uses representative layers, not how often each op runs per full VLA/Gemma inference |
| Target throughput (FPS) / latency budget | open | product-level; not in repo |
| Power / energy budget, edge platform | open | product-level; not in repo |
| Batch / sequence-length distributions | assumption | programs are small / batch-1; broaden when known |
| Weight/activation memory footprint per model | open | derivable from case-study dims; not yet tabulated end-to-end |
