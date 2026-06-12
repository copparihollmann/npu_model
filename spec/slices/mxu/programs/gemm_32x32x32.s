# gemm_32x32x32.s — the first e2e MXU driving example.
#
# Pure MXU datapath: a 32x32x32 GEMM tile on MXU0, weight-stationary.
# VMEM/DMA staging is OUT of this slice's scope — we assume the operands are
# already resident in MRF (in the full bundle, dma.load + vload put them there).
#
# Mnemonic syntax is the npu_model assembly (see
# npu_model/configs/programs/asm/parameterized_fused_matmul_bias64x64.S).
# The static schedule inserts an explicit `delay <latency>` after each blocking
# MXU op; because each delay fully covers the op's latency class, the schedule is
# bank-conflict-free by construction (no two MXU ops are in flight at once).
#
# Pre:   activation A (32x32 fp8) in m0 ; weight B (32x32 fp8) in m2
# Post:  result C (32x32 bf16) in m6
# Cost:  32 + 96 + 32 = 160 cycles (push + matmul + pop), MXU0

vmatpush.weight.mxu0 w0, m2      # B -> weight slot w0 (weight-stationary)   [timing.mxu_push_pop = 32]
delay 32
vmatmul.mxu0 acc0, m0, w0        # acc0 = A @ B  (overwrite accumulator)      [timing.mxu0_matmul   = 96]
delay 96
vmatpop.bf16.acc.mxu0 m6, acc0   # C = acc0 -> MRF (lower out)                [timing.mxu_push_pop = 32]
delay 32

# ---------------------------------------------------------------------------
# K-tiled accumulate variant (32x32x64): demonstrates the inout accumulator and
# weight-stationary weight swap. Pre: A0=m0,A1=m1,B0=m2,B1=m3 ; Post: C=m6.
#
#   vmatpush.weight.mxu0 w0, m2
#   delay 32
#   vmatmul.mxu0 acc0, m0, w0        # acc0  = A0 @ B0   (overwrite)
#   delay 96
#   vmatpush.weight.mxu0 w0, m3      # swap resident weight tile
#   delay 32
#   vmatmul.acc.mxu0 acc0, m1, w0    # acc0 += A1 @ B1   (accumulate, inout vd)
#   delay 96
#   vmatpop.bf16.acc.mxu0 m6, acc0
#   delay 32
#   # Cost: 2*(32+96) + 32 = 288 cycles
# ---------------------------------------------------------------------------
