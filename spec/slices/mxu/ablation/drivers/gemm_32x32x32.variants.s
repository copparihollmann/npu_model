# gemm_32x32x32.variants.s — the SAME 32x32x32 weight-stationary GEMM tile expressed in
# each ISA variant. Pre: activation A in m0 (fp8), weight B in m2 (fp8). Post: C (bf16) in m6.
# Scale e0 used only on fp8 stores. `delay` omitted for brevity (see ../../programs/ for the
# fully-scheduled baseline). The point is the INSTRUCTION-SURFACE contrast, not re-simulation.

# ===== C3_baseline (14-op ISA; GEMM path uses 3 instrs) =====================
vmatpush.weight.mxu0 w0, m2
vmatmul.mxu0 acc0, m0, w0
vmatpop.bf16.acc.mxu0 m6, acc0

# ===== C2_collapsed (4-op ISA; unit/dtype/acc folded into mod) ==============
mxu.push.weight w0, m2, mod=unit0
mxu.matmul acc0, m0, w0, mod=unit0.clear
mxu.pop m6, acc0, e0, mod=unit0.bf16

# ===== C1_risc_min (3-op uniform ISA over unified tile regs) ================
tld   tw0, m2, mod=weight.unit0
mma   ta0, m0, tw0, mod=unit0.clear
tst   m6, ta0, e0, mod=bf16

# ===== C4_cisc_macro (1-op ISA) ============================================
tilegemm m6, m0, m2, k=1, mode=unit0
# K-tiled (K=2) contrast: C4 stays ONE instruction; C1/C2/C3 grow ~linearly in K:
#   C4:  tilegemm m6, m0, m2, k=2, mode=unit0
#   C3:  push.weight; matmul(clear); push.weight; matmul.acc; pop   (5 instrs)
