# Ablation metrics & hypotheses

## Independent variables
- **Variant** (family A), spanning *both sides* of the baseline:
  `C4_cisc_macro` (1) · `C1_risc_min` (3) · `C2_collapsed` (4) · `C3_baseline` (14) ·
  `C7_capable` (24) · `C5_expanded` (36) · `C6_scattered` (42). Underlying axes are independent
  — a fair ablation must control for *why* an ISA has more opcodes:
  - **opcode surface** (size) — 1 → 42
  - **per-op semantic complexity** — highest at C4 (macro)
  - **scatter / diversity** — C5 (regular) vs C6 (irregular), *size-matched* → C5↔C6 isolates
    diversity from size (`ablation.py diversity`).
  - **capability vs packaging** — C7 (every added opcode justified) vs C5 (more opcodes, zero
    new capability) → C7↔C5 isolates *justified* complexity from *bloat* (`ablation.py capability`).
    This is the control that makes "more opcodes" fair: extra ops in C7 buy a quantified win.
- **Style** (family B): `concise` · `structured` · `verbose` presentation of the same ISA.

## Dependent variables (per variant × style)
| Metric | Source | Status |
|---|---|---|
| opcode count | `ablation.py opcodes` | ✅ measured |
| description tokens | `ablation.py present` (swap in a real tokenizer) | ✅ measured (approx) |
| one-shot RTL **functional correctness** vs `npu_model` oracle | `evaluate()` hook | ⬜ hook |
| **perf overhead** vs Layer B (target **≤10%**) | `evaluate()` hook | ⬜ hook |
| RTL-gen **pass@k** / retries to correctness | wrap `rtl_generate()` | ⬜ hook |
| **tokens-per-correct-RTL** (efficiency = quality ÷ cost) | derived | ⬜ |

## Separating the two things the professor called out
- **Token efficiency** = description tokens (family B) and tokens-per-correct-RTL.
- **Evaluator** = does the spec presentation carry enough to *check* correctness, independent
  of generation? Hold `rtl_generate` fixed (or use a reference RTL) and vary style to isolate
  the evaluator's dependence on verbosity.

## Hypotheses to test
1. **RTL-gen correctness is non-monotone in opcode surface** — a U/∪ shape across
   C4→C1→C2→C3→C5→C6. Both extremes are hard: `C4` (1 macro, heavy hidden semantics) and
   `C6` (large + scattered); the uniform middle (`C1`/`C2`) is easiest. Now testable both
   *below* and *above* the baseline.
2. **Size vs scatter are different costs.** `C5` (large, regular) vs `C6` (size-matched,
   irregular): if `C6` is markedly harder than `C5`, *diversity* — not raw count — is the
   dominant difficulty driver (the professor's "scattered info / low learning data" claim).
2b. **Justified complexity is cheaper than bloat.** `C7` (24 ops, all added ones earn a win)
   vs `C5` (36 ops, none): if `C7` generates correct RTL more reliably *per added opcode*, the
   penalty is for *gratuitous* complexity, not capability. A capable ISA can beat a bloated
   one even with a comparable surface — the real design lesson. Also: do C7's fused ops shrink
   the *driver* (fewer instructions / MRF round-trips) enough to offset their RTL-gen cost?
3. **Low-resource penalty.** Variants far from training distributions (novel `tld/tst/mma`,
   `tilegemm`, C6's ad-hoc fused ops) need more verbose specs to hit correctness — a
   verbose×variant interaction.
4. **Verbose helps generation, less so evaluation.** `verbose` lifts functional pass-rate for
   exotic variants but adds little for the evaluator once `structured` is present.
5. **Token efficiency favors `C1`/`C2`** (correct RTL per token); `C4` is token-cheap but
   low-correctness, `C5`/`C6` are token-expensive AND lower-correctness.

## Confounds to control
- Same capability + same driver (`drivers/gemm_32x32x32.variants.s`) across variants.
- Same oracle (`npu_model` MXU) and same ≤10% perf bar (Layer B) across variants.
- Variant encodings are clearly *allocated* (not the baseline ISA) so a model can't pattern-match.
- Replace the approximate token estimate with the target model's tokenizer before reporting.

## Extending to the whole ISA
Add op nodes for VPU/XLU/LSU/DMA/scalar to each variant (or new variant dirs); the harness,
metrics, and hooks are unchanged. The MXU run is the methodology shakedown.
