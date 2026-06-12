# ISA-complexity ablation (MXU substrate, built to extend to the whole ISA)

Goal (from the professor's direction): test how **ISA complexity** affects the ability to
**one-shot generate correct, performant RTL** from a spec — separating **token efficiency**
from **evaluator** quality, and probing the **concise-vs-verbose / low-resource** question.

We run it on the **MXU slice** first (small, self-contained, clean oracle) but the harness
and variant format are **ISA-agnostic** — adding the whole ISA later is just more op nodes.

## Two ablation families

- **(A) ISA *design* complexity** — *different instruction sets for the same MXU capability.*
  Moves two axes: **opcode surface** (how many distinct instructions) × **per-op semantic
  complexity**. Not monotone — that's the interesting part.

  | Variant | Opcodes | Idea | Where on the axes |
  |---|---:|---|---|
  | `C4_cisc_macro` | **1** | `tilegemm` macro fuses push+matmul+pop over K | tiny surface, max per-op semantics |
  | `C1_risc_min` | **3** | `tld`/`tst`/`mma` over a *unified* tile register file, explicit operands | small, uniform |
  | `C2_collapsed` | **4** | baseline with unit/dtype/acc folded into a modifier field | small |
  | `C3_baseline` | **14** | the real ISA: unit & dtype baked into the opcode | medium, regular |
  | `C7_capable` | **24** | baseline + fine-control ops, **every added opcode justified** | above baseline, JUSTIFIED |
  | `C5_expanded` | **36** | slot + acc **indices** baked into the opcode (no new capability) | large, REGULAR, **bloat** |
  | `C6_scattered` | **42** | C5 size + mixed formats/order + some fused ops | large, **IRREGULAR** |

  The three points *above* the baseline separate the **reasons** an ISA grows (a fair ablation
  must — more opcodes should buy something):
  - **C7_capable** — more opcodes *and more capability*: each added op (transpose-fused matmul,
    fused matmul→MRF, fused requant, bias-on-drain, 16×16 sub-tile) has a quantified win in
    `attrs.rationale`. The "complexity that pays its way" arm.
  - **C5_expanded** — more opcodes, *same capability* (selector re-encoded into the opcode):
    the **bloat control**. Its only reason is an encoding/decode trade-off.
  - **C6_scattered** — more opcodes + *irregular packaging* (mixed formats, shuffled operands):
    the **scatter control**. C5↔C6 are size-matched, so that contrast isolates diversity.

  C5/C6/C7 are generated deterministically by `harness/gen_variants.py`. Use
  `ablation.py capability` to see justified-vs-bloat and `ablation.py diversity` for size-vs-scatter.

- **(B) ISA *description* complexity** — the **same** ISA rendered at three verbosity levels
  (`concise` mnemonic line / `structured` node YAML / `verbose` prose+Verilog), with token
  accounting. Tests token-cost vs RTL-gen quality and how much the evaluator depends on it.

## How it's built (extensible)

Each variant is a **swappable op-node set** over the **shared substrate**
(`../machine/{types,state,spec}.yaml` + `shared/{types,state}.yaml`), validated by the
**same** core meta-model (`../../schema/core.py`). So:

- variants reuse one state/type/timing model → differences are purely the *instruction layer*;
- the RISC/CISC variants abstract MXU-local storage as `state.mxu_tile_regs` (a reference
  *view* of weight slots + accumulators — same hardware);
- folded fields live in a `type.mod` modifier immediate (the thing complex variants move out
  of the opcode);
- **whole-ISA extension** = drop more op nodes into a variant; the harness is unchanged.

```
ablation/
  README.md  metrics.md
  shared/        types.yaml (type.mod, type.tile_any) · state.yaml (state.mxu_tile_regs)
  variants/      C1_risc_min/ C2_collapsed/ C4_cisc_macro/   (C3 = ../../machine/isa.yaml)
  drivers/       gemm_32x32x32.variants.s  (one GEMM tile in every variant's ISA)
  harness/       ablation.py  (validate · opcodes · present · run)
```

## Run it

```
python spec/slices/mxu/ablation/harness/gen_variants.py        # (re)generate C5/C6
python spec/slices/mxu/ablation/harness/ablation.py validate    # all 6 variants validate
python spec/slices/mxu/ablation/harness/ablation.py opcodes     # family (A): opcode surface
python spec/slices/mxu/ablation/harness/ablation.py capability  # justified vs bloat (do extra ops earn their keep?)
python spec/slices/mxu/ablation/harness/ablation.py diversity   # size vs scatter (two axes)
python spec/slices/mxu/ablation/harness/ablation.py present      # family (B): token table
python spec/slices/mxu/ablation/harness/ablation.py run          # full matrix (RTL hooks TODO)
```

Current measured output (counts + tokens are real; RTL-gen/eval are hooks):

```
(A) opcodes:   C4=1  C1=3  C2=4  C3=14  C5=36  C6=42
size vs scatter:   opcodes formats arities dir_sigs    <- C5 regular, C6 scattered at ~same size
    C5_expanded         36       1       2        4
    C6_scattered        42       5       3       11
(B) ~tokens:        concise  structured  verbose
    C3_baseline        230       2335      2037
    C5_expanded        392       4976      4369
    C6_scattered       458       6219      5465
```

## What's a hook (what you/your model wire in)

`harness/ablation.py` has two intentional hooks:
- `rtl_generate(prompt)` — one-shot LLM RTL generation from a presentation.
- `evaluate(rtl, variant)` — functional check vs the `npu_model` oracle + perf vs Layer B
  (the **≤10% abstraction-overhead** bar). Returns `{functional, perf_ok}`.

Everything else (variant validation, opcode surface, presentation rendering, token cost,
the experiment matrix) runs today. See `metrics.md` for the full metric set and hypotheses.
