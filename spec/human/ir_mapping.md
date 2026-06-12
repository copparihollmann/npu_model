# Node → SSA-IR lowering contract

The machine spec is a typed node graph (`META.md`). Because every node has a stable
`id`, a `kind`, by-id `refs`, and (for `op`) typed operands with `direction` +
a structured `effect`, lowering to an SSA-style IR is **mechanical**: walk nodes,
emit IR objects, resolve `refs` to IR handles. This document is the contract; it does
**not** implement an IR (see `ir_requirements.md` for what an IR must abstract).

## Mapping

| Spec node kind | IR object | Notes |
|----------------|-----------|-------|
| `type` | IR **type** | dtypes / operand classes / tile types |
| `state` | IR **location** (memory/regfile/buffer/flag) | the things SSA values live in/flow through |
| `op` | IR **operation** | operands → IR operands/results; `effect` → op body |
| `op` operand `direction: use` | IR **operand (use)** | reads; SSA use edge |
| `op` operand `direction: def`/`inout` | IR **result/def** | writes; SSA def edge |
| `op.effect` `{def, expr}` | IR **op body** (value DAG) | `expr` AST → IR value graph; `{lit}` → constant |
| `edge` | IR **dataflow value/movement** | the data-movement graph |
| `unit` | IR **module/instance** | `refs.visible_state`/`latency_class` carried as attrs |
| `port` / `channel` | IR **port/channel** (+ protocol) | `refs.accesses`/`endpoints` → IR location handles |
| `timing_class` | IR **latency attr** | `cycles` or `formula` |
| `param` / `constraint` | IR **attribute / contract** | hard vs target via `attrs.kind_of` |
| `knob` | IR **parameter / transform-target** | `refs.varies` → the IR nodes a transform edits |
| `test_intent` / `coverage_goal` | IR **test_intent / coverage_goal** | `refs.covers` → ops/locations |
| `claim` | IR **provenance annotation** | carry `status`+`refs.about` onto the IR node(s) |
| dialect kind (e.g. `kernel`) | IR **dialect op/region** | lowers via the dialect's own rules |

## Worked example (already extractable today)

`python spec/schema/load.py --ssa mxu_matmul` prints, from the nodes:

```
op.vmatmul.mxu0:      uses=[vs1, vs2]      defs=[vd]
    vd <- {op: matmul, args: [vs1, vs2]}
op.vmatmul.acc.mxu0:  uses=[vd, vs1, vs2]  defs=[vd]      # vd is inout (accumulate)
    vd <- {op: add, args: [vd, {op: matmul, args: [vs1, vs2]}]}
```

Lowering reads each operand's `location` (`state.mrf`, `state.mxu_weight_slots`,
`state.mxu_accum_buffers`) and `type` to place SSA values in the right IR locations,
then emits the `effect` expr as the op's value DAG. `vmatmul.acc`'s `inout` `vd`
becomes a read-modify-write on the accumulator location — a textbook SSA
def-with-prior-use.

## Invariants the lowering relies on (enforced by `schema/load.py`)

- ids unique; every `refs` id resolves; operand `type`/`location` resolve to
  `type`/`state` nodes; effect `def` is a def/inout operand; effect operand names are
  declared. These guarantees are what make the walk safe.
