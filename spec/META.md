# META — how the machine spec is composed (authoring conventions)

The `machine/*.yaml` files are a **typed graph of nodes**. One envelope, a fixed
**core kind** vocabulary, by-id **references**, and per-accelerator **dialects**.
This is what lets the spec (a) validate, (b) lower mechanically to an SSA-style IR
(`human/ir_mapping.md`), and (c) grow by adding *data* (and dialects), not bespoke
structure. Validator: `python spec/schema/load.py` (typed model in `schema/core.py`).

## The node envelope (every entry, every machine file)

```yaml
- id: <prefix>.<name>      # unique, namespaced (op.vadd.bf16, state.mrf, type.bf16)
  kind: <core | dialect>   # discriminator from the registry
  status: normative | reference | assumption | open | rejected
  provenance: {sources: [source.*], confidence: high|medium|low}   # optional
  doc: "human note"        # optional
  attrs: {...}             # kind-specific fields (required ones enforced; extras allowed)
  refs:  {role: id | [ids]}# typed edges; every id MUST resolve to a node
  ext:   {<dialect>: {...}}# dialect extension (namespace must be a registered dialect)
  operands: [...]          # op kind only
  effect:   [...]          # op kind only
```

Required: `id`, `kind`, `status`. Files are `{nodes: [...]}` or a bare list. The
loader merges ALL machine files into one graph (ids are globally unique).

## Core kinds (fixed, versioned — the IR depends on these)

`type · param · unit · state · port · channel · op · timing_class · edge ·
constraint · knob · test_intent · coverage_goal · claim · source · decision ·
axis · artifact`. Each declares required/optional attrs and allowed ref roles in
`schema/core.py:CORE_KINDS`. **Do not add core kinds for accelerator-specific
concepts — add a dialect.**

## References (the graph edges)

Every cross-reference is a by-id `refs:` entry with a role, validated to resolve
(dangling = error). Examples: `op.refs.latency_class -> timing_class`,
`operand.location -> state`, `knob.refs.varies -> [nodes]`,
`claim.refs.about -> [nodes]`, `constraint.refs.subject -> node`.
**Derive, don't duplicate:** state read/written-by is *not* stored — it is implied
by op operands.

## `op` nodes — operands + effect (SSA)

```yaml
operands:
  - {name: vd,  type: type.mrf_tile_bf16, location: state.mrf, direction: def}   # write
  - {name: vs1, type: type.mrf_tile_bf16, location: state.mrf, direction: use}   # read
effect:
  - {def: vd, expr: {op: add, args: [vs1, vs2]}}        # small typed AST
```
`direction` ∈ `{def, use, inout}` → SSA defs/uses. `effect` is a list of
`{def: <operand>, expr}`; `expr` is `{op: <name>, args: [<operand-name> | nested-expr
| {lit: <value>}]}`. The validator checks each `def` is a def/inout operand and
every operand name in `expr` is declared. SSA-critical families (LSU/VPU/MXU/DMA)
carry operands+effect; trivial scalar/control ops may use a prose `attrs.sem`.

## Dialects (extensibility without touching core)

A dialect (`dialects/<name>.yaml`) registers extra kinds and/or `ext` schemas:
```yaml
dialect: npu
kinds:
  - {name: kernel, id_prefix: kernel, required_attrs: [operation],
     ref_roles: {uses_ops: [op], reads: [state], writes: [state]}}
```
List active dialects in `manifest.yaml:dialects`. Nodes use a dialect kind directly,
or attach `ext: {npu: {...}}` to a core node. **Designers extend a dialect; core
stays small and lowerable.**

## Adding things (recipes)

- **A new instruction** → add an `op` node (operands+effect if SSA-critical;
  `latency_class` ref to an existing `timing_class`).
- **A new parameter/state/component** → add a `param`/`state`/`unit` node; wire refs.
- **An accelerator-specific concept** → add/extend a dialect kind, not core.
- **Always** run `python spec/schema/load.py` (and `python spec/lint.py`) — 0 errors.
