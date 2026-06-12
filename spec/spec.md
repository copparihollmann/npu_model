# Hardware Accelerator Spec Definition

## Purpose

This page defines what we mean by a **hardware accelerator specification** in this project.

A spec is not just prose, an ISA table, a simulator, RTL documentation, or a Claude prompt. It is a **versioned architecture contract** that can be read by humans, checked by tools, refined into IR, and used to generate models, tests, scoreboards, assertions, and RTL-generation tasks.

The target flow is:

```text
Natural language intent
    ↓
Human-readable architecture spec
    ↓
Structured machine-readable spec
    ↓
Executable spec / reference semantics
    ↓
IR-ready spec
    ↓
LHWIR / hardware IR
    ↓
Models, tests, RTL-generation tasks, verification artifacts
```

The executable model or oracle sits **beside** this chain. It can be an authoring source and an evaluation oracle, but the IR should be generated from the spec, not from the oracle directly.

---

## Core definition

A hardware accelerator spec is a **typed, versioned, provenance-carrying graph of architectural claims**.

It defines:

1. what the accelerator does,
2. what state exists,
3. how software or other hardware interacts with it,
4. what operations are legal,
5. what each operation means,
6. how data moves,
7. how time, ordering, and concurrency behave,
8. what constraints an implementation must satisfy,
9. what is fixed versus tunable,
10. what must be verified,
11. how the spec maps to IR,
12. how conflicting sources are reconciled.

In this project, a spec is both:

* a **human architecture contract**, and
* a **machine-readable generation contract**.

---

## What the spec is not

A spec is not:

* a collection of informal notes,
* a standalone ISA table,
* a block diagram,
* a performance model,
* an RTL implementation,
* a simulator,
* a testbench,
* a Claude prompt,
* a copy of the manual design.

Those can all be sources or projections of the spec, but they are not the spec itself.

The spec is the source of truth from which those artifacts are derived or checked.

---

# 1. Separation of levels

The spec must separate three levels.

| Level                             | Question                                     | Included in spec?                                                   |
| --------------------------------- | -------------------------------------------- | ------------------------------------------------------------------- |
| **Architecture**                  | What is visible and guaranteed?              | Yes, primary content.                                               |
| **Microarchitecture constraints** | What implementation properties are required? | Yes, when they affect correctness, compatibility, or evaluation.    |
| **Implementation**                | How is the RTL built internally?             | No, unless externally visible or required for benchmark comparison. |

Examples:

| Statement                                                               | Classification                                                |
| ----------------------------------------------------------------------- | ------------------------------------------------------------- |
| “The ISA has 32-bit fixed-width instructions.”                          | Architecture                                                  |
| “VMEM has 8 banks and 32-byte conflict granularity.”                    | Architecture / microarchitecture constraint                   |
| “MXU0 has 32×32 PEs.”                                                   | Microarchitecture constraint if required; otherwise reference |
| “This FSM has state `WAIT_DMA` encoded as `3'b010`.”                    | RTL implementation detail                                     |
| “`dma.wait.chN` must not complete before channel N transfer completes.” | Architecture contract                                         |
| “The VPU implementation uses this exact module hierarchy.”              | Implementation detail                                         |

A good spec constrains the **contract**, not one accidental implementation.

---

# 2. Spec maturity ladder

A spec evolves through maturity levels.

| Level  | Name                             | Meaning                                                         |
| ------ | -------------------------------- | --------------------------------------------------------------- |
| **S0** | Natural-language intent          | Informal goals, workload intent, design rationale, constraints. |
| **S1** | Human-readable architecture spec | Clear prose specification with normative language.              |
| **S2** | Structured machine-readable spec | Typed nodes, attributes, references, provenance, constraints.   |
| **S3** | Executable spec                  | Reference semantics, checks, and simulatable behavior.          |
| **S4** | IR-ready spec                    | Mechanically lowerable into LHWIR / hardware IR.                |
| **S5** | Frozen benchmark spec            | Versioned, controlled spec used for experiments and baselines.  |

The project should avoid jumping directly from S0 to RTL.

The intended path is:

```text
S0 → S1 → S2 → S3 → S4 → IR → models/tests/RTL tasks
```

A spec is **S2-incomplete** unless the structured graph validates successfully.

---

# 3. Human spec vs machine spec

The spec has two synchronized representations:

| Representation   | Role                                                 | Source-of-truth rule                              |
| ---------------- | ---------------------------------------------------- | ------------------------------------------------- |
| `human/*.md`     | Prose, rationale, explanation, reviewability         | Canonical for intent and rationale.               |
| `machine/*.yaml` | Values, claims, typed nodes, references, constraints | Canonical for values and machine-consumed claims. |

Hard rule:

> `machine/*.yaml` is canonical for values and claims.
> `human/*.md` is canonical for prose and rationale.
> They may not disagree on critical constants or normative behavior.

If they drift, the validator must fail.

This two-representation model is only safe if the machine graph is validated and every human-facing value can be traced back to the structured graph.

---

# 4. Normative language

Every requirement must use explicit requirement language.

| Word             | Meaning                              |
| ---------------- | ------------------------------------ |
| **MUST / SHALL** | Required for correctness.            |
| **SHOULD**       | Expected unless justified otherwise. |
| **MAY**          | Optional implementation choice.      |
| **MUST NOT**     | Prohibited.                          |

Avoid vague language such as “probably,” “ideally,” or “should work” in normative sections.

If a behavior is uncertain, mark it as `open` or `assumption`.

---

# 5. Claim status

Every nontrivial claim must carry a status.

| Status         | Meaning                                                        |
| -------------- | -------------------------------------------------------------- |
| **normative**  | Must be implemented and tested.                                |
| **reference**  | Observed in a source, useful for comparison, but not required. |
| **assumption** | Inferred from incomplete evidence; should be reviewed.         |
| **open**       | Known ambiguity or unresolved architectural decision.          |
| **rejected**   | Considered and intentionally excluded.                         |

This distinction is mandatory.

The spec must not silently mix:

* intended behavior,
* implemented behavior,
* model behavior,
* inferred behavior,
* future desired behavior.

---

# 6. Source bundle, provenance, and reconciliation

The spec must track where claims came from.

Sources may include:

* natural-language brainstorm,
* architecture notes,
* ISA reference,
* executable model,
* RTL,
* final presentation,
* tests,
* performance model,
* prior LLM-generated attempts,
* manual designer notes.

Each source should be represented as a node.

```yaml
- id: source.npu_model
  kind: source
  status: reference
  attrs:
    type: executable_model
    trust: high_for_semantics_medium_for_rtl_exactness
    role: architecture_semantics_and_latency
```

Each claim should reference its sources.

```yaml
- id: claim.vmem.conflict_policy
  kind: claim
  status: normative
  refs:
    about: [state.vmem]
    sources: [source.npu_model, source.human_spec]
  attrs:
    confidence: high
    text: "VMEM conflicts are software-managed and statically illegal."
```

---

## Multi-source reconciliation rule

When two sources disagree on a normative value or behavior, the spec **must not silently pick one**.

It must record a `discrepancy` or `decision` node naming:

* the affected node,
* both sources,
* the conflicting claims,
* the selected normative source if already ratified,
* the current disposition.

Example:

```yaml
- id: discrepancy.over_issue_policy
  kind: decision
  status: open
  refs:
    about: [unit.mxu0, unit.mxu1, unit.vpu, unit.lsu]
    sources: [source.npu_spec, source.npu_model]
  attrs:
    npu_spec: "Frontend stalls when target unit is busy."
    npu_model: "Raises on over-issue to a busy non-DMA unit."
    normative_choice: pending
    disposition: "Ratify before benchmark freeze."
```

Hard rule:

> Every value or behavior with more than one source must either agree across sources or have a `discrepancy` / `decision` node. No silent winner is allowed.

This is one of the highest-value parts of the spec. Specs rot when disagreements are hidden.

---

# 7. Authored vs generated artifacts

Every machine-readable artifact must declare whether it is hand-authored or generated.

```yaml
origin: authored | generated
generated_by: optional script/path
generated_from: optional list[input_artifacts]
```

Rules:

* Authored files may be edited manually.
* Generated files must not be hand-edited.
* Generated files must record the generator and inputs.
* Regenerating a file must be reproducible.
* If a generated artifact is modified manually, it must either become authored or the edit must move into the generator/input.

This matters for variant ISAs, generated flavors, generated scaffolds, and derived test artifacts.

---

# 8. Spec as a typed graph

The machine-readable spec should be a typed graph.

Every entity is a node. Relationships are explicit references between nodes.

A node has at minimum:

```yaml
id: globally.unique.id
kind: node_kind
status: normative | reference | assumption | open | rejected
attrs: {}
refs: {}
```

Recommended full schema:

```yaml
id: string
kind: string
status: string
title: optional string
summary: optional string
attrs: map
refs: map
children: optional list[id]
parent: optional id
source: optional list[id]
owner: optional string
maturity: optional string
version: optional string
tags: optional list[string]
origin: authored | generated
generated_by: optional string
generated_from: optional list[id]
ext: optional map[dialect_name, map]
```

Rules:

* `id` must be stable across revisions.
* Every `refs` target must resolve.
* Every normative node must be testable or explicitly justified as non-testable.
* Every open node must have a decision owner or disposition.
* Every assumption must be traceable to source evidence.
* Machine-readable values must not disagree with human-readable prose.
* Generated files must not be manually edited.
* Any fact derivable from another node must not be stored redundantly.

---

# 9. Core kind set

The core kind set is versioned and intentionally small.

Prefer:

```text
small core + dialect extensions + decomposed nodes
```

over:

```text
large monolithic *_model nodes
```

The current implemented core is approximately 18 kinds. Additions to the core should be deliberate and versioned.

Core kinds that are generally justified:

| Kind                       | Purpose                                                                |
| -------------------------- | ---------------------------------------------------------------------- |
| `source`                   | Provenance artifact.                                                   |
| `claim`                    | Atomic statement about the design.                                     |
| `discrepancy` / `decision` | Conflicting source claims or unresolved ratification item.             |
| `param`                    | Named design parameter or constant.                                    |
| `type`                     | Data type, tile type, operand class, immediate type.                   |
| `state`                    | Architectural state: registers, memories, queues, flags, buffers.      |
| `unit`                     | Functional unit or architectural component.                            |
| `op`                       | Operation or instruction semantics.                                    |
| `port`                     | Unit-local port.                                                       |
| `channel`                  | Communication path or transaction channel.                             |
| `edge`                     | Dataflow or movement relation.                                         |
| `timing_class`             | Latency or timing formula.                                             |
| `constraint`               | Hard requirement, functional rule, target, PPA constraint, assumption. |
| `knob`                     | Design-space parameter / legal transformation target.                  |
| `test_intent`              | Required test or verification scenario.                                |
| `coverage_goal`            | Required coverage point.                                               |
| `artifact`                 | Downstream deliverable generated from or linked to the spec.           |
| `axis`                     | Cross-cutting concern such as RAS, security, power, QoS.               |

Potential core additions, if versioned:

| Proposed kind | Why it may deserve core status                                  |
| ------------- | --------------------------------------------------------------- |
| `interface`   | Externally visible boundary distinct from local ports/channels. |
| `protocol`    | Reusable transaction/handshake rule set.                        |
| `contract`    | Explicit assume/guarantee relationship.                         |
| `oracle`      | Reference model/equivalence source.                             |

Avoid adding `schedule_model` and `memory_model` as monolithic node kinds unless absolutely necessary. Scheduling and memory behavior should usually be represented as views/bundles composed from `param`, `state`, `op`, `edge`, `timing_class`, `constraint`, and `test_intent`.

---

# 10. Dialect extension model

Domain-specific concepts should live in dialects.

A node may include an `ext` field:

```yaml
- id: kernel.gemm
  kind: kernel
  status: normative
  refs:
    uses_ops: [op.vmatmul.mxu0, op.vmatpop.bf16.acc.mxu0]
  ext:
    npu:
      dataflow: weight_stationary
```

Dialect extensions must not be arbitrary free-form blobs. Each dialect must declare:

```yaml
dialect:
  name: npu
  version: 0.1.0
  kind_extensions:
    kernel:
      id_prefix: kernel.
      required_attrs: [operation]
      optional_attrs: [dataflow, shapes, tolerance]
      ref_roles: [uses_ops, reads, writes]
```

Dialect registration contract:

* dialect name must be registered,
* dialect version must be recorded,
* dialect-specific required attributes must validate,
* dialect-specific reference roles must resolve,
* dialect extensions must not override core semantics silently.

---

# 11. Derive, do not duplicate

Any fact derivable from another node must not be stored redundantly.

Examples:

* A state’s readers/writers are derived from `op.operands[].location` and `direction`.
* A coverage map is derived from `test_intent.refs.covers`.
* A unit’s used state can be derived from ops executed on that unit.
* A dataflow graph can be derived from `edge` nodes and op effects.
* Regeneration impact can be derived from `knob.refs.varies` plus artifact dependencies.

Rule:

> Redundant storage is a lint error unless explicitly cached with provenance and invalidation metadata.

This prevents the spec from drifting internally.

---

# 12. Required top-level sections

A complete accelerator spec must contain the following sections.

---

## 12.1 Overview and intent

Purpose: explain what the design is and why it exists.

Must include:

* design name,
* target domain,
* intended workloads,
* design goals,
* non-goals,
* target integration context,
* expected software stack,
* expected implementation target,
* architectural philosophy.

Example questions:

* Is this an inference accelerator, packet processor, video accelerator, DSP, or general compute fabric?
* What workloads justify the architecture?
* What is intentionally out of scope?
* Is the design controlled by software, firmware, static schedules, command queues, or autonomous hardware?

---

## 12.2 Source bundle and provenance

Purpose: document the evidence used to construct the spec.

Must include:

* source list,
* source role,
* trust level,
* claim ledger,
* discrepancies,
* known conflicts,
* known model defects,
* open ratification decisions.

A source can be high-trust for one purpose and low-trust for another.

Example:

```yaml
trust: high_for_functional_semantics_medium_for_cycle_exactness
```

This is critical when the RTL, executable model, slides, and ISA docs disagree.

---

## 12.3 Workloads and application contract

Purpose: define what the accelerator is designed to run.

Must include:

* kernel families,
* tensor shapes,
* data types,
* layouts,
* batching assumptions,
* performance-critical kernels,
* representative programs,
* input/output contracts,
* acceptance tolerances,
* workload gaps.

A computer architect’s spec should not only describe hardware. It should describe the workload contract that motivates the hardware.

---

## 12.4 Architectural state

Purpose: define all state that affects visible behavior.

Must include:

* scalar registers,
* vector/tensor registers,
* memories,
* queues,
* buffers,
* flags,
* CSRs/MMIO registers,
* program counter / control state,
* reset behavior,
* visibility,
* addressing rules,
* initialization rules.

Each `state` node should specify:

```yaml
id: state.vmem
kind: state
status: normative
attrs:
  category: memory
  visibility: software_visible
  capacity: 1 MiB
  banks: 8
  addressing: byte_offset
  reset_value: undefined
  conflict_policy: software_managed
```

State visibility should be explicit:

| Visibility               | Meaning                                                            |
| ------------------------ | ------------------------------------------------------------------ |
| `software_visible`       | Software can observe or manipulate it.                             |
| `architectural_internal` | Not directly software visible, but affects architectural behavior. |
| `microarchitectural`     | Implementation detail, not part of architecture.                   |
| `debug_visible`          | Exposed only for debug/perf counters.                              |

Reset value may be intentionally undefined. That is valid if explicit.

Examples:

```yaml
reset_value: zero
reset_value: undefined
reset_value: randomized_by_model_for_verification_only
```

Randomization in a model is a verification aid, not necessarily architecture.

---

## 12.5 Programmer’s model / software-visible interface

Purpose: define how software controls the accelerator.

Must include:

* address spaces,
* MMIO map,
* CSR map,
* command format,
* instruction memory loading,
* start/done protocol,
* status/error reporting,
* polling or interrupt behavior,
* memory ordering with host,
* ABI conventions,
* lifecycle from reset to completion.

If host transport is not yet defined, the spec must explicitly say so and create `decision` nodes.

Example:

```yaml
- id: decision.host_transport
  kind: decision
  status: open
  attrs:
    question: "How does the host start execution and observe completion?"
    recommended: "MMIO START/STATUS/ERROR control block"
    affects: [software, verification, soc_integration]
```

---

## 12.6 Operations / ISA semantics

Purpose: define what operations exist and what they mean.

For instruction-driven accelerators, this is the ISA. For command-queue accelerators, these are commands. For streaming accelerators, these are transactions or kernels.

Each `op` node must include:

```yaml
id: op.name
kind: op
status: normative
attrs:
  family: matrix | vector | dma | scalar | control
  syntax: optional
  encoding: optional
  blocking: true | false
  illegal_cases: []
refs:
  executes_on: unit
  latency_class: timing.some_latency
operands:
  - name: src
    type: type.some_type
    location: state.some_state
    direction: use
  - name: dst
    type: type.some_type
    location: state.some_state
    direction: def
effect:
  - def: dst
    expr: {op: some_semantic_function, args: [src]}
```

Each operation must define:

* operands,
* operand types,
* operand locations,
* use/def/inout direction,
* preconditions,
* postconditions,
* side effects,
* illegal cases,
* timing class,
* blocking/pipelined behavior,
* resource usage,
* semantics,
* verification obligations.

An opcode table without effects is not sufficient.

When encoding exists, encoding legality is normative. This includes:

* reserved-zero rules,
* operand-field constraints,
* selector bits hidden in otherwise reserved fields,
* pair-register legality,
* invalid encodings,
* immediate range constraints.

Example:

```yaml
encoding_legality:
  reserved_zero:
    - vs2 == 0
  pair_register:
    low_register_only: true
    illegal: r == 63
```

---

## 12.7 Data types and numerical semantics

Purpose: define exact numeric behavior.

Must include:

* data types,
* bit layouts,
* type views,
* conversion rules,
* rounding modes,
* saturation behavior,
* overflow/underflow behavior,
* NaN/Inf behavior,
* accumulation precision,
* quantization/dequantization,
* tolerance policy,
* bit-exact versus tolerance-bound operations.

Each type should be represented explicitly.

```yaml
- id: type.fp8_e4m3
  kind: type
  status: normative
  attrs:
    category: dtype
    bits: 8
    exp: 4
    mantissa: 3
    signed: true
    saturation: true
```

For low-precision accelerators, this section is mandatory. Numeric ambiguity is one of the easiest ways for generated RTL to pass trivial tests and fail real workloads.

---

## 12.8 Memory model

Purpose: define how data is addressed, moved, ordered, and observed.

Must include:

* memory spaces,
* address composition,
* capacity,
* banking,
* bank-conflict policy,
* alignment,
* access granularity,
* ordering rules,
* consistency rules,
* DMA semantics,
* completion semantics,
* initialization semantics,
* illegal memory behavior.

The memory model should generally be a **reading bundle / view**, not one monolithic `memory_model` node. It is composed from:

* `state`,
* `channel`,
* `edge`,
* `op`,
* `timing_class`,
* `constraint`,
* `test_intent`.

For each conflict or invalid access, define whether it is:

* statically illegal,
* dynamically trapped,
* stalled,
* undefined,
* implementation-defined,
* silently ignored.

Silent ambiguity is not allowed.

---

## 12.9 Execution and scheduling model

Purpose: define how operations execute over time.

Must include:

* issue model,
* ordering model,
* pipeline model,
* latency classes,
* overlap rules,
* blocking rules,
* structural hazards,
* dependency rules,
* fences/barriers/waits,
* delay slots if any,
* dynamic versus static scheduling,
* determinism assumptions.

The scheduling model should generally be a **reading bundle / view**, not one monolithic `schedule_model` node. It is composed from:

* `param`,
* `op`,
* `unit`,
* `timing_class`,
* `constraint`,
* `test_intent`.

The spec must distinguish:

| Concept                | Meaning                                      |
| ---------------------- | -------------------------------------------- |
| Functional ordering    | What result is visible.                      |
| Timing class           | When result is expected to complete.         |
| Resource occupancy     | Whether a unit can accept another operation. |
| Schedule legality      | Whether a program is valid.                  |
| Implementation latency | Actual RTL timing, if constrained.           |

---

## 12.10 Component architecture

Purpose: describe the main architectural units and their responsibilities.

Must include:

* units,
* hierarchy,
* visible state per unit,
* responsibilities,
* supported operations,
* local buffers,
* interfaces,
* latency/resource classes,
* topology constraints,
* what is architectural versus reference.

Example:

```yaml
- id: unit.mxu0
  kind: unit
  status: normative
  refs:
    visible_state: [state.mxu_weight_slots, state.mxu_accum_buffers]
    latency_class: timing.mxu0_matmul
  attrs:
    role: matrix_engine
    topology:
      value: systolic_array
      status: reference
    responsibility:
      - fp8_fp8_to_bf16_matmul
      - local_accumulation
```

A component diagram is useful, but it is not enough. Each component needs explicit state, interfaces, operations, and contracts.

---

## 12.11 Contract layer vs oracle layer

A spec may present a component in two layers:

| Layer              | Meaning                                                                                                                 | Status      |
| ------------------ | ----------------------------------------------------------------------------------------------------------------------- | ----------- |
| **Contract layer** | The clean normative datapath or behavior an implementer should design and lower to.                                     | `normative` |
| **Oracle layer**   | The incidental complexity or exact behavior an existing model/RTL exhibits and the implementation is evaluated against. | `reference` |

This pattern lets the spec simplify without lying.

When a component uses this split, the spec must declare an **abstraction-fidelity budget**:

```yaml
abstraction_fidelity:
  governing_metric: cycles | accuracy | utilization | area | energy
  max_deviation: 10%
  oracle: source.npu_model
  applies_to: [unit.mxu0]
```

Rule:

> A contract-faithful implementation may differ structurally from the oracle, but its deviation from the oracle must stay within the declared fidelity budget.

Default target: **≤10%** on the governing metric, unless the benchmark says otherwise.

This is especially useful when an existing model/RTL contains incidental complexity that should not be forced into every generated implementation.

---

## 12.12 Interfaces, ports, channels, and protocols

Purpose: define how components communicate.

Must include:

* external interfaces,
* internal ports,
* channels,
* endpoints,
* direction,
* width,
* transaction type,
* valid/ready behavior,
* backpressure,
* reset semantics,
* ordering,
* error behavior,
* clock/reset domain,
* protocol contracts.

Separate:

| Object      | Meaning                               |
| ----------- | ------------------------------------- |
| `interface` | Externally visible boundary.          |
| `port`      | Unit-local access point.              |
| `channel`   | Communication path between endpoints. |
| `protocol`  | Rules governing interaction.          |

Example:

```yaml
- id: channel.dma
  kind: channel
  status: normative
  refs:
    endpoints: [state.dram, state.vmem]
  attrs:
    direction: read_write
    count: 8
    protocol: "request {dram_addr, vmem_addr, length}; <=1 outstanding per channel"
```

Protocols should be first-class concepts, even if represented as `constraint` nodes in the current core.

---

## 12.13 Constraints

Purpose: define what must hold.

Constraints should be typed.

| Constraint kind | Meaning                                            |
| --------------- | -------------------------------------------------- |
| `hard`          | Fixed architectural value.                         |
| `functional`    | Required behavior.                                 |
| `timing`        | Latency or ordering requirement.                   |
| `resource`      | Capacity, bandwidth, port, queue, or banking rule. |
| `target`        | Performance target.                                |
| `ppa`           | Area/power/frequency target.                       |
| `qoi`           | Quality-of-implementation target.                  |
| `assumption`    | Modeling assumption.                               |

Example:

```yaml
- id: constraint.no_over_issue
  kind: constraint
  status: normative
  attrs:
    kind_of: functional
    text: "MUST NOT issue to a non-DMA unit while busy."
```

Every hard constraint should be machine-checkable where possible.

---

## 12.14 Quality-of-implementation targets

Purpose: define how good an implementation should be.

Must include, when relevant:

* cycle count,
* throughput,
* utilization,
* energy/op,
* power,
* area,
* frequency,
* timing slack,
* memory traffic,
* stall breakdown,
* bandwidth efficiency,
* comparison against manual/reference design.

Each target should be labeled:

* required,
* target,
* reference,
* stretch,
* open.

Do not mix correctness requirements with performance aspirations.

---

## 12.15 Verification obligations

Purpose: define what must be checked.

The spec defines **what must be verified**. The verification flow defines **how and when to verify it**.

Must include:

* directed tests,
* random tests,
* negative tests,
* protocol assertions,
* numerical tests,
* memory-ordering tests,
* schedule legality tests,
* coverage goals,
* equivalence modes,
* oracles.

Example:

```yaml
- id: test.neg.dma_busy_channel
  kind: test_intent
  status: normative
  refs:
    covers: [op.dma.load, op.dma.store]
  attrs:
    test_type: negative
    expect: error
```

Coverage goals should also be first-class:

```yaml
- id: cov.every_mnemonic
  kind: coverage_goal
  status: normative
  attrs:
    text: "Every op node executed at least once."
```

Every normative behavior should map to at least one verification obligation, or be explicitly marked as untested with justification.

---

## 12.16 Error handling and undefined behavior

Purpose: remove ambiguity around invalid behavior.

Must define:

* illegal instructions,
* invalid encodings,
* invalid operands,
* misalignment,
* out-of-bounds memory,
* bank conflicts,
* over-issue,
* unsupported shapes,
* illegal schedules,
* numeric out-of-domain cases,
* DMA misuse,
* protocol violations.

For each invalid case, specify the behavior:

| Behavior                 | Meaning                                    |
| ------------------------ | ------------------------------------------ |
| `static_reject`          | Tool/compiler/spec checker rejects it.     |
| `runtime_error`          | Hardware/model reports an error.           |
| `halt`                   | Execution stops with cause.                |
| `stall`                  | Hardware waits until legal.                |
| `undefined`              | No guarantee; tests must avoid it.         |
| `implementation_defined` | Implementation must document its behavior. |

Example:

```yaml
error_behavior:
  bank_conflict:
    classification: illegal
    response: runtime_error_or_static_reject
    silent_corruption: prohibited
```

Undefined behavior must be intentional, documented, and minimized.

---

## 12.17 Design-space knobs and flavors

Purpose: define what can vary legally.

A spec should not freeze every design choice. It should identify legal variation points.

Each `knob` must define:

* choices or range,
* default,
* affected nodes,
* whether it changes architecture,
* whether it changes microarchitecture,
* whether it is software-compatible,
* what must be regenerated when it changes,
* what it buys.

Example:

```yaml
- id: knob.vmem.bank_count
  kind: knob
  status: normative
  refs:
    varies: [state.vmem]
  attrs:
    choices: [4, 8, 16]
    default: 8
    affects: [microarchitecture]
    sw_compatible: false
    regenerate: [bank_conflict_model, schedules, tests]
    rationale: "Explore SRAM banking vs conflict pressure."
    choice_semantics:
      4: {kind: packaging, expected_benefit: lower_area}
      8: {kind: baseline, expected_benefit: balanced}
      16: {kind: packaging, expected_benefit: higher_bandwidth}
```

Each knob or flavor must declare, per choice, whether it changes:

| Variation type   | Meaning                                                             |
| ---------------- | ------------------------------------------------------------------- |
| `capability`     | Adds/removes externally visible behavior.                           |
| `packaging`      | Same behavior, different encoding, structure, cost, or performance. |
| `implementation` | Same architecture, different RTL/microarchitecture.                 |
| `benchmark`      | Used only for controlled experiments.                               |

Rule:

> A variation that adds complexity without capability, measurable cost benefit, performance benefit, verification benefit, or benchmark value is bloat, not a design point.

Flavors are named configurations of knobs.

```yaml
flavor:
  id: flavor.baseline
  knobs:
    knob.vmem.bank_count: 8
    knob.tile.shape: 32x32
    knob.vpu.lanes_bf16: 16
```

---

## 12.18 IR mapping

Purpose: define how the spec lowers into IR.

Every spec concept should map to an IR object.

| Spec concept               | IR object                                                           |
| -------------------------- | ------------------------------------------------------------------- |
| `type`                     | IR type                                                             |
| `state`                    | IR location / memory / register file / buffer                       |
| `op`                       | IR operation                                                        |
| `op.operands`              | IR operands and results                                             |
| `op.effect`                | IR value graph / semantic body                                      |
| `unit`                     | IR module / instance                                                |
| `port` / `channel`         | IR port / channel                                                   |
| `protocol`                 | IR protocol contract                                                |
| `timing_class`             | IR latency attribute                                                |
| `constraint`               | IR contract / verifier condition                                    |
| `edge`                     | IR data movement                                                    |
| `knob`                     | IR transform target                                                 |
| `test_intent`              | IR test obligation                                                  |
| `coverage_goal`            | IR coverage target                                                  |
| `claim`                    | IR provenance annotation                                            |
| `discrepancy` / `decision` | IR unresolved-decision metadata or compile-time error before freeze |

This mapping should be mostly mechanical.

If a concept cannot be mapped to IR, either the spec is too informal or the IR is missing a required abstraction.

---

## 12.19 Cross-cutting axes

Some architecture concerns span many nodes. They should be represented explicitly as `axis` nodes, even when initially open.

Required cross-cutting axes:

* interconnect / scale-out,
* reliability / RAS,
* security / isolation,
* power / clock / thermal,
* QoS / preemption / virtualization,
* capability discovery / versioning,
* debug / observability,
* physical-design constraints,
* DFT / testability,
* software compatibility.

Example:

```yaml
- id: axis.power_clock_thermal
  kind: axis
  status: open
  attrs:
    what_is_known:
      - single_clock_domain
    undefined:
      - clock_gating
      - power_domains
      - dvfs
      - thermal_throttling
    affects:
      - physical_design
      - soc_integration
```

Open does not mean ignored. It means surfaced and tracked.

---

## 12.20 Downstream artifacts

The spec should define what artifacts are expected downstream.

Examples:

| Artifact                | Purpose                                             |
| ----------------------- | --------------------------------------------------- |
| IR generation           | Lower structured spec to LHWIR.                     |
| Functional model        | Execute architectural semantics.                    |
| Resource/cost model     | Estimate cycles, stalls, utilization, energy, area. |
| Virtual platform        | Run software-visible device model.                  |
| RTL prompt/scaffold     | Generate direct or IR-mediated RTL tasks.           |
| RTL testbench           | Check generated RTL.                                |
| Assertions              | Enforce protocol/timing/memory contracts.           |
| Scoreboards             | Compare outputs/traces to oracles.                  |
| Coverage plan           | Track exercised behavior.                           |
| PPA flow                | Synthesis/timing/power evaluation.                  |
| Driver/runtime contract | Software integration.                               |

Each artifact should say:

* what it depends on,
* what spec nodes it consumes,
* when it must be regenerated.

---

# 13. Reading bundles

Some sections are one coupled story. If one file in the bundle changes, the whole bundle must be revalidated.

Example bundles:

| Bundle                            | Files / concepts                          | Why coupled                                                         |
| --------------------------------- | ----------------------------------------- | ------------------------------------------------------------------- |
| **Semantics**                     | ISA, numerics, state                      | Operation effects depend on state and numeric interpretation.       |
| **Execution / memory / dataflow** | scheduling, timing, memory, data movement | Legal schedules depend on memory/resource behavior.                 |
| **Programmer interface**          | programmer model, host control, errors    | Software-visible behavior depends on lifecycle and error reporting. |
| **Performance / DSE / quality**   | constraints, design space, QoI, workloads | Knobs must be evaluated against workload and quality targets.       |
| **Components / interfaces**       | units, ports, channels, protocols         | Structure and communication must agree.                             |
| **Verification**                  | test obligations, coverage, oracles       | Every normative behavior must be checked.                           |
| **Traceability**                  | provenance, claims, decisions, IR mapping | Claims must remain source-backed through lowering.                  |

Rule:

> A change to any member of a reading bundle triggers validation of the full bundle.

---

# 14. Machine-readable file layout

Recommended spec bundle layout:

```text
spec/
  README.md
  META.md
  manifest.yaml
  lint.py

  human/
    00_intent.md
    SPEC.md
    workloads.md
    numerics.md
    sw_interface.md
    ir_requirements.md
    ir_mapping.md
    SPEC_DEFINITION.md

  machine/
    provenance.yaml
    types.yaml
    state.yaml
    spec.yaml
    isa.yaml
    interfaces.yaml
    constraints.yaml
    design_space.yaml
    test_obligations.yaml
    cross_cutting.yaml
    downstream.yaml
    kernels.yaml

  schema/
    core.py
    load.py

  dialects/
    npu.yaml

  slices/
    <slice_name>/
      human/
      machine/
      manifest.yaml

  assets/
    images/
```

`human/` explains the design.
`machine/` defines the typed graph.
`schema/` validates the graph.
`dialects/` extends the core spec vocabulary.
`slices/` contains isolated sub-bundles for focused experiments.
`assets/` contains diagrams and referenced images.

---

# 15. Validation as export gate

The spec must ship a runnable validator.

The validator is a hard export gate:

```text
exit 0 required
```

A spec that does not pass validation is **S2-incomplete** and must not be used as a frozen benchmark input.

The validator must check at least:

* all IDs are unique,
* every `refs` ID resolves,
* status vocabulary is valid,
* kind vocabulary is valid,
* dialect namespaces are registered,
* dialect-required attributes are present,
* every op operand type resolves,
* every op operand location resolves,
* every effect `def` is a declared `def` or `inout` operand,
* every effect operand name is declared,
* generated files declare generator and inputs,
* generated files are not manually edited,
* human↔machine critical constants agree,
* every multi-source disagreement has a `discrepancy` or `decision` node,
* every normative behavior has a verification obligation or explicit waiver,
* every knob declares affected nodes and regeneration impact,
* every reading bundle is complete,
* manifest file set is complete.

Validation is not optional documentation hygiene. It is part of the spec.

---

# 16. Minimal required node schema by kind

## `type`

A `type` defines data representation.

Required fields:

```yaml
id
kind: type
status
attrs:
  category: dtype | tile | operand_class | address | immediate
  bits: optional
  shape: optional
  element: optional
  layout: optional
  numeric_semantics: optional
```

---

## `state`

A `state` node defines architectural storage.

Required fields:

```yaml
id
kind: state
status
attrs:
  category: register | register_file | memory | buffer | queue | flag | csr
  visibility: software_visible | architectural_internal | microarchitectural | debug_visible
  capacity/count/width/shape
  addressing: optional
  reset_value
  access_semantics
```

---

## `unit`

A `unit` defines an architectural component.

Required fields:

```yaml
id
kind: unit
status
refs:
  visible_state: []
  supported_ops: []
  interfaces: []
attrs:
  role
  responsibility
  topology: optional
  implementation_constraints: optional
```

---

## `op`

An `op` defines executable semantics.

Required fields:

```yaml
id
kind: op
status
attrs:
  family
  syntax: optional
  encoding: optional
  encoding_legality: optional
  blocking: optional
  illegal_cases: []
refs:
  executes_on: unit
  latency_class: timing_class
operands:
  - name
    type
    location
    direction: use | def | inout
effect:
  - def
    expr
```

---

## `channel`

A `channel` defines communication.

Required fields:

```yaml
id
kind: channel
status
refs:
  endpoints: []
attrs:
  direction
  width: optional
  protocol
  ordering
  backpressure
  error_behavior
```

---

## `constraint`

A `constraint` defines a requirement or target.

Required fields:

```yaml
id
kind: constraint
status
refs:
  subject: optional
attrs:
  kind_of: hard | functional | timing | resource | target | ppa | qoi | assumption
  property: optional
  value: optional
  text: optional
```

---

## `knob`

A `knob` defines legal variation.

Required fields:

```yaml
id
kind: knob
status
refs:
  varies: []
attrs:
  choices or range
  default
  affects
  sw_compatible
  regenerate
  rationale
  choice_semantics
```

---

## `test_intent`

A `test_intent` defines a required verification scenario.

Required fields:

```yaml
id
kind: test_intent
status
refs:
  covers: []
attrs:
  test_type
  oracle
  applies_to
  expected
  tolerance: optional
```

---

## `decision` / `discrepancy`

A `decision` or `discrepancy` records unresolved or resolved source conflicts.

Required fields:

```yaml
id
kind: decision | discrepancy
status: open | normative | rejected
refs:
  about: []
  sources: []
attrs:
  source_a_claim
  source_b_claim
  normative_choice
  disposition
  owner: optional
  freeze_blocking: true | false
```

---

# 17. Consistency rules

The spec is valid only if these rules hold.

1. Every node ID is unique.
2. Every reference resolves.
3. Every normative claim has provenance.
4. Every multi-source disagreement has a `discrepancy` or `decision` node.
5. Every normative behavior has a verification obligation.
6. Every operation has typed operands and defined effects or explicitly references an external semantic function.
7. Every operation with encoding defines encoding legality.
8. Every state object has reset/initialization semantics, even if intentionally undefined.
9. Every memory has addressing, capacity, ordering, and error behavior.
10. Every timing claim is either normative, reference, assumption, or open.
11. Every design-space knob declares rationale, affected nodes, and regeneration impact.
12. Every generated file declares origin, generator, and inputs.
13. Generated files are not hand-edited.
14. Human-readable values and machine-readable values do not disagree.
15. Every open issue is explicit.
16. Every assumption is traceable.
17. Every downstream artifact declares dependencies.
18. Every IR mapping either exists or is marked missing.
19. No hidden implementation knowledge is required to implement the normative spec.
20. No redundant derivable facts are stored without cache/invalidation metadata.
21. The validator passes.

---

# 18. Good-spec checklist

Before a spec can be used as a benchmark, it should answer:

* What workloads is this accelerator for?
* What state exists?
* What operations exist?
* What does every operation do?
* What are the legal and illegal cases?
* What are the memory spaces?
* What are the ordering rules?
* What are the interfaces?
* What timing is architecturally visible?
* What is implementation-defined?
* What is open?
* What source conflicts exist?
* What normative choices were made?
* What is a design-space knob?
* What does each variation buy?
* What must be tested?
* What is the oracle?
* What maps to IR?
* What artifacts are generated from this spec?
* Does the validator pass?
* Can another person implement the design without reading the RTL?
* Would direct-Claude and LHWIR-mediated flows receive the same normative information?

---

# 19. NPU-specific instantiation

For the Atlas / Tapeout NPU, the general spec definition instantiates as:

| General spec concept    | NPU instance                                                                 |
| ----------------------- | ---------------------------------------------------------------------------- |
| Workload contract       | GEMM, attention, RMSNorm, softmax, activations, requantization               |
| Data types              | FP8 E4M3, BF16, FP32 reference                                               |
| State                   | XRF, ERF, MRF, IMEM, VMEM, DRAM, DMA flags, MXU buffers                      |
| Units                   | Frontend, SALU, MXU0, MXU1, VPU, XLU, LSU, DMA                               |
| Ops                     | RV32-style scalar ops, tensor ops, matrix ops, vector ops, DMA ops           |
| Memory model            | IMEM/VMEM/DRAM, DMA-only DRAM↔VMEM, 32-byte conflict granularity             |
| Scheduling              | Single-issue, in-order, static latency-annotated execution                   |
| Knobs                   | MXU topology/count, VMEM banks/capacity, VPU lanes, DMA channels, tile shape |
| Verification            | Kernel tests, negative tests, protocol assertions, numeric edge coverage     |
| Oracle                  | `npu_model`, Python workload references, later IR simulator                  |
| IR target               | LHWIR architectural-semantic layer, then structural/microarch lowering       |
| Reconciliation examples | DMA encoding, PC units, ERF scale semantics, over-issue behavior             |

The NPU is not the definition of a spec. It is the first concrete instance of the spec formalism.

---

# Final rule

For this project, a spec is valid only if it can drive automation.

A valid spec should be able to generate or configure:

```text
IR
functional simulator
resource/cost model
virtual platform
RTL-generation task
testbench
assertions
scoreboards
coverage goals
baseline prompt/scaffold
failure triage metadata
```

Therefore, the spec is not merely documentation.

It is the root artifact for architecture, generation, modeling, verification, and comparison.
