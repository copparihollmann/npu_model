# Atlas NPU Subset — Architecture Specification Bundle

**Version 0.1.0 · Maturity S2 · self-contained for evaluation.**

A versioned architecture contract for the Atlas NPU subset — a statically-scheduled,
FP8/BF16, tile-oriented accelerator for edge VLA / Physical-AI inference.

## Folder layout

```
spec/
  README.md          ← start here (front door)
  META.md            ← how the machine spec is composed (envelope, kinds, refs, dialects, effect)
  manifest.yaml      ← machine index: dialects, core kinds, lineage, source-of-truth, reading bundles
  lint.py            ← export gate:  python spec/lint.py
  human/             ← S0/S1 human narrative (.md) — AUTHORED, canonical for prose
    00_intent.md       design intent + canonical pipeline (read 1st)
    workloads.md       what it must run: kernels, shapes, rooflines (read 2nd)
    SPEC.md            the canonical spec, §0–§22 (read 3rd; §0 lists reading bundles)
    numerics.md  sw_interface.md  ir_requirements.md  ir_mapping.md   (annexes)
  machine/           ← S2 TYPED NODE GRAPH (.yaml) — AUTHORED, canonical for VALUES/CLAIMS
    types.yaml state.yaml spec.yaml isa.yaml interfaces.yaml constraints.yaml
    design_space.yaml test_obligations.yaml cross_cutting.yaml downstream.yaml
    provenance.yaml kernels.yaml      (one envelope; loader merges all into one graph)
  schema/            ← typed core meta-model + loader/validator (core.py, load.py)
  dialects/          ← registered dialects extending the core (npu.yaml)
  assets/images/     ← diagrams referenced by the narrative
```

Every `machine/` entry is a **node** `{id, kind, status, attrs, refs, ...}` (see
`META.md`). The whole machine spec is a typed graph with by-id references — it
validates (`python spec/schema/load.py`) and lowers mechanically to an SSA-style IR
(`human/ir_mapping.md`). Accelerator-specific concepts are added via **dialects**,
never by changing the core kinds.

## What comes from where (lineage)

```
GIVEN (external, outside spec/)        AUTHORED here (hand-written)          OUTPUT (future, NOT here)
  npu_model   ─┐                         human/   (S0,S1 .md narrative)        IR
  npu_spec/   ─┤  distilled into ───►     machine/ (S2 .yaml, canonical) ──►   models · cost models
  tests/      ─┤                            ▲         │                         tests · scoreboards
  speed_of_light ┘                          └─ lint ──┘  (keeps them agree)     RTL-generation tasks
```

- **Given** sources live *outside* `spec/`; this bundle is distilled from them.
  `npu_model` is also the **evaluation oracle**.
- **Authored**: `human/` and `machine/` are **both hand-written**. Nothing here is
  auto-generated yet. `lint.py` keeps them consistent.
- **Output**: IR / models / tests are **future** artifacts generated *from* `machine/`
  (the IR), enumerated in `machine/downstream.yaml`. They do not exist in this bundle.
- **Supersession**: this bundle **supersedes `npu_spec/`** as the contract; inside it,
  **`machine/*.yaml` supersedes `human/*.md`** for any value/claim.

Full machine-readable lineage, per-file roles, and reading bundles: `manifest.yaml`.

## Source of truth

`machine/*.yaml` is canonical for **every value and claim**; `human/SPEC.md` is
canonical for **prose/rationale**. They may never disagree — `lint.py` enforces it.

## Validate (export gate)

```
python spec/lint.py        # exit 0 = bundle internally consistent
```
Checks: YAML parses, manifest file-set complete, critical constants agree between
`human/SPEC.md` and `machine/spec.yaml`, ISA mnemonics unique, provenance well-formed.

## Status, scope, open items

- **Scope:** architecture + required microarch *constraints*, **repo ISA only**.
  Not microarch realization / SoC-PHY / DV / PPA — those follow-on artifacts are
  enumerated in `human/SPEC.md §21` / `machine/downstream.yaml`.
- **Four open ratification decisions** before freeze (recommended defaults in §2):
  scalar width, VMEM bus width, DRAM aperture, host transport + STATUS/ERROR CSRs.
- **Model defects vs `npu_spec`** (npu_spec is normative, fix the model):
  `dma.config`/`dma.wait` funct7, PC units (word vs byte), ERF scale (E8M0 vs integer)
  — `machine/downstream.yaml:model_defects`.
- **§22 cross-cutting axes** (interconnect/RAS/security/power/QoS/capability) are
  recorded as explicit `open`/undefined — `machine/cross_cutting.yaml`.

## Relationship to `npu_spec/`

`npu_spec/` is the prior **human-only** expert spec and the primary source this bundle
distills and **supersedes**. This bundle adds the machine-readable layer, provenance,
e2e register, and — crucially — **reconciliation against the executable model** (which
caught real disagreements between `npu_spec/` and `npu_model`). Section coverage and
per-claim provenance: `machine/provenance.yaml`.
