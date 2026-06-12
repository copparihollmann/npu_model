"""
core.py — the typed core meta-model for the Atlas spec bundle.

Every spec entity is a NODE sharing one envelope:

    id:         namespaced unique id            (e.g. op.vadd.bf16, state.mrf)
    kind:       discriminator from the registry  (core kind or dialect kind)
    status:     normative|reference|assumption|open|rejected
    provenance: {sources: [source-ids], confidence: high|medium|low}
    doc:        human note
    attrs:      kind-specific typed fields
    refs:       {role: id | [ids]}               (typed graph edges -> resolve to nodes)
    ext:        {dialect: {...}}                  (dialect extension, validated by that dialect)
    operands:   [ {name,type,location,direction} ]   (op kind only)
    effect:     [ {def: name, expr: <expr>} ]        (op kind only; small typed AST)

`id` + `kind` + `refs` make the whole bundle a typed graph -> mechanically lowerable
to an SSA-style IR (see human/ir_mapping.md). This module has NO third-party deps
(stdlib only) so the bundle stays portable; load.py adds PyYAML to read the files.

The kind vocabulary is a CLOSED, versioned core (the IR depends on it). Accelerator-
specific concepts are added by DIALECTS that register extra kinds / ext schemas
(see register_kind / register_ext) — designers extend a dialect, never this core.
"""
from __future__ import annotations
from dataclasses import dataclass, field

META_VERSION = "0.1.0"

STATUSES = {"normative", "reference", "assumption", "open", "rejected"}
CONFIDENCE = {"high", "medium", "low"}
DIRECTIONS = {"def", "use", "inout"}


@dataclass(frozen=True)
class KindSpec:
    """Declarative schema for one node kind. Dialects add more of these."""
    name: str
    id_prefix: str                      # convention: id should start with "<prefix>."
    required_attrs: tuple = ()
    optional_attrs: tuple = ()
    # role -> expected target kind(s); None = any kind. value may be id or [ids].
    ref_roles: dict = field(default_factory=dict)
    has_operands: bool = False          # op-style nodes carry operands + effect
    has_effect: bool = False
    dialect: str = "core"               # "core" or a dialect name


# ---------------------------------------------------------------------------
# The fixed CORE kind vocabulary (~16). Closed + versioned.
# ---------------------------------------------------------------------------
CORE_KINDS: dict[str, KindSpec] = {k.name: k for k in [
    KindSpec("type",         "type",
             required_attrs=("category",),                       # dtype|operand_class|unit|tile
             optional_attrs=("bits", "exp", "mantissa", "signed", "layout", "element", "of", "notes")),
    KindSpec("param",        "param",
             required_attrs=("value",), optional_attrs=("unit", "choices", "notes")),
    KindSpec("unit",         "unit",
             required_attrs=("role",),
             optional_attrs=("responsibility", "topology", "latency_class", "notes"),
             ref_roles={"visible_state": ("state",), "latency_class": ("timing_class",),
                        "ports": ("port", "channel")}),
    KindSpec("state",        "state",
             required_attrs=("category", "visibility"),          # register_file|memory|buffer|flag|register
             optional_attrs=("count", "width_bits", "capacity", "banks", "geometry",
                             "reset_value", "names", "special", "conflict_policy",
                             "conflict_granularity_bytes", "addressing", "notes"),
             ref_roles={"element_type": ("type",), "per_unit": ("unit",)}),
    KindSpec("port",         "port",
             required_attrs=("direction",),
             optional_attrs=("protocol", "width_bits", "width_bytes", "bytes_per_beat",
                             "core_cycles_per_beat", "count", "notes"),
             ref_roles={"of_unit": ("unit",), "accesses": ("state",), "carries": ("type",)}),
    KindSpec("channel",      "channel",
             required_attrs=("direction",),
             optional_attrs=("protocol", "count", "realized_as", "notes"),
             ref_roles={"endpoints": ("state", "unit", "port")}),
    KindSpec("op",           "op",
             required_attrs=("family",),
             optional_attrs=("exu", "encoding", "blocking", "illegal_cases", "sem", "notes"),
             ref_roles={"latency_class": ("timing_class",), "unit": ("unit",)},
             has_operands=True, has_effect=True),
    KindSpec("timing_class", "timing",
             required_attrs=(), optional_attrs=("cycles", "formula", "notes")),
    KindSpec("edge",         "edge",
             required_attrs=(),
             optional_attrs=("payload", "latency", "notes"),
             ref_roles={"src": ("state", "unit", "port"), "dst": ("state", "unit", "port"),
                        "via": None, "payload_type": ("type",)}),
    KindSpec("constraint",   "constraint",
             required_attrs=("kind_of",),                        # hard|functional|target|ppa|assumption|qoi
             optional_attrs=("property", "op", "value", "unit", "text", "label", "basis", "notes"),
             ref_roles={"subject": None}),
    KindSpec("knob",         "knob",
             required_attrs=(),
             optional_attrs=("choices", "range", "default", "affects", "sw_compatible",
                             "regenerate", "notes"),
             ref_roles={"varies": None}),
    KindSpec("test_intent",  "test",
             required_attrs=("test_type",),
             optional_attrs=("oracle", "applies_to", "shapes", "expect", "tolerance", "notes"),
             ref_roles={"covers": None, "tolerance_ref": None}),
    KindSpec("coverage_goal", "cov",
             required_attrs=("text",), optional_attrs=("notes",), ref_roles={"covers": None}),
    KindSpec("claim",        "claim",
             required_attrs=("text",),
             optional_attrs=("confidence", "unresolved_questions", "notes"),
             ref_roles={"about": None, "sources": ("source",)}),
    KindSpec("source",       "source",
             required_attrs=("type", "trust"), optional_attrs=("path", "role", "notes")),
    KindSpec("decision",     "decision",
             required_attrs=("question",),
             optional_attrs=("recommended", "alt", "affects", "notes")),
    KindSpec("axis",         "axis",
             required_attrs=("title",),
             optional_attrs=("what_is_known", "undefined", "recommended_dir", "affects", "notes")),
    KindSpec("artifact",     "artifact",
             required_attrs=("name", "owner"),
             optional_attrs=("file", "scope", "notes"),
             ref_roles={"depends_on": ("decision", "artifact")}),
]}


# ---------------------------------------------------------------------------
# Dialect registration — the controlled-extensibility tier.
# ---------------------------------------------------------------------------
def fresh_registry() -> dict[str, KindSpec]:
    """A mutable copy of the core registry that dialects extend."""
    return dict(CORE_KINDS)


def register_kind(registry: dict[str, KindSpec], spec: KindSpec) -> None:
    if spec.name in registry and registry[spec.name].dialect == "core":
        raise ValueError(f"dialect may not redefine core kind '{spec.name}'")
    registry[spec.name] = spec


@dataclass
class Node:
    id: str
    kind: str
    status: str = "normative"
    provenance: dict = field(default_factory=dict)
    doc: str = ""
    attrs: dict = field(default_factory=dict)
    refs: dict = field(default_factory=dict)
    ext: dict = field(default_factory=dict)
    operands: list = field(default_factory=list)
    effect: list = field(default_factory=list)
    src_file: str = ""                  # provenance for error messages

    @staticmethod
    def from_dict(d: dict, src_file: str = "") -> "Node":
        known = {"id", "kind", "status", "provenance", "doc", "attrs", "refs",
                 "ext", "operands", "effect"}
        attrs = dict(d.get("attrs") or {})
        # tolerate flat authoring: unknown top-level keys fold into attrs
        for k, v in d.items():
            if k not in known:
                attrs[k] = v
        return Node(
            id=d.get("id", ""), kind=d.get("kind", ""),
            status=d.get("status", "normative"),
            provenance=d.get("provenance") or {}, doc=d.get("doc", ""),
            attrs=attrs, refs=d.get("refs") or {}, ext=d.get("ext") or {},
            operands=d.get("operands") or [], effect=d.get("effect") or [],
            src_file=src_file,
        )


def _expr_operand_names(expr) -> set:
    """Collect operand-name references used inside an effect expr (a small AST)."""
    names: set = set()
    if isinstance(expr, str):
        names.add(expr)
    elif isinstance(expr, dict):
        if "lit" in expr:
            return names                       # literal, no operand ref
        for a in expr.get("args", []) or []:
            names |= _expr_operand_names(a)
    elif isinstance(expr, list):
        for a in expr:
            names |= _expr_operand_names(a)
    return names


def validate_graph(nodes: list[Node], registry: dict[str, KindSpec]) -> list[str]:
    """Return a list of error strings ([] == valid). Pure; no I/O."""
    errs: list[str] = []
    by_id: dict[str, Node] = {}
    for n in nodes:
        loc = f"{n.src_file}:{n.id or '<no-id>'}"
        if not n.id:
            errs.append(f"{loc}: node missing 'id'"); continue
        if n.id in by_id:
            errs.append(f"{loc}: duplicate id (also in {by_id[n.id].src_file})"); continue
        by_id[n.id] = n

    def kind_target_ok(target_id: str, expected) -> bool:
        if target_id not in by_id:
            return False
        if expected is None:
            return True
        return by_id[target_id].kind in expected

    for n in nodes:
        loc = f"{n.src_file}:{n.id}"
        ks = registry.get(n.kind)
        if ks is None:
            errs.append(f"{loc}: unknown kind '{n.kind}' (not core or registered dialect)"); continue
        if n.status not in STATUSES:
            errs.append(f"{loc}: bad status '{n.status}'")
        conf = (n.provenance or {}).get("confidence")
        if conf is not None and conf not in CONFIDENCE:
            errs.append(f"{loc}: bad provenance.confidence '{conf}'")
        if ks.id_prefix and not n.id.startswith(ks.id_prefix + "."):
            errs.append(f"{loc}: id should start with '{ks.id_prefix}.' for kind '{n.kind}'")
        # attrs: required are enforced; extra attrs are allowed (free metadata the IR
        # may ignore) so designers are not over-constrained. The hard guarantees are
        # on ids, refs, operands and effect below.
        for req in ks.required_attrs:
            if req not in n.attrs:
                errs.append(f"{loc}: kind '{n.kind}' requires attr '{req}'")
        # refs
        for role, target in n.refs.items():
            if role not in ks.ref_roles:
                errs.append(f"{loc}: ref role '{role}' not allowed for kind '{n.kind}'"); continue
            expected = ks.ref_roles[role]
            for tid in (target if isinstance(target, list) else [target]):
                if not kind_target_ok(tid, expected):
                    exp = f" (expected kind in {expected})" if expected else ""
                    errs.append(f"{loc}: ref '{role}' -> '{tid}' does not resolve{exp}")
        # operands / effect (op-style nodes)
        if n.operands and not ks.has_operands:
            errs.append(f"{loc}: kind '{n.kind}' may not carry operands")
        opnames = set()
        for o in n.operands:
            nm = o.get("name")
            if not nm:
                errs.append(f"{loc}: operand missing 'name'"); continue
            opnames.add(nm)
            if o.get("direction") not in DIRECTIONS:
                errs.append(f"{loc}: operand '{nm}' bad direction '{o.get('direction')}'")
            t = o.get("type")
            if t is not None and not kind_target_ok(t, ("type",)):
                errs.append(f"{loc}: operand '{nm}' type '{t}' is not a type node")
            l = o.get("location")
            if l is not None and not kind_target_ok(l, ("state",)):
                errs.append(f"{loc}: operand '{nm}' location '{l}' is not a state node")
        defs = {o["name"] for o in n.operands if o.get("direction") in ("def", "inout")}
        for e in n.effect:
            d = e.get("def")
            if d not in defs:
                errs.append(f"{loc}: effect writes '{d}' which is not a def/inout operand")
            for ref_nm in _expr_operand_names(e.get("expr")):
                if ref_nm not in opnames:
                    errs.append(f"{loc}: effect expr references unknown operand '{ref_nm}'")
        # ext must name a registered dialect
        for dia in n.ext:
            if not any(k.dialect == dia for k in registry.values()):
                errs.append(f"{loc}: ext namespace '{dia}' is not a registered dialect")
    return errs
