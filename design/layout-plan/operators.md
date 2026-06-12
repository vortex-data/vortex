<!--
SPDX-License-Identifier: Apache-2.0
SPDX-FileCopyrightText: Copyright the Vortex contributors
-->

# Layout Plan — Operators & Demand, in detail

Detailed design for Piece 2 (operators, edges, demand), refined onto a DuckDB-like push-based
pipeline execution model. Overview and decision log: [`plan-sketch.md`](./plan-sketch.md),
[`requirements.md`](./requirements.md).

## 1. Execution model: DuckDB mapping

| DuckDB | Here | Notes |
|---|---|---|
| `DataChunk` | `Patch` — a value stamped with the contiguous `RowRegion` it covers | Regions live in a declared row domain (root rows, values rows, zone indices). |
| Source / Operator / Sink | `Source` / `StreamOp` / `Sink` | Same push-based roles, same result enums. |
| Pipeline | Maximal Source→StreamOp*→Sink chain | Pipeline breakers: `Materialize`, accumulator collectors, the output collector. |
| Pipeline dependencies | Consumer pipelines depend on the `Materialize`/collector sinks they read | Everything else overlaps freely. |
| Morsel-driven parallelism | **Bands × pipelines** | The executor lazily instantiates pipeline instances per row *band* (lazy unroll, D5); patches are the morsels flowing within an instance. |
| Global / local operator state | Plan-shared state (SIP, materialized values, dynamic-filter versions) / per-band instance state | |
| `Blocked` + interrupt callbacks | Async: sources are I/O-driven futures on the Vortex runtime | Pipelines are spawned tasks; channels at pipeline boundaries. |

A crucial difference from DuckDB: DuckDB pipelines read *all* columns at the source of one
pipeline. Here, filter-column subgraphs and projection-column subgraphs are **separate
pipelines over the same rows**, coupled only through (a) exact mask edges and (b) SIP. The old
hardcoded prune → filter → project protocol becomes *scheduling policy*: prune pipelines are
cheap and scheduled first, projection sources may be dispatched late (to exploit refined
demand) or early (when demand is already dense) — a cost decision, not a correctness structure.

## 2. Edges: the patch stream contract

```rust
/// One item on an edge. `region` is in the edge's RowDomain, in that domain's coordinates.
pub struct Patch<V> {
    region: Range<u64>,
    value: PatchValue<V>,
}

pub enum PatchValue<V> {
    Data(V),          // Bytes | ArrayRef | Mask | AccState | ByteRanges
    Pruned,           // header-only: this region will never produce data (selection-killed)
}
```

**Stream invariant (proposed default, P-Q2.5: dense in-order coverage).** Every edge stream
emits patches with monotonically increasing, contiguous, non-overlapping regions that exactly
cover the edge's assigned range. A range eliminated by selection is emitted as a zero-cost
`Pruned` patch. Consequences:

- Alignment operators make progress by matching region frontiers only — no watermark
  side-channel, no plan-coupled inference.
- Ordered output is free (emit in region order); progress is observable per edge.

**Patch boundaries (proposed default, P-Q2.6: producer-chosen, consumers slice).** Producers
emit natural sizes (whatever a resolved byte range decodes to; chunk extents), bounded by a
`max_patch_rows` cap. Consumers needing alignment cut inputs at `min(next frontier)` across
ports — array/mask slicing is zero-copy. The planner may additionally annotate an edge with
preferred boundaries (e.g. zone-aligned for `Aggregate` stat substitution) as a hint.

**Sharing (proposed default, P-Q2.7).** An edge with ≥2 consumers is a bounded broadcast
channel (per-consumer cursors over a shared ring; producer paced by the slowest consumer). The
optimizer must insert an explicit `Materialize` breaker on shared edges that (a) are small
(dict values), or (b) sit on a *diamond* where one consumer is transitively gated on another
consumer's output — bounded lock-step broadcast on a diamond can deadlock, so this is a plan
*validation rule*, not a runtime heuristic.

**Parallelism (proposed default, P-Q2.8: two levels).** Engine-visible partitions (the
existing `Partition`/scan APIs) each instantiate independent pipeline instances over disjoint
root-row ranges from the same shared plan; SIP and `Materialize` results are shared across
them. Within an instance: per-column pipelines run as separate tasks, and `Concat` readahead
polls N child bands concurrently. Repeated execution (R7) is just instantiating new bands from
the same plan.

## 3. Operator interface

```rust
pub enum OpResult { NeedMoreInput, HaveMoreOutput, Finished }

/// Pipeline heads. I/O-driven; registered with the segment source at instantiation
/// (static ranges) or upon receiving a ByteRanges patch (computed ranges).
pub trait Source: Send {
    fn poll_next(&mut self, cx: &mut Context<'_>) -> Poll<VortexResult<Option<Patch>>>;
}

/// Streaming transformers. Push-based; may buffer internally; emit through `out`.
pub trait StreamOp: Send {
    fn push(&mut self, port: PortId, patch: Patch, out: &mut PatchSink)
        -> VortexResult<OpResult>;
    /// The given input port is exhausted (its range fully covered).
    fn finish(&mut self, port: PortId, out: &mut PatchSink) -> VortexResult<OpResult>;
}

/// Pipeline breakers and terminals.
pub trait Sink: Send {
    fn sink(&mut self, patch: Patch) -> VortexResult<SinkResult>;   // may report Blocked
    fn finalize(&mut self) -> VortexResult<()>;
}
```

Operators additionally implement two *plan-time* hooks used by the optimizer/unroller (§5):

```rust
/// Translate demand on an output edge into demand on each input edge (backward pass, D14).
fn demand_in(&self, out: &EdgeDemand, port: PortId) -> EdgeDemand;

/// Declare SIP interactions for validation/EXPLAIN: (reads, refines) per domain.
fn sip_signature(&self) -> SipSignature;
```

## 4. The operator catalog

### I/O and decode

**`ReadSegment { segment_id, ranges: Static(Vec<ByteRange>) | Computed(port) }` → `Bytes`**
(Source). `Static` registers with the `SegmentSource` at instantiation — preserving today's
eager-registration/coalescing contract (C3) — and emits `Bytes` patches as reads resolve.
`Computed` registers upon receiving each `ByteRanges` patch; this is the second I/O stage,
expressed purely as a data dependency. Whole-segment reads are the degenerate
`Static([0..len])`.

**`Decode { dtype, ctx }`** : `Bytes → Array` (StreamOp). Whole-segment: one patch in, one
array out (today's `SerializedArray` path, array tree from inline metadata or parsed from the
segment).

**`DecodePartial { array_tree }`** : `Bytes → Array` (StreamOp, stateful). Holds the array
tree; tracks which buffers have arrived; emits array patches for row ranges whose required
buffers are all present (fixed-width: rows decodable as soon as their data+validity byte ranges
land). This is how a flat leaf "returns smaller patches".

**`PlanIo { array_tree }`** — two forms:
- *Plan-time function* `plan_io(array_tree, demand_summary) -> Vec<ByteRange>`: used by the
  unroller to choose `Static` ranges for a band from current demand annotations.
- *Runtime op* : `Mask → ByteRanges` (StreamOp): for second-stage reads gated on exact masks
  (post-filter projection fetch).

Required from vortex-array (Piece 5 details this): per serialized encoding, a conservative
`buffer_ranges_for_rows(rows) -> Vec<ByteRange>` walk of the array tree. v1 scope: whole-buffer
pruning and fixed-width row striding; variable-width staging (offsets first, then data) is a
follow-up.

### Compute

**`Eval { expr }`** : `Array → Array` (StreamOp, stateless per patch). Applies a scalar-fn
expression via `apply()` (whole-expr pushdown into encoded arrays preserved, D16). Adaptivity
(D13): consults selection SIP for the patch's region — below a density threshold it
slices/filters the input before evaluating (today's `FlatReader` threshold, now per patch).

**`ExecMask`** : `Array(bool) → Mask`. Mask edges may carry a `refines: selection@domain`
marker (IR-visible, implemented inline): emission also intersects into the SIP selection.

**`MaskAnd`** : `Mask × Mask → Mask` (aligning). Exactness still requires intersecting
conjunct masks even though SIP made the conjuncts skip each other's dead rows.

**`Filter`** : `Array × Mask → Array` (aligning). Emits `data.filter(mask_slice)`; a `Pruned`
input region or all-false mask slice emits `Pruned`.

**`Pack { fields }`** : `Array^n (× Mask validity) → Array` (aligning, struct assembly).
Buffers per-port frontiers; emits packed patches over the common covered prefix, slicing
children zero-copy (P-Q2.6).

**`Concat`** : children in order → parent domain (chunked). Re-stamps child-local regions by
static offset; forwards in order. Readahead = executor policy: child bands k+1..k+N
instantiated while k drains (the "chunked buffers children" behavior).

**`Take`** : `(codes: Array) × (values: materialized) → Array` (dict gather). Values arrive
via a `Materialize` breaker (below), not a streaming port.

### SIP / domain operators

**`PruneZones { pruning_predicate }`** : `Array(zone stats) → Mask(zone domain)` followed by a
static `MapDomain{zone→rows}` re-stamp, terminating in a selection-refining sink. A pure SIP
producer: its output is conservative (may-contain), so it never feeds `MaskAnd` — eliminating
it from a plan changes performance, never results. Dynamic-filter updates (R4) re-enqueue
exactly this subtree on version bump — the one enumerable re-dispatch trigger.

**`DemandValues`** : `Array(codes) → ()` (values-domain demand publisher) — only needed when a
values domain is too large to materialize wholesale; v1 default is whole-values
`Materialize`.

### Aggregation (R2, D8)

**`Aggregate { fn, slices }`** : `Array (× Mask) → AccState` (StreamOp, stateful). Folds
in-order patches into the current slice's accumulator (reusing
`vortex-array::aggregate_fn`); the dense-coverage invariant tells it exactly when a slice is
complete (input frontier passes the slice end), emitting one `AccState` patch per row-index
slice, in order, with no shuffle. Bands are aligned to slice boundaries when slices exist;
otherwise per-band partials merge positionally via **`MergeAcc`**.

*Stat substitution* is a plan rewrite: when slice boundaries align with zone boundaries and
selection proves a zone fully included (or excluded), the data subtree under `Aggregate` is
replaced by a fold over the zone-map stats column (COUNT from row counts, MIN/MAX from zone
min/max, SUM where tracked); partially-selected zones keep the data path (v1: substitute only
when *all* zones in the slice are decided; hybrid per-zone splicing is a follow-up).

### Breakers and terminals

**`Materialize`** : Sink+Source pair (pipeline breaker). Fully buffers (or spills) its input;
multiple downstream pipelines read it, including random access. Uses: dict values (plan-proven
consumption — this *replaces* `DictReader`'s unsound per-expression cache: expression-on-values
nodes are ordinary `Eval` nodes above the shared `Materialize`, and pushing a *fallible*
expression to the values side is guarded by the existing `has_all_values_referenced` flag),
shared-edge diamonds (P-Q2.7), and any optimizer-chosen cache point.

**`Collect`** : terminal sink adapting patches to the engine-facing `ArrayStream` (ordered =
region order, free under the invariant) or to the partial-`AccState` stream for aggregates.

## 5. Demand, in detail (D14)

```rust
pub struct EdgeDemand {
    domain: DomainId,
    rows: DemandSummary,            // block-quantized row set + count (16-row granularity ok)
    kind: Static | Computed { gate: EdgeId },  // Computed: exact rows known only at runtime
}
```

**Backward propagation.** At unroll time the planner seeds demand at terminals and propagates
backward through `demand_in`:

| Operator | `demand_in` rule |
|---|---|
| `Collect` | requested range ∩ current selection upper bound |
| `Eval` | identity (row-aligned) |
| `Filter` | data port: region ∩ selection upper bound; mask port: full region |
| `Pack` | identity per field port |
| `Concat` | split by child extents, shift coordinates |
| `Take` | codes port: identity; values: AllValues (v1) or `Computed(codes edge)` |
| `Aggregate` | slices ∩ selection; only the fields the aggregate reads |
| `ReadSegment` | terminal: demand + array tree ⇒ byte ranges via `plan_io` |

**Refresh.** Demand annotations are recomputed incrementally at *phase boundaries* — i.e.
whenever the unroller instantiates the next band, and after selection-refining pipelines for a
band complete — always reading the current selection SIP. Refresh is cheap because summaries
are block-quantized.

**Dispatch policy (where SIP pays off).** When instantiating a band's projection-column
sources, the unroller chooses per segment, using demand density + storage profile (latency,
coalescing config):

- dense demand → `Static` whole-segment read *now* (your "projection sees most rows are needed
  and just starts the download");
- sparse demand + final mask imminent → `Computed` read gated on the mask edge (`PlanIo`);
- in between → `Static` with sub-ranges from the current conservative demand (safe: demand is
  monotone, an over-approximation never under-fetches; late refinement only means we fetched
  more than the minimum).

Correctness never depends on the choice (C1); it only moves bytes.

## 6. Worked example

Query: filter `a > 5 AND b = 'x'`, project `{a, c}`, `sum(c)` over row slices `[0..N)` (one
slice), tree `zoned(struct{a: flat, b: dict(values: flat, codes: flat), c: flat})`, one band
shown:

```
                 ┌─ ReadSegment(zones.a) ─ Decode ─ PruneZones(a>5) ─→ selection@root (SIP)
   prune         └─ ReadSegment(zones.b) ─ Decode ─ PruneZones(b='x') ─→ selection@root (SIP)

   filter a      ReadSegment(a) ─ DecodePartial ─ Eval(a>5) ─ ExecMask ──→ Ma [refines SIP]
   filter b      ReadSegment(b.values) ─ Decode ─ Materialize(V)
                 ReadSegment(b.codes) ─ Decode ──┐
                 Eval(='x') over V ── Take(codes)┴─ ExecMask ─→ Mb [refines SIP]
   combine       MaskAnd(Ma, Mb) ─→ M
   project a     (shares the filter's a-array edge via broadcast; Filter(a, M))
   project c     PlanIo(c.array_tree) ←M ─ ReadSegment(c, Computed) ─ DecodePartial
                 ─ Filter(c, M) ─┬─ Pack{a,c} ─ Collect
   aggregate     Aggregate(sum) ─┘(c edge shared) ─→ AccState ─ Collect(partial)
```

Scheduling: prune pipelines run first (tiny metadata reads, refine SIP summaries); `a` and `b`
filter pipelines run next, skipping zones SIP killed (their `ReadSegment`s were instantiated
with ranges already excluding pruned zones); `c` is fetched sub-segment via the mask-gated
`PlanIo` because demand was sparse — or flipped to a whole-segment `Static` read by the
dispatch policy if SIP shows dense survival. Every coupling shown is either an exact edge or
SIP; remove all SIP writes and the result is identical, only slower (C1).

## 7. Defaults adopted in this document (veto points)

- **P-Q2.5** Dense in-order coverage per edge, `Pruned` markers for killed ranges.
- **P-Q2.6** Producer-chosen patch boundaries, consumers slice zero-copy; planner hints only.
- **P-Q2.7** Bounded broadcast for shared edges; mandatory `Materialize` on small or
  diamond-gated shared edges (plan validation rule).
- **P-Q2.8** Two parallelism levels: engine partitions instantiate independent band pipelines
  from one shared plan (shared SIP/Materialize); intra-band parallelism from per-column
  pipelines + `Concat` readahead.
