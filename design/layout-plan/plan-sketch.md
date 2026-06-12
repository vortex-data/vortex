<!--
SPDX-License-Identifier: Apache-2.0
SPDX-FileCopyrightText: Copyright the Vortex contributors
-->

# Layout Plan (scan v2) — Design Sketch

Living document; updated as each piece is designed. Requirements and the decision log live in
[`requirements.md`](./requirements.md). We are designing piece by piece; later sections are
placeholders.

## Design spine

A query `(filter conjuncts, projection incl. scalar fns + aggregates, selection, limit)` plus a
**layout tree** (one or more files) is compiled into an **operator DAG over segments**, then
executed by a **phase-driven executor** that lazily unrolls and optimizes the DAG as execution
proceeds.

Four core concepts:

1. **Operator DAG.** Nodes are physical operators (segment reads, decodes, expression evals,
   I/O planning, aggregation); edges carry exact, correctness-bearing values (bytes, arrays,
   masks, accumulator states). Shared work — e.g. dict values used by both a filter conjunct
   and the projection — is a single shared subgraph (CSE), which also removes the unsound
   per-expression values cache in today's `DictReader`.
2. **SIP selection & demand.** Two shared, monotone, conservative side-channel structures per
   row domain. *Selection* (downward-only): which rows can still match the filter. *Demand*
   (upward-only): which rows some consumer will need materialized. Used purely for efficiency —
   skipping work, planning I/O, starting downloads early — never for correctness (exact masks
   stream on edges).
3. **Row domains.** Root rows, dict-values rows, and zone indices are distinct coordinate
   spaces with static mappings (zone→row-range, chunk offset) or data-dependent mappings
   (codes→values — itself an operator). Zone pruning and dict pushdown are the same mechanism:
   refinement propagated through a domain mapping.
4. **Phased lazy unroll.** The plan starts layout-shaped and coarse. The executor unrolls and
   optimizes fragments incrementally — first chunk of each row section first, more unrolled
   while execution is in flight — so plan size is never O(all chunks) up front, and information
   produced by early phases (zone-map reads, selection refinement) informs how later fragments
   are optimized (e.g. whether to plan sub-segment reads).

## Piece 1 — SIP selection & demand (drafted, reviewed)

**Semantics.**

- Two structures per row domain (D9), both monotone (C4):
  - `Selection`: rows move *maybe → cannot-match* as refiners (filter conjuncts, zone pruning,
    index probes, dynamic-filter re-pruning) publish results. Conjuncts consult it before
    evaluating so they skip rows another conjunct already excluded — this **replaces adaptive
    conjunct ordering** with shared state.
  - `Demand`: rows move *maybe → needed* as consumers (projection column reads, aggregate
    inputs, codes→values mapping) declare interest. Demand may **over-approximate** (block
    quantized, e.g. 16-row blocks): extra fetched bytes are wasted I/O, never wrongness.
- SIP is advisory only (C1/C2, D12). Exact masks stream on DAG edges; a consumer that needs the
  final filter result depends on its mask edge, not on SIP. No finality bookkeeping is needed:
  reads of SIP are safe at any time because the contents are conservative.
- Limit / early termination = executor cancellation of plan regions (dropping registered I/O
  futures already cancels reads today), not SIP retraction.

**Representation (per domain).** Two-level: a root vector of per-block summaries
(`AllMatch | NoMatch | Mixed{count}`) with an O(1)-maintained total count, plus lazily-allocated
leaf bitmaps for `Mixed` blocks. Coarse refiners (zone pruning) only ever touch summaries; fine
refiners touch their block's bitmap. Demand may stay summary-only (block granularity
configurable, e.g. 16/64/1024 rows). Queries: `count(region)` cheap; `summary(region,
granularity)` cheap (feeds I/O planning and adaptive operators); `exact(region)` materializes a
`Mask` on request.

**API sketch.**

```rust
struct DomainId(u32);
struct RowRegion { domain: DomainId, range: Range<u64> }

impl Selection {
    fn refine(&self, region: &RowRegion, mask: &Mask);   // monotone intersect
    fn count_upper(&self, region: &RowRegion) -> u64;
    fn summary(&self, region: &RowRegion, granularity: BlockSize)
        -> impl Iterator<Item = BlockState>;
    fn exact(&self, region: &RowRegion) -> Mask;         // may materialize
}

impl Demand {
    fn require(&self, region: &RowRegion);               // monotone union, quantized OK
    fn demanded(&self, region: &RowRegion, granularity: BlockSize)
        -> impl Iterator<Item = BlockState>;
}
```

## Piece 2 — Operator taxonomy & edge types (in progress)

**Execution model (D15).** Operators are **stateful stream transformers**; an edge is an
ordered stream of **patches** — values stamped with the `RowRegion` they cover. Examples that
motivated this: a flat leaf doing sub-segment I/O emits decoded patches as byte ranges resolve
(it need not wait for the whole region); a struct operator aligns child streams that advance at
different rates; a chunked operator buffers child streams for readahead. Aggregates are
naturally streaming folds emitting one `AccState` per row-index slice.

**Demand (D14).** Demand is an annotation on edges computed by a backward propagation pass at
plan/unroll time: outputs declare needs (projection: selected rows of its fields; aggregates:
inputs at slices ∩ selection); each operator translates output-demand into input-demand
(`Filter` → source at selection upper bound; `Take` → codes-domain demand; codes→values is
`Computed` from the codes edge, symmetric with `ReadSegment{Computed}`). Annotations are
refreshed at phase boundaries from current selection SIP, which is how a projection "sees" dense
demand and starts whole-segment downloads early. Runtime adaptivity beyond that lives inside
operators reading SIP (D13).

**Edge value types (draft).** Each edge is a stream of patches of one of:

| Type | Description |
|------|-------------|
| `Bytes` | Segment bytes or sub-range buffers (`BufferHandle`). |
| `Array` | A Vortex array (possibly still encoded), stamped with its `RowRegion`. |
| `Mask` | Exact selection for a `RowRegion` (correctness-bearing, streamed). |
| `AccState` | Mergeable aggregate accumulator state, one per row-index slice. |
| `ByteRanges` | A computed I/O plan for (part of) a segment. |

**Node taxonomy (draft).**

I/O:

- `ReadSegment { segment_id, range: Static(..) | Computed(ByteRanges edge) } → Bytes` — static
  ranges are registered eagerly for coalescing (C3); computed ranges form a second I/O stage by
  data dependency (no explicit stage barriers).
- `PlanIo { array_tree } → ByteRanges` — reads demand/selection SIP and maps demanded rows to
  buffer byte ranges within a flat segment using the inline array-tree metadata (R6, D7).

Decode:

- `Decode { dtype, ctx } : Bytes → Array` — whole-segment decode (today's path).
- `DecodePartial { array_tree } : Bytes × ByteRanges → Array` — assemble an array view from a
  subset of buffers.

Compute:

- `Eval { expr } : Array → Array` — scalar-function expression evaluation.
- `ExecMask : Array(bool) → Mask`
- `MaskAnd : Mask × Mask → Mask`
- `Filter : Array × Mask → Array`, `Take : Array × Array(indices) → Array` (dict gather)
- `PackStruct`, `Concat` — assembly.

SIP / domains:

- `RefineSelection { region } : Mask → ()` — publish a conjunct's result into selection.
- `PruneZones { pruning predicate } : Array(stats) → Mask(zone domain)` — plus static zone→row
  expansion into root selection.
- `DemandValues : Array(codes) → ()` — data-dependent demand mapping into a values domain.

Aggregation (R2, D8):

- `Aggregate { fn, slices } : Array × Mask → stream of AccState` — streaming fold, one state
  per row-index slice, emitted in row order.
- `MergeAcc : AccState* → AccState`
- Stat substitution is a rewrite: `Aggregate` over a column becomes `Aggregate` over the
  zone-map stats column when slice boundaries align with zones and selection proves the zone
  fully included/excluded.

**Resolved (rounds 5–6).** Q2.1: operators are stateful streams (D15). Q2.2: provisionally
opaque `Eval` per conjunct/field-expr (D16). Q2.3: demand propagated along edges at plan time
(D14). Q2.4: adaptivity inside operators reading SIP (D13).

**Detailed design: see [`operators.md`](./operators.md)** — the DuckDB-style push-based
pipeline model (Source/StreamOp/Sink, patches as region-stamped morsels, bands as the lazy
unroll unit), the full operator catalog, demand propagation rules, dispatch policy, and a
worked example. Stream-semantics points Q2.5–Q2.8 are resolved there as **stated defaults
(P-Q2.5..P-Q2.8, §7)** pending veto.

**The two planes are specified in [`patches-and-sip.md`](./patches-and-sip.md)**: the data
plane (`Patch` — region-stamped Vortex arrays with embedded selection, the DataChunk
analogue; stream invariant; edge typing) and the demand plane (`SelectionSip` / `DemandSip` /
`EdgeDemand` — concrete representations, writer/reader tables, propagation and refresh,
dispatch policy, and the cross-plane contract).

## Piece 3 — Lowering contract for layouts (not started)

How each layout encoding contributes DAG fragments and rewrite rules; how lazy unroll calls into
it per phase; how index layouts plug in as selection refiners.

## Piece 4 — Executor & phases (not started)

The plan→execute loop: dispatch waves, lazy unroll scheduling, cancellation/limit, preserving
eager-registration coalescing, ordered vs. unordered emission, repeated execution over row
ranges (R7), dynamic-filter re-dispatch.

## Piece 5 — Sub-segment I/O planning for FlatLayout (not started)

`PlanIo` in detail: what `vortex-array` must expose per encoding to map demanded rows to buffer
byte ranges; interaction with coalescing; cost model for whole-segment vs. staged reads.

## Piece 6 — Lowerings for existing layouts (not started)

Concrete DAG fragments for flat, chunked, struct, dict, zoned, row-idx, file-stats; parity
checklist against current reader behaviors (incl. struct validity, `row_idx`, repartitioning).

## Piece 7 — Aggregate pushdown details (not started)

Accumulator state wire contract with engines (DataFusion/DuckDB), exactness guards for stat
substitution, slice/zone alignment.

## Piece 8 — Migration & testing (not started)

Parallel-stack strategy (D2), differential testing against the existing reader path, benchmark
gates, switchover criteria.
