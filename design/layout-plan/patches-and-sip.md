<!--
SPDX-License-Identifier: Apache-2.0
SPDX-FileCopyrightText: Copyright the Vortex contributors
-->

# Layout Plan — The Two Planes: Data Patches & Demand SIP

Detailed specification of the two communication planes in the plan/execute model:

- **Data plane** — exact, correctness-bearing values flowing on DAG edges as region-stamped
  patches of Vortex arrays (the DuckDB `DataChunk` analogue).
- **Demand plane (SIP)** — approximate, monotone, conservative knowledge about which rows can
  still match and which rows will be needed, used *only* to optimize I/O and skip work.

The invariant tying them together: **delete the entire demand plane and every query returns
identical results, only slower.** Context: [`operators.md`](./operators.md),
[`plan-sketch.md`](./plan-sketch.md), [`requirements.md`](./requirements.md).

---

## Part A — The data plane: `Patch`

### A.1 Domains and regions

```rust
/// A coordinate space within a plan. Row domains: root rows, a dict instance's values rows,
/// a zoned instance's zone indices, per-chunk local rows. Byte domains: a segment's byte
/// offsets (for Bytes/ByteRanges edges). Declared at plan time (or on unroll for lazily
/// discovered domains); every edge belongs to exactly one domain.
pub struct DomainId(u32);

/// Contiguous half-open range in the edge's domain (domain implied by the edge).
pub type Region = Range<u64>;
```

### A.2 The patch type

```rust
pub struct Patch {
    /// The slice of the edge's domain this patch covers.
    region: Region,
    payload: Payload,
}

pub enum Payload {
    /// A Vortex array of ANY encoding (never force-canonicalized), with an optional
    /// embedded selection — see A.3.
    Array { array: ArrayRef, sel: Option<Mask> },
    /// An exact filter result for the region: mask.len() == region.len().
    Mask(Mask),
    /// Raw bytes for a byte-domain region (resolved segment read).
    Bytes(BufferHandle),
    /// An I/O plan: byte ranges a downstream Computed ReadSegment must fetch.
    ByteRanges(Vec<Region>),
    /// Mergeable accumulator state for one row-index slice (slice-ordered, dense:
    /// a fully-pruned slice still emits its identity state so merging stays positional).
    Acc { slice: u32, state: AccState },
    /// Header-only: the region is proven empty (selection-killed / pruned). Zero cost,
    /// carries progress so aligners and aggregates can advance their frontiers.
    Pruned,
}
```

Relationship to vortex-array: `array` is an ordinary `ArrayRef` — encodings are preserved
end-to-end (dict stays dict, ALP stays ALP) so `Eval`'s `apply()` keeps pushing whole
expressions into encoded forms; `Mask` is `vortex_mask::Mask`; `AccState` comes from
`vortex_array::aggregate_fn`. Patch boundaries never force `ChunkedArray` wrappers mid-plan;
only `Collect` assembles engine-facing chunks.

### A.3 Embedded selection (the DuckDB selection-vector analogue)

A data patch is in one of two canonical states:

| State | Condition | Meaning |
|---|---|---|
| **aligned** | `sel == None`, `array.len() == region.len()` | position `i` ↔ row `region.start + i` |
| **compacted** | `sel == Some(m)`, `m.len() == region.len()`, `array.len() == m.true_count()` | array holds only the selected rows, in order |

This lets `Filter` be *representational*: it attaches the mask and may defer the physical
gather, exactly like DuckDB flowing a selection vector instead of copying. Operator rules:

- `Eval` runs on either state unchanged (scalar fns are position-wise).
- `Pack` requires all field ports at identical `(region, sel)`; if sels differ it compacts to
  the intersection first (mask intersect + `filter`).
- `Aggregate` consumes compacted patches (count = `array.len()`), or aligned + a mask port.
- `Collect` emits compacted — the scan output *is* the selected rows.
- `Compact` is the explicit flattening op the planner can insert when deferral stops paying.

### A.4 The stream invariant (formal)

For an edge with assigned set `S` (a sorted set of regions: the whole range for most edges;
the requested byte ranges for a `Bytes` edge):

1. Patches are emitted with **ascending, non-overlapping regions whose union is exactly `S`**.
2. A region with no data is emitted as `Pruned` (or an identity `Acc`) — never silently
   skipped. Progress is therefore observable from the stream alone: a consumer's *frontier*
   (end of last region) is total knowledge of what will never change.
3. `finish(port)` arrives after the final patch; receiving it with frontier ≠ `end(S)` is a
   plan bug.
4. Any patch may be **sliced zero-copy at any interior boundary** (arrays, masks, and `sel`
   all slice O(1)); producers choose sizes (≤ `max_patch_rows`), consumers slice to align
   (P-Q2.6).

### A.5 Edge typing & validation

Every edge declares `EdgeType { domain, payload_kind, dtype?: DType }`. Plan validation
checks port compatibility (like DuckDB chunk types), plus: mask edges marked
`refines: selection@domain` must target a domain registered in the plan; shared edges must
satisfy the broadcast/Materialize rule (P-Q2.7).

---

## Part B — The demand plane (SIP)

### B.1 Lifecycle and ownership

Per **execution** (one `execute(row_range, dynamic-filter bindings)` over the shared plan) a
`SipContext` is created: one `SelectionSip` per row domain, one `DemandSip` per domain that has
*runtime* demand publishers, and demand annotations on edges. It is shared across all engine
partitions and bands of that execution (cross-band signals like "this zone is dead" reach
everyone). `Materialize` caches may be plan-scoped when their inputs are execution-invariant
(dict values: yes; anything downstream of a filter or dynamic expression: no).

### B.2 `SelectionSip` — "can this row still match?" (monotone ↓)

```rust
pub struct SelectionSip {
    block_size: u32,                    // e.g. 1024 rows
    blocks: Vec<BlockSlot>,
    count_upper: AtomicU64,             // maintained total upper bound
}

struct BlockSlot {
    state: AtomicU8,                    // AllMaybe | NoneSelected | Mixed
    count_upper: AtomicU32,             // upper bound of surviving rows in this block
    fine: OnceLock<RwLock<BitBuffer>>,  // per-row bits, materialized on first Mixed refine
}
```

Operations (all conservative; relaxed atomics suffice because any stale read is an
over-approximation):

- `refine(region, &Mask)` — intersect: allocate `fine` on first fine-grained write, popcount
  delta updates counts, possibly transition `Mixed → NoneSelected`. **Monotone: intersect
  only, never union.**
- `refine_blocks(region, false)` — coarse kill from `PruneZones` / file stats: state
  transitions only, no bitmaps ever allocated.
- `count_upper(region) -> u64`; `summary(region, granularity) -> impl Iterator<Item =
  BlockSummary>` (granularity ≥ 16 rows; sub-block summaries read 16-row lanes of `fine` when
  materialized, else the block state).
- `upper_mask(region) -> Mask` — `AllMaybe` blocks expand to all-true. Deliberately named
  *upper*: SIP never yields an exact mask; exact masks are data-plane values.

| | Writers | Readers |
|---|---|---|
| per patch | mask edges marked `refines` (write-through on emission) | `Eval` strategy choice (density threshold) |
| per band/phase | `PruneZones`, dynamic-filter re-prune (version bump), engine `Selection` seeding at execute() | unroller: skip dead bands, demand refresh, dispatch policy |

### B.3 `DemandSip` and `EdgeDemand` — "will someone need these rows?" (monotone ↑)

Demand is mostly **computed**, not stored: a backward pass over edges. The stored form:

```rust
pub struct EdgeDemand {
    domain: DomainId,
    rows: DemandSet,
    refreshed_at: PhaseStamp,           // memoization key for incremental refresh
}

pub enum DemandSet {
    All,                                // dense: fetch everything in the edge's range
    Blocks(BlockSet),                   // quantized (16-row blocks): conservative union
    Computed { gate: EdgeId },          // exact rows known only when `gate` resolves
}
```

A mutable `DemandSip` (per-domain `Vec<AtomicU8>` of 16-row block flags + count — summary-only,
no bitmaps, because **over-approximation only ever over-fetches**) exists only for domains with
runtime publishers: today that is values domains fed by `DemandValues(codes)` when the
optimizer decides a values segment is too large to materialize wholesale.

**Backward propagation.** Seeded at terminals (`Collect`: requested range ∩
`selection.summary`; `Aggregate`: slices ∩ selection, only its input fields), propagated
through each operator's `demand_in` (rules table in `operators.md` §5), memoized per
`(edge, PhaseStamp)`. **Refresh triggers:** (a) a new band is unrolled, (b) a band's
selection-refining pipelines complete, (c) a dynamic filter bumps its version. Refresh is
cheap: summaries in, summaries out.

**Consumption — the dispatch policy.** When instantiating a `ReadSegment` the unroller
resolves `EdgeDemand` against the storage profile:

- `All` / dense `Blocks` → `Static(whole segment)` now — eager registration, coalescing (C3);
- sparse `Blocks` → `Static(plan_io(array_tree, blocks ∩ selection.summary))` — sub-segment
  ranges from conservative demand;
- `Computed{gate}` → mask-gated `PlanIo` + `ReadSegment(Computed)` — the second I/O stage.

Monotonicity guarantees the choice is always safe: demand can only grow toward `All`, and
selection can only shrink what is *worth* fetching — neither can make an already-issued read
wrong, only suboptimal.

### B.4 The contract between the planes

1. **Correctness flows only through patches.** `Filter`, `MaskAnd`, `Pack`, `Aggregate`,
   `Collect` consume edges exclusively. The only operators that read SIP do so to pick between
   result-identical strategies (`Eval`) or to shape I/O (`ReadSegment`/`PlanIo`/unroller).
2. **SIP is conservative in both directions.** Selection over-approximates survivors; demand
   over-approximates need. Every stale or quantized read errs toward fetching/computing more.
3. **Writes are monotone; cancellation is separate.** Limit kills bands/pipelines via the
   executor (dropping registered reads cancels them, as today); SIP never retracts.
4. **Enumerable and explainable.** Each operator's `sip_signature()` declares what it reads
   and refines per domain; plan validation checks every `refines` marker targets a registered
   domain, and EXPLAIN prints both planes (data edges, and SIP read/refine sets per node).
