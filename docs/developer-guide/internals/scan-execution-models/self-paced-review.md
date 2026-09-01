# Review of the Self-Paced Execution Proposal

A design review of [self-paced plan execution](self-paced.md) and its
[implementation plan](self-paced-implementation-plan.md). It accepts the frame — hand-written
resumable execution nodes, prefix-only progress, fixed outer morsels, a central scheduler — and
argues about the contract inside it.

Findings are ranked by severity. Each names the defect, the evidence, and a concrete replacement.

## Status

These findings have been merged into the design and the phase plan. This document is retained as
the rationale record: the code references, measurements of scale, and reasoning behind each choice
live here rather than cluttering the design.

| Finding | Where it landed |
| --- | --- |
| F1 row domains | New "Row domains" section; invariants 1-4; settled choice 3; plan Phase 1 |
| F2 per-scan tier | New "Three state tiers" section; settled choice 4; plan Phase 3 |
| F3 capping and retention ownership | "Parent alignment"; invariant 15; settled choice 11; plan Phase 1/5 |
| F4 epochs | "Widening and epochs" now flags it; open-questions table; plan Phase 0 gates Phase 2 |
| F5 credit deadlock | "Progress guarantees"; settled choice 12; plan Phase 4 |
| F6 Yield and minimum prefix | "Progress obligations"; invariants 6 and 11; plan Phase 1 |
| F7 split discovery | New "Morsel boundary discovery" section; plan Phase 6 |
| F8 Take strategies | TakeExec now states a default and a sub-root model; plan Phase 9 |
| F9 sealed-demand suffix | `mask_offset` on SealedDemand |
| F10 summary redundancy | "Coarse demand summaries" trimmed to two facts and one cache |
| F11 batch-carried demand | ExecBatch's mask is debug-only |
| F12 estimated counts | Marked scheduling-only and omittable |

## What the current design gets right

The `DemandLedger`/`ReadCatalog` split resolves the three problems that mattered most in earlier
drafts, and it resolves them better than a narrower fix would have:

- **Read discovery is decoupled from mask resolution.** Today's overlap between filter and
  projection I/O is structural, not incidental: `vortex-layout/src/plan/plans/segment_scan.rs:123`
  issues `segment_source().request(..)` synchronously at `execute` time and awaits the mask only at
  line 140, so `vortex-scan-v2/src/tasks.rs:88-93` gets the whole projection subtree's reads in
  flight while the filter is still running. A contract built on a concrete demand mask would have
  serialized those two phases per morsel. Static `describe_reads` plus shared `ReadKey` preserves
  the overlap and additionally makes the filter/projection dedupe explicit rather than accidental.
- **`DriveResult` plus `DriveContext` registration** lets one drive expose independent I/O and CPU
  work from different children. An exclusive `MoreIo`/`RunCpu`/`Batch` enum could not, and made the
  anti-serialization rule unenforceable prose.
- **`wait_for_credit` returning a `CreditTicket`** puts resource waits on the same wake path as I/O
  and CPU, so a node blocked on bytes has something to name in its `WaitSet`.
- **Invariant 9 and "visit every missing child before blocking"** correctly assign fan-out to the
  operator rather than to the driver, which is the only place it can live given `&mut self` drive.

The implementation plan's Phase 0 oracle, per-phase exit criteria, and decision gates are the right
shape. The findings below are about what those phases will hit.

## F1. Make the row domain and its transforms first-class

Severity: **high**. This is an API-shape problem that Phase 9 discovers after Phases 1-8 have
hardened around it, and fixing it collapses four other problems into one mechanism.

### The immediate defect

`BatchRequest` carries `demand: SealedDemand<'a>` and that is the only way to drive a child.
Invariant 1 restricts construction to `DemandLedger`, and the ledger divides *the morsel row space*
into blocks. But `TakeExec` drives a values child in a lookup domain and `ListPackExec` drives an
element child in an element domain. Neither is a subrange of the morsel, and the ledger holds no
predicates there, so it has no basis on which to seal anything. Phase 9 says "translate outer
prefixes into element ranges" without saying what request type carries them.

### The domain concept already exists, five times over

Every operator that changes coordinates already hand-rolls its own translation, in its own
encoding:

| Operator | Existing state | Translation |
| --- | --- | --- |
| `RowIdx` | `RowIdxData::row_offset` — "the row offset applied to the child domain" | `child = parent - offset` |
| `Concat` | `ConcatData::row_offsets: Arc<[u64]>` | `child = parent - chunk_offset` (`concat.rs:175`) |
| `Zoned` | `zone_len` | `zone = row / zone_len` (`zoned.rs:304-305`) |
| `ListPack` | `elements_range_from_offsets` | `elements = offsets[start]..offsets[end]` |
| `ListPack` | inline `row_range.end + 1` | offsets child needs one extra row |
| `Take` | codes child vs. values child | `value = codes[row]` |

`vortex-scan-v2/src/splits.rs` then hand-rolls a *sixth* copy: `collect_plan_splits` descends
`Pack` children only where `child.row_count() == plan.row_count()` (identity), adds
`row_offset + chunk_offset` for `Concat` (shift), and takes only `Take`'s codes child (skipping the
one child whose domain differs). And the read catalog needs a seventh, currently described as "a
lookup-domain or nested operator may instead provide a conservative group or a gate".

Naming the concept once replaces all seven.

### The model

A **domain** is a row universe: two nodes share a domain when a row in one *is* a row in the other.
A **domain map** is the transform on a parent-child edge. Pure renumbering stays inside a domain;
only a genuine change of row universe crosses into a new one.

~~~rust
/// A row universe. Allocated during morsel preparation.
#[derive(Copy, Clone, PartialEq, Eq, Hash)]
struct DomainId(u32);

enum DomainMap {
    /// child row r is parent row r.
    /// Pack fields, Eval input, Take codes, ListPack validity, Zoned data, RowIdxPartition.
    Identity,
    /// child = parent - offset. Concat children, RowIdx.
    Shift { offset: i64 },
    /// Shift, plus `extra` trailing rows. ListPack offsets (n rows need n+1 offsets).
    Fence { offset: i64, extra: u64 },
    /// child = parent / stride. Zoned evidence. Crosses into a new domain.
    Coarsen { stride: u64 },
    /// Monotone and contiguous, resolved by a gate. ListPack elements. New domain.
    MonotoneGated { gate: GateId },
    /// Arbitrary gather, resolved by a gate. Take values. New domain.
    GatherGated { gate: GateId },
}
~~~

The set is closed even for third-party layouts, because they lower to these same operators.

Four operations are needed, and they are what every one of the seven hand-rolled copies is doing:

~~~rust
impl DomainMap {
    /// Child range covering a parent range. Read coverage and child requests.
    fn map_range(&self, parent: Range<u64>) -> VortexResult<Range<u64>>;
    /// Child demand for a parent demand.
    fn map_demand(&self, parent: SealedDemand<'_>) -> VortexResult<OwnedDemand>;
    /// Largest parent prefix satisfied by a child committed to `child_end`.
    /// Defined only when `prefix_preserving()`.
    fn unmap_frontier(&self, child_end: u64) -> VortexResult<u64>;
    fn prefix_preserving(&self) -> bool;
    fn is_static(&self) -> bool;
}
~~~

`unmap_frontier` is the one that makes alignment work across a coordinate change: for `ListPack` it
is a search over decoded offsets for the largest `k` with `offsets[k] <= child_end`.

### The property that matters: only one map breaks prefix progress

| Map | Static | Prefix-preserving | Exact for fallible work |
| --- | :---: | :---: | :---: |
| `Identity` | ✔ | ✔ | ✔ |
| `Shift` | ✔ | ✔ | ✔ |
| `Fence` | ✔ | ✔ | ✔ |
| `Coarsen` | ✔ | ✔ | ✖ |
| `MonotoneGated` | ✖ | ✔ | ✔ |
| `GatherGated` | ✖ | ✖ | ✔ |

This is a much sharper statement than "coordinate-changing operators are hard". Everything except
`Take`'s values child composes under one uniform rule, and `ListPack` — which the current draft
groups with `Take` as a "coordinate-changing operator" — is in the easy class. Its element child is
monotone and contiguous, so an outer prefix maps to an element prefix and ordinary prefix progress
applies once the gate resolves.

### Derivation and the sealing invariant

Sealing is a claim about *finality*, not about row space, and translation preserves finality. So
`SealedDemand` gains a domain and a derivation, legal only for the operator that declared the map:

~~~rust
struct SealedDemand<'a> {
    epoch: DemandEpoch,
    domain: DomainId,
    rows: Range<u64>,
    mask: &'a Mask,
    mask_offset: usize,   // see F9
}

impl SealedDemand<'_> {
    fn derive(&self, map: &DomainMap) -> VortexResult<OwnedSealedDemand>;
}
~~~

The naive invariant "a derived demand must be implied by its parent's" is wrong: `Coarsen` maps a
row set to the set of *covering* zones, which is a superset in row terms. Two rules instead:

1. **Completeness.** The derived demand covers every child row that any demanded parent row depends
   on. Violating this produces wrong values.
2. **Minimality for fallible work.** A map used to derive demand for fallible computation must
   contain no child row that no demanded parent row depends on. `Identity`, `Shift`, `Fence`,
   `MonotoneGated`, and `GatherGated` satisfy this exactly. `Coarsen` does not, so it may drive only
   infallible metadata work — which is already true in practice, since it feeds evidence.

That is checkable, and it replaces the current draft's reliance on prose about what operators
"should" do with speculative rows.

The framework also produces a *better* derivation than today's code in at least one place.
`list_pack.rs` drives the offsets child with `MaskFuture::new_true(row_count + 1)` regardless of
outer demand. The exact `Fence` derivation is `d | (d << 1)` — for a sparse filter, materially
fewer offsets.

### `Take`'s values child becomes a sub-root

Because `GatherGated` is the one map that is not prefix-preserving, the values child is not driven
inside the parent's prefix cursor at all. It becomes a **sub-root**: its own domain, driven by its
own prefix cursor over `0..values_row_count`, with a sparse demand mask equal to the gather set.

This is well-formed. The gather set derives from a sealed outer demand and decoded codes, both
final, so the value-domain demand is sealed the moment the gate expands. The value domain has no
predicates, so there is nothing to wait for. Below that point, ordinary prefix progress applies
normally, including when the values subtree is itself a `Concat`.

It also answers F8: the three `Take` strategies are just three widths of gather mask. Full
materialization is an all-true mask (what `take.rs` does today), sparse gather is the exact code
set, and incremental is one immediately-sealed value-domain demand **per outer prefix**, deduplicated
by the `ScanState` value cache from F2. That last form is what makes incremental lookup work without
any general widening machinery: the demand never widens, there are simply successive independent
demands over a shared cache.

### What this collapses

- **F1**: `Take` and `ListPack` become expressible, and `ListPack` turns out to be easy.
- **Catalog coverage**: compose maps from a read's owning node up to the ledger domain. All-static
  and prefix-preserving gives exact block coverage; a gated map on the path means group coverage
  until the gate expands. Mechanical, not per-operator judgement, and no second mechanism.
- **F7**: `collect_plan_splits`'s operator switch becomes "walk edges whose map is static and
  prefix-preserving, translating boundaries; stop at gated maps" — which is exactly what its seven
  hand-written cases already compute.
- **Row identity**: absolute row number is composition of `Shift` maps to the file domain, which is
  precisely what `RowIdxData::row_offset` already stores. Row-index execution stops being a special
  coordinate rule.

### Cost on the fast path is zero

For a flat, chunked, or struct scan every edge is `Identity` or `Shift` and the whole graph is one
domain. `Coarsen`, `MonotoneGated`, and `GatherGated` are the only maps that allocate a new
`DomainId`, so the common case pays a branch on a copy type and nothing else.

Move this into **Phase 1**, with the primitives. Adding a domain parameter after eight phases of
row-space assumptions is the expensive version of this change.

## F2. There is no per-scan tier, and three separate costs land on its absence

Severity: **high**.

"Ownership boundaries" names immutable `PlanRef` and per-morsel `ExecGraph`. Preparation is
explicitly per-morsel: "This work is performed once per plan use and morsel." Anything a node would
compute identically in every morsel is therefore recomputed for every morsel. Three costs converge
here:

1. **Dictionary value domains.** `vortex-layout/src/plan/plans/take.rs` executes the values child
   over `0..values_plan.row_count()` with an all-true mask on every call. Phase 9 keeps this as one
   of three options. Under a 100,000-row morsel a 1,000,000-row file rebuilds the whole value
   domain ten times.
2. **Catalog construction.** For a wide table the catalog is `columns × segments-per-morsel` entries
   per morsel. A 1,000-column table with 8,000-row segments is ~13,000 entries built and thrown
   away per morsel. The risk register lists "Static catalog is too large" with the mitigation
   "measure entry count; group homogeneous reads if necessary", but no phase owns that measurement
   and no exit criterion bounds it.
3. **Lazy plan lowering.** `vortex-layout/src/plan/children.rs` populates `OnceCell` children on
   first access. That cache is on `PlanRef` and shared, which is correct today but is exactly the
   "runtime caches leaking into reusable plan data" the design warns about — it needs a stated home.

All three are the same missing tier. Name it:

| Tier | Lifetime | Sharing | Contents |
| --- | --- | --- | --- |
| `PlanRef` | cross-scan | immutable | operators, dtypes, row domains, expressions |
| `ScanState` | one scan | `Arc`, keyed by plan identity | catalog spine, resolved metadata, dictionary value domains, zone maps, read-store handles |
| `ExecGraph` | one morsel | owned, never shared | cursors, tails, tickets, scratch |

Two consequences worth building in from Phase 3 rather than retrofitting:

- The catalog should be built **once per scan** with per-morsel *views*, not rebuilt per morsel.
  Segment identity and row coverage are morsel-independent facts; only necessity and lifecycle are
  per-morsel. That turns finding 2 from a measurement risk into a non-issue.
- `ScanState` entries must be bounded and have an eviction policy. Dictionary value domains are the
  obvious unbounded case, and F8 depends on this existing.

`layout27` had this tier ("runtime state is keyed by plan identity in a scan state cache") and the
current proposal drops it. `index.md`'s comparison row asserting runtime state lives "only in the
execution graph and scheduler" should be corrected alongside.

## F3. Parent tail retention: ownership is ambiguous and the cap mechanism goes unused

Severity: **high**.

### The fragmentation

For a `Pack` over K children with independent natural boundaries, min-of-heads alignment emits the
**union** of all K boundary sets. The doc's own example is this at small scale — two children, four
output batches — and is presented as the useful meaning of "return whatever size it likes":

~~~text
field A: [0..4) [4..10)
field B: [0..3) [3..8) [8..10)
Pack:    [0..3) [3..4) [4..8) [8..10)
~~~

At scale, a 20-field struct over a 100,000-row morsel with per-child 8,000-row pacing can emit ~250
batches averaging ~400 rows rather than 12 of 8,000. Note this is not a regression against today —
`collect_plan_splits` (`vortex-scan-v2/src/splits.rs:89-98`) already unions the boundaries of every
row-equivalent `Pack` child — but self-pacing does not fix it either. It re-pays the same cost at
batch granularity and adds K tail buffers on top.

### The unused mechanism

The batch contract already makes the request end a hard bound ("a node never exceeds the sealed
request"). A parent can therefore cap a child instead of slicing it afterwards:

1. Round one goes wide to every child, so all their I/O is in flight, and min-of-heads sets `L`.
2. Later rounds issue `rows.end = frontier + L` to *all* children.

Children that already hold decoded data past `L` return exactly `L` — slicing a decoded array is
free — so the union collapses to one boundary and **the parent retains nothing**. A child that
genuinely cannot stop at `L` returns shorter and the parent re-learns `L`. The parent distinguishes
the two without new API: `batch.rows.end == request.rows.end` means capped, anything shorter is a
real constraint.

### The ownership question this settles

The struct example says Pack "retains a[8k..64k)" and the frontier table charges it to "Pack/a".
That ambiguity is worth resolving in the child's favour, and capping does it:

> A child that decodes more than the parent asked for keeps the surplus in **node-local** decoded
> state. It is charged to that node's decoded credit and released by that node. Parent-owned tails
> exist only where a child cannot re-slice its own output.

This matters because the child is the only party that knows whether re-slicing is free, and it is
the party that can release. It also removes most of the exposure in F5, and it makes decision 6 of
the implementation plan's final list ("What exact memory is charged to the node, scheduler result
store, and parent after a batch is sliced?") answerable by rule rather than case by case.

The 64,000-row indivisible decode in the worked example still forces retention — the doc is right
that no API manufactures granularity the encoding does not expose. Capping does not fix that case;
it fixes every case where the child *could* have stopped and was not asked to.

Add to Phase 5 validation: a K-child `Pack` whose children share a boundary emits one batch per
boundary, not K.

## F4. The epoch machinery is unmotivated; settle reachability in Phase 0

Severity: **medium-high**, because the unresolved fork sits on a correctness path.

`DemandEpoch` appears in `SealedDemand`, invariant 2, Phase 2 work items 6 and validation, Phase 7
item 10, and the risk register. But no document names an operation that widens demand, and the
resolution is left open: "The implementation must either restart the uncommitted suffix under that
epoch or define snapshot semantics that defer the change to a later scan." Those are very different
implementations, and Phase 7 is the most expensive place to find out you need the first one.

The evidence suggests widening is not currently reachable:

- `Selection` (`vortex-scan/src/selection.rs:18`) is fixed when the scan is constructed.
- Pruning, evidence, and predicates all intersect, so they only narrow.
- The only dynamic predicate in the tree, `DynamicFilterPhysicalExpr`
  (`vortex-datafusion/src/persistent/opener.rs:1124`), is used as `file_pruning_predicate` and
  applied before the scan opens. DataFusion's dynamic filters tighten as they resolve.

The one construct that looks like widening is incremental `Take`: as successive outer prefixes
arrive, more of the value domain is needed. F1 shows this is not widening at all — each outer prefix
mints an independent, immediately-sealed demand in the value domain, and the `ScanState` value cache
(F2) makes the overlap free. So the apparent counterexample resolves without epochs.

Make this a **Phase 0 question**: is there any supported or planned API through which demand can
widen after a scan opens? If no, delete `DemandEpoch` from `SealedDemand` and replace the machinery
with one debug assertion that intersections never widen. If yes, name the operation and pick
restart-or-snapshot before Phase 2, because the choice changes what `ExecGraph` must be able to
discard.

Phase 0 already commits to deciding "which current behavior is contractual and which is merely an
implementation artifact". This belongs in that list.

## F5. Credit deadlock is handled for speculation but not for retained decoded state

Severity: **medium**.

The design covers the speculation-versus-required case well: a reserved progress allowance for
blocking reads (Phase 4), "the scheduler must be able to stop further speculation and reserve
progress credit for blocking work", and invariant 8 requiring every `Blocked` to name a condition
that can change.

The uncovered case is hold-and-wait on **decoded** credit across morsels. A `PackExec` holding K-1
child tails is charged for them while needing decoded credit to advance child K, and child K's
progress is the only thing that releases them. Invariant 8 is a local check: each morsel's `Blocked`
names a live condition, and yet no morsel can advance because every one holds partial state. The
progress reserve is defined against *reads*, not against decoded bytes and retained tails.

Add the standard pool rule as a settled design choice:

- Credits are reserved **per morsel at admission**; a morsel is admitted only if its worst case can
  be granted.
- The **oldest in-flight morsel is never denied credit** in any class. It can always drain and
  release, so global progress follows by induction on morsel age.

F3's capping removes most tails and therefore most of the exposure, but the rule is still needed for
`ListPack` element buffers and `Take` value caches, which retain by construction.

## F6. `Yield` and minimum prefix length are unbounded

Severity: **medium**.

`Yield` is a fourth non-terminal outcome — "useful local progress was made, but the transition
budget was exhausted" — and nothing constrains a `Yield` → `Yield` loop. Invariant 8 constrains
`Blocked` only. Separately, a node returning a one-row prefix forever satisfies every batch
invariant. Phase 1 validates that "a perpetually ready node yields after its transition budget",
which is the opposite property: it proves `Yield` happens, not that it terminates.

Two additions:

- A `Yield` must be accompanied by evidence of progress — at minimum a monotonically increasing
  transition counter, ideally a frontier that moved. A node returning `Yield` twice with no frontier
  change and no ticket state change is a debug assertion.
- A node returns at least `min(request.rows.len(), MIN_PREFIX_ROWS)` unless bounded by an
  indivisible unit or a credit. Track drives-per-committed-row as a metric with a debug ceiling.

Phase 11 already measures "Yield count" and "no-progress wakes"; make the no-progress case an
assertion in debug builds rather than only a metric.

## F7. `splits.rs`'s central operator switch is never replaced, though F1 makes it free

Severity: **medium**.

Phase 6 item 2 says "Continue using current split discovery as the source of outer morsel ranges",
and nothing later replaces it. `collect_plan_splits` (`vortex-scan-v2/src/splits.rs:62-123`) is a
hard-coded switch over `Zoned`, `Eval`, `RowIdx`, `Take`, `Pack`, `RowIdxPartition`, and `Concat` —
the same central-type-switch pattern `plan-v2.md` already flags as needing "a layout vtable hook or
registry so third-party layouts can produce plans without editing a central module". Extensibility
is one of the design's stated motivations, and this is the one place it is left intact.

F1 supplies the replacement. Split discovery becomes: walk edges whose `DomainMap` is static and
prefix-preserving, translating boundaries through the map, and stop at gated maps. That is exactly
what the seven hand-written cases already compute — `Pack`'s `child.row_count() == plan.row_count()`
test is an `Identity` check, `Concat`'s `row_offset + chunk_offset` is a `Shift`, and taking only
`Take`'s codes child is skipping a `GatherGated` edge. Combined with catalog coverage it also makes
decision 8 of the final list ("Are morsels always row-count ranges, or should physical boundaries
align or cap them?") answerable from data the design already holds.

Add it as a Phase 6 work item with its own exit criterion, and remove split discovery's operator
switch in the same change that proves the derived boundaries match today's.

Related sequencing note: `RebatchExec` lands in Phase 10, but it is independently valuable, needs
none of the state machine, and decouples the public batch size from the 100,000-row unit against
*today's* executor. Shipping it in Phase 0 alongside the baseline harness would deliver one of the
design's five real benefits before any of the risk, and give Phase 11's batch-size-distribution
metric a stable reference point.

## F8. `Take` still lists three strategies as peers

Severity: **medium**. Largely answered by F1; what remains is choosing the default.

Phase 9: "choose full, sparse, or incremental value-domain materialization." Under F1 these are not
three architectures but three widths of the same value-domain gather mask, so the choice is a
policy knob rather than a design fork. They are still not equivalent:

- Full materialization is what `take.rs` does today and is unbounded in dictionary cardinality
  (F2).
- Sparse gather is correct **per prefix** and wrong if read as "collect all codes first", which
  would defeat prefix progress for the whole subtree.
- Incremental is per-prefix gather with cross-prefix reuse, which needs the `ScanState` cache.

Pick a default and name the fallback:

> Default to per-prefix code collection followed by a sparse gather in the lookup domain, backed by
> a bounded `ScanState` value cache. Fall back to full materialization only when the domain is below
> a byte threshold — which is also the common case and the fast path.

## F9. `SealedDemand` cannot represent a suffix without allocating

Severity: **low-medium**, but concrete. The `mask_offset` field in F1's struct is this fix.

~~~rust
struct SealedDemand<'a> {
    rows: Range<u64>,
    mask: &'a Mask,
}
~~~

The batch contract requires "subsequent requests begin at the previous rows.end", so the coordinator
re-mints a `SealedDemand` for the unconsumed suffix after every prefix. With `mask: &'a Mask`
interpreted relative to `rows.start`, that requires a sliced mask — an allocation per prefix, on the
hot path, at roughly 12 prefixes per morsel per operator edge. Carrying
`mask_offset: usize` instead makes a suffix a field update, with the underlying sealed mask shared
for the morsel's lifetime. `Shift` derivation becomes a field update for the same reason, so the
common `Concat` edge stays allocation-free too.
This also interacts with block size: sealed windows are 1,024 rows and prefix targets are 8,192, so
state whether `SealedDemand.rows` is always block-aligned (the contiguous sealed frontier) and
whether a batch prefix may end mid-block. The invariants currently imply it may, which is fine, but
it should be written down because it determines whether the ledger or the operator does the slicing.

## F10. `BlockDemandSummary` holds four encodings of one fact

Severity: **low**.

By the document's own definitions, `maybe_nonempty[i]` is true exactly when the candidate mask is
non-empty, and `upper_counts[i]` is the exact current candidate count — so
`maybe_nonempty[i] == (upper_counts[i] > 0)`. The tri-state `Zero`/`All`/`Mixed` summary is a third
encoding of the same information plus "is it saturated", and `sealed_nonempty` folds in block state.

Phase 2's own rationale is that "the optimization cannot become a second source of truth". Keep the
two that are not derivable — `upper_counts` (exact) and block state — and derive the rest. If
`maybe_nonempty` exists as a `BitSet` purely so the scheduler can scan many blocks with one bitwise
pass, say that, because it is a cache of a derived value and needs a stated coherence rule.

## F11. `ExecBatch.demand` is redundant

Severity: **low**.

Invariant 3 says `demand` is exactly the sealed mask sliced to `rows`. The requester issued the
request and holds the sealed mask, so it can compute the slice — which it must do anyway to split
retained state. Carrying it on the batch creates a second source of truth and costs a mask slice per
batch per edge. `ExecBatch { rows, values, retained_bytes }` suffices; keep `demand` behind
`debug_assertions` as the cross-check the driver already performs at every edge.

Dropping `selection` from the earlier draft and adding `retained_bytes` were both right.

## F12. `expected_counts` needs a stated source, or a stated absence

Severity: **low**.

`expected_counts: Vec<f32>` is "derived from remaining predicate selectivities". The available
source is `FilterExpr::report_selectivity` (`vortex-scan-v2/src/filter.rs:97`), which records one
rate per conjunct after that conjunct runs, globally rather than per block. Applying a global rate
uniformly across blocks carries no per-block information, so `expected_counts` can only order reads
across *different* predicate sets, not across blocks under the same one.

Phase 2 already states expected counts never enter a correctness branch. Add that they may be
omitted entirely in the first implementation, and record what would justify adding them.

## Suggested edits

### Settled design choices

- Amend 6: open demand is owned by `DemandLedger`; projection planning may consume immutable open
  snapshots and summaries for candidate I/O and explicitly safe discovery work. Exact or fallible
  value execution receives sealed demand, and the operator owning an edge's `DomainMap` may
  **derive** sealed demand across it. *(F1)*
- Add: the row domain and its transforms are first-class. Every edge declares a `DomainMap`, and one
  map serves demand derivation, catalog coverage, morsel-boundary discovery, and row identity.
  *(F1)*
- Add: prefix progress composes across every map except `GatherGated`; a gather child is driven as a
  sub-root with its own cursor over its own domain. *(F1)*
- Amend 10: parents own alignment and use the request end as a **cap**; a child that decodes past
  the cap retains the surplus itself. Parent-owned tails are the exception. *(F3)*
- Amend 11: separate horizons and budgets, plus per-morsel credit reservation and a
  never-denied oldest morsel. *(F5)*
- Add: state has three tiers — immutable `PlanRef`, per-scan `ScanState`, per-morsel `ExecGraph` —
  and the read catalog spine is per scan, not per morsel. *(F2)*
- Add: every non-terminal drive outcome must carry evidence of progress. *(F6)*
- Consider deleting the epoch concept entirely, pending F4.

### Correctness invariants

- Amend 1: only `DemandLedger` constructs sealed demand, and only the operator owning an edge's
  `DomainMap` derives across it. Derivation must be **complete** — covering every child row a
  demanded parent row depends on — and, when it drives fallible work, **minimal**, covering no
  others. `Coarsen` is not minimal and may drive only infallible metadata work. *(F1)*
- Amend 8: every `Blocked` names a viable condition, **and** every `Yield` advances a transition
  count or frontier. *(F6)*
- Amend 12: retained data is charged to the node that can release it; a child that overshoots a cap
  charges itself. *(F3)*
- Add: a batch is at least `MIN_PREFIX_ROWS` unless bounded by an indivisible unit or a credit.
  *(F6)*
- Drop 4's reliance on a batch-carried mask; it becomes a debug-mode driver check. *(F11)*

### Implementation plan

- **Phase 0**: add the widening-reachability question to the contractual-behavior list *(F4)*; add
  `RebatchExec` against the current executor *(F7)*; record read-overlap between filter and
  projection as a baseline metric, since it is the thing most easily lost.
- **Phase 1**: add `DomainId`, `DomainMap`, and demand derivation to the primitives, and make the
  mock nodes exercise a non-identity map so the simulator cannot bake in row-space assumptions
  *(F1)*; add the `Yield` and minimum-prefix assertions *(F6)*.
- **Phase 3**: express catalog coverage as map composition rather than per-operator judgement
  *(F1)*; build the catalog spine per scan with per-morsel views, and give catalog entry count an
  exit criterion rather than a risk-register line *(F2)*.
- **Phase 5**: add the shared-boundary `Pack` test and implement capping *(F3)*.
- **Phase 6**: replace `splits.rs`'s operator switch with a `DomainMap` walk *(F1, F7)*.
- **Phase 9**: `ListPack` moves to the prefix-preserving class and should be portable well before
  `Take`; state the `Take` default rather than three options *(F1, F8)*.
