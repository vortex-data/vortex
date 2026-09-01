# Self-Paced Executor Reference

This document explains the implemented self-paced executor piece by piece: what each part is, why
it has the shape it has, and the trait every part hangs off. It then shows how each layout concept
— Flat, Chunked, Struct, and a future Dict — enters the executor. It complements the
[tutorial](self-paced-executor-tutorial.md) (concepts in historical order), the
[handover](self-paced-plan-exec-handover.md) (state and next work), and the
[findings](self-paced-plan-exec-findings.md) (every measurement referenced here). File paths are
relative to `vortex-layout/src/plan/exec/`.

The one-sentence architecture: **a scheduler that only knows "morsel in, ordered batches out", a
pluggable policy that computes each morsel's demand mask, and a per-field vtable that moves
demand down and results up — everything else is a kernel.**

## How a scan executes

Before the piece-by-piece detail, the two flows those pieces compose into.

### Plan level: from a file to a batch stream

```text
SourcePlan + ScanQuery                     immutable; built once, predicates coalesced
        |
StructScanPipeline::new                    wire topology: one ConcatDomain per field,
        |                                  shared projected-chunk emission boundaries
        |
morsel ranges                              natural splits merged; computed once per scan
        |
run_pipeline_sharded(pipeline, ranges, threads)
        |          reused pool, one shared atomic morsel cursor
        |
   thread 1..N, each:                      a fast thread pulls more morsels;
        loop {                             a straggler blocks nobody
            i = cursor.fetch_add(1)
            pipeline.execute(ctx, ranges[i], sink)   -- emits 1..k batches
            ctx.end_morsel()                          -- drop the morsel's decode cache
        }
        |
batches sorted by (morsel index, emission index)  ->  ordered output stream
```

The scheduler never learns what a pipeline does; the pipeline never learns which thread runs it
or in what order morsels complete.

### Morsel level: the pipeline's phases

One `execute` call moves a morsel through three phases; demand shrinks monotonically in the
first, is immutable after the second, and streams out during the third:

```text
+---------------------------------------------------------------------+
| FILTERING            DemandPolicy::morsel_demand                    |
|   for each conjunct, in policy order:                               |
|     push_demand(field)      cut the morsel, price each chunk        |
|     decode demanded chunks  via the per-thread cache                |
|     predicate kernel        full / sparse / dense regime            |
|     pull_mask -> demand     adopted subset; no intersection         |
|   (a zero-demand chunk is never read; None demand means all-true)   |
+-------------------+--------------------------------+----------------+
                    | demand nonempty                | demand empty
                    v                                v
+-----------------------------------+  +------------------------------+
| SEALED    selection := demand     |  | SEALED EMPTY                 |
+-------------------+---------------+  | emit one dense zero-value    |
                    v                  | batch; done                  |
+-----------------------------------+  +------------------------------+
| EMITTING   cut + price every projected field ONCE for the morsel,  |
|            then one batch per span between consecutive chunk       |
|            boundaries shared by every projected field:             |
|   selection and selected count come from zero-copy slices and the  |
|   already-priced segment counts (no recounting, no re-cutting)     |
|   for each projected field:                                        |
|     take its pre-cut span segments -> decode -> pull_array         |
|     release the span's decoded chunks                              |
|   pack_struct_array -> sink(ExecBatch)                             |
+-------------------+------------------------------------------------+
                    v
                  DONE      end_morsel() clears the remaining cache
```

The invariants that make this correct: demand only shrinks while filtering and never changes
after sealing; span cuts are boundaries of *every* projected field, so no projected chunk
straddles a cut and each span's chunks have no later use in the morsel; batches leave the sink in
row order within the morsel, and the scheduler restores cross-morsel order by index.

### Morsel level: the reactor's state machines

The reactor makes the same flow explicit as data — four interlocking machines instead of three
inline phases. **Result slots** hold every value exactly once:

```text
         offer            claim                complete
Empty ---------> Offered ---------> Running ---------> Ready(value)
                    |                  |
                    | revoke           | failure
                    v                  v
                  Empty              Failed
```

**Tasks** move through the scheduler: offered (with `Promote` and `Revoke` updates while queued),
claimed into a `RunnableTask` that owns cloned inputs and holds leases, evaluated by any worker,
and returned as a `Completion` the owner adopts. **Morsels** are driven by repeated `advance`
calls:

```text
advance(budget) --> Budgeted    budget expired: call advance again immediately
               \--> Quiescent   nothing to do until an outstanding task completes
               \--> Retired     final batch emitted; remaining leases drain
```

**Resources** (one per shared segment) track availability
`Absent -> Reading -> SegmentReady -> Decoding -> ArrayReady` crossed with lifetime: pinned while
any joined morsel or claimed lease uses them, reusable while an unresolved morsel still might,
dead otherwise. The external driver loop ties the machines together: `advance`, apply the task
updates, claim and evaluate admitted work, feed completions back, repeat until `Retired`.

The pipeline collapsed all of this into the three inline phases above — same model, no machinery
— which is why it is both the fastest mode and the harder one to observe; the reactor remains the
observable, contract-checking form.

## Part 1: the pieces and why they exist

### The plan: `SourcePlan`, `ChunkPlan`, `FlatPlan` (`model.rs`)

```rust
struct FlatPlan {
    field: FieldId,
    segment: SegmentId,
    root_coverage: Range<u64>,
    row_count: usize,
    estimated_bytes: Option<usize>,
    encoding: FlatEncoding,      // RawI64 | Serialized { dtype, read_ctx, array_tree }
}
```

A `FlatPlan` is one physical leaf: "rows N..M of field F live in segment S, decoded like E". A
`SourcePlan` is field names plus a list of `ChunkPlan`s, each holding one `FlatPlan` per field.
`SourcePlan::try_from_layout` validates a reopened `Struct(Chunked(Flat))` footer into this form.

**Why it exists this way.** V1 holds the same information implicitly, spread across reader
objects, and re-derives it per call. Making it one immutable value means every later decision —
cutting, pricing, skipping — is arithmetic over data, and the plan can be built once and shared.
The rule that goes with it: **planning does no compute**. An eager plan-time materialization of
the per-morsel segment cutting was built and measured slower (it serialized ~100ns/segment
arithmetic that sixteen threads otherwise do in parallel), so the plan stays purely descriptive.

### The query: `ScanQuery`, `Conjunct`, `Predicate` (`model.rs`)

A query is `conjuncts: Vec<Conjunct>` (a `FieldId` plus a comparison) and `projection:
Vec<FieldId>`. Before execution, `coalesce_same_field_predicates` intersects compatible predicates
on the same field algebraically — two bounds on `l_shipdate` become one `RangeExclusive`.

**Why.** Every conjunct pass costs a decode traversal over the demanded rows; predicates that can
be fused in the query representation should never reach the executor twice. On TPC-H Q6 this cut
predicate tasks from 2,290 to 1,374 and halved aggregate predicate latency.

### Demand: `Option<BitBuffer>`

Demand is the morsel's still-alive row set. It is a plain bit buffer — not an array, not a mask
object — and `None` means "all rows", kept symbolic.

**Why.** Materializing an all-true buffer per morsel was measurable waste on sub-millisecond
scans, and most morsels of an unfiltered scan never need physical bits at all. `BitBuffer` is the
cheapest representation that supports the three operations demand actually needs: `count_range`
(pricing), `slice` (cutting), and `&` (intersection).

### The per-thread context: `PipelineCtx` (`pipeline.rs`)

```rust
struct PipelineCtx<'a> {
    source: &'a dyn SegmentSource,
    session: &'a VortexSession,
    decoded: HashMap<SegmentId, ArrayRef>,
}
```

`decoded_chunk(plan)` reads and decodes a chunk once per thread, keyed by segment identity.
`release(segment)` drops one cached decode, and `end_morsel()` drops them all.

**Why per-thread rather than shared.** A shared cache needs locks or an owner, which is the
coordinator problem all over again. Per-thread caching needs neither, and because morsel groups
tend to end on natural splits, the cost is at most a handful of duplicate boundary decodes per
run (measured: one on ClickBench Q45). The cache is also what makes filter/projection sharing
work: a field used by both decodes once (FineWeb Q10 reads 238 MB against V1's 358 MB).

**Why the cache is scoped, not scan-lived.** An unbounded cache retains every chunk a thread
touches — memory proportional to the whole scan. Instead the executor releases each emission
span's chunks the moment its batch is emitted, and the scheduler clears the rest between
morsels, so executor-retained decoded memory is bounded by one thread's current working set. A
chunk shared with an adjacent morsel is re-read: the same bounded, deterministic duplicate the
cross-thread boundary case already accepts. Emitted batches keep their own refcounted views, so
end-to-end residency is the consumer's pace plus the working set — which is what streaming
output is for.

### The scheduler: `run_pipeline_sharded` (`pipeline.rs`)

Roughly seventy lines: a lazily created, **reused** thread pool per thread-count, one shared
`AtomicUsize` morsel cursor that threads `fetch_add` from, results tagged with their morsel index
and sorted at the end.

**Why so small.** The scheduler's entire knowledge of execution is `dyn MorselPipeline`, so there
is nothing else for it to do. Why self-scheduling instead of pre-assigned contiguous groups: fixed
groups left tail imbalance whenever morsel counts were few or uneven (ClickBench dashboard
1.06 -> 0.82, Q40 1.22 -> 0.67 after the switch). Why a reused pool: per-run thread spawns were
the dominant fixed cost of sub-millisecond scans. Order restoration by index keeps the ordered
output contract without any cross-thread coordination during the scan.

### Trait 1 of 3: `MorselPipeline` — all the scheduler sees

```rust
type BatchSink<'s> = dyn FnMut(ExecBatch) -> VortexResult<()> + Send + 's;

trait MorselPipeline: Send + Sync {
    fn execute<'a, 'c>(
        &'a self,
        ctx: &'a mut PipelineCtx<'c>,
        range: Range<u64>,
        sink: &'a mut BatchSink<'_>,
    ) -> BoxFuture<'a, VortexResult<()>>;
}
```

**Why it exists.** The reactor generation proved that coupling the scheduler to node structure
puts every node change on the scheduler's critical path — and the coordinator that resulted was
89% busy while workers starved. Behind one trait, any node graph is schedulable and no new node
ever touches scheduling. Output streams through the sink as ordered dense-prefix batches — the
struct pipeline emits one per shared projected-chunk span — and emitting a single whole-morsel
batch remains the valid degenerate stream.

### Trait 2 of 3: `DemandPolicy` — how a morsel's demand gets computed

```rust
trait DemandPolicy: Send + Sync {
    fn morsel_demand<'a, 'c>(
        &'a self,
        ctx: &'a mut PipelineCtx<'c>,
        fields: &'a FieldSet<'a>,
        query: &'a ScanQuery,
    ) -> BoxFuture<'a, VortexResult<Option<BitBuffer>>>;
}
```

Three implementations, all output-identical (conjunction commutes, and every result is adopted as
a subset of the demand it was evaluated under — the hash gate checks this on every run):

- **`CascadeDemand`**: conjuncts in query order against shrinking demand. A chunk whose demanded
  rows price to zero is neither read nor decoded; a mask pulled up from a cascade round is
  already a subset of the current demand, so it is adopted directly with no intersection.
- **`EagerDemand`**: every conjunct over every row, then intersect. Exists as the baseline the
  cascade must beat and because dense demand makes gating cost more than it avoids.
- **`AdaptiveDemand`** (default): the cascade with two measured behaviors folded in. Conjuncts
  run most-selective-first using survival rates accumulated across morsels in lock-free atomics
  (unobserved conjuncts keep query order via a neutral prior); and any conjunct whose current
  demand is at least half dense switches to full-evaluate-and-intersect, because the dense
  crossover was measured directly (ClickBench Q02 at 87.7% survival: cascade 2.38 vs eager 1.31).

**Why a trait.** The crossovers are workload properties, not code properties. Swapping the policy
must touch nothing but the policy object — and did, repeatedly, during the experiments.

### Trait 3 of 3: `FieldDomain` — row-domain relationships as a vtable

```rust
trait FieldDomain: Send + Sync {
    fn push_demand<'a>(&'a self, range: &Range<u64>, demand: Option<&BitBuffer>)
        -> VortexResult<Vec<ChildSegment<'a>>>;
    fn pull_mask(&self, range: &Range<u64>, parts: Vec<(Range<usize>, BitBuffer)>)
        -> VortexResult<BitBuffer>;
    fn pull_array(&self, segments: &[ChildSegment<'_>], arrays: Vec<ArrayRef>,
        true_count: usize, range_rows: usize, shared_mask: Option<&Mask>)
        -> VortexResult<ArrayRef>;
}

struct ChildSegment<'p> {
    plan: &'p FlatPlan,           // the physical leaf to read
    chunk_local: Range<usize>,    // the overlap in the child's coordinates
    parent_local: Range<usize>,   // the overlap in the parent's coordinates
    demanded: usize,              // demanded rows in this overlap — the price
    demand: Option<BitBuffer>,    // demand restricted to the overlap (None = all)
}
```

Every parent/child row relationship is one **down demand transform** (`push_demand`: cut the
parent range into child segments and price each) and two **up transforms** (`pull_mask`,
`pull_array`: reassemble child results in parent coordinates). `FieldSet` is the per-morsel view
that hands callers the right vtable per field; policies and projection speak only to it and
cannot tell relationship kinds apart.

**Why this shape.**

- *Pricing inside the cut* is what makes skipping free: callers drop `demanded == 0` segments
  before any read, and `pull_array`'s coverage check sums the already-priced counts instead of
  re-scanning bits.
- *Per-chunk dispatch, never per row* is why the abstraction costs nothing measurable: after the
  vtable refactor, FineWeb moved ~0.32 -> ~0.34 geometric mean and TPC-H was unchanged.
- *Modeled on the layout's native metadata* — prefix sums, offsets, refcounts — rather than any
  materialized row mapping, which is the executable form of the design's `DomainMap` idea.

### The kernels (`evaluate.rs`)

Plain functions with no scheduling opinions: `decode_flat` (raw or serialized chunk to
`ArrayRef`), `pack_struct_array`, and the predicate kernels with three demand regimes:

```text
full demand        vectorized multiversioned collector, no mask consulted
sparse  (<= 1/5)   iterate set bits, evaluate only demanded rows
dense-but-partial  two vectorized passes: full evaluation, then AND with demand
```

**Why three regimes.** Each boundary is a measured crossover: sparse iteration wins when demand
is rare; consulting the demand bit per row loses to two vectorized passes once demand is dense
(this switch is what made five-conjunct chains competitive under the cascade); and full demand
should never touch a mask at all — an all-match "optimistic pre-scan" fast path was tried and
removed because the scalar loop lost to the multiversioned collector.

### The output: `ExecBatch`

`coverage` (dense root-row range), `selection` (the sealed demand sliced to the batch's span, as
a boolean array), `array` (the packed compact values). The selection *is* the sealed demand —
filtering and output selection are one object. The pipeline streams one batch per shared
projected-chunk span; the reactor modes still emit one whole-morsel batch, the valid degenerate
stream.

### The reactor generation (`model.rs`, `slots.rs`, `graph.rs`, `reactor.rs`, `baseline.rs`)

The first executor modeled execution as an explicit task graph, and it remains in the tree as the
validated-contract reference and as the `pooled`/`owned` modes (`VORTEX_SELF_PACED_SHARD_MODE` in
`baseline.rs`). Its pieces:

- **Write-once slots** (`slots.rs`): a five-state machine per result —
  `Empty -> Offered(task) -> Running(task) -> Ready(value) | Failed`, with `revoke` returning an
  offered slot to empty. Every transition checks task ownership, which is what made wrong-type,
  duplicate, stale, and revoked completions mechanically rejectable.
- **Offers, claims, leases** (`model.rs`, `reactor.rs`): an offered `Task` is descriptive (slot
  identifiers only); `claim` clones resolved inputs into an immutable `RunnableTask` and acquires
  leases, so workers never touch the mutable store and revocation stays safe.
- **Resource nodes** (`graph.rs`): scan-wide state per physical segment. Lifetime is a
  three-line classification — joined users or leases pin it, unresolved users keep it reusable,
  otherwise it is dead — which answers "can this decoded array be dropped" without a graph walk.
- **Mask summaries** (`model.rs`): every boolean result carries `len`, `true_count`, and its bit
  buffer, so planning and sealing never scan an array; `CachedPredicate` additionally records
  exactly which rows it evaluated, making partial-predicate reuse coverage-safe.

**Why it still exists.** These contracts are what the experiment set out to test, and they all
held. What did not hold was the execution architecture around them: one coordinator serializing
thousands of small transitions. The pipeline keeps the model (morsels, demand, skipping,
dedup-by-segment) and discards the machinery — the slot/offer/claim apparatus does not exist in
pipeline mode. The reactor is the proof of correctness properties; the pipeline is the proof that
they can be had cheaply.

### Correctness enforcement (`tests.rs`, the harness)

Four independent layers: `run_eager`, a trivially correct reference executor every mode is
differentially tested against; row-count plus ordered-output-hash gates before any timing, with
per-iteration row re-checks; a per-iteration cold-scan I/O invariant (each run must re-read its
warmup's unique-segment floor, byte-exact for self-paced); and a DuckDB oracle over the original
Parquet for seventeen workloads. Unit tests cover misaligned children, empty demand, shared
resources, and revocation.

## Part 2: how each concept enters the executor

A concept enters by answering three questions: *how does demand reach my rows* (`push_demand`),
*how do my masks come back* (`pull_mask`), *how do my values come back* (`pull_array`). The
answers sort every layout concept into one of three kinds:

```text
identity        same row domain           -> share the demand handle; zero-copy up   (Struct)
metadata map    child cut from static     -> a pure FieldDomain                      (Chunked)
                offsets/counts
staged map      child addresses need a    -> a node that decodes the fact, then runs (Dict, List)
                decoded fact first           a second FieldDomain cycle
```

### Flat: the leaf

Flat is not a `FieldDomain` — it is what the domains bottom out in. A `ChildSegment` names a
`FlatPlan`; executing it is `ctx.decoded_chunk(plan)` plus a kernel over the decoded values. Flat
contributes three things: a physical identity (`SegmentId`, the dedup and cache key), a coverage
(`root_coverage`, what cutting arithmetic consumes), and a decode recipe (`FlatEncoding`). Adding
a new leaf encoding means extending `decode_flat` and nothing else — no trait, no scheduler, no
policy change.

### Chunked: `ConcatDomain`, the metadata-map archetype

The chunked (concatenation) relationship is implemented entirely on the chunk-offset prefix sums
the layout already stores:

- **`push_demand`**: `partition_point` binary-search to the first overlapping chunk, walk the
  overlaps, verify they tile the range, and price each with `count_range` plus a demand `slice`.
  Cost: `O(log chunks + overlaps)`, independent of row count.
- **`pull_mask`**: parts arrive in parent order and tile the range, so reassembly is ordered
  `append_buffer` — with a zero-copy fast path when one segment covers the whole morsel.
- **`pull_array`**: slice each decoded chunk to its overlap; one part passes through, several
  become a `ChunkedArray`. Then three exits in cheapness order: all rows demanded — return
  unfiltered; the field gathered the whole range under partial demand — filter by the parent's
  `shared_mask`, built once per morsel; otherwise concatenate the per-segment demand slices and
  filter by that (a lazy `FilterArray`, matching V1's output materialization).

**Why this is the archetype**: every decision is arithmetic over `root_coverage` values that
exist in the plan. Nothing is decoded to *decide* anything. And because each field owns its own
`ConcatDomain`, fields with mutually unaligned chunk boundaries need no alignment step — cutting
is root-row arithmetic per field (unit-tested with fields chunked `[0,3,10)` against
`[0,6,10)`).

### Struct: the identity relationship

Struct is deliberately *not* a `FieldDomain` either, because there is no transform to write: its
children share its row domain. Its two halves live in `StructScanPipeline::execute`:

- **Down**: compute the morsel's demand once via the `DemandPolicy`, cut and price every
  projected field once against that shared mask, and build one selection `Mask` per emission
  span, shared by every field that gathers the span (`shared_mask`). Identity means share, not
  copy — the span loop only consumes what the morsel-level cut produced.
- **Up**: `pack_struct_array` assembles the field arrays into a `StructArray` without copying
  values.

A nested struct is the composition of identities — which is why the restricted executor can
simply flatten fields; a nested output shape would pack twice, nothing more.

### Dict: how it would look

A dictionary field is two children in different row domains: **codes** (one per row — the row
domain) and **values** (one per distinct value — the dictionary domain). The codes side is
ordinary: codes chunks form a `ConcatDomain`, and demand reaches them exactly as it reaches any
Flat field. What is new is that the *values* work cannot be priced from metadata: which value
pages matter depends on which codes survive — the design documents call this a gated (or
`GatherGated`) edge.

That makes Dict the staged-map archetype. The clean composition keeps `FieldDomain` pure-metadata
and stages two cycles inside the node:

```rust
struct DictField {
    codes: ConcatDomain,     // row domain -> code chunks (static metadata)
    values: ConcatDomain,    // dictionary domain -> value chunks (static metadata)
    // scan-wide decoded-values cache, keyed by (SegmentId, coverage): morsels
    // share dictionaries, so value pages outlive any one morsel.
}
```

- **Filtering on a dict field** needs no new demand machinery: evaluate the predicate once over
  the values domain (small), producing a matching-code set; the per-row kernel becomes code-set
  membership over the decoded codes. To the `DemandPolicy` this is just another conjunct — the
  demand algebra is untouched, only the kernel differs. This is also the cheap path: the values
  domain is usually orders of magnitude smaller than the row domain.
- **Projecting a dict field** stages: (1) `codes.push_demand(range, demand)` and decode the
  surviving code segments — a normal metadata cycle; (2) compute the distinct surviving codes —
  the gather set, the data-dependent fact; (3) treat the gather set as an immediately-sealed
  demand over the dictionary domain and run `values.push_demand(0..dict_len, gather_mask)` — a
  second, ordinary metadata cycle that prices and skips value pages exactly like chunks; (4)
  decode the demanded value pages (through the scan-wide cache) and `take(values, codes)` upward.

Two things follow from the staging. First, the values domain is a *sub-root*: its demand is
sealed the moment the gather set exists, because nothing else can shrink it — so no new demand
states are needed. Second, the seam holds without modification: `push_demand` stays synchronous
arithmetic in both cycles; the only await points are the two decode stages, which live in the
node exactly where Flat's decode already lives. The alternative — making `pull_array` async and
handing it the context so a domain can read — was rejected because it would let data dependencies
leak into the vtable that every pure-metadata relationship shares.

List is the same staged shape with a different fact: offsets instead of codes, run-expansion of
masks instead of gather sets, and the down transform reads `offsets[k]..offsets[k+1]` per
demanded outer row.

### The checklist for a new concept

1. Same row domain as the parent? Share the demand handle by refcount and pack zero-copy upward.
   No trait implementation needed (Struct).
2. Child mapping computable from plan metadata? Implement `FieldDomain` — cut, price, reassemble
   (Chunked; any fixed-arithmetic mapping).
3. Child mapping requiring a decoded fact? Build a node that decodes the fact under demand, then
   runs a second `FieldDomain` cycle in the child's domain, with a scan-wide cache when the child
   domain outlives morsels (Dict, List).
4. Register it: a projected field's entry in `StructScanPipeline` is a `Box<dyn FieldDomain>`; a
   new pipeline shape is a new `MorselPipeline`. The scheduler, the demand policies, and the
   kernels do not change — that separation is the point of the three traits.
