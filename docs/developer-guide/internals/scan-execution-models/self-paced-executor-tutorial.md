# The Self-Paced Execution Model, From First Principles

This tutorial teaches the experimental self-paced execution model to a reader who knows Layout
V1 and nothing else. It introduces one concept at a time, in the order the experiments
introduced them, and ties each back to its V1 counterpart. The
[findings report](self-paced-plan-exec-findings.md) has every benchmark table; the
[handover](self-paced-plan-exec-handover.md) has the code map; the
[executor reference](self-paced-executor-reference.md) explains each implementation piece and how
new layout concepts plug in. Here the goal is understanding.

## 1. What you already know: V1 in three sentences

In V1, a scan asks the layout tree for its **splits** (`register_splits` unions the chunk
boundaries of every field the query touches), then turns each split into an independent task on
the Tokio runtime. Each task calls `filter_evaluation(row_range, expr, mask)` to get a survivor
mask and `projection_evaluation(row_range, expr, mask)` to get the output rows, and each layout
node (struct, chunked, flat) implements those vtable methods by translating the row range into
its children's coordinates. A 15M-row file with ~1,800 splits therefore becomes ~1,800 futures,
each a black box that reads, decodes, filters, and projects its own little row range.

Hold on to two properties of this design, because the whole experiment is a reaction to them:

- **The unit of work is the split**, and there are thousands of them. Every split pays the
  future/scheduling machinery, and two splits that need the same segment don't know about each
  other.
- **Filtering and projecting are one opaque call per split.** The engine cannot see "predicate
  A eliminated 99% of rows, so don't bother reading column B for this region" across the
  boundary of a split, and it cannot share partially-computed filter state.

## 2. The experiment's question, and its restricted world

The self-paced experiment asks: if execution could see *inside* the scan — which rows are still
alive, which segments serve which predicates — could it do less work and go faster?

To make that tractable it restricts the world to one layout shape:
**`Struct(Chunked(Flat<i64>))`** — a struct of non-nullable i64 fields, each field a sequence of
flat chunks, all fields' chunks aligned at the same row boundaries. No compression, no nulls, no
strings (string datasets are ingested by hashing strings to i64). Real datasets (FineWeb,
ClickBench, TPC-H lineitem, gnomAD genomics) are converted into this shape so the *scheduling*
question can be studied without the *encoding* question. Everything below lives inside this
restriction; the last section says what lifting it takes.

## 3. Concept: the plan (`SourcePlan`)

V1 discovers structure lazily by walking reader objects. The experiment instead builds one
explicit, immutable description of the file up front:

```
SourcePlan
├── field_names: ["url_hash", "text_len", ...]
├── row_count: 14_868_862
└── chunks: [ChunkPlan { root_coverage: 0..8192, fields: [FlatPlan, FlatPlan, ...] }, ...]
      where FlatPlan = { field, segment_id, root_coverage, row_count, encoding }
```

A `FlatPlan` is one physical leaf: "rows 8192..16384 of field 3 live in segment 1042". That's
the whole plan — pure metadata, no data, built once per file. Everything the executor does is
phrased against it. (V1 analogue: the information `register_splits` and the readers hold
implicitly, made explicit and queryable.)

**Rule learned the hard way (section 12): planning does no compute.** The plan describes; the
executor works.

## 4. Concept: the morsel

The **morsel** is the self-paced unit of scheduling and ordering: a contiguous root-row range,
formed by merging 16 consecutive natural splits (so ~1,800 V1 splits become ~116 morsels). A
morsel may span several chunks. Its output is a sequence of dense-prefix batches in row order:

```
ExecBatch { coverage: 524288..655360, selection: BoolArray, array: StructArray }
```

A morsel *streams* those batches out rather than holding its whole result: streaming bounds
retained output memory, hands downstream consumers work before the morsel finishes, and is what
makes time-to-first-batch measurable. The pipeline executor emits one batch per span between
chunk boundaries shared by every projected field, releasing each span's decoded chunks at
emission; the reactor modes still emit one batch per morsel, the valid degenerate stream.

Why merge? Each unit of work pays fixed machinery cost; fewer, bigger units amortize it. Why
not merge everything into one? Parallelism needs at least as many units as cores, and output
should stream. The experiments ended with an *adaptive* merge (`clamp(splits/32, 1, 16)`)
because a fixed 16 collapsed compact datasets (8 splits -> 1 morsel -> 1 core).

**The fairness contract** for every number in these docs: V1 runs over the *unmerged* natural
splits, exactly as production V1 would; only self-paced gets morsels; both scan the same
serialized bytes with the same query, and row counts plus an ordered output hash are validated
before any timing.

## 5. Concept: demand

This is the one genuinely new idea; everything else is scheduling. **Demand** is a bitmask over
a morsel's rows meaning "these rows are still alive". It starts all-true and only ever shrinks:

```
morsel rows:      [r0 r1 r2 r3 r4 r5 r6 r7]
initial demand:    1  1  1  1  1  1  1  1
after A > 5:       0  1  1  0  1  0  1  1     <- conjunct A evaluated on all 8 rows
after B == 3:      0  1  0  0  0  0  1  0     <- B evaluated ONLY on the 5 surviving rows
projection:       read/decode/copy only what covers rows r1, r6
```

Three consequences, each worth money:

1. **Later predicates evaluate fewer rows.** On FineWeb Q06, later conjuncts evaluated 25K rows
   and skipped 29.7M row-visits.
2. **Whole segments can be skipped.** Before reading a chunk for predicate B or for projection,
   count demand in that chunk's range; zero means don't read it. The empty-result shapes read
   only the first filter column (1,823 requests vs V1's 7,292).
3. **The final demand mask *is* the selection** for the output batch — filtering and output
   selection are the same object.

V1 has a cousin of this (the mask threaded through `filter_evaluation`), but per split and
opaque; demand is morsel-wide state the executor can inspect, count, and route work by.

## 6. Generation 0: the reactor (what the handover left us)

The original executor modeled everything as a task graph: every read, decode, predicate,
selection, and pack was a **task** flowing through offer -> claim -> complete states, with
results in **slots**, morsels subdivided into per-chunk **fragments** so demand could advance
segment-by-segment, and cached predicate results (with explicit evaluated-row coverage) shared
between consumers. One **coordinator** thread owned all mutable state; a 16-thread pool
evaluated claimed tasks.

It was correct, observable, and **2.5x slower than V1** on the headline workload. The rest of
this tutorial is what the measurements said and what each redesign changed.

## 7. Measure before believing: phase timing

We added wall-clock attribution to the coordinator loop (drain / advance / schedule / dispatch /
wait) plus a timestamp on every worker completion. Finding: the coordinator was **89% busy**
(advance 34%, completion handling 28%, dispatch 24%) and each finished worker result sat ~17us
in a queue before being absorbed. The workers were starving behind the coordinator.

We then applied every micro-optimization the code audits suggested — allocation-free mask
adoption, batched state transitions, skipped scheduler passes. Q06 moved 2.53x -> 2.32x.
**Lesson: reducing work on a serialized path barely moves wall time. You must parallelize the
path or delete it.**

## 8. Generation 1: sharded coordinators (2.32x -> 1.40x)

Observation: almost all coordinator state is *morsel-local*. So partition the morsel list into
N contiguous groups, give each group its own private `Execution` and coordinator thread, share
the worker pool. Because morsel boundaries land on chunk boundaries, no segment straddles
groups — the sharded run performed byte-identical I/O. Four shards: **1.40x**.

## 9. Generation 2: owned execution (1.40x -> 0.79x)

If four self-contained coordinators work, the coordinator/worker split itself is the question.
**Owned mode**: 16 threads, each owns a morsel group and runs the *whole* loop inline —
coordinates its own demand state and executes every read, decode, predicate, and selection
itself. No pool, no completion channel, no dispatch, no queue. Thread count now equals V1's.
Q06 became a win (0.79) and the model collapsed to something simple: **morsel-driven
self-coordination**. The cross-thread communication was the cost; the coordination logic never
was.

## 10. Generation 3: the pipeline (0.79x -> 0.41x)

Owned mode still ran the reactor's task-graph machinery per morsel. The final rebuild keeps the
execution *model* (morsels, demand, skipping) and discards the task graph. It is defined by
three traits — this is the part worth learning, because it is the extensibility story:

**(a) `MorselPipeline` — all the scheduler knows.**

```rust
trait MorselPipeline {
    fn execute(&self, ctx: &mut PipelineCtx, morsel: Range<u64>, sink: &mut BatchSink)
        -> Future<()>;
}
```

The sink receives the morsel's output as ordered dense-prefix batches — the struct pipeline
emits one per shared projected-chunk span — so a morsel's results flow out before the morsel
finishes and nothing accumulates a whole morsel's output.

The scheduler is ~40 lines: threads pull morsel indices from one shared atomic counter (work
stealing — a fast thread takes more morsels; order is restored by index), each on a reused
pool, each with a `PipelineCtx` holding a per-thread decoded-chunk cache (so a field used by
filter *and* projection decodes once). Adding any new node or pipeline shape never touches this.

**(b) `DemandPolicy` — how the morsel's demand mask gets computed.**

```rust
trait DemandPolicy {
    fn morsel_demand(&self, ctx, fields: &FieldSet, query) -> Future<Option<BitBuffer>>;
}
```

The struct node computes demand once per morsel and shares the same refcounted mask with every
child. Implementations are swappable: `cascade` (conjuncts in order against shrinking demand,
skipping empty chunks), `eager` (all conjuncts in full, intersect), and the default `adaptive`
(order conjuncts by observed survival, most selective first; switch any conjunct to
full-evaluate-and-intersect when demand is >= 50% dense, because gating dense demand costs more
than it avoids — both behaviors are measured crossovers, and all policies are output-identical
by construction and by the hash gate).

**(c) `FieldDomain` — row-domain relationships as two vtable transforms.**

Every parent/child row relationship in a layout is expressible as a *down demand transform* and
*up result transforms*:

```rust
trait FieldDomain {
    fn push_demand(&self, range, demand) -> Vec<ChildSegment>;  // down: cut + price
    fn pull_mask(&self, range, parts)    -> BitBuffer;          // up: masks -> parent domain
    fn pull_array(&self, segments, arrays, ...) -> ArrayRef;    // up: arrays -> parent domain
}
```

`push_demand` cuts a parent row range into child segments — each with its coordinates in both
domains and its **demanded row count**, so callers skip empty children before any read. Each
relationship is modeled on the layout's own metadata, never a materialized mapping:

| Relationship | Model | Down | Up |
| --- | --- | --- | --- |
| struct (zip) | none — same row domain | share the demand handle by refcount | zero-copy struct pack |
| chunked (concat) | chunk-offset prefix sums | binary search + `count_range` + mask slice | ordered append / chunk assembly |
| list (future) | its offsets buffer | two offset loads; run-expand masks | per-run reduce |
| filter/demand itself | bitmap + rank | `count_range` / `select` | — |

Children with mutually **unaligned** chunk boundaries just work, because alignment is root-row
arithmetic, not a precondition (unit-tested with fields chunked `[0,3,10)` vs `[0,6,10)`).
Dispatch happens per *chunk*, never per row, so the whole trait seam measured ~0-5% — the
abstraction is effectively free.

Result: Q06 at **0.41**. The attribution cornerstone is the wide select-all shape: both engines
read byte-identical data, nothing is avoidable, and the pipeline is still ~2.5x faster — the
residual advantage is purely cheaper scheduling units (tens of self-scheduled morsels vs
thousands of per-split futures) plus inline execution.

## 11. V1 -> self-paced translation table

| V1 concept | Self-paced counterpart |
| --- | --- |
| split | morsel (merged splits; adaptive merge factor) |
| per-split Tokio future | thread pulling morsels from a shared cursor, executing inline |
| `register_splits` | the plan's chunk coverage + the harness's split catalog |
| `filter_evaluation` per split | `DemandPolicy` per morsel (inspectable, ordered, chunk-skipping) |
| `projection_evaluation` per node | `FieldDomain::push_demand` + `pull_array` |
| mask argument | demand: morsel-wide, shrinking, countable |
| reader-internal range translation | `FieldDomain` down/up transforms over native metadata |

## 12. Two rules that came from failed experiments

- **Planning does no compute.** Pre-materializing the segment cutting at plan time measured
  *slower* (it serialized ~100ns/segment arithmetic that threads do in parallel, and per-scan
  planning amortizes nothing). Deleted. Planning wires topology and shares demand handles; the
  splits are computed once; all compute happens on the owning threads.
- **Fixed cost per morsel is its own budget.** Sub-millisecond scans (genomics dataset) exposed
  per-morsel constants: per-run thread spawns, an all-true mask allocation, per-field `Mask`
  construction, redundant coverage bit-scans — each individually invisible on a 10ms scan. All
  removed (reused pool, mask-free full evaluation, single-segment zero-copy paths, one shared
  selection `Mask` per morsel). Also: 5-iteration medians are noise at this scale; sub-ms
  shapes use 100-iteration medians.

## 13. What flowed back into production V1

The I/O audit caught V1 reading up to **2.7x the file size** on shared filter/projection scans:
`FlatReader::array_future` rebuilt its "shared" future on every call, so filter and projection
(and every split subdividing a chunk) re-read and re-decoded the same segment. The fix is the
pipeline's dedup idea under V1's memory discipline: memoize the future behind a `WeakShared` —
shared while any evaluation is live, freed when the last consumer drops, so scan memory stays
flat. Committed independently off `develop` (`worktree-v1-flat-reader-dedup`). All comparisons
above are against the *fixed* V1.

## 14. How we know it's correct

Four layers, in increasing independence: (1) every benchmark run validates identical row counts
and an ordered full-output hash between V1 and self-paced before timing, and every timed
iteration re-checks row counts; (2) the pipeline is tested against `run_eager`, a trivially
correct reference, plus unit tests for misaligned children, empty demand, and the FlatReader
dedup/release semantics; (3) a per-iteration **no-caching invariant**: each run must re-read at
least its cold warmup's unique-segment bytes (byte-exact under deterministic policies); (4) an
external oracle — 17 workloads checked against **DuckDB over the original parquet**, all row
counts exact.

## 15. Where it stands, and how to refine it

| Suite | vs fixed V1 | Geomean |
| --- | --- | ---: |
| FineWeb (18 shapes) | 18/18 wins | ~0.33 |
| TPC-H SF10 (3) | 3/3 wins | ~0.63 |
| ClickBench (25 shapes) | 25/25 wins | ~0.56 |
| statpopgen (6, sub-ms) | 3 wins, 2 ties, 1 open | — |

Q06 arc: 2.53 -> 2.32 (micro-opts) -> 1.40 (sharded) -> 0.79 (owned) -> **0.41 (pipeline)**.

Refinement plan, in order:

1. **statpopgen Q02 anomaly**: eager policy runs 1.31 but the logically equivalent in-policy
   dense switch runs 2.62 — that delta shouldn't exist; a samply profile is captured.
2. **TPC-H Q6 makespan**: 29 huge morsels bound wall time at 2 serial morsels/thread; needs
   intra-morsel parallelism or byte/CPU-aware roll-up.
3. **Stream morsel output** *(pipeline: done)*: `MorselPipeline` emits ordered `ExecBatch`
   prefixes through a batch sink — one per shared projected-chunk span — and releases each
   span's decoded chunks at emission, with the per-morsel cache clear bounding executor memory
   to the working set. Remaining: the reactor's `AdvanceResult` prefixes, and measuring
   time-to-first-batch and peak retained output in the harness.
4. **Real I/O**: everything here is in-memory. Agenda: ranged/multi-get `SegmentSource` (run
   coalescing — up to 86x fewer requests on wide scans), per-thread async read-ahead,
   writer-side chunk sizing for small-segment datasets.
5. **Lift the restriction**: unaligned real files need only per-field `FieldDomain` instances
   (the seam is proven); then a list node over its offsets buffer; then compressed encodings,
   nulls, general expressions.
6. **Ship the V1 fix** (independent PR), and deepen the oracle to value-level checks.
