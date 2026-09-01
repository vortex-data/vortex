# Scan Execution Model Comparison

This section compares four ways to turn a stored layout into scan results:

1. the established V1 `LayoutReader`;
2. the plan-native executor in `vortex-scan-v2`;
3. the `layout27` design and the hybrid path at the tip of `ji/layout27`; and
4. a proposed demand-bounded, self-paced executor.

The comparison is deliberately about the complete path:

```text
layout -> plan -> execution system -> output batches
```

That distinction matters. A good plan representation does not by itself provide scheduling,
backpressure, memory control, or a useful batching contract. Conversely, a mature executor can be
difficult to optimize if planning and mutable runtime state are represented by the same objects.

The **current working direction** is
[morsel-based plan execution](morsel-based-plan-execution.md): one stateful exec-node graph per
fixed row morsel, a lazy `IO | Plan` stream whose planning state remains internal to the morsel,
resumable value execution, and explicit retirement. The comparison and earlier proposals below
remain as design history and evidence. Its [documents-to-use
section](morsel-based-plan-execution.md#documents-to-use) is the short reading map.

Its P1 spine is now **implemented** in the `vortex-morsel` crate and measured against the V1
`LayoutReader`: see [P1 findings](morsel-prototype-p1-findings.md) for what was built, what the
numbers say, and which parts of the plan's evaluation matrix could not be run here — gate E1 as
written was *not* evaluated, because rows B and C do not exist in this repository. The raw
evaluation output is in [P1 evaluation output](morsel-prototype-p1-eval.md).

Those P1 numbers came from synthetic fixtures. **Real TPC-H at SF=1** — `tpchgen` data, real
decimals and dates, written through the btrblocks compressing pipeline, running the real scan
portions of Q1/Q6/Q12/Q14/Q15/Q19 — is measured in
[real TPC-H results](morsel-prototype-tpch-findings.md): the prototype is ~1.3x faster than V1 at
one thread and ~1.5x at four, and the cross-morsel decode reuse that mattered on synthetic
fixtures turns out to be neutral on a real file except on width-divergent schemas.
[The handoff](morsel-prototype-handoff.md) says how to re-run all of it on other hardware and
which conclusions are host-specific.

```{toctree}
---
maxdepth: 1
---

layout-reader-v1
plan-v2
layout27
self-paced
morsel-reactor
scheduler-visible-work
morsel-reactor-ideas
self-paced-plan-exec-experiment
self-paced-executor-tutorial
self-paced-executor-reference
self-paced-plan-exec-findings
self-paced-plan-exec-handover
self-paced-plan-exec-learnings
scan-execution-framework
scan-execution-graph-model
scan-execution-graph-next-discussion
morsel-prototype-p1-findings
morsel-prototype-p1-eval
morsel-prototype-tpch-findings
morsel-prototype-tpch-eval
morsel-prototype-tpch-sweep
morsel-prototype-handoff
scan-execution-demand-and-operators
scan-execution-design
scan-execution-design-one-pager
morsel-based-plan-execution
self-paced-implementation-plan
self-paced-review
```

## Executive comparison

| Property | V1 `LayoutReader` | Current plan v2 | `layout27` | Proposed self-paced model |
| --- | --- | --- | --- | --- |
| Layout representation | Stored `Layout` tree | Stored `Layout` tree | Stored `Layout` tree | Stored `Layout` tree |
| Physical plan | Implicit in reader tree | Generic, rewriteable `PlanRef` | Generic `ScanPlanRef` | Generic, immutable `PlanRef` |
| Per-scan executor | The reader tree itself | Recursive futures created per call | Prepared tasks; the `ji/layout27` tip delegates to V1 | Separate mutable `ExecNode` state-machine tree or arena |
| Expression pushdown | Reader-specific and repeated at execution boundaries | Generic plan rewrites | `ScanPlan::try_push_expr` | Generic plan rewrites before opening execution |
| Worker work unit | Precomputed split issued by the scan driver | Precomputed split issued by scan-v2 | Precomputed fixed morsel admitted by the scheduler | Configurable fixed morsel, normally around 100,000 rows |
| Intra-work-unit progress | One exact reader result for the split | One exact recursive plan result for the split | Multi-step tasks ending in one exact morsel result | Resumable state machines returning child-sized prefixes, for example 8,000 rows |
| Execution transition | Future resolving to a mask or array | Recursive future resolving to an array | Read step followed by a continuation | Run-to-quiescence `drive`: `Batch`, `Blocked`, `Done`, or `Yield`; work is registered through tickets |
| Child output size | Exactly requested cardinality | Exactly requested cardinality | Exactly one requested morsel | Any non-empty prefix within demand and memory bounds |
| Parent alignment | Guaranteed by exact child requests | Guaranteed by exact child requests | Guaranteed by morsel tasks | Parent caps the request end; children that overshoot retain their own surplus |
| Coordinate translation | Per-reader arithmetic | Per-operator arithmetic | Per-plan arithmetic | One declared `DomainMap` per edge, shared by demand, coverage, boundaries, and row identity |
| Runtime state location | `LayoutReader` implementations | Context plus some plan data and futures | Dedicated scan state and prepared handles | Per-scan `ScanState` for reusable facts, per-morsel `ExecGraph` for progress |
| I/O scheduling | Eager future construction and source-level sharing | Eager future construction and source-level sharing | Explicit reads, phases, lanes, priorities, and byte admission | Per-scan read catalog with morsel views, dynamic gates, lazy demand scoring, deduplication, and byte credits |
| Mask stability at projection | A shared future resolves the final split mask | A shared `MaskFuture` resolves the final split mask | Selection and demand are explicit in prepared tasks | Projection planning sees immutable open snapshots; exact value execution receives sealed demand |
| Split dependence | Required for pacing and parallelism | Required for pacing and parallelism | Required as morsel boundaries | Required only for outer morsel parallelism, not internal batching |
| Backpressure boundary | Stream of completed splits | Stream of completed splits | Morsel scheduler | Every parent-child edge inside a morsel plus root rebatching |

## Work boundaries: morsel versus batch

There are two distinct boundaries to compare:

- the **worker boundary** assigns a disjoint split or morsel, normally around 100,000 rows, to one
  execution activation; and
- the **intra-worker boundary** controls how that activation advances and returns smaller arrays.

"Caller" below means the scan driver immediately above the reader or plan, not the application
using the scan API.

Five questions distinguish the models:

1. Who chooses the fixed worker range?
2. Who chooses the next dense prefix inside that range?
3. Is the inner choice made before execution or while data is being read?
4. Must the subtree satisfy the whole worker range in one result?
5. Does scheduling operate only between worker ranges, or also between parent and child nodes?

The ranges below describe dense row coverage. A mask can make the returned array compact. For
example, satisfying dense rows `[0..1,000)` with 12 demanded rows returns 12 values, but still
advances the execution frontier by 1,000 rows.

### V1: a split imposed top-down

Before execution, the V1 scan driver asks the reader tree for natural split boundaries and may
subdivide large spans. It creates one split task for each resulting range. A call such as:

```text
projection_evaluation(rows = [0..100,000), mask)
```

requires the reader subtree to account for the complete `[0..100,000)` range. A chunked reader can
divide that request among several child readers internally, but its future resolves only when it
can return the compact values for the whole split. A child cannot return `[0..32,768)` and ask its
parent to resume the suffix later.

The split therefore serves three roles at once:

- a unit of scan concurrency;
- an internal pacing boundary; and
- usually one output-stream batch.

Backpressure applies when the root stream waits before starting or yielding another split. It does
not apply independently at every parent-child edge inside the reader tree.

### Plan v2: the same top-down unit over a generic plan

Plan v2 changes how the physical work is represented and how natural boundaries are discovered,
but not this execution contract. Scan-v2 selects a split and calls the root plan with its exact
range and mask:

```text
root.execute(rows = [0..100,000), mask)
```

Structural operators derive exact child requests from that envelope. `Pack` asks all row-equivalent
children for `[0..100,000)`. `Concat` partitions the range at chunk boundaries and gathers every
overlapping child result. `ListPack` reads enough offsets and elements to reconstruct that entire
outer range. The root future still produces exactly the split's selected cardinality.

The plan is generic and rewriteable, but execution remains **root-paced**: a boundary chosen before
the recursive calls controls every row-equivalent subtree below it.

### `layout27`: a fixed morsel with finer scheduling

In the full `layout27` design, split hints are converted into fixed morsels before their value tasks
run. The central scheduler chooses which ready task or continuation to admit next, using lanes,
priorities, read dependencies, and byte budgets. It does not normally renegotiate the morsel's row
end.

For a `[0..100,000)` morsel, the scheduler can interleave work such as:

```text
evidence setup
  -> evidence probe for [0..100,000)
  -> residual predicate read for [0..100,000)
  -> projection read for [0..100,000)
```

A `ReadTask` may return `Continue` and expose a second set of data-dependent reads. That makes the
steps inside one morsel dynamic, but the final array still satisfies the preselected morsel. The
scheduler owns **when and in what phase** work runs; the preplanning layer still owns **which dense
rows constitute the unit**.

At the `ji/layout27` tip described here, ordinary scans use bound V1 readers, so their effective
unit of work remains the V1 split.

### Proposed model: a fixed morsel containing child-chosen prefixes

The outer scheduler first assigns a configurable fixed morsel, such as `[0..100,000)`, to one
execution activation. Inside that morsel, the proposed model divides ownership of each batch
boundary:

- the parent specifies the maximum outstanding range, immutable sealed demand, and soft size
  target;
- resource credits provide a hard upper bound; and
- the child chooses how much of the range's next contiguous prefix it can efficiently produce now.

Suppose a parent has an outstanding request for `[0..100,000)`. A segment child may stop at a page
boundary and return `[0..32,768)`. The successful result commits that prefix. The parent then
continues with `[32,768..100,000)`; it does not retry or recompute the first prefix.

For a row-wise parent, child boundaries need not match:

```text
outstanding parent request: [0..10)

field A returns: [0..4)
field B returns: [0..3)

Pack emits:       [0..3)
Pack retains:     A's [3..4) tail
next request:     begins at row 3
```

The child chooses only the prefix end. It cannot change the start, skip rows, exceed the parent
range, or return a disconnected interval. This restriction gives layouts useful sizing freedom
without requiring a general interval join or unbounded reordering buffers in every parent.

The prefix is committed only by a `Batch` result. Before that, one run-to-quiescence `drive`
call may register reads for one child and CPU work for another through stable tickets. It returns
`Blocked` only after exposing all independent work, and completion events merely wake it to
inspect durable ticket state. Static reads are described once for the whole morsel; data-dependent
operators expand explicit gates when new offsets, codes, or evidence become available.

The resulting inner unit is **negotiated and edge-local** rather than fixed globally:

- a leaf stops at a natural physical boundary;
- a parent may shorten that result to align siblings;
- backpressure limits how far each edge advances;
- a root rebatcher combines or slices internal prefixes for stable consumer batches; and
- fixed morsels open independent parallel graphs, but no longer dictate every internal array
  boundary.

This is the intended meaning of "each layout can return an array of whatever size it likes": it can
choose any safe, non-empty prefix within the request and resource budget, while its parent assumes
responsibility for slicing, buffering, and alignment.

## What each approach optimizes for

### V1: behavior and coverage

V1 has the widest set of mature layout-specific behaviors. It can specialize dictionary, list,
struct, chunked, zoned, and row-index reads while overlapping projection registration with filter
resolution. Its cost is architectural: the stateful reader tree is simultaneously the physical
plan, expression partitioner, executor, child cache, and split provider.

### Plan v2: a clean physical IR

Plan v2 gives optimization a layout-independent operator tree. `Concat`, `Pack`, `Take`,
`ListPack`, `Eval`, and `SegmentScan` describe work rather than mirroring layout types. Execution,
however, is still externally paced: every call names an exact row range and mask, and every child
must return exactly that selected cardinality.

### `layout27`: explicit preparation and scheduling

The broader `layout27` design cleanly separates immutable scan plans from per-scan state and makes
I/O dependencies visible to a scheduler. It introduces useful concepts such as selection versus
demand, prepared read routes, continuations, evidence, read phases, priorities, and byte budgets.
At commit `9734b85de4`, the `ji/layout27` branch uses a hybrid path for ordinary scans: expressions
are pushed into `ScanPlan`s, but bound readers delegate actual pruning, filtering, and projection to
V1 `LayoutReader` methods.

### Self-paced execution: local batching and global control

The proposed model keeps plan v2's operator IR and adopts `layout27`'s explicit runtime and I/O
ideas. Static reads are catalogued once, exact mask refinement stays in a root demand ledger, and
projection planning may use immutable open snapshots to offer candidate I/O while exact or fallible
value execution uses sealed windows. A drive call registers any mix of scheduler-owned I/O and CPU
work and runs until it returns a prefix batch, blocks on tickets, finishes, or yields for fairness.
Parents own cursors that slice and align child batches, while a root rebatcher adapts natural
internal batches to consumer-facing sizes. Several fixed morsels provide outer scan parallelism;
self-paced execution happens independently inside each one.

## Decision matrix

| Requirement | Best source to retain | Reason |
| --- | --- | --- |
| Proven layout semantics | V1 | It is the compatibility baseline for complex layouts and masks. |
| Rewritable physical operators | Plan v2 | Operators are independent of the layout that produced them. |
| Immutable plan and per-scan state separation | `layout27` | Preparation and state initialization are explicit. |
| Scheduler-visible I/O | `layout27` | Required reads, prefetches, phases, priorities, and bytes are first-class. |
| Natural, variable output sizes | Proposed model | Prefix progress lets each subtree select an efficient batch size. |
| Correct row-wise composition | Proposed model | Parent cursors make alignment an explicit invariant. |
| Stable consumer batches | Proposed model | Root rebatching isolates consumers from internal fragmentation. |

## Recommendation

Adopt the proposed model as an evolution of plan v2, not as a fifth layout reader API:

```text
LayoutRef
  -> layout-specific lowering
PlanRef                       immutable and rewriteable
  -> open scan
ScanState                     domains, edge maps, ReadCatalog spine, reusable caches
  -> assign configurable fixed morsels
DemandLedger                  refine exact masks and seal windows
MorselExec                    drive one mutable graph per worker unit
  -> register I/O and CPU tickets
  -> Batch | Blocked | Done | Yield
  -> self-paced ExecBatch values
Root RebatchExec
  -> ArrayStream              consumer-sized output
```

The implementation should retain:

- V1 as the semantic oracle during migration;
- plan v2's generic operator identities and rewrites;
- `layout27`'s selection/demand distinction and explicit read scheduling, refined into an open
  demand ledger plus sealed execution demand; and
- fixed split hints only for creating independent execution graphs, not for forcing every node to
  return a fixed-size array.

The implementation should avoid:

- storing mutable stream cursors or per-scan caches in `PlanRef`;
- rebuilding per-scan facts, such as the read catalog or a dictionary value domain, once per morsel;
- allowing a child to return an arbitrary disconnected interval;
- letting a parent's batch count scale with its child count when those children could have aligned;
- reimplementing coordinate translation per operator, per catalog, and per split walker;
- treating a zero-length value array as lack of progress when its dense row coverage advanced; and
- exposing small layout-native fragments directly to scan consumers.

## Why prefix progress is the key restriction

"Any array size" must not mean "any rows." If a child could return an arbitrary interval, every
parent would need a general interval join, unbounded reordering buffers, and substantially more
complex error handling. The useful freedom is narrower:

> A node may return any non-empty prefix of the outstanding row request that fits its natural
> boundaries and current resource credits.

That rule preserves streaming progress and bounded parent state while still allowing flat
segments, chunks, pages, dictionaries, and list elements to choose efficient units.

## Migration outline

1. Establish a differential semantic and performance baseline, settle whether demand can widen, and
   add root rebatching against the current executor.
2. Declare row domains and edge maps, replacing the coordinate arithmetic five operators already
   hold.
3. Prove prefix, cursor, ticket, capping, and drive invariants in a deterministic simulator.
4. Implement the exact DemandLedger and its coarse scheduler summaries.
5. Prepare a per-scan ReadCatalog with morsel views, lazy scoring, and dynamic gates.
6. Build a minimal ticket scheduler and per-morsel resource-credit model.
7. Port SegmentScan, Concat, Eval, and Pack behind an exact-result compatibility adapter.
8. Derive morsel boundaries from edge maps, retiring the central split switch.
9. Add an unfiltered self-paced morsel root, then integrate pruning and sealed filter demand.
10. Add scheduler-owned CPU concurrency and byte-bounded struct wavefronts.
11. Port the prefix-preserving domain operators — ListPack, Zoned, row-index — then Take's gather
    sub-root.
12. Complete ordering, limits, cancellation, and stream integration.
13. Switch the default only after differential, memory, and performance qualification.

The detailed contracts are in [self-paced scan execution](self-paced.md). Phase dependencies, exit
criteria, tests, and rollout gates are in the
[self-paced implementation plan](self-paced-implementation-plan.md). The evidence and reasoning
behind the contract's choices are in the [design review](self-paced-review.md).
