# Scan Scheduler Design

## Table of Contents

1. [What We Are Trying to Solve](#1-what-we-are-trying-to-solve)
2. [What Makes It Hard](#2-what-makes-it-hard)
3. [How We Differ From Other Engines](#3-how-we-differ-from-other-engines)
4. [Core Design](#4-core-design)
    - [The Layout Tree](#41-the-layout-tree)
    - [The Layout Writer](#42-the-layout-writer)
    - [Multi-Layout Assembly](#43-multi-layout-assembly)
    - [Lazy Deserialization](#44-lazy-deserialization)
    - [The SplitPlan](#45-the-splitplan)
    - [The Segment Registry](#46-the-segment-registry)
    - [The Worker Loop](#47-the-worker-loop)
    - [Split Formation](#48-split-formation)
    - [Ordered Output](#49-ordered-output)
5. [The Global Scheduler](#5-the-global-scheduler)
    - [Why Global](#51-why-global)
    - [Scan Registration and Lifecycle](#52-scan-registration-and-lifecycle)
    - [Cross-Scan Segment Deduplication](#53-cross-scan-segment-deduplication)
    - [Fairness and Priority](#54-fairness-and-priority)
    - [Backpressure](#55-backpressure)
    - [The Global Worker Loop](#56-the-global-worker-loop)
6. [Optional Extension: Multiple I/O Sources](#6-optional-extension-multiple-io-sources)
7. [Optional Extension: I/O Coalescing](#7-optional-extension-io-coalescing)
8. [Optional Extension: io_uring and Kernel-Bypass Polling](#8-optional-extension-io_uring-and-kernel-bypass-polling)
9. [Optional Extension: Zone Map Lookahead](#9-optional-extension-zone-map-lookahead)
10. [Optional Extension: Filter-Projection Pipeline Gap](#10-optional-extension-filter-projection-pipeline-gap)
11. [Optional Extension: Adaptive Conjunct Reordering](#11-optional-extension-adaptive-conjunct-reordering)
12. [What Each Component Does and Does Not Know](#12-what-each-component-does-and-does-not-know)

---

## 1. What We Are Trying to Solve

We have a columnar file format built as a **tree of layout nodes**. Each node performs a structural role:

- **ChunkedLayout** — concatenates rows from multiple children sequentially.
- **StructLayout** — zips independent columns together; children share row-space.
- **ZoneMapLayout** — wraps a data child with a small metadata segment (min/max per region) that can skip the data child
  entirely when a predicate is provably false.
- **Leaf segments** — physical segments of compressed bytes, readable from some I/O source.

The layout tree is a description of structure. It is shared between reading and writing: the writer produces a layout
tree populated with segment references, and the reader navigates the same tree to evaluate queries. At read time,
multiple independently-written layouts may be assembled into a single tree spanning many files, partitions, or sources.

A query is expressed as an **expression tree** that is pushed into the layout tree. Each node claims the sub-expressions
it can evaluate. The results — typed vectors, not always boolean — flow back up through the tree as `Compute` nodes in a
per-split execution plan. The scheduler executes this plan without understanding expression semantics.

The system must:

1. Read only the segments actually required to answer the query.
2. Parallelize the scan across many CPU cores.
3. Preserve sort order in output when data is pre-sorted.
4. Adapt to the structure of the layout tree without any hardcoded knowledge of zone maps, filters, projections, or
   column types.
5. Remain correct and efficient when columns have segments of different and misaligned sizes.
6. Share I/O resources efficiently across multiple concurrent scans over many files.
7. Support lazy deserialization of layout trees so that wide tables do not pay deserialization cost for subtrees not
   touched by the query.

---

## 2. What Makes It Hard

### 2.1 Variable and Misaligned Segment Boundaries

Different columns have segments of different sizes, and boundaries do not align across columns:

```
Column A: |----10K----|------20K------|----10K----|
Column B: |---8K---|----12K----|---8K---|---12K---|
Column C: |----------25K----------|------15K------|
```

A fixed-size parallel work unit (a "split") that crosses a segment boundary forces the column reader to decompress two
separate segments mid-split, disrupting decompression state and segment pin lifetimes. Simply using fixed-size splits
produces incorrect decompression and uneven work distribution.

### 2.2 I/O Is Conditionally Determined at Runtime

A ZoneMapLayout node creates a runtime branch: read the zone map segment, evaluate the pruning predicate, and then
either skip or proceed to the data child. The set of required data segments is not fully known before zone map segments
have been read and evaluated. Static prefetch planning is incomplete.

### 2.3 The Expression Residual Lives in the Plan, Not the Scheduler

When a filter like `(a.x + b.y) > 10` is partitioned over a StructLayout, child `a` evaluates `a.x` and returns a
`Vector<int>`, child `b` evaluates `b.y` and returns a `Vector<float>`. Neither is boolean. The assembly of
`($va + $vb) > 10` is not done by the scheduler — it is a `Compute` node baked into the per-split plan by the layout
tree during planning. The scheduler fires this node when both inputs arrive. It does not know what the node evaluates.

### 2.4 Sorted Output Requires Ordered Emission

Parallel scans naturally produce results out of order. If the data is pre-sorted and the query requires sorted output,
the system must emit splits in their original row order — not completion order. A split that completes early must wait
for all preceding splits before its output can be released downstream.

### 2.5 Layout Nodes Cannot Build the Execution Plan Themselves

Two different paths through the layout tree may terminate in the same physical segment — for example, two columns whose
segments happen to share storage, or a column referenced by both a filter and a projection expression. A layout node
cannot know about other nodes' segment requirements. Only a central registry that sees all paths can detect the overlap
and issue a single read.

### 2.6 The Scheduler Must Also Drive I/O Polling

In a thread-per-core model, a separate thread for ring polling wastes a core. The scheduler loop and the I/O polling
loop must be the same thread. This means the scheduler cannot block waiting for I/O — it must make progress on whatever
work is available (other splits, other completions) while reads are in flight.

### 2.7 Multiple Concurrent Scans Share a Device

A server running many concurrent scans over many files shares a single I/O device (or a bounded pool of I/O connections
to object storage). Without a global scheduler each scan operates independently, leading to device oversubscription,
missed coalescing opportunities, and unfair bandwidth allocation.

### 2.8 Wide Tables Must Not Pay Full Deserialization Cost

A layout tree for a table with thousands of columns, serialized as a flatbuffer, may be megabytes of metadata. A query
that touches two columns should not deserialize the subtrees for the other 998. The layout tree must support lazy
deserialization driven by the query.

---

## 3. How We Differ From Other Engines

### ClickHouse

Stores each column in a separate file per part, with a mark file providing granule-level (8192-row) indexing. Sorted
output is achieved by assigning one thread per part and merging streams with a k-way heap. Parallelism is bounded by
part count. Zone maps do not exist at the granule level. Expression push-down stops at the storage layer boundary. No
dynamic conjunct reordering. No cross-query segment deduplication.

### DuckDB

Stores tables as row groups (~122K rows) in a single file with a userspace LRU buffer manager. Sorted output uses a
reorder buffer over row groups. Zone maps (min/max per row group) are evaluated before I/O. Conjunct ordering is static.
The Parquet reader bypasses the buffer manager entirely and has no shared segment deduplication across columns.
Concurrent scans share the buffer manager but not the I/O scheduler.

### Umbra

Uses pointer swizzling to make the buffer manager zero-overhead for hot data. Split-driven parallelism with NUMA
awareness. LLVM-compiled query code operates on raw pointers. No layout tree concept — storage is managed directly. No
dynamic conjunct reordering. No global I/O scheduler across queries.

### This Design

Six things none of the above provide:

1. **Layout-tree-aware split formation**: split boundaries are derived from the union of all segment boundaries across
   the layout tree, with coordinate translation at ChunkedLayout nodes.
2. **Per-split execution plans (SplitPlan) with a shared segment registry**: each split gets its own plan; deduplication
   and coalescing happen in a registry that sees all active splits across all concurrent scans.
3. **Context-free layout nodes**: nodes produce `SplitPlan` nodes (SegmentReads and Compute closures). They do not know
   if their result is used as a filter or projection, and the scheduler does not know what any Compute node evaluates.
4. **Single worker loop combining scheduling and I/O polling**: no separate polling thread; the worker loop services all
   active scans, flushes reads, polls for completions, and advances plans in a single tight loop.
5. **Global scheduler with cross-scan deduplication and device-level coordination**: a single scheduler manages all
   concurrent scans, enforcing fairness and coalescing I/O across scan boundaries.
6. **Lazy layout deserialization**: layout trees are navigated top-down driven by the query; subtrees for unreferenced
   columns or non-overlapping row ranges are never deserialized.

---

## 4. Core Design

### 4.1 The Layout Tree

The layout tree describes the structure of stored data. It is the same structure at read and write time — what differs
is whether leaf segments are being read from or written to.

Every layout node implements a two-phase read interface:

```rust
trait LayoutNode {
    // Planning phase (once per query):
    // Pre-compute how the expression maps onto this subtree.
    // Returns an opaque QueryPlan stored on the Scan object.
    // The scheduler never inspects it — it is passed back to
    // build_split_plan for each split.
    // May return a no-op plan if partitioning is cheap enough
    // to redo per split.
    fn plan(&self, expr: &Expr) -> Box<dyn QueryPlan>;

    // Execution plan phase (once per split):
    // Given the query plan and a row range, produce a SplitPlan —
    // a graph of SegmentRead and Compute nodes the scheduler can execute.
    // All expression residuals are baked in as Compute closures.
    // The scheduler executes the plan without understanding its contents.
    fn build_split_plan(
        &self,
        plan: &dyn QueryPlan,
        row_range: Range,
    ) -> SplitPlan;

    // Boundary phase (once per query, for split formation):
    // Report natural split points in this node's row-space.
    fn boundaries(&self, row_range: Range) -> BTreeSet<RowIdx>;
}
```

Node responsibilities:

- **ChunkedLayout**: translates between global and local row-space. The only node that changes coordinate systems.
  Deserializes only children whose row range overlaps the split's row range (see section 4.4). Stores `row_count` per
  child to support range intersection without deserialization.
- **StructLayout**: fans out to children, each evaluating its assigned sub-expression. Bakes residual cross-column
  expressions (e.g. `a.x + b.y > 10` after children return `a.x` and `b.y`) as `Compute` nodes in the plan. Deserializes
  only children referenced by the expression.
- **ZoneMapLayout**: emits a `SegmentRead` for the zone map segment and a `Compute` node that evaluates the pruning
  predicate. Attaches a conditional edge: SKIP cancels the data child's subgraph; PROCEED enables it. The data child
  receives the original expression unchanged. The zone map is a guard, not an expression transformer.
- **Leaf**: emits a `SegmentRead` for its physical segment and a `Compute` closure that decompresses the relevant rows
  and evaluates any pushed-down expression (e.g. binary search bounds on a sorted segment).

**What layout nodes do not know**: whether their result is used as a filter or projection, whether other nodes share
their segment, which split or scan they belong to, what I/O source will serve the request, and what the scheduler will
do with their output.

### 4.2 The Layout Writer

The write path uses the same layout tree structure. Each node gains a write method:

```rust
trait LayoutNode {
    // ... read methods above ...

    // Write path:
    // Accept a batch of rows, write segments via the sink,
    // and return a new Layout with SegmentRefs populated.
    fn write(
        &self,
        rows: &RecordBatch,
        sink: &dyn SegmentSink,
    ) -> Arc<dyn LayoutNode>;
}

trait SegmentSink: Send + Sync {
    // Accept compressed bytes, return a reference to where they were written.
    fn write(&self, bytes: Bytes) -> SegmentRef;
}
```

`write` returns a new layout node of the same type, with the same structure, but with `SegmentRef`s now populated on
each leaf. This returned layout is exactly what you hand to the read path later. The structure (ChunkedLayout,
StructLayout, ZoneMapLayout nesting) is unchanged; only the leaf references are filled in.

`ZoneMapLayout` computes zone map statistics as rows flow through it and writes the zone map segment before the data
segment, so both refs are available in the returned layout.

The `SegmentSink` is the write-path counterpart to `SegmentSource` on `SegmentRead`. A sink backed by a local file, an
S3 multipart upload, or an in-memory buffer all implement the same trait.

### 4.3 Multi-Layout Assembly

At read time, multiple independently-written layouts — each from a different file, partition, or source — are assembled
into a single tree. The assembly point is always `ChunkedLayout`, which concatenates children sequentially:

```
ChunkedLayout                              ← assembled at query time, in-memory
  ├── StructLayout  (partition_2024_01)    ← root of one written layout
  │     ├── ZoneMapLayout
  │     │     ├── LeafSegment [source: S3("bucket/p1/zm_price")]
  │     │     └── LeafSegment [source: S3("bucket/p1/price")]
  │     └── LeafSegment     [source: S3("bucket/p1/user_id")]
  └── StructLayout  (partition_2024_02)    ← root of another written layout
        ├── ZoneMapLayout
        │     ├── LeafSegment [source: local("/data/p2/zm_price")]
        │     └── LeafSegment [source: local("/data/p2/price")]
        └── LeafSegment     [source: local("/data/p2/user_id")]
```

The assembled `ChunkedLayout` node is `Materialized` (built in-memory). Its children are the roots of per-file layouts,
which may themselves be `Deferred` (lazily deserialized from flatbuffers — see section 4.4). The scanner sees a single
tree; the distinction between assembled and deserialized nodes is invisible to it.

Each `LeafSegment` carries its own `SegmentSource`, so different files reading from different I/O sources (S3, local
disk, memory) coexist naturally within the same tree. The scheduler routes each `SegmentRead` to the correct source
without any special handling.

### 4.4 Lazy Deserialization

Layout trees are serialized using flatbuffers. For wide tables, deserializing the full tree upfront is wasteful — a
query touching two of a thousand columns should not pay to deserialize 998 column subtrees.

Each structural node stores its children as `LazyChild` rather than direct pointers:

```rust
enum LazyChild {
    // Built in-memory (writer path, or assembled at query time)
    Materialized(Arc<dyn LayoutNode>),

    // Backed by a flatbuffer sub-slice — not yet deserialized.
    // OnceLock ensures at-most-one deserialization under concurrent access.
    Deferred {
        buf: Bytes,                      // zero-copy sub-slice of parent buffer
        cache: OnceLock<Arc<dyn LayoutNode>>,
    },
}

impl LazyChild {
    fn get(&self) -> &Arc<dyn LayoutNode> {
        match self {
            LazyChild::Materialized(node) => node,
            LazyChild::Deferred { buf, cache } => {
                cache.get_or_init(|| deserialize_layout_node(buf))
            }
        }
    }
}
```

Structural nodes call `child.get()` only when the query requires that child:

```rust
impl LayoutNode for ChunkedLayout {
    fn build_split_plan(&self, plan: &dyn QueryPlan, row_range: Range) -> SplitPlan {
        let mut result = SplitPlan::new();
        let mut offset = 0;
        for (row_count, child) in &self.chunks {
            let child_range = offset..offset + row_count;
            if child_range.overlaps(&row_range) {
                // child.get() called only for overlapping chunks
                let local = row_range.intersect(&child_range).shift(-offset);
                result.merge(child.get().build_split_plan(plan, local));
            }
            // Non-overlapping chunks: child.get() never called
            offset += row_count;
        }
        result
    }
}
```

The `row_count` stored alongside each `ChunkedLayout` child is available without deserialization — it is a top-level
field in the flatbuffer `ChunkEntry` table. This is the only metadata required to drive row-range intersection.
Similarly, `StructLayout` stores field names as top-level fields, allowing it to identify which children the expression
references before deserializing any of them.

**Mixed trees**: an assembled `ChunkedLayout` may have `Materialized` children (the per-file roots) that are themselves
roots of trees with `Deferred` children. The two levels of laziness compose transparently — `LazyChild::get()` behaves
identically in both cases.

**Concurrency**: `OnceLock` ensures that if two splits race to deserialize the same child, exactly one does the work and
the other waits. After initialization, all accesses are lock-free reads.

### 4.5 The SplitPlan

`build_split_plan` returns a `SplitPlan` — a graph of nodes the scheduler can walk and execute. The scheduler
pattern-matches on the node enum to decide what to do. The content of `Compute` nodes (the closure) is opaque to the
scheduler; it just calls it when inputs are ready.

```rust
struct SplitPlan {
    nodes: Vec<SplitPlanNode>,
    edges: Vec<(NodeId, NodeId, Edge)>,
}

enum SplitPlanNode {
    SegmentRead {
        source: Arc<dyn IOSource>,
        cache_key: CacheKey,
        coalesce_hint: Option<ByteRange>,
    },
    Compute {
        inputs: Vec<NodeId>,
        function: Box<dyn Fn(&[EvalResult]) -> EvalResult + Send>,
    },
}

enum Edge {
    Unconditional,
    Conditional(Box<dyn Fn(&EvalResult) -> bool + Send>),
}
```

`NodeId` is an index into `nodes`. The scheduler walks the graph by node state:

```
Waiting   — not all inputs complete yet
Ready     — all inputs complete, not yet executed
Submitted — SegmentRead in flight with the I/O source
Complete  — result available
Cancelled — upstream Conditional edge evaluated to false
```

`Compute` nodes with no `SegmentRead` ancestors (pure expression residuals assembled from other `Compute` outputs)
become `Ready` without any I/O firing — they transition as soon as their input `Compute` nodes complete.

The scheduler does not distinguish zone map evaluations, filter evaluations, decompression, or projection. They are all
`Compute` nodes. The graph topology — which edges are conditional, which nodes are high in the graph — is what gives
zone map evaluation its early-cancellation power, not any special-casing in the scheduler.

### 4.6 The Segment Registry

The segment registry is a **global** shared structure — not scoped to a single scan. It is the deduplication and routing
layer between all active split plans and the I/O layer.

```rust
struct SegmentRegistry {
    // (SourceId, CacheKey) → entry, across all scans
    segments: ConcurrentHashMap<(SourceId, CacheKey), SegmentEntry>,
}

struct SegmentEntry {
    state: SegmentState,   // Pending | Submitted | Complete(data)
    waiters: Vec<(ScanId, SplitId, NodeId)>,
}
```

When a `SegmentRead` node is registered:

- If already complete (cached): the downstream `Compute` node is immediately ready — no I/O needed.
- If already submitted (another split in any scan requested the same segment): add to the waiter list. No new I/O is
  issued.
- If new: add to the waiter list and add to the pending pool for the owning source.

When a segment read completes, the registry fires all waiting compute nodes across all splits in all scans that were
waiting on that segment.

When a split is cancelled, it withdraws from all segments it was waiting on. If a segment has no remaining waiters
across any scan, its pending I/O is dropped.

### 4.7 The Worker Loop

The scheduler loop and the I/O polling loop are the same thread. Each worker thread services splits from all active
scans, not a dedicated subset.

```
loop {
    // 1. Fill windows across all active scans
    for scan in global_scheduler.active_scans() {
        while scan.active_splits.len() < scan.window_size {
            let Some(split) = scan.dispatcher.next() else { break };
            let split_plan = layout_root.build_split_plan(&scan.query_plan, split.row_range);
            for node in split_plan.unconditional_segment_reads() {
                segment_registry.register(node, scan.id, split.id);
            }
            scan.active_splits.push(split.id, split_plan);
        }
    }

    // 2. Flush pending reads — write to SQ ring or enqueue to source
    //    Pending pool is global: reads from all scans are coalesced together
    for (source_id, reads) in segment_registry.drain_pending_by_source() {
        sources[source_id].enqueue(reads, &completion_sink);
    }

    // 3. Poll for completions — from all sources, for all scans
    for completed in completion_sink.drain() {
        let waiters = segment_registry.complete(completed.cache_key);
        for (scan_id, split_id, node_id) in waiters {
            let plan = global_scheduler.scan(scan_id).split_plan(split_id);
            let result = plan.evaluate_compute(node_id, &completed.data);
            advance_plan(scan_id, split_id, node_id, result);
        }
    }

    // 4. Emit complete splits to each scan's output queue
    for scan in global_scheduler.active_scans() {
        scan.active_splits.retain(|plan| {
            if plan.is_complete() {
                scan.output_queue.push(plan.split_id, plan.take_output());
                false
            } else { true }
        });
    }

    if global_scheduler.all_scans_done() { break; }
    if global_scheduler.nothing_in_flight() { spin_loop_hint(); }
}
```

`advance_plan` propagates a compute result through the split's plan: marking downstream nodes as Ready, adding
newly-enabled `SegmentRead` nodes to the pending pool, cancelling subgraphs when a conditional edge evaluates to false,
and storing typed intermediate results for subsequent `Compute` nodes.

### 4.8 Split Formation

Splits are formed once per scan before the worker loop starts.

**Step 1 — Collect boundaries**: call `boundaries()` recursively on the layout tree. `ChunkedLayout` translates
child-local boundaries to global coordinates. `StructLayout` unions boundaries from all required children.
`ZoneMapLayout` reports its region boundaries. Leaves report their own start and end. The result is a `BTreeSet<RowIdx>`
of all natural split points.

**Step 2 — Coalesce**: greedily merge adjacent small splits up to a target row budget (e.g. 10,000 rows), provided the
merge does not cause any column reader to cross more than one additional segment boundary.

**Step 3 — Split**: split units that exceed the target budget at column-segment-aligned midpoints.

The result is a list of row ranges where every split lies within exactly one segment per required column, and splits are
roughly equal in row count.

### 4.9 Ordered Output

Each scan has its own output queue — a priority queue keyed by split ID. Splits are assigned IDs in row order at
formation time. The queue emits splits strictly in ID order: a completed split is held until all lower-ID splits have
been emitted.

Splits whose entire plan was cancelled (e.g. zone map SKIP with empty selection) emit an empty output record. They still
occupy a slot in the output queue — split IDs have no gaps.

---

## 5. The Global Scheduler

### 5.1 Why Global

The motivating case is a server running many concurrent scans over many different files — each scan is an independent
query, reading independent columns from independent files. The scans do not share data. However, they share a single I/O
device (or a bounded pool of connections to object storage), CPU, and memory. Without a global scheduler, each scan runs
its own independent worker loop with no awareness of the others.

**Oversubscription**: N scans each maintaining a window of W splits may collectively have N×W segment reads in-flight.
An NVMe device has a finite command queue depth (typically 1024). An S3 endpoint has rate and connection limits.
Exceeding these causes queuing inside the device or HTTP stack rather than in the scheduler, where it can be managed. A
global scheduler enforces a ceiling on total in-flight reads.

**No global coalescing**: two scans reading adjacent byte ranges on the same device — even in different files — cannot
be coalesced unless something sees both reads simultaneously. A global pending pool sees all reads from all scans before
flushing, enabling cross-scan coalescing. This is particularly valuable on object storage where a merged range request
avoids an entire round trip.

**Unfairness**: a large analytical scan with a wide window fills the device queue with its reads, starving smaller
interactive scans. Global window budget allocation ensures every scan gets a fair share of device bandwidth proportional
to its priority.

**Cross-scan segment deduplication** is also possible when two scans happen to read the same file — two queries against
the same table running concurrently, shared metadata, or zone map segments. This is an incidental benefit rather than
the primary motivation, but it falls out of the global registry for free.

### 5.2 Scan Registration and Lifecycle

```rust
struct Scan {
    id: ScanId,
    layout_root: Arc<dyn LayoutNode>,
    query_plan: Box<dyn QueryPlan>,    // opaque, produced by layout_root.plan()
    dispatcher: SplitDispatcher,
    active_splits: HashMap<SplitId, SplitPlan>,
    window_size: usize,
    output_queue: OutputQueue,
    priority: ScanPriority,
    state: ScanState,            // Running | Draining | Done
}
```

A scan transitions to `Draining` when its dispatcher is exhausted but splits are still in-flight. It transitions to
`Done` when its output queue has emitted all splits.

### 5.3 Cross-Scan Segment Deduplication

The segment registry is keyed by `(SourceId, CacheKey)` globally. In the common case of scans over independent files,
deduplication will not fire. When it does — two queries against the same table, shared metadata, zone map segments read
by multiple scans of the same file — a single read satisfies all waiters:

```
Scan A registers (source_0, segment_42)
  → state: Pending, waiters: [(A, split_3, node_7)]

Scan B registers (source_0, segment_42)   // same file, same segment
  → state: already Pending
  → waiters: [(A, split_3, node_7), (B, split_1, node_4)]

One read submitted. On completion:
  → fires (A, split_3, node_7)
  → fires (B, split_1, node_4)
```

The registry does not know whether A and B scan the same file — they are just tuples in the waiter list. Deduplication
applies whenever possible without special-casing.

### 5.4 Fairness and Priority

**Per-scan window size** is the primary fairness mechanism:

```
total_window_budget = MAX_GLOBAL_INFLIGHT_READS  // e.g. 256

for each active scan:
    scan.window_size = total_window_budget
                     × scan.priority_weight / sum(all priority_weights)
    scan.window_size = clamp(scan.window_size, MIN_WINDOW, MAX_WINDOW)
```

Window sizes are recomputed on each scan registration and deregistration. A scan not filling its window cedes budget to
others.

**Priority tiers**:

```
Interactive   weight: 4  — user-facing queries, latency goal
Bulk          weight: 2  — batch analytics, throughput goal
Background    weight: 1  — maintenance, compaction
```

**I/O ordering within the pending pool**:

```
score = scan_priority_weight
      + head_split_bonus      // blocking this scan's output queue?
      + zone_map_bonus        // small segment, high cancellation leverage?
      - gap_penalty           // far beyond this scan's output frontier?
```

### 5.5 Backpressure

**I/O oversubscription**: total in-flight reads capped at `MAX_GLOBAL_INFLIGHT_READS`. When reached, no scan's window
advances until completions arrive.

**Buffer pool pressure**: when pinned and cached segments exceed a memory threshold, all scan window sizes are reduced
proportionally. Scans with many slow in-flight splits are reduced first.

**Output queue depth**: if a scan's output queue grows large (consumer is slow), that scan's window advancement is
paused.

### 5.6 The Global Worker Loop

Worker threads are not assigned to specific scans. Any worker thread may advance any scan's splits in any iteration. All
worker threads share the global segment registry and completion sink — a concurrent hash map and a lock-free queue
respectively. No scan-level locking is needed in the hot path.

For N worker threads and M > N active scans, scans are round-robined across threads per loop iteration. Scans stalled on
I/O (no ready compute nodes, nothing to flush) are skipped cheaply.

---

## 6. Optional Extension: Multiple I/O Sources

**What it adds**: different layout nodes can read from different I/O sources — local files, object storage, memory —
without the scheduler knowing anything about source types.

**The interface**:

```rust
trait IOSource: Send + Sync {
    fn id(&self) -> SourceId;
    fn coalesce_policy(&self) -> CoalescePolicy;

    // Enqueue reads. Completion delivered via CompletionSink.
    // For async-native sources (S3), spawns a task internally.
    // For io_uring, the worker loop drives submission (see section 8).
    fn enqueue(&self, reads: Vec<ReadRequest>, sink: &CompletionSink);
}
```

Each `SegmentRead` node in a `SplitPlan` carries an `Arc<dyn IOSource>`. The scheduler groups pending reads by
`SourceId` before calling `enqueue`. Sources on different files, even within the same split, are dispatched
independently.

**Implementations**:

- `SyncFileSource`: calls `pread()` directly. Simple baseline.
- `MemorySource`: slices an `Arc<[u8]>` synchronously, pushes to sink inline.
- `ObjectStoreSource`: spawns a Tokio task per coalesced range, pushes to sink on HTTP response.
- `IoUringSource`: writes to SQ ring; CQ polling happens in the worker loop (section 8).

**Without this extension**: assume all reads come from a single local file via `pread()`. The scheduler calls `pread()`
directly in step 2 of the worker loop.

---

## 7. Optional Extension: I/O Coalescing

**What it adds**: adjacent segment reads from the same source are merged into a single larger read. With a global
scheduler, coalescing happens across splits from different scans — adjacent byte ranges in the same file needed by
different queries become one read.

**How it works**: the pending pool is sorted by byte offset per source. Adjacent reads within `max_gap_bytes` are
merged. The source splits the result back into per-`CacheKey` `Bytes` slices and the registry delivers each to its
waiters.

```
Pending (from two scans, same file):
  Scan A, split 3: [64KB @ offset 0]
  Scan B, split 1: [64KB @ offset 64KB]
  Scan A, split 4: [64KB @ offset 192KB]

Gap between entries 2 and 3: 64KB — exceeds max_gap for NVMe (32KB)

Result: one 128KB read @ 0     → sliced: A/split_3 and B/split_1
        one  64KB read @ 192KB → A/split_4
```

**Coalesce policy per source**:

```
Local NVMe:       max_gap=32KB,  max_size=4MB
Object storage:   max_gap=2MB,   max_size=64MB
Memory:           disabled
```

**Without this extension**: each `SegmentRead` becomes one `pread()`. Correct, but cross-scan coalescing opportunities
are lost.

---

## 8. Optional Extension: io_uring and Kernel-Bypass Polling

**What it adds**: eliminates submission syscalls and completion interrupts for local file I/O. The worker thread polls
the CQ ring directly — no context switches, no kernel signals.

**The three overheads eliminated**:

```
Submission syscall   → IORING_SETUP_SQPOLL: kernel thread polls SQ ring;
                       submissions are pure writes to shared memory
Completion interrupt → worker thread polls CQ ring each loop iteration
Device interrupt     → IORING_SETUP_IOPOLL: kernel polls NVMe completion queue
                       (requires O_DIRECT and NVMe polling queue)
```

**Integration**: the `IoUringSource` does not own the ring. The ring is owned by the worker thread. The source writes
requests to a lock-free queue. The worker loop flushes them into the SQ ring in step 2 and drains the CQ ring in step 3:

```
// Step 2
for request in uring_source.drain_pending() {
    ring.push_sqe(build_sqe(request));  // pure memory write with SQPOLL
}

// Step 3
for cqe in ring.completion() {
    completion_sink.push(CompletedRead::from_cqe(cqe));
}
```

**One ring per thread**: with multiple worker threads, each thread owns its own ring. No cross-thread ring contention.
Reads coalesce within each thread's pending pool.

**For Tokio-based engines**: use `tokio-uring` or a bridge — dedicated worker threads run the polling loop and push
completed splits to a channel consumed by the async executor.

**Without this extension**: use `pread()`. Correct but adds syscall overhead per read.

---

## 9. Optional Extension: Zone Map Lookahead

**What it adds**: zone map segments for all splits in the active window are registered immediately when the window is
filled, before any data segment reads. SKIP results cancel downstream data reads before they are ever submitted.

**Why this matters**: zone map segments are small and are often already cached from a prior scan of the same file,
making their evaluation synchronous and free. Even when cold, they are small enough that many can be read in a single
coalesced I/O. A SKIP result may cancel megabytes of data reads. Running zone maps ahead maximises the opportunity to
shrink the data read set before committing to it.

**The scheduling policy**: zone map `SegmentRead` nodes are unconditional — they appear at the top of the `SplitPlan`
graph and are registered in step 1 of the worker loop. Data segment reads are conditional on the downstream `Compute`
node's PROCEED edge and only enter the pending pool after it fires.

**Without this extension**: zone maps are still evaluated (they are `Compute` nodes in the plan) but no special
lookahead is given. Data reads may be submitted before earlier zone maps complete.

---

## 10. Optional Extension: Filter-Projection Pipeline Gap

**What it adds**: when filter and projection expressions touch different columns, the filter phase runs ahead of
projection. The filter produces a `SelectionMask` (small) for each split; projection reads and decompresses only
surviving rows.

**The gap**:

```
target_gap = filter_throughput × projection_io_latency
```

Also bounded by shared segment memory pressure — segments used by both filter and projection columns must remain pinned
until projection completes the split.

**Interaction with the global scheduler**: the gap controller is per-scan. When a scan's shared segment pins accumulate,
the global scheduler reduces that scan's window size, limiting shared pin growth without the scan needing global
visibility.

**Shared segment pinning**: segments are classified at plan time as FilterOnly, ProjectionOnly, or Shared. Shared
segments are pinned by the filter phase and released when projection drops the split result (RAII). The gap controller
limits how many splits hold shared pins simultaneously.

**SameColumn case**: when a column appears in both filter predicate and projection output, the decompressed values may
be cached. At high selectivity (>50% rows survive), cache the full decompressed vector and apply a gather at projection
time. At low selectivity (<5%), decompress only surviving rows sparsely at projection time from the still-pinned
segment.

**Without this extension**: filter and projection run in lockstep within each split. Correct but leaves projection I/O
latency on the critical path.

---

## 11. Optional Extension: Adaptive Conjunct Reordering

**What it adds**: filter expressions with multiple conjuncts are reordered at runtime. The conjunct that eliminates the
most rows per byte read runs first, cancelling the most downstream I/O.

**The oracle**: a shared lock-free structure, one entry per conjunct (keyed by ExprId), shared across all scans:

```
ema_selectivity:  exponential moving average of rows_out / rows_in
ema_io_cost:      exponential moving average of bytes_read / rows_in
ema_variance:     variance of selectivity
partition_stats:  per-partition buckets for skew handling
```

**Scoring**:

```
score = (1 - ucb_selectivity) / io_cost
where ucb_selectivity = ema_selectivity - EXPLORE_FACTOR × std_dev
```

The UCB term rewards conjuncts with few observations. A global oracle converges faster because all concurrent scans
contribute observations.

**Three reordering loops**:

- **Cross-split** (millisecond scale): re-sort conjuncts by score before building each split's plan. Triggered when the
  oracle's generation counter changes.
- **Intra-split racing** (microsecond scale): independent conjuncts have their `SegmentRead` nodes submitted in
  parallel. When the first completes and the selection mask empties, remaining in-flight reads for this split are
  cancelled.
- **I/O priority** (sub-microsecond scale): the pending pool is a priority queue; reads for more selective conjuncts are
  flushed first.

**Thompson sampling**: each conjunct maintains a Beta distribution over its selectivity. Worker threads sample
independently, providing exploration without an explicit budget.

**Without this extension**: conjuncts are evaluated in static plan order. Correct; this is a performance optimisation
only.

---

## 12. What Each Component Does and Does Not Know

This table is the key invariant of the design. If a component starts needing information it should not have, that is a
sign the abstraction boundary is wrong.

```
Component              Knows                              Does Not Know
──────────────────────────────────────────────────────────────────────────────────
Layout node            Its own structure                  Other nodes' segments
                       Its own I/O source (on leaves)     Whether result is filter/projection
                       How to partition an expression     Other splits or scans
                       How to decompress its segment      Window size or scan priority
                       Which children are needed          What the scheduler does with output
                         (from expression + row range)

LazyChild              Its flatbuffer bytes               The query
                       Whether it is materialized         Other children
                       How to deserialize itself          Split formation

SplitPlan              Graph structure (nodes + edges)    What any Compute closure evaluates
                       Node states                        Scan or split identity
                       Which SegmentReads are             Expression semantics
                         unconditional

Segment registry       (SourceId, CacheKey) → waiters     Expression semantics
                       Segment state                       Column types
                       Which scans/splits are waiting      Scan priority
                                                           SplitPlan structure

Worker loop            All active scans                    Expression semantics
                       Split window per scan               Column types
                       SplitPlan node states               I/O source internals
                       Completion routing                  Whether a node is a zone map
                       Output ordering per scan

Global scheduler       All active scans and priorities     Expression semantics
                       Window budget allocation            Layout tree structure
                       Backpressure signals                Column types
                       Scan lifecycle                      I/O source internals

I/O source             How to read bytes                   SplitPlan structure
                       Coalescing policy                   Split or scan IDs
                       How to push to CompletionSink       Expression semantics

Selectivity oracle     Per-conjunct runtime stats          Layout tree structure
(extension)            EMA + variance + partition buckets  I/O sources
                       Generation counter                  Split or scan IDs

Output queue           Split ID ordering (per scan)        Expression results
                       Buffer of out-of-order splits       I/O sources
                       Next expected split ID              Other scans' queues
```

The worker loop and global scheduler are the only components that cross multiple boundaries. They coordinate by routing
opaque tokens — `SegmentId`, `ExprId`, `SplitId`, `ScanId` — rather than by understanding what any of them mean
semantically. The `SplitPlan` is the contract between the layout tree (which builds it) and the scheduler (which
executes it): structure is visible, semantics are not.