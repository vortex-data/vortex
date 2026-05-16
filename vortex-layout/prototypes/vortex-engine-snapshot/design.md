# Design: A Vortex-Native Query Engine

**Status:** Draft
**Author:** (you)
**Last updated:** 2026-05-11

## 1. Summary

This document describes the design of a single-node query engine written in Rust that uses Vortex arrays as both the on-disk file format and the in-memory representation. The engine targets workloads that are dominated by merge, join, aggregate, and top-K operations over large, sorted, unique-keyed tables. It draws its execution model from Google's F1 Query and Napa, its compute model from Vortex's lazy/encoded array semantics, and its concurrency model from thread-per-core systems with worker-local async runtimes for I/O.

The unifying architectural property: **the layout tree of a Vortex file is the bottom of the query plan**, and physical operators above it are themselves expressible as virtual layouts. Late materialization, demand propagation, and sortedness-aware execution are first-class properties of the system, not optimizations bolted onto a Volcano interpreter.

## 2. Goals and non-goals

### Goals

- Sub-second query latency on multi-billion-row sorted Vortex tables for the target workloads.
- Linear thread scaling on commodity multi-core hardware.
- Minimum-bytes-read against object storage; aggressive I/O coalescing and pruning.
- Late materialization end-to-end; values are only resolved when an operator actually needs them.
- Encoding-aware execution: compute on bit-packed, dictionary, FSST, and other encoded forms without canonicalizing.
- Compatibility with the broader Rust/Arrow ecosystem at the boundary (Substrait in, Arrow out).
- Implementable in phases; each phase produces a usable engine.

### Non-goals

- Distributed execution. The architecture should not preclude it, but the initial implementation is single-node.
- General-purpose SQL coverage. The target workloads are read-mostly analytical; full ANSI SQL semantics can be deferred to the frontend.
- Write-path / ingestion. The engine reads Vortex; writes happen out of band.
- Becoming a DataFusion replacement. We may ride DataFusion's frontend; we replace only the execution layer.

## 3. Background and motivation

### 3.1 Why Vortex

Vortex is a columnar format whose layouts are hierarchical, lazy, and segment-backed. Unlike Parquet, layouts can be composed arbitrarily, and statistics are first-class at every level of the tree. Compute kernels can operate on encoded forms directly, falling back to canonical (Arrow) form only when necessary. The natural execution model for Vortex is one that pushes operators *into* the layout tree, not one that reads everything into Arrow and then operates.

### 3.2 Why the F1/Napa execution model

For sorted, unique-keyed data, sort-merge dominates hash-based execution. Joins, aggregates, top-K, dedup, and set operations are all variants of k-way merge over sorted streams. Sort-merge is also asymptotically the best fit for object-storage-backed workloads, because reads are linear and predictable. F1 Query and Napa have demonstrated this at planet scale for over a decade.

Key techniques from this lineage:

- **Tournament trees (loser trees)** for k-way merge with cached comparison state.
- **Offset-value coding (OVC)** for amortized-O(1) key comparisons across the entire pipeline.
- **In-sort aggregation** that folds duplicate-key collapsing into the merge itself.
- **Progressive partitioning** for per-query work distribution against skewed data.
- **Materialized views as just another sorted forest** indistinguishable from base tables.

### 3.3 Why not DuckDB-style pipelines

DuckDB's pipeline abstraction exists to manage pipeline breakers (hash builds, sorts). In a sort-merge-default world, almost nothing is a pipeline breaker, so pipelines are an abstraction with little to do. We adopt the same pull-based vectorized execution model but with a flat operator tree rather than a pipeline graph.

### 3.4 Why not Tokio everywhere

Multi-threaded Tokio gives concurrent I/O at the cost of cross-thread work-stealing for compute. For analytical workloads where each partition does substantial CPU work, this thrashes caches and breaks linear scaling. We use thread-per-core for compute and confine async to I/O within each worker.

## 4. Architecture overview

```
         ┌──────────────────────────┐
         │   Logical plan (Substrait │
         │   or DataFusion logical)  │
         └────────────┬─────────────┘
                      │ planner
                      ▼
         ┌──────────────────────────┐
         │  Physical operator tree  │
         │  + DemandState channels  │
         └────────────┬─────────────┘
                      │ progressive_partition()
                      ▼
   ┌────────────┬─────────┴───────────┬────────────┐
   │ Partition  │ Partition           │ Partition  │
   │ 0          │ 1                   │ N-1        │
   │ (clone +   │ (clone +            │ (clone +   │
   │ RangeBound)│ RangeBound)         │ RangeBound)│
   └─────┬──────┴───────┬─────────────┴─────┬──────┘
         │              │                   │
   ┌─────▼──────┐ ┌─────▼──────┐     ┌─────▼──────┐
   │ Worker 0   │ │ Worker 1   │ ... │ Worker N-1 │
   │ (core 0)   │ │ (core 1)   │     │ (core N-1) │
   │            │ │            │     │            │
   │ • current- │ │ • current- │     │ • current- │
   │   thread   │ │   thread   │     │   thread   │
   │   runtime  │ │   runtime  │     │   runtime  │
   │ • memory   │ │ • memory   │     │ • memory   │
   │   pool     │ │   pool     │     │   pool     │
   │ • I/O sched│ │ • I/O sched│     │ • I/O sched│
   └─────┬──────┘ └─────┬──────┘     └─────┬──────┘
         │              │                   │
         └──────────────┼───────────────────┘
                        ▼
              ┌──────────────────┐
              │ Gather MergeK    │
              │ (final merge if  │
              │  ordered output) │
              └──────────────────┘

  Shared across all workers (cross-thread state):
    • Adaptive concurrency limiter (per backend)
    • Footer cache
    • Layout metadata cache
```

The compute path within each worker is fully synchronous-feeling from outside. Async exists inside leaf operators (`LayoutWalker`, `Materialize`, `AsyncMap`) and is driven by the worker's local single-thread runtime.

## 5. Core abstractions

### 5.1 SortKey

```rust
enum SortKey {
    /// Sorted on real columns with declared direction and collation.
    Natural { columns: Vec<ColumnRef>, directions: Vec<Direction> },
    /// Implicit positional order in some row domain.
    RowIndex { domain: RowDomainId },
    /// No guaranteed order.
    Unordered,
}
```

`SortKey` is a property carried on every stream and reasoned about by the planner. It is also propagated through operators: a `Filter` preserves the input `SortKey`; a `MergeK` requires `Natural` on a matching prefix and emits the same; a `HashAggregate` (when added) emits `Unordered`.

### 5.2 SortedStream

The core operator trait:

```rust
trait SortedStream: Send {
    async fn next_batch(&mut self) -> Result<Option<Batch>>;
    
    /// Advance past all rows with sort key < `target`. Default: no-op.
    async fn seek(&mut self, target: &Key) -> Result<()> { Ok(()) }
    
    fn sort_key(&self) -> &SortKey;
    fn row_domain(&self) -> Option<RowDomainId>;
    fn required_input_order(&self) -> Vec<OrderRequirement>;
    fn output_order_for(&self, inputs: &[SortKey]) -> SortKey;
}
```

The method is `async fn` so that operators may await on I/O or on lazy array resolution. Most operators do not actually suspend on `next_batch`; the state machines the compiler generates are near-zero overhead in the no-suspend case.

`seek` enables demand propagation in the form of monotone position advancement; see §10.

### 5.3 Batch and AsyncArray

A `Batch` is the unit of inter-operator dataflow. It carries positions, optional OVC annotations, and lazy values.

```rust
struct Batch {
    /// Position range or set within the row domain.
    positions: PositionRange,
    /// The structurally-typed array; columns may be unresolved.
    array: AsyncArray,
    /// Optional offset-value codes for the leading sort key.
    ovcs: Option<vortex::Array>,
    n_rows: usize,
}

struct AsyncArray {
    dtype: DType,
    /// Underlying Vortex layout — values may be on disk or in flight.
    layout: Arc<dyn Layout>,
    /// Per-segment resolution state.
    segments: Arc<[SegmentState]>,
}

enum SegmentState {
    Resolved(Arc<Bytes>),
    Pending(Shared<BoxFuture<'static, Result<Arc<Bytes>>>>),
    Unrequested { range: ByteRange },
}

impl AsyncArray {
    /// Force materialization of one or more columns.
    async fn resolve(&self, columns: &[ColumnRef]) -> Result<vortex::Array>;
    /// Return canonical if all segments are already resolved; else None.
    fn try_canonical(&self) -> Option<vortex::Array>;
}
```

A batch flowing through several operators stays unresolved until an operator's kernel actually reads values. Filters and projections that only rearrange or select structure never trigger resolution. This is the foundation of late materialization.

### 5.4 PositionRange and row domains

A *row domain* is the positional space of some leaf source (a Vortex file, an intermediate result, an in-memory array). Within a domain, positions are well-defined and stable: position 17 always refers to the 18th row of that source.

```rust
struct RowDomainId(u64);  // globally unique identifier per leaf

enum PositionRange {
    Contiguous { domain: RowDomainId, lo: u64, hi: u64 },
    /// Compact sorted run-length-encoded selection.
    Sparse { domain: RowDomainId, indices: vortex::Array /* primitive u64 */ },
}
```

Operators that preserve the row domain (filter, project, materialize) carry positions through unchanged or filter them down. Operators that *create* a new domain (sort, hash join, hash aggregate, group-by output) terminate the old domain and produce a new one. Once a domain is terminated, positions in that domain are no longer addressable from downstream operators.

This means **demand propagation works within a domain but not across domain boundaries**. Boundaries are explicit in the operator tree.

### 5.5 DemandState

Per pipeline segment (contiguous chain of operators sharing a row domain), a shared `DemandState` carries backward-flowing demand:

```rust
struct DemandState {
    domain: RowDomainId,
    /// Upper bound on positions of interest; monotonically decreasing.
    max_position: AtomicU64,
    /// Excluded ranges within [0, max_position).
    excluded: Mutex<IntervalSet>,
}

impl DemandState {
    fn limit_to(&self, max: u64);             // upper-bound advancement
    fn exclude(&self, range: Range<u64>);     // range exclusion
    fn is_needed(&self, pos: u64) -> bool;
    fn next_needed_after(&self, pos: u64) -> Option<u64>;
}
```

Updates are constrained to be **monotone**: the upper bound only ever decreases, and exclusions only ever accumulate. This makes the state cheap to maintain (one atomic decrement, one append to a small interval set) and trivial to consult (binary search). The constraint excludes some pathological demand patterns ("I no longer want every fifth row") but covers all the cases that arise from real operators: TopK saturation, group completion, zone-map pruning, filter discovery.

`DemandState` is created at each pipeline boundary by the planner and shared via `Arc` to all operators within that pipeline. Leaf operators consult it before issuing reads.

## 6. Physical operators

The full operator inventory. All implement `SortedStream`.

### 6.1 LayoutWalker

Walks a Vortex layout tree and emits position batches, with metadata-level pruning applied via zone maps. Does *not* fetch payload data; only fetches the metadata needed to make pruning decisions.

```rust
struct LayoutWalker {
    layout: Arc<dyn Layout>,
    demand: Arc<DemandState>,
    filter_predicate: Option<Expression>,  // for zone-map level pruning only
    cursor: WalkCursor,
}
```

Output: `Batch` whose `array` is unresolved and whose `positions` describe a contiguous or sparse range from the layout. The walker is what makes "the layout tree is part of the plan" concrete; reading a file is just `LayoutWalker::next_batch` to exhaustion.

### 6.2 Materialize

Takes position batches from a child stream and resolves the requested columns, doing the actual I/O.

```rust
struct Materialize {
    child: Box<dyn SortedStream>,
    columns: Vec<ColumnRef>,
    layout: Arc<dyn Layout>,
    io: Arc<IoScheduler>,
    
    // Internal prefetch window (see §8 and §9).
    in_flight: VecDeque<PendingBatch>,
    target_depth: AdaptiveDepth,
}
```

`Materialize` is the only place in a sort-style pipeline where I/O actually happens for value bytes. Multiple `Materialize` operators can exist in a single plan, one per "demand tier" (zone-map columns, filter columns, projection columns).

### 6.3 MergeK

The workhorse. N-way merge over sorted child streams.

```rust
struct MergeK {
    children: Vec<Box<dyn SortedStream>>,
    mode: MergeMode,
    key: SortKeyPrefix,
    tournament: LoserTree,  // OVC-annotated
}

enum MergeMode {
    Union,
    Intersect,
    InnerJoin { join_key_prefix: usize },
    LeftOuterJoin { join_key_prefix: usize },
    Dedup,
    Aggregate { aggs: Vec<AggregateFunction> },
}
```

A single operator subsumes join, group-by, dedup, and set operations. The mode parameter selects the duplicate-handling policy. The tournament tree caches comparison state and uses OVCs for amortized-O(1) successor comparisons.

For `InnerJoin` and `LeftOuterJoin` modes, `MergeK` implements the zig-zag optimization: when one side advances past a key, the other side is `seek`ed forward.

### 6.4 StreamAggregate

A degenerate single-child specialization of `MergeK::Aggregate` that's worth having separately because its inner loop is much tighter. Collapses adjacent equal-keyed rows into one.

```rust
struct StreamAggregate {
    child: Box<dyn SortedStream>,
    group_key_prefix: usize,
    aggs: Vec<AggregateFunction>,
}
```

Memory footprint: O(one row + aggregate accumulator state). No hash table.

### 6.5 Filter and Project

Order-preserving, row-domain-preserving operators.

```rust
struct Filter {
    child: Box<dyn SortedStream>,
    predicate: Expression,
    demand: Arc<DemandState>,  // publishes range exclusion on zone-level wins
}

struct Project {
    child: Box<dyn SortedStream>,
    exprs: Vec<NamedExpression>,
}
```

Filter operates on encoded arrays when possible via Vortex's compute kernel registry. Project rewrites the struct schema and inserts derived columns; if it touches the sort key, the output `SortKey` is downgraded to `Unordered` (or `RowIndex`, if the underlying row domain survives).

### 6.6 TopK

Bounded top-K via a small loser tree.

```rust
struct TopK {
    child: Box<dyn SortedStream>,
    k: usize,
    ordering: Vec<(ColumnRef, Direction)>,
    heap: BoundedLoserTree<Row>,
    saturated: bool,
    demand: Arc<DemandState>,
}
```

When the input is sorted on the ordering columns, TopK saturates after K rows and can publish a `limit_to` to `DemandState`, causing the underlying scan to stop. When the input is unsorted on the ordering columns, TopK degenerates to a full pass with a K-sized heap; still cheap, just no early stop.

### 6.7 PartitionSplit and RangeBound

```rust
struct PartitionSplit {
    child: Box<dyn SortedStream>,
    boundaries: Vec<Key>,  // N-1 boundaries produce N output streams
}

struct RangeBound {
    child: Box<dyn SortedStream>,
    lo: Option<Key>,
    hi: Option<Key>,
}
```

`PartitionSplit` is the dual of `MergeK`: it slices one stream into N by key range. `RangeBound` is what gets injected by the partitioner when a single operator tree is cloned per partition; it clips the leaf scans to that partition's range. On `RangeBound`, `seek(lo)` is called once at construction; `next_batch` returns `None` once past `hi`.

### 6.8 IndexedLookup

For auxiliary row domains in Vortex layouts (DictLayout, run-end encoding, sparse layouts).

```rust
struct IndexedLookup {
    primary: Box<dyn SortedStream>,   // in the primary domain
    values: Arc<dyn Layout>,          // value-domain layout
    strategy: LookupStrategy,
}

enum LookupStrategy {
    Eager,                    // small value domain, gather immediately
    Lazy,                     // defer until upstream actually reads
    Bulk { capacity: usize }, // build a code→value map for the partition
    ScanMerge,                // when value domain is also sorted on relevant key
}
```

Strategy selection is the planner's job, driven by value-domain size, expected upstream selectivity, and whether the value domain has useful order.

### 6.9 AsyncMap

For user-defined per-row async work (e.g. `http.fetch(url)`).

```rust
struct AsyncMap {
    child: Box<dyn SortedStream>,
    expr: AsyncExpression,
    max_concurrency: usize,
    output_order: OutputOrder,
    timeout: Duration,
    max_per_row_bytes: usize,
}
```

Internally maintains a sliding window of outstanding futures and emits an `AsyncArray` whose segments correspond to in-flight responses. Downstream operators await on resolution as needed; if the downstream filter discards a row before its response is resolved, the future is dropped (and ideally cancelled).

### 6.10 Sort, HashJoin, HashAggregate (Phase 2+)

Not part of the initial implementation, but the interface is designed to accommodate them:

- `Sort`: classic external sort using tournament-tree run generation and `MergeK` for the merge phase. Spills runs to Vortex files on local disk. Produces OVC-annotated output.
- `HashJoin`: separate `HashBuild` and `HashProbe` operators; build is a pipeline breaker producing an `Unordered`-keyed hash table; probe consumes it. Inserted by the planner when one or both inputs are `Unordered` and the cost of sorting exceeds the cost of hashing.
- `HashAggregate`: similar pattern, for grouping over `Unordered` input.

The planner picks sort-merge or hash-based variants based on input order properties and cardinality estimates.

## 7. Execution model

### 7.1 Thread-per-core, partition-per-thread

The runtime spawns one OS thread per core, each pinned. There is no cross-thread work stealing for compute. Each thread runs one partition's operator tree to completion, then takes another partition from a work queue if any remain (see §7.4).

```rust
fn execute(plan: PhysicalPlan, config: ExecConfig) -> Result<OutputStream> {
    let n_workers = config.workers.unwrap_or_else(num_cpus::get);
    let partitions = progressive_partition(&plan, n_workers * config.partition_multiplier)?;
    let work_queue = WorkQueue::from(partitions);
    
    std::thread::scope(|scope| {
        let handles: Vec<_> = (0..n_workers)
            .map(|core_id| scope.spawn(move || worker_loop(core_id, &work_queue)))
            .collect();
        gather_outputs(handles)
    })
}
```

### 7.2 Progressive partitioning

Implemented per the Napa paper. Algorithm sketch:

1. Walk the Vortex layout tree at each leaf scan. Each layout level carries size statistics in its zone maps.
2. Start at the root of each leaf's layout. Compute initial size estimates and propose `n_workers * k` split points.
3. For each candidate partition, check its estimated cost against an error bound. If tight enough, freeze.
4. For partitions whose bounds are too loose, descend selectively into the layout tree to refine — only the layouts contributing meaningfully to that partition.
5. Stop when all partitions meet the error bound or budget is exhausted.

This is structurally identical to Napa's algorithm, except that the B-tree is a Vortex layout tree. The size statistics are read from zone maps (which Vortex already carries).

For multi-scan queries, partition the *most expensive* scan finely and propagate range bounds to other scans. The planner derives derived partition boundaries by translating through join predicates.

### 7.3 Worker-local async runtime

Each worker spawns a `tokio::runtime::Builder::new_current_thread()` (or `glommio` / `monoio`) runtime. The runtime is single-threaded by construction; no work-stealing.

```rust
fn worker_loop(core_id: usize, queue: &WorkQueue) -> Result<Vec<Batch>> {
    pin_to_core(core_id);
    let rt = tokio::runtime::Builder::new_current_thread()
        .enable_io()
        .enable_time()
        .build()?;
    let mut output = Vec::new();
    
    while let Some(partition) = queue.next() {
        let result = rt.block_on(async {
            let mut stream = partition.build_stream()?;
            let mut batches = Vec::new();
            while let Some(b) = stream.next_batch().await? {
                batches.push(b);
            }
            Ok(batches)
        })?;
        output.extend(result);
    }
    
    Ok(output)
}
```

The runtime is used exclusively to drive async I/O within leaf operators. Upper operators are async-by-trait but rarely suspend.

### 7.4 Work queue

When `n_partitions > n_workers`, partitions are dispatched via a longest-job-first work queue.

```rust
struct WorkQueue {
    /// Sorted descending by estimated cost.
    pending: Mutex<BinaryHeap<Partition>>,
}
```

Contention on the mutex is negligible because partitions are coarse-grained (tens to hundreds of milliseconds of work each). For more sophisticated workloads, replace with `crossbeam::deque::Injector` or a lock-free priority queue.

### 7.5 Backpressure

Within a worker, backpressure is implicit:

- Upper operators pull from below synchronously via `next_batch`.
- Leaf operators maintain a bounded prefetch window (§9).
- When the consumer is slow, the leaf's window fills up and stops issuing new I/O.
- Memory pools (§9) block allocation when full, pausing the worker until reclaim runs.

Across workers, the gather merge at the top has a small bounded queue per partition (typically 2-4 batches). Producer-consumer mismatch causes the producer to block on queue full, which is fine because that worker can pick up another partition if any remain.

## 8. I/O subsystem

### 8.1 Per-worker I/O scheduler

Each worker has its own I/O scheduler:

```rust
struct IoScheduler {
    backend: Arc<dyn ObjectStore>,
    limiter: Arc<AdaptiveLimiter>,  // shared across workers
    pending: PriorityQueue<IoRequest>,
    in_flight: HashMap<RequestId, JoinHandle<Result<Bytes>>>,
}

struct IoRequest {
    range: ByteRange,
    priority: Priority,
    deadline: Option<Instant>,
}

enum Priority {
    Critical,   // blocking next_batch
    High,       // current pipeline tier
    Normal,     // prefetch
    Background, // speculative
}
```

The scheduler accepts requests, coalesces adjacent ranges, acquires permits from the shared adaptive limiter, and issues reads. Higher-priority requests preempt lower-priority ones in the issue order, though already-in-flight reads cannot be cancelled cheaply against most backends.

### 8.2 Shared adaptive concurrency limiter

One per (backend, endpoint) pair, shared across all workers.

```rust
struct AdaptiveLimiter {
    current_limit: AtomicUsize,
    in_flight: AtomicUsize,
    waiters: Mutex<PriorityQueue<Waker>>,
    latency_p99: SlidingWindow,
    throttle_observed: AtomicBool,
}
```

The limit adjusts via AIMD: additive increase on success, multiplicative decrease on observed throttling (HTTP 503, SlowDown, significant p99 latency increase). Acquisition is priority-ordered.

This is the only piece of cross-worker state in the I/O path, and it pays for itself: per-worker static limits waste capacity when workers are unevenly busy, and global throttle-then-back-off is necessary for cloud storage anyway.

### 8.3 Prefetch window in leaf operators

Each `LayoutWalker` and `Materialize` operator maintains an internal prefetch window of N pending batches whose I/O is in flight.

```rust
impl Materialize {
    async fn next_batch(&mut self) -> Result<Option<Batch>> {
        // Top up the window before consuming.
        while self.in_flight.len() < self.target_depth.get() {
            let Some(positions) = self.child.next_batch().await? else { break };
            let reads = self.plan_reads(&positions);
            let handle = self.io.submit_many(reads);
            self.in_flight.push_back(PendingBatch::new(positions, handle));
        }
        
        let Some(head) = self.in_flight.pop_front() else { return Ok(None) };
        
        // Speculative decode of new head.
        if let Some(next) = self.in_flight.front() {
            next.start_decode_in_background();
        }
        
        Ok(Some(head.resolve().await?))
    }
}
```

Depth is adaptive: track observed I/O completion-to-consumption gaps and adjust. Initial depth 8, range [1, 32].

### 8.4 Coalescing and prefetching

Read coalescing happens at the I/O scheduler. Adjacent byte ranges within a configurable threshold (1 MB for S3, 4 KB for NVMe) are merged into single requests. This is essential for sparse reads against object storage where per-request overhead dominates.

Speculative prefetch is conservative: only prefetch ranges within the demand window. The demand state guides what is and isn't worth speculating on.

### 8.5 Footer caching and metadata

Vortex footers and top-level layout metadata are cached aggressively in-memory across queries. Cache key: (URL, ETag/version). The cache is shared across workers. Cache size budgeted (default: 256 MB) with LRU eviction.

### 8.6 Hedged reads

For high-tail-latency backends, the I/O scheduler optionally issues duplicate reads to alternate endpoints once a request exceeds a deadline (typically p95 of recent latencies). The first to complete wins; the other is cancelled. Only applied to Critical/High priority requests, not prefetch.

### 8.7 Tier-priority scans

For a scan with a filter and a projection, the scan is decomposed into priority tiers:

1. **Tier 1 (Critical)**: zone-map walk. Reads tiny metadata, decodes fast, produces `DemandState` updates (range exclusions for pruned zones).
2. **Tier 2 (High)**: filter-column materialization. Reads only the columns in the filter predicate, for zones that survived tier 1.
3. **Tier 3 (Normal)**: projection-column materialization. Reads only projection columns, for positions that survived tier 2.

The I/O scheduler honors these priorities, ensuring that tier 1 reads complete before tier 3 even starts (for the same partition). This is what makes selective queries dramatically cheaper than full scans.

## 9. Memory management

### 9.1 Hierarchical pools

```rust
struct MemoryPool {
    used: AtomicUsize,
    limit: usize,
    parent: Option<Arc<MemoryPool>>,
    cv: Condvar,
    reclaimable: Mutex<Vec<Weak<dyn Reclaimable>>>,
}
```

Hierarchy: process pool → worker pool → operator pool. Each level enforces its own limit; allocations must succeed at every level up to the root.

### 9.2 Reservation API

Operators that know they're about to allocate a significant chunk *reserve* first:

```rust
trait MemoryPool {
    fn reserve(&self, n_bytes: usize) -> Result<Reservation>;
    /// Block until either reserve succeeds or reclaim runs.
    fn reserve_blocking(&self, n_bytes: usize) -> Result<Reservation>;
}
```

Reservations let the pool know what's coming, triggering pre-emptive reclaim instead of reactive failure. Allocations are then drawn from the reservation.

### 9.3 Sync blocking via Condvar

Memory pools use `Condvar` rather than waker-based async signaling. When `reserve_blocking` cannot succeed:

1. It first attempts local reclaim (calls `reclaim` on registered `Reclaimable` operators, largest first).
2. If reclaim doesn't free enough, the thread blocks on the condvar.
3. Another operation freeing memory in the same pool calls `notify_one`, waking the thread.

This works because the worker thread has nothing better to do — there is no other compute work for it that doesn't also need memory. The local async runtime continues to drive I/O on the thread while compute is paused, which is the right behavior: I/O completion will refill caches and may itself trigger reclaim.

### 9.4 Reclaim policies

Operators register as `Reclaimable` if they hold significant state:

```rust
trait Reclaimable {
    fn reclaimable_bytes(&self) -> usize;
    fn reclaim(&mut self, target_bytes: usize) -> Result<usize>;
}
```

- `Sort`: spill current in-memory runs to local-disk Vortex files.
- `HashBuild`: spill build-side partitions.
- `IndexedLookup` with `Bulk` strategy: evict entries; rebuild on next access.
- `AsyncMap`: reduce concurrency, cancel pending fetches.
- `StreamAggregate`, `MergeK`: not reclaimable; they hold only frontier state.

### 9.5 Vortex allocator integration

Every allocation Vortex performs for in-memory arrays goes through a `vortex::Allocator` trait, which we wrap to route through the operator's pool. Pool inheritance: when an operator produces a derived array, the new array inherits the operator's pool.

Canonicalization (forcing a Vortex array into its Arrow form) is the highest-risk allocation; tracked explicitly and counted against the reservation at the point the canonicalization is requested, not at completion.

## 10. Late materialization and demand propagation

### 10.1 Pipeline segments

A *pipeline segment* is a maximal chain of operators sharing a row domain. Boundaries are:

- Joins, aggregates, sorts (domain-changing operators).
- The output (the final consumer).

Each segment has its own `DemandState`. Demand propagation is intra-segment; it does not cross boundaries.

### 10.2 Demand forms

Only two forms are supported, both monotone:

- **Upper bound** (`limit_to(p)`): no rows past position p are needed.
- **Range exclusion** (`exclude(lo..hi)`): no rows in [lo, hi) are needed.

These compose: exclusions accumulate, the bound only decreases. The combined state is queryable in O(log n) where n is the number of intervals.

### 10.3 Who publishes what

- **TopK** publishes `limit_to(advancing_bound)` as its heap fills.
- **LIMIT** publishes `limit_to(k)` immediately on construction.
- **Filter** publishes `exclude(zone)` when a zone-level predicate prunes a zone entirely.
- **StreamAggregate** publishes `exclude(group_range)` when a group is finalized (the input positions for that group are no longer needed).
- **MergeK** (in join modes) publishes `seek(advancing_key)` on its idle side, which translates to position advancement at the leaf.

### 10.4 Who consumes

- **LayoutWalker** consults demand before traversing each zone. Excluded zones are skipped entirely; reaching the upper bound terminates the walk.
- **Materialize** consults demand before planning reads for a batch; positions that have been excluded since the walker produced the batch are dropped from the read plan.

### 10.5 EV-based predicate ordering

Within a `Filter` operator with multiple conjuncts, predicates are ordered by `cost / (1 - selectivity)`, with both estimated online from observed batches. Cheaper or more selective predicates run first; this minimizes total work.

Across tiers (zone map vs. filter columns vs. projection columns), priority is fixed by the planner via the I/O scheduler priority levels. There is no need for online EV reordering across tiers because tiers are structurally different: zone maps are always cheaper than column data.

### 10.6 LayoutWalker + Materialize as the late-materialization primitives

The split into `LayoutWalker` (produces positions) and `Materialize` (produces values) is what makes late materialization default rather than optional:

- Position-only pipelines (filter chains, demand-publication chains) never call `Materialize`.
- Each `Materialize` is inserted by the planner at the *latest* point where values are actually needed.
- Multiple `Materialize` operators can exist in a single plan, each at a different tier and pulling different column sets.

This mirrors C-Store/Vertica's position list model, but with the additional benefit that the position lists are themselves Vortex arrays and can be operated on with Vortex compute.

## 11. Vortex integration

### 11.1 Layouts as walkable trees

The engine treats every Vortex layout type as a node it can walk:

- `FlatLayout`: leaf, single chunk.
- `ChunkedLayout`: produces positions chunk-by-chunk.
- `ColumnarLayout`: each child is a column; positions span the row range.
- `ZonedLayout`: zone-map-driven pruning at this level.
- `DictLayout`: walks the codes; values are an indexed-lookup child.

Each layout exposes:

```rust
trait Layout {
    fn dtype(&self) -> &DType;
    fn n_rows(&self) -> u64;
    fn child_layouts(&self) -> &[Arc<dyn Layout>];
    fn zone_stats(&self) -> Option<&ZoneStats>;
    fn segment_ranges(&self, positions: &PositionRange) -> Vec<ByteRange>;
    fn decode(&self, bytes: &[Arc<Bytes>], positions: &PositionRange) -> Result<vortex::Array>;
}
```

`LayoutWalker` is generic over `dyn Layout` and dispatches per node type. Custom layouts plug in automatically.

### 11.2 Encoding-aware compute

Filter predicates and projection expressions dispatch to Vortex's compute kernel registry. The kernel is selected based on the encoding of the input array; canonicalization is a fallback, not a default.

A non-exhaustive list of encoding-specific kernels we need:

- Bit-packed integer comparison (range, equality)
- Dictionary code comparison (using the dictionary's order if sorted)
- FSST symbol-prefix comparison
- ALP-encoded float arithmetic
- Run-end-encoded sum/count aggregation

### 11.3 OVC integration with Vortex encodings

OVC values are computed in the encoded domain wherever possible:

- Bit-packed integers: OVC offset is the bit position of the differing bit; value is the differing byte.
- Dictionary codes: OVC is computed on the codes, valid as long as the dictionary is sorted.
- FSST: OVC starts from the FSST symbol comparison; falls back to character compare only when symbols are equal.

This requires Vortex to expose these comparisons at the encoded level. Where not yet available, contributions back to Vortex are expected.

### 11.4 Custom virtual layouts (Phase 2)

In Phase 2, the engine introduces virtual layouts that *are* operators:

- `MergeLayout`: child layouts merged on a key. Materializing this layout runs `MergeK`.
- `FilterLayout`: child layout with a predicate applied.
- `GroupByLayout`: child layout with streaming aggregation.

These can be serialized and persisted as new Vortex files, giving us incremental view maintenance for free: a materialized view is just a saved virtual layout tree.

## 12. Comparison to other systems

### 12.1 DataFusion

Similar in physical-plan shape (pull-based vectorized operators, partitioning as a first-class property). Different in:

- Batches are `vortex::Array`, not `RecordBatch`. Stays encoded end-to-end.
- Tracks `SortKey` and OVC annotations as plan properties.
- Sync-presenting compute, async only at leaves (DataFusion is async all the way through).
- Thread-per-core, not Tokio multi-thread.

We may ride DataFusion's logical planning and optimization rules at the frontend, replacing the execution layer. The interface boundary is the `ExecutionPlan` trait; our operators implement it but produce Vortex arrays.

### 12.2 DuckDB

Similar in vectorized execution model and selection-vector-style late materialization. Different in:

- Sort-merge default vs. hash-default.
- Storage is Vortex, not DuckDB's row groups.
- No pipeline abstraction; flat operator tree.
- External (object-store) workloads are first-class; DuckDB's I/O is more of an afterthought.

### 12.3 F1 Query / Napa

We adopt the execution philosophy almost wholesale:

- LSM-style forest of sorted files = Vortex tree of sorted files.
- Tournament tree + OVC.
- Progressive partitioning.
- In-sort aggregation.
- Materialized views as just-another-sorted-source.

We do not adopt: Napa's distributed coordination, queryable-timestamp / freshness machinery, georeplication. Those are for a future distributed phase.

### 12.4 Velox

Similar in vectorized building-block approach and Vector-level encoding awareness. Different in:

- Velox is a library of operators consumed by Presto/Spark engines; we are a complete engine.
- We choose sort-merge as the default; Velox is more hash-default.
- Our Batch is a Vortex array; Velox has its own Vector type.

Velox's memory management and reservation API are direct inspirations.

## 13. Implementation phases

### Phase 0: Foundations (weeks 1-4)

- `SortKey`, `SortedStream`, `Batch`, `AsyncArray`, `DemandState`, `RowDomain` types.
- `LayoutWalker` for basic layouts (Flat, Chunked, Columnar).
- `Materialize` with no prefetch window (synchronous reads).
- Synchronous single-threaded execution over a local Vortex file.
- Test query: `SELECT * FROM file.vtx WHERE col > 100`.

### Phase 1: Sort-merge core (weeks 5-10)

- `MergeK` with tournament tree, no OVC yet.
- `StreamAggregate`.
- `Filter`, `Project`, `TopK`.
- `RangeBound`, `PartitionSplit`.
- Single-threaded execution of: merge-join-aggregate over two sorted Vortex files.
- Test query: top-10 by sum-of-amount, joined on customer_id, two sorted files.

### Phase 2: Parallelism and I/O (weeks 11-18)

- Progressive partitioning algorithm against Vortex layout statistics.
- Thread-per-core dispatcher with work queue.
- Worker-local Tokio runtime.
- Per-worker I/O scheduler with coalescing.
- Prefetch window in `Materialize`.
- Shared adaptive concurrency limiter.
- Footer caching.
- Test workload: TPC-H-scale sort-merge queries against object storage.

### Phase 3: Demand and late materialization (weeks 19-24)

- `DemandState` plumbing.
- `seek()` propagation in `MergeK` and `TopK`.
- Tier-priority scans (zone map → filter → projection).
- Selection-vector decode in Vortex layouts.
- Test: verify byte-level I/O reduction on selective queries.

### Phase 4: OVC and encoding-aware compute (weeks 25-30)

- OVC computation in tournament tree.
- OVC propagation through `Filter`, `Project`, `MergeK`.
- Encoded-domain comparison kernels in Vortex (likely upstream contributions).
- Performance measurement vs. Phase 1-3 baseline.

### Phase 5: Memory and reclaim (weeks 31-34)

- Pool hierarchy with reservations.
- Condvar-based blocking allocation.
- `Reclaimable` trait and implementations.
- Vortex allocator integration.

### Phase 6: Async expressions and indexed lookup (weeks 35-40)

- `AsyncMap` operator with concurrency window.
- `IndexedLookup` with all four strategies.
- Custom layout support in `LayoutWalker`.

### Phase 7: Hash-based fallback (weeks 41-46)

- `Sort` operator (external, tournament-tree run gen, MergeK merge phase).
- `HashBuild` / `HashProbe`.
- `HashAggregate`.
- Planner integration: cost-based choice between sort-merge and hash variants.

### Phase 8: Virtual layouts and materialized views (future)

- `MergeLayout`, `FilterLayout`, `GroupByLayout`.
- Serialization to Vortex files.
- Incremental view maintenance.

## 14. Open questions and risks

### 14.1 OVC propagation through Vortex encodings

How completely can OVCs be computed from encoded comparisons across all Vortex encoding types? Bit-packed and dictionary are clear; FSST is partial; ALP is unclear. May require Vortex API extensions.

### 14.2 Custom layout extensibility

Vortex supports user-defined layouts. We want them to plug into `LayoutWalker` seamlessly, but our walker needs structural information (segment ranges, zone stats) that custom layouts might not expose uniformly. Need a stricter `Layout` trait contract than Vortex currently mandates.

### 14.3 Async I/O abstraction choice

`tokio::current_thread`, `glommio`, and `monoio` are all viable. `glommio` is most aligned philosophically (thread-per-core, io_uring) but less mature. `tokio` is safest but has more cross-thread overhead even in current-thread mode. Decision deferred to Phase 2 prototyping.

### 14.4 Materialized view freshness

Once virtual layouts are persistable, we have materialized views — but the consistency story (when does a view see which base table version?) is intricate and modeled on Napa's queryable-timestamp machinery, which we said was out of scope. Need to revisit when Phase 8 approaches.

### 14.5 Planner integration with DataFusion

We want to ride DataFusion's frontend, but DataFusion's optimizer doesn't reason about OVC, our `DemandState`, or tier-priority scans. We'll need either (a) custom optimizer rules, or (b) a separate physical planning phase that runs after DataFusion's optimizer. Probably (b).

### 14.6 Skew tolerance under progressive partitioning

Progressive partitioning produces evenness based on *size estimates*. For queries whose cost is wildly non-uniform across keys (e.g. a join where some keys have 1M matches and others have 1), size-based partitioning gives uneven *runtime*. Need to investigate either cost-model-driven partitioning or runtime re-partitioning under work queue.

### 14.7 Backpressure across partitions

Within a worker, backpressure is implicit via pull. Across the gather merge at the top, we have small queues per partition. What happens when one partition produces 10x more output than others? The gather queue fills, the producer worker blocks, but it could productively be running another partition. Need to model this carefully.

## 15. Appendix: example query traces

### 15.1 Top-10 with predicate on sorted Vortex file

```
SELECT customer_id, total
FROM orders_sorted_by_total
WHERE region = 'NA'
ORDER BY total DESC
LIMIT 10
```

Plan:
```
TopK(k=10, order=total DESC)
  └── Filter(region = 'NA')
      └── Materialize(columns=[customer_id, total, region])
          └── LayoutWalker(orders_sorted_by_total.vtx)
```

Execution (single partition for simplicity):

1. `LayoutWalker` consults `DemandState`; initial state is "everything needed". Walks the root `ChunkedLayout`, emits position batch [0..10000).
2. `Materialize` plans reads for `[customer_id, total, region]` over positions [0..10000). Issues 3 byte-range reads (one per column). Awaits.
3. Reads complete. `Materialize` returns a `Batch` whose `AsyncArray` is now resolved.
4. `Filter` applies `region = 'NA'`. Suppose 30% pass. Output batch has positions [0..10000) with selection. Output `SortKey` is `Natural(total DESC)` because filter preserves order.
5. `TopK` consumes. Heap fills to 10 after a few batches. The 10th-best `total` value becomes the new `limit_to` upper bound.
6. `TopK` updates `DemandState.max_position` based on what zones can still contain values exceeding the 10th-best total. (For data sorted by total DESC, once we've passed total = current 10th-best, no future row can beat it — so `limit_to` advances to the current scan position.)
7. `LayoutWalker` consults demand on its next iteration. If position bound has narrowed to the current position, the walker terminates.
8. `TopK` emits final 10 rows.

Bytes read: roughly the first chunk's worth, regardless of file size. For a 100M-row file with 1000-row chunks, that's ~0.1% of the file.

### 15.2 Merge-join with streaming aggregate

```
SELECT customer_id, SUM(amount)
FROM orders JOIN customers USING(customer_id)
WHERE customers.region = 'NA'
GROUP BY customer_id
```

Plan (after partitioning, per worker):

```
StreamAggregate(group=[customer_id], sum=amount)
  └── MergeK(mode=InnerJoin, key=[customer_id])
      ├── Materialize(columns=[customer_id, amount])
      │   └── LayoutWalker(orders.vtx, range=[lo, hi))
      └── Filter(region = 'NA')
          └── Materialize(columns=[customer_id, region])
              └── LayoutWalker(customers.vtx, range=[lo, hi))
```

Execution: see §15.1 for individual operator behavior. The merge frontier holds at most one row per side; the aggregate holds at most one in-progress group. Memory per worker is dominated by the prefetch windows in the two `Materialize` operators.

After all workers complete, the gather phase runs `MergeK(mode=Union)` over partition outputs (a final merge on customer_id) to produce globally sorted output. If the consumer doesn't need order, the gather phase just concatenates.
