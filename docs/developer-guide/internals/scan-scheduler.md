# Scan Scheduler

:::{note}
This is an implementation design for scheduler-aware ScanPlan execution. It describes the resource
coordination shape that the scan runtime is growing toward.
:::

The ScanPlan scan path needs a resource scheduler that can coordinate work across files, partitions,
and concurrent scans. The scheduler should be explicit and embeddable: a host engine can share one
scheduler across many scans to enforce global limits, or create a fresh scheduler for each query to
isolate resource usage.

The design uses one shared `ScanScheduler` object for resource arbitration and one per-scan runtime
for query semantics.

The existing `DataSource` / `ScanRequest` / `DataSourceScan` API remains the public query-engine
boundary for this phase. The scheduler and morsel runtime sit behind that boundary, so the first
implementation can improve scan execution without introducing a second scan API that mostly duplicates
the current one.

## Goals

- Bound scan resource usage across concurrent scans.
- Allow DataFusion users to choose a shared scheduler, a new scheduler per query, or an unbounded
  mode.
- Give DuckDB a simple global scheduler owned by the extension session.
- Keep ScanPlan planning and morsel ordering local to each scan.
- Make I/O planning explicit enough that future evidence, predicate, and projection reads can be
  deduplicated, batched, and prioritized without relying on hidden unpolled futures inside layout
  readers.
- Keep storage backends pluggable: local files, object stores, HTTP range sources, memory buffers,
  and future `io_uring`-backed sources should all fit behind the same scheduler-visible shape.
- Make cancellation and permit release reliable when a stream is dropped early.
- Keep scheduler APIs independent of layout internals so other `DataSource` implementations can use
  the same resource controls.

## Non-goals

- Do not make a process-global singleton the only way to schedule scans.
- Do not put query semantics, filter ordering, evidence planning, or output ordering into the global
  scheduler.
- Do not replace the `DataSource` scan API in the first scheduler implementation. If the public API
  changes later, it should be because the ScanPlan runtime needs capabilities that cannot be added
  compatibly to `ScanRequest` or `DataSourceScan`.
- Do not require every scan integration to expose the same configuration surface immediately.
- Do not solve cluster-wide distributed admission control. The scheduler is process-local.
- Do not design an opaque I/O path in the first implementation. If a future custom `ScanPlan` needs
  non-segment I/O, add that as a small extension point next to `SegmentRequest`.

## Core Model

There are three layers:

1. `ScanScheduler`
   Arbitrates global resources such as I/O bytes, decoded bytes, request concurrency, decode task
   concurrency, and per-scan fairness.

2. `ScanTicket`
   Represents one logical scan registered with a scheduler. It carries scan identity, cancellation,
   priority, metrics, and per-scan limits.

3. Per-scan `MorselScanRuntime`
   Owns the ScanPlan graph, evidence/read/aggregate plans, morsel queue, row ordering, limit
   handling, dynamic filters, and the choice of which work is useful next.

`DataSource::scan` constructs this per-scan runtime internally and returns the existing
`DataSourceScan` wrapper. Query engines do not need to know the internal ScanPlan topology.

The scheduler decides whether work may run. The per-scan runtime decides what work should run.

```text
DataFusion / DuckDB
        |
        v
DataSource::scan(request)
        |
        v
resolve scheduler provider
        |
        v
ScanScheduler::register_scan(meta) -> ScanTicket
        |
        v
MorselScanRuntime
        |
        +-- plan next useful morsel
        +-- acquire scheduler permits
        +-- run evidence / read / decode / aggregate work
        +-- release permits on completion or drop
```

## Scheduler Ownership

`ScanScheduler` is an ordinary shared object:

```rust
pub struct ScanScheduler {
    config: ScanSchedulerConfig,
    state: ScanSchedulerState,
}
```

The object is normally used behind `Arc<ScanScheduler>`.

```rust
let scheduler = Arc::new(ScanScheduler::new(config));
```

Scheduler ownership is selected by a provider:

```rust
pub enum ScanSchedulerProvider {
    /// Use one scheduler for every scan that shares this provider.
    Shared(Arc<ScanScheduler>),

    /// Construct a new scheduler whenever a logical scan starts.
    PerScan(ScanSchedulerConfig),

    /// No resource limits. Useful as the compatibility default and for tests.
    Unbounded,
}
```

The provider is resolved when a logical scan starts, not when a table or data source is registered.
This matters for DataFusion, where a table can be registered once and executed many times.

```rust
impl ScanSchedulerProvider {
    pub fn scheduler_for_scan(&self, meta: &ScanMeta) -> Arc<ScanScheduler>;
}
```

## Session Integration

The scheduler provider should be stored on `VortexSession`, following the same pattern as
`RuntimeSession`.

```rust
pub struct ScanSchedulerSession {
    provider: Arc<ScanSchedulerProvider>,
}

pub trait ScanSchedulerSessionExt: SessionExt {
    fn scan_scheduler_provider(&self) -> Arc<ScanSchedulerProvider>;

    fn with_scan_scheduler(self, scheduler: Arc<ScanScheduler>) -> Self;

    fn with_new_scan_scheduler_per_scan(self, config: ScanSchedulerConfig) -> Self;

    fn with_unbounded_scan_scheduler(self) -> Self;
}
```

The default can be `Unbounded` initially, so adopting the scheduler does not silently introduce new
resource limits. Integrations can opt into bounded scheduling explicitly.

The scheduler types should live in `vortex-scan`, not `vortex-layout`, because the resource policy
belongs to the scan API layer and should be reusable by non-layout sources. ScanPlan-specific code in
`vortex-layout` can consume tickets and permits through the public scan scheduler API without making
the scheduler understand layout-specific plan types.

## DataFusion Integration

DataFusion should expose scheduler control in the table/source builders.

```rust
impl VortexDataSourceBuilder {
    pub fn with_scan_scheduler(mut self, scheduler: Arc<ScanScheduler>) -> Self;

    pub fn with_scan_scheduler_provider(
        mut self,
        provider: Arc<ScanSchedulerProvider>,
    ) -> Self;

    pub fn with_new_scan_scheduler_per_query(
        mut self,
        config: ScanSchedulerConfig,
    ) -> Self;
}
```

The same options should be available on `VortexTable` and `VortexFormatFactory` so users who
register tables through DataFusion's listing format path can still control scheduling.

For DataFusion, `DataSource::open` creates a single Vortex scan for partition zero. A per-query
scheduler can therefore be resolved immediately before calling
`DataSourceRef::scan`. If DataFusion later produces multiple Vortex scan plans for one query and
those scans should share a per-query scheduler, the integration should propagate a scheduler through
DataFusion's `TaskContext` or another query-scoped extension and use that as the provider result.

Recommended DataFusion modes:

```rust
// One scheduler across an application, tenant, or SessionContext.
let scheduler = Arc::new(ScanScheduler::new(config));
let source = VortexDataSource::builder(data_source, session)
    .with_scan_scheduler(scheduler)
    .build()
    .await?;

// A fresh scheduler each time this table is scanned.
let source = VortexDataSource::builder(data_source, session)
    .with_new_scan_scheduler_per_query(config)
    .build()
    .await?;
```

Benchmark environment variables can map onto these APIs, but they should not be the primary control
surface:

```text
VORTEX_SCAN_SCHEDULER=unbounded|shared|per-query
VORTEX_SCAN_MAX_MORSEL_SLOTS=...
```

## DuckDB Integration

DuckDB can use one scheduler in the extension's global session.

```rust
static SCAN_SCHEDULER: LazyLock<Arc<ScanScheduler>> =
    LazyLock::new(|| Arc::new(ScanScheduler::new(ScanSchedulerConfig::duckdb_default())));

static SESSION: LazyLock<VortexSession> = LazyLock::new(|| {
    let session = VortexSession::default()
        .with_handle(RUNTIME.handle())
        .with_scan_scheduler(Arc::clone(&SCAN_SCHEDULER));
    vortex_geo::initialize(&session);
    session
});
```

This matches DuckDB's current extension shape: a global runtime and global Vortex session. It still
keeps the scheduler explicit and testable.

## Work Requests and Permits

Scan work should acquire scheduler permits before consuming bounded resources.

The first implementation should not require every `PreparedRead`, `PreparedEvidence`, or `PreparedAggregate`
to expose pending I/O, decoded-size estimates, or cost statistics. Those estimates are useful, but
they are also hard to get right and would make the initial ScanPlan API more rigid. The scan runtime
already knows the coarse unit of scheduling: the morsel. The MVP scheduler should admit morsels and
let each admitted morsel run its evidence/read/aggregate pipeline internally.

The MVP `WorkRequest` should be coarse:

```rust
pub struct WorkRequest {
    pub class: ScanWorkClass,
    pub slots: u32,
}

pub enum ScanWorkClass {
    FileOpen,
    Morsel,
    OutputConversion,
}

impl ScanScheduler {
    pub fn register_scan(&self, meta: ScanMeta) -> ScanTicket;

    pub async fn acquire(
        &self,
        ticket: &ScanTicket,
        request: WorkRequest,
    ) -> VortexResult<WorkPermit>;
}
```

Richer byte/task fields can be added once the runtime has instrumentation showing which resource
limits matter in practice:

```rust
pub struct PreparedCostHint {
    pub estimated_io_bytes: Option<u64>,
    pub estimated_decoded_bytes: Option<u64>,
    pub estimated_cpu_units: Option<u64>,
}
```

If those hints are added, they should remain advisory. A prepared handle that does not provide hints should
still be schedulable with default morsel accounting.

`WorkPermit` is RAII. Dropping it releases every reserved resource. This is required for early
limit termination, query cancellation, stream drop, and panic-safe cleanup.

```rust
pub struct WorkPermit {
    scheduler: Arc<ScanScheduler>,
    reservation: ReservationId,
}

impl Drop for WorkPermit {
    fn drop(&mut self) {
        self.scheduler.release(self.reservation);
    }
}
```

Once byte accounting exists, large work should be allowed to resize reservations after the actual
memory footprint is known:

```rust
impl WorkPermit {
    pub async fn grow_decoded_bytes(&mut self, bytes: u64) -> VortexResult<()>;

    pub fn shrink_decoded_bytes(&mut self, bytes: u64);
}
```

This will let the scan reserve from estimates first, then correct accounting after decoding.

## Explicit Segment Request Model

The ScanPlan path makes segment requests explicit enough for scheduling while keeping physical I/O
inside the segment source. Layouts still refer to logical segments by
`SegmentId`, and the scheduler should stay at that same abstraction level. It should know which
registered source owns the segment and roughly how many bytes the segment costs, but it should not
need the segment's physical byte location:

```rust
pub struct SegmentRequest {
    pub source: SegmentSourceId,
    pub segment: SegmentId,
    pub bytes: u64,
    pub phase: ScanIoPhase,
    pub priority: ScanPriority,
    pub cancel_group: CancelGroup,
}

pub enum ScanIoPhase {
    EvidenceSetup,
    EvidenceProbe,
    PredicateRead,
    ProjectionRead,
    AggregateRead,
}
```

`SegmentId` is not a physical I/O address. It is a layout-local reference. A `VortexFile` binds
that reference when it instantiates a ScanPlan tree:

```text
footer segment map + opened byte source
        |
        v
SegmentId -> SegmentInfo { bytes, cacheability, source-local metadata }
```

For normal Vortex files, the source is the `VortexReadAt` returned by `VortexOpenOptions` or
`FileSystem::open_read`. For a custom ScanPlan, the source might be an HTTP range reader, an
in-memory reader, or another backend that can provide segment payloads.

The first implementation should only support segment requests. A future non-segment I/O hook can be
added next to `SegmentRequest` if a custom source cannot present its work as segment payloads, but
leaving that hook out of the initial API keeps the scheduler boundary smaller.

There are two stages for making this request model authoritative:

1. **Intermediate: scheduled morsel futures.**
   Constructing a morsel future synchronously registers the segment requests that future will later
   await. The returned future owns those segment futures, so the scheduler can construct work ahead
   of time, observe its byte cost, reorder or drop it, and only poll it when useful.

2. **End state: strict scheduler-backed resolution.**
   Plans describe their exact segment requests before execution, the scheduler/source submits them,
   and execution reads through a context backed by the submitted request set. In this mode, reading
   a segment that was not declared is an error or an explicitly metered late request.

The intermediate stage preserves the useful old pre-registration behavior, but moves it out of
layout-reader side effects and into an explicit scheduler context.

## Segment Source Registration

Segment sources are registered against a scan ticket and receive scheduler-local identities:

```rust
impl ScanTicket {
    pub fn register_segment_source(
        &self,
        source: Arc<dyn ScheduledSegmentSource>,
        meta: SegmentSourceMeta,
    ) -> SegmentSourceId;
}
```

Equivalently, this can be spelled as `ScanScheduler::register_segment_source(&ticket, ...)` if the
implementation wants the scheduler to own all mutation directly. The ergonomic API should keep the
source tied to the ticket, because source identity is only meaningful within the scheduler context
that is arbitrating the scan.

`SegmentSourceId` should be opaque. A shared scheduler may internally deduplicate physical sources
by an optional stable `SegmentSourceKey`, but correctness must not depend on that global
deduplication. The minimum guarantee is scan-local identity: all requests with the same
`SegmentSourceId` target the same registered source and may be deduped or batched together.

For a prepared `VortexFile`, source registration happens during file preparation, before layout
plans produce runtime segment requests. Layout-specific plans should not know how a file was
opened. A flat layout can continue to store `segment_id`; the prepared file state translates that ID to a
`SegmentRequest` using the bound segment table.

Custom ScanPlans that own independent I/O register their own sources during preparation or state
initialization. For example, an HTTP-backed plan can register a source that maps `SegmentId`s to
HTTP range requests internally and produce `SegmentRequest`s against the returned
`SegmentSourceId`.

## Batching and Coalescing

`ScanScheduler` should own logical scheduling, not physical coalescing.

The intermediate scheduler should expose a scheduling context to plan execution constructors:

```rust
pub struct ScheduleCtx<'a> {
    ticket: &'a ScanTicket,
    source_id: SegmentSourceId,
    source: Arc<dyn ScheduledSegmentSource>,
    in_flight: &'a SegmentFutureCache,
}

impl ScheduleCtx<'_> {
    pub fn request_for_segment(&self, segment: SegmentId) -> VortexResult<SegmentRequest>;

    pub fn request_segment(&mut self, request: SegmentRequest) -> SegmentFuture;
}
```

`request_segment` is synchronous. It dedupes by `(SegmentSourceId, SegmentId)`, submits to the
registered source when needed, and returns a shared future for the logical segment payload. Adjacent
morsels and different prepared handles that touch the same segment receive clones of the same shared future
while it remains in flight.

The scheduler should therefore:

- construct scheduled morsel futures ahead of polling;
- let those constructors call `ScheduleCtx::request_segment` for the I/O they will later await;
- cache in-flight segment futures by `(SegmentSourceId, SegmentId)`;
- group pending reads by `SegmentSourceId`;
- prioritize grouped reads based on phase, frontier, memory pressure, cancellation, and observed
  predicate selectivity;
- submit ordered batches or windows of segment requests to each source.

For the current intermediate implementation, scans without a pushed-down limit should default to an
unbounded planning window and a bounded launch window. Constructing the planned morsel registers
segment futures for the scan window, while only
the launch window controls how many morsels are actively polled and decoded. Ordered scans use the
same planning and launch machinery, but projection completions are buffered behind an ordered
emission frontier. Scans with a pushed-down limit should continue using a `1/1` plan/launch window
until limit accounting can be preserved with a wider frontier.

The `ScheduledSegmentSource` should own physical coalescing and submission:

```rust
pub trait ScheduledSegmentSource: Send + Sync {
    fn segment_info(&self, id: SegmentId) -> VortexResult<SegmentInfo>;

    fn capabilities(&self) -> SegmentSourceCapabilities;

    fn submit(&self, batch: SegmentBatch) -> Vec<SegmentHandle>;
}
```

Different backends make different tradeoffs. Local files, object stores, HTTP range readers, memory
buffers, and an `io_uring` implementation have different queue depths, alignment constraints,
cancellation behavior, request overheads, and tolerance for over-reading. The scheduler can hand a
source nearby segment requests in a useful order, but the source decides whether to merge their
underlying byte ranges into one physical request, issue them independently, or use a backend-specific
submission queue.

The in-flight future cache is a scan/runtime data structure, not a decoded-data cache. Its job is to
avoid duplicate physical submission while scheduled morsels overlap. Once no scheduled future or
plan state retains interest, the in-flight entry may be dropped. Longer-lived reuse belongs either
in plan state, such as a decoded flat array or zoned stats table, or in the segment cache described
below.

This preserves the existing `VortexReadAt` abstraction. A default `ReadAtSegmentSource` can wrap
`Arc<dyn VortexReadAt>` and use the source's `coalesce_config()` and `concurrency()` as physical
submission policy behind a file-backed `ScheduledSegmentSource`. The current `FileSegmentSource`
can then become a compatibility adapter over the same machinery rather than the scheduler-visible
abstraction.

The important invariant is that dedupe and coalescing never cross `SegmentSourceId` unless an
implementation has proven that two IDs share the same physical source. A byte range on one HTTP
endpoint is not interchangeable with the same range on another endpoint.

## Segment Cache

The segment cache should cache segment payloads, not entire files and not arbitrary coalesced byte
ranges. The unit stored in the cache is the exact buffer a layout segment would receive after a
physical read has been sliced back to the segment boundary.

The cache key must be source/file scoped. A raw `SegmentId` is only meaningful inside one footer's
segment map. Reusing `SegmentId(0)` across two files must not collide. A scheduler-aware key should
therefore include source identity plus the logical segment:

```rust
pub struct SegmentCacheKey {
    pub source: SegmentSourceId,
    pub segment_id: SegmentId,
}
```

If a shared scheduler later wants cross-scan cache reuse for the same object, it can translate
`SegmentSourceId` to an optional stable `SegmentSourceKey`, such as an opened-file identity with
size and version metadata. That optimization is separate from correctness. The scan-local key is
sufficient for deduping and cache lookup within one prepared scan.

Cache lookup should happen before physical I/O submission:

1. The runtime produces a cacheable `SegmentRequest`.
2. The scheduler dedupes exact logical requests.
3. The source adapter checks the segment cache for each cacheable request.
4. Cache hits complete the logical segment request without consuming physical I/O queue depth.
5. Cache misses are submitted to the underlying `ScheduledSegmentSource`.
6. When a physical read completes, the source stores the exact segment slice, not the coalesced
   super-range.

This can be implemented as a cached source adapter:

```rust
CachedSegmentSource {
    cache: Arc<dyn SegmentCache>,
    inner: Arc<dyn ScheduledSegmentSource>,
}
```

The adapter mirrors today's `SegmentCacheSourceAdapter`, but it works with explicit
`SegmentRequest`s instead of `SegmentFuture`s. It also preserves the current
`InitialReadSegmentCache` behavior: when footer parsing already fetched bytes that cover whole
segments, the prepared file can seed those segment entries and avoid issuing later reads.

Segment cache admission should remain a cache policy decision at first. The scheduler should observe
hits, misses, and stores for metrics, and it may eventually coordinate a shared cache memory budget,
but it does not need to own eviction to schedule scans correctly.

## Scheduled Morsel Futures

`ScanPlan` and prepared handles serve different purposes:

- `ScanPlan` is the expanded layout tree with capabilities. It answers whether a layout can push an
  expression, produce evidence, read values, split work, or answer statistics.
- A prepared handle is a reusable compiled route through that tree for one purpose, such as reading
  a projection expression or producing one predicate's evidence. It should not own frontier state
  and should not have an `execute_next(len)` API.

The drive/cursor owns frontier state and chooses explicit morsel ranges. Prepared handles execute
explicit work:

```rust
pub struct MorselScope<'a> {
    pub range: Range<u64>,
    pub rows: RowScope<'a>,
}

pub struct ScheduledRead<'a> {
    pub range: Range<u64>,
    pub phase: ScanIoPhase,
    pub bytes: u64,
    pub future: BoxFuture<'a, VortexResult<ArrayRef>>,
}

pub trait ScheduledPreparedRead {
    fn schedule_morsel<'a>(
        &'a self,
        scope: MorselScope<'a>,
        state: &'a Self::State,
        cx: &'a mut ScheduleCtx<'_>,
        local: &'a mut ExecutionCtx,
    ) -> VortexResult<ScheduledRead<'a>>;
}
```

The exact Rust shape may differ, but the important property is that `schedule_morsel` is not an
`async fn`. It runs immediately, requests every segment the returned future will await, and returns
a future that may be polled later. For example, a flat leaf requests its segment before returning
the decode future:

```rust
fn schedule_morsel(...) -> VortexResult<ScheduledRead<'_>> {
    let request = cx.request_for_segment(flat.segment_id())?;
    let segment = cx.request_segment(request);
    Ok(ScheduledRead::new(async move {
        let bytes = segment.await?;
        decode_flat(bytes)
    }))
}
```

This avoids a pure "declare requests, then ignore them during execution" layer. The scheduled
future captures the segment futures that define its I/O lifetime. Dropping an unpolled scheduled
morsel releases its interest; polling it later awaits the already-registered segment futures.

The scheduler can construct scheduled work ahead until one or more thresholds are reached:

- in-flight segment bytes;
- in-flight scheduled morsels;
- projected output or intermediate memory;
- maximum distance ahead of the contiguous morsel frontier.

It can then choose which scheduled futures to poll using phase, readiness, observed selectivity,
frontier pressure, and byte size. Evidence can run farther ahead because output is small and it
sharpens later work. Predicate reads should be ordered by expected selectivity per byte. Projection
reads should stay near the accepted-row frontier so the scan does not retain an entire filtered
stream before emitting output.

## End-State Prepared-Handle Introspection

The stricter end state can add explicit request introspection on top of scheduled morsel futures.
In that model, prepared handles describe the segments they would need before execution, the
scheduler submits those requests, and execution receives a resolver backed by the submitted request
set:

```rust
pub trait PreparedRead {
    fn segment_requests(
        &self,
        range: Range<u64>,
        rows: RowScope<'_>,
        state: &Self::State,
        cx: &mut SegmentPlanCtx,
    ) -> VortexResult<SegmentRequests> {
        Ok(SegmentRequests::unknown())
    }
}

pub trait PreparedEvidence {
    fn segment_requests(
        &self,
        req: &EvidenceRequest<'_>,
        state: &Self::State,
        cx: &mut SegmentPlanCtx,
    ) -> VortexResult<SegmentRequests> {
        Ok(SegmentRequests::unknown())
    }
}
```

Leaf prepared handles can provide exact requests. A flat leaf reports the segment bound to its `segment_id`.
Zoned evidence reports the shared stats-table setup read separately from cheap per-morsel probes.
Struct and apply prepared reads compose child requests. Chunked prepared reads use `selection` and `demand` to include
only the chunks that actually require data, preserving the current selected-but-undemanded behavior
where default filler can be produced without expanding or reading a child.

In strict mode, prepared handles that return `unknown` cannot use the strict resolver without falling back to
an explicit late request path. That fallback should be observable in metrics and should eventually
disappear from core layouts. This is why the scheduled-morsel-future model is the better
intermediate step: it makes I/O registration authoritative without requiring every prepared handle
to expose a perfect request set on day one.

## Morsel Pipeline

Long term, the per-scan runtime should manage a state machine per morsel:

```text
Planned
  -> EvidenceReady
  -> PredicateReady
  -> ProjectionReady
  -> Emitted | Pruned
  -> Released
```

Evidence, predicate, and projection work should be pipelined rather than globally phased:

- Run shared evidence setup far ahead when it is cheap and likely to prune later work.
- Run evidence probes ahead within an evidence-memory budget.
- Schedule residual predicate reads using observed cost and selectivity.
- Schedule projection reads with output backpressure so the scan does not filter the entire stream
  and retain all surviving masks before producing batches.
- Re-run cheap `recheck_before_projection` evidence immediately before projection when dynamic
  predicate versions changed while the morsel was in flight.

The runtime, not the global scheduler, owns predicate semantics. It tracks masks, predicate
versions, limits, output ordering, and aggregate state. The scheduler only sees resource classes,
segment requests, priorities, reservations, cancellation state, and source IDs.

Predicate ordering should be adaptive. The runtime can keep per-predicate statistics such as:

- evidence prune rate;
- residual selectivity;
- bytes read per evaluated row;
- latency;
- cache hit rate;
- downstream projection bytes avoided.

These observations feed future priority decisions. A residual predicate that is cheap and highly
selective should run before an expensive low-selectivity predicate. A predicate whose evidence setup
is already cached may become cheap enough to run earlier. These are per-scan policy decisions, not
layout-node behavior.

## Morsel Frontier

The scheduler-aware runtime should own the per-file morsel frontier. Each prepared file tracks the
set of morsels that may still read state. When a morsel is emitted or pruned, the runtime advances
the contiguous completed frontier and calls release hooks on prepared reads and scan plans.

This is required for lookahead. Without a frontier, running evidence and predicate work far ahead can
leave decoded chunks, flat arrays, zone maps, and masks retained longer than intended. The release
frontier lets layouts keep only the working set:

```text
unfinished: [m3, m7, m8]
completed:  [m0, m1, m2, m4, m5, m6]
frontier:   end(m2)
```

The frontier advances only through contiguous completed morsels. Later completed morsels cannot
release earlier state until the gap closes.

## Resources to Control

The first implementation should control active execution:

- Maximum morsels in flight per scan.
- Maximum morsels in flight across a shared scheduler.

This intentionally approximates the current scan behavior: scans without a pushed-down limit can run
several morsels concurrently. Ordered scans keep the same work window, but emit projection results
through an ordered frontier. Scans with a pushed-down limit should run with a narrower launch
window. The default launch window should be proportional to available scan parallelism:

```text
no limit: max_morsels_in_flight = 4 * available_parallelism
limit:    max_morsels_in_flight = 1
```

The shared scheduler can apply the same window globally, per scan, or both. For example, a
DataFusion user can choose one shared scheduler with `4 * available_parallelism` total morsel slots
to cap the whole process, or create a new scheduler per query to isolate resource accounting.

Later implementations can add:

- I/O bytes in flight.
- Decoded/intermediate bytes in flight.
- Number of outstanding I/O operations.
- Number of decode/CPU tasks spawned by the scan path.
- Scheduler-aware segment cache admission.
- Per-scan weights and priorities.
- Storage-class-specific concurrency, such as separate local disk and object store limits.
- Output batch memory handoff, where permits live until the query engine consumes the batch.

Output memory is the hardest resource to account for because ownership leaves the scan runtime.
The first byte-accounting implementation should bound intermediate scan memory and treat output
batch accounting as a follow-up.

## Fairness

A shared scheduler must avoid letting one large scan submit enough work to starve smaller scans.
The initial policy should combine global limits with per-scan windows:

- Each `ScanTicket` has a maximum number of in-flight morsels.
- Global slot semaphores bound aggregate morsel concurrency.
- Work is admitted only when both the per-scan and global limits allow it.

This is simpler than a centralized global work queue and avoids making the scheduler responsible for
query semantics. Weighted fair scheduling can be added later if the per-scan windows are not enough.

## Morsel Runtime

The scan should move toward an explicit per-scan runtime. The MVP can still execute one whole
morsel after acquiring one coarse scheduler permit, but the runtime boundary should be chosen so it
can later split a morsel into evidence, predicate, projection, emit, and release work without
changing the public `DataSource` API.

```rust
pub struct MorselScanRuntime {
    scheduler: Arc<ScanScheduler>,
    ticket: ScanTicket,
    plan: ScanRuntimePlan,
    state: ScanRuntimeState,
}
```

`ScanRuntimePlan` is internal to the scan implementation. It contains the files, expanded ScanPlan
trees, pushed expressions, prepared evidence handles, prepared reads, prepared aggregate handles,
and reusable per-file state.
It is not a replacement public scan API.

MVP execution loop:

```text
while output is still required:
    claim next morsel
    acquire per-morsel scheduler permits
    read evidence needed for pruning or satisfaction
    update row selection
    read residual filter columns if needed
    evaluate residual filter
    read projected values or update aggregate state
    emit batch or aggregate partial
    release permits
```

Target execution loop:

```text
while output is still required:
    choose explicit morsel ranges from the drive/cursor
    construct scheduled evidence/predicate/projection futures until budget is full
    each scheduled future registers its segment requests through ScheduleCtx
    poll the most useful scheduled futures based on phase, bytes, selectivity, and frontier
    update evidence, predicate masks, projection demand, or output state as futures complete
    advance the per-file frontier and release state behind it
```

The scheduler should not know that the work is "zoned evidence" or "dict prepared read". It should see
resource classes, source IDs, segment requests, slot counts, cancellation state, and priorities.
The per-scan runtime maps layout-specific plan behavior into those generic scheduler inputs.

## Cancellation

`ScanTicket` owns a cancellation token.

```rust
impl ScanTicket {
    pub fn cancel(&self);

    pub fn is_cancelled(&self) -> bool;
}
```

Cancellation should happen when:

- The engine drops the stream.
- A limit has been satisfied.
- A scheduler admission wait is cancelled.
- The host engine explicitly cancels the query.

Queued work must observe the ticket before starting. Running work must release permits on drop.

## Metrics

The scheduler should expose per-scheduler and per-scan metrics:

- Permit wait time by resource.
- Morsels admitted, completed, cancelled, and skipped.
- Per-scan queue/admission delay.

Later byte/task accounting should add bytes reserved, peak bytes reserved, I/O operations admitted,
and decode tasks admitted.

DataFusion should attach these to the existing scan metrics path where possible. DuckDB can expose
them through tracing or debug logs first.

## Implementation Plan

1. Add scheduler API to `vortex-scan`.
   Include `ScanScheduler`, `ScanSchedulerConfig`, `ScanSchedulerProvider`, `ScanSchedulerSession`,
   `ScanTicket`, `WorkRequest`, and `WorkPermit`.

2. Wire the scan to register one ticket per `DataSource::scan` call.
   Store the ticket and scheduler in the `DataSourceScan` so all partitions from the same scan
   share one resource view.

3. Add permits around morsel execution.
   Start with one scheduler slot per in-flight morsel. Do not require `PreparedRead`,
   `PreparedEvidence`, or `PreparedAggregate` to expose cost estimates in the MVP. Keep byte accounting and output batch
   memory accounting out of the MVP.

4. Add DataFusion builder controls.
   Support shared scheduler and per-query scheduler modes on `VortexDataSource`, `VortexTable`, and
   `VortexFormatFactory`.

5. Add DuckDB global scheduler.
   Store a shared scheduler in the extension's global `VortexSession`.

6. Add benchmark env vars.
   Use them to compare unbounded, shared, and per-query scheduler modes under TPC-H and ClickBench.

7. Add fairness and cancellation tests.
   Tests should cover permit release on stream drop, per-scan windows, shared limits across two
   scans, and per-query isolation.

8. Add scheduler-scoped segment source registration.
   A prepared `VortexFile` should register its opened segment source and keep a bound segment table
   that maps `SegmentId` to `SegmentInfo`. Physical byte locations stay inside the registered
   source.

9. Add a scheduler-owned in-flight segment future cache.
   Key it by `(SegmentSourceId, SegmentId)`. `ScheduleCtx::request_segment` should synchronously
   submit or reuse the logical segment request and return a shared future for the segment payload.

10. Convert prepared-handle execution to scheduled morsel future construction.
    `PreparedRead` and `PreparedEvidence` should expose synchronous future constructors for explicit
    morsel ranges. Constructing the future registers all segment futures it will await. The drive
    can then construct work ahead until byte/frontier/memory thresholds are full and decide which
    futures to poll.

11. Route segment requests through source batches.
    The per-scan runtime should dedupe logical segment reads, group pending requests by
    `SegmentSourceId`, and submit ordered batches to the source. The default source adapter should
    wrap `VortexReadAt` and own physical coalescing using that backend's `coalesce_config()` and
    `concurrency()`.

12. Move segment-cache lookup into the segment request path.
    Add `SegmentCacheKey` to cacheable `SegmentRequest`s and implement a cached source adapter that
    checks the segment cache before submitting physical I/O, then stores exact segment slices on
    misses. Preserve initial-read cache seeding during file preparation.

13. Split whole-morsel execution into pipeline work.
    Add explicit morsel states for evidence, residual predicate reads, projection, emit, and
    release. Use observed selectivity and I/O cost to reprioritize predicate work within each scan.

14. Drive the morsel frontier.
    Track completed/pruned morsels per file and call prepared-read/scan-plan release hooks as the
    contiguous frontier advances.

15. Add strict end-state segment resolution.
    Add exact request introspection and a scheduler-backed read context. In strict mode, execution
    reads only submitted segments from that context; undeclared segment reads are errors or explicit
    late requests with metrics. Start with flat, zoned, dictionary, struct/apply, and chunked
    composition, then remove late fallback from core layouts.

## Open Questions

- Should the default scheduler remain unbounded permanently, or should ScanPlan scans eventually use bounded
  defaults?
- How should DataFusion propagate one per-query scheduler across several Vortex scan plans in the
  same physical plan?
- Should scheduler config be part of the public stable scan API or remain integration-specific until
  the ScanPlan scan is more mature?
- How should output batch memory be accounted once ownership moves into DataFusion or DuckDB?
- Should segment cache memory share the scheduler's decoded/intermediate budget, or have a separate
  cache budget coordinated by the same scheduler?
- Should `SegmentSourceId` be strictly scan-local, or should a shared scheduler expose optional
  cross-scan source keys for deduping reads against the same opened object?
- How much physical coalescing feedback should a `ScheduledSegmentSource` report back to the
  scheduler for adaptive policy and metrics?
