# Scan Scheduler

:::{note}
This is an implementation design for the ScanNode-backed scan path. It describes the scheduler
shape the V2 scan should grow into, not the current behavior of the released scan API.
:::

The ScanNode scan path needs a resource scheduler that can coordinate work across files, partitions,
and concurrent scans. The scheduler should be explicit and embeddable: a host engine can share one
scheduler across many scans to enforce global limits, or create a fresh scheduler for each query to
isolate resource usage.

The design uses one shared `ScanScheduler` object for resource arbitration and one per-scan runtime
for query semantics.

The existing `DataSource` / `ScanRequest` / `DataSourceScan` API remains the public query-engine
boundary for this phase. The scheduler and morsel runtime sit behind that boundary, so the first
implementation can improve V2 execution without introducing a second scan API that mostly duplicates
the current one.

## Goals

- Bound scan resource usage across concurrent scans.
- Allow DataFusion users to choose a shared scheduler, a new scheduler per query, or an unbounded
  mode.
- Give DuckDB a simple global scheduler owned by the extension session.
- Keep ScanNode planning and morsel ordering local to each scan.
- Make cancellation and permit release reliable when a stream is dropped early.
- Keep scheduler APIs independent of layout internals so other `DataSource` implementations can use
  the same resource controls.

## Non-goals

- Do not make a process-global singleton the only way to schedule scans.
- Do not put query semantics, filter ordering, evidence planning, or output ordering into the global
  scheduler.
- Do not replace the `DataSource` scan API in the first scheduler implementation. If the public API
  changes later, it should be because the V2 runtime needs capabilities that cannot be added
  compatibly to `ScanRequest` or `DataSourceScan`.
- Do not require every scan integration to expose the same configuration surface immediately.
- Do not solve cluster-wide distributed admission control. The scheduler is process-local.

## Core Model

There are three layers:

1. `ScanScheduler`
   Arbitrates global resources such as I/O bytes, decoded bytes, request concurrency, decode task
   concurrency, and per-scan fairness.

2. `ScanTicket`
   Represents one logical scan registered with a scheduler. It carries scan identity, cancellation,
   priority, metrics, and per-scan limits.

3. Per-scan `MorselScanRuntime`
   Owns the ScanNode graph, evidence/read/aggregate plans, morsel queue, row ordering, limit
   handling, dynamic filters, and the choice of which work is useful next.

`DataSource::scan` constructs this per-scan runtime internally and returns the existing
`DataSourceScan` wrapper. Query engines should not need to know whether a data source is implemented
by the legacy `LayoutReader` path or the V2 ScanNode runtime.

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

The default should be `Unbounded` initially, so enabling the V2 scan does not silently introduce new
resource limits. Integrations can opt into bounded scheduling explicitly.

The scheduler types should live in `vortex-scan`, not `vortex-layout`, because the resource policy
belongs to the scan API layer and should be reusable by non-layout sources. ScanNode-specific code in
`vortex-layout` can consume tickets and permits through the public scan scheduler API without making
the scheduler understand layout-specific node types.

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

For the current V2 DataFusion path, `DataSource::open` creates a single Vortex scan for partition
zero. A per-query scheduler can therefore be resolved immediately before calling
`DataSourceRef::scan`. If DataFusion later produces multiple Vortex scan nodes for one query and
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
VORTEX_SCAN_IMPL=v2
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

The first implementation should not require every `ReadPlan`, `EvidencePlan`, or `AggregatePlan`
to expose pending I/O, decoded-size estimates, or cost statistics. Those estimates are useful, but
they are also hard to get right and would make the initial ScanNode API more rigid. The V2 runtime
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
pub struct PlanCostHint {
    pub estimated_io_bytes: Option<u64>,
    pub estimated_decoded_bytes: Option<u64>,
    pub estimated_cpu_units: Option<u64>,
}
```

If those hints are added, they should remain advisory. A plan that does not provide hints should
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

## Resources to Control

The first implementation should control:

- Maximum morsels in flight per scan.
- Maximum morsels in flight across a shared scheduler.

This intentionally approximates the current scan behavior: unordered scans can run several morsels
concurrently, while ordered scans and scans with a pushed-down limit should run with a narrower
window. The default should mirror the existing `ScanBuilder` concurrency factor:

```text
unordered/no-limit: max_morsels_in_flight = 4 * available_parallelism
ordered or limit:   max_morsels_in_flight = 1
```

The shared scheduler can apply the same window globally, per scan, or both. For example, a
DataFusion user can choose one shared scheduler with `4 * available_parallelism` total morsel slots
to cap the whole process, or create a new scheduler per query to preserve the old per-query
behavior.

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

The V2 scan should move toward an explicit per-scan runtime:

```rust
pub struct MorselScanRuntime {
    scheduler: Arc<ScanScheduler>,
    ticket: ScanTicket,
    plan: ScanRuntimePlan,
    state: ScanRuntimeState,
}
```

`ScanRuntimePlan` is internal to the V2 implementation. It contains the files, expanded ScanNode
trees, pushed expressions, evidence plans, read plans, aggregate plans, and reusable per-file state.
It is not a replacement public scan API.

Execution loop:

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

The scheduler should not know that the work is "zoned evidence" or "dict read plan". It should only
see resource classes, slot counts, and cancellation state in the MVP. Later versions can add byte
estimates, CPU estimates, and priorities as advisory hints.

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

2. Wire the V2 scan to register one ticket per `DataSource::scan` call.
   Store the ticket and scheduler in the V2 `DataSourceScan` so all partitions from the same scan
   share one resource view.

3. Add permits around V2 morsel execution.
   Start with one scheduler slot per in-flight morsel. Do not require `ReadPlan`, `EvidencePlan`,
   or `AggregatePlan` to expose cost estimates in the MVP. Keep byte accounting and output batch
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

## Open Questions

- Should the default scheduler remain unbounded permanently, or should V2 eventually use bounded
  defaults?
- How should DataFusion propagate one per-query scheduler across several Vortex scan nodes in the
  same physical plan?
- Should scheduler config be part of the public stable scan API or remain integration-specific until
  the V2 scan is more mature?
- How should output batch memory be accounted once ownership moves into DataFusion or DuckDB?
- Should segment cache memory share the scheduler's decoded/intermediate budget, or have a separate
  cache budget coordinated by the same scheduler?
