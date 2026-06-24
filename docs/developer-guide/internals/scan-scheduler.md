# Scan Scheduler

This document describes the current ScanPlan V2 scheduler and I/O pipeline. It is
an implementation guide, not a design sketch.

The scheduler is split across three layers:

- `vortex-scan::scheduler` owns the process/query-level scheduler object,
  scheduler provider, and read-byte budget configuration.
- `vortex-file::multi::scan_v2` owns the per-partition ScanPlan runtime. It
  plans morsels, queues evidence/predicate/projection work, and decides which
  queued task is useful next.
- `vortex-file::segments` and `vortex-file::read` own segment future
  registration, logical read deduplication, physical range coalescing, and
  backend request concurrency.

The global scheduler is deliberately not a central work queue. It does not know
about predicates, layouts, row masks, or query semantics. The scan runtime makes
those decisions locally, then uses scheduler-visible read bytes and task lanes to
control how much work is launched.

## Execution Shape

The normal DataFusion V2 path is:

```text
DataFusion DataSource::open(partition)
        |
        v
VortexDataSource builds ScanRequest
        |
        v
DataSourceRef::plan_morsel_partitions or DataSourceRef::scan
        |
        v
ScanSchedulerProvider::scheduler_for_scan
        |
        v
partition_work_stream
        |
        +-- plan morsels into task queues
        +-- register segment futures synchronously
        +-- admit tasks by lane/frontier/read bytes
        +-- poll task futures on the Vortex runtime
        +-- emit arrays in ordered or unordered mode
```

`DataSource::plan_morsel_partitions` is used when the engine can consume many
output partitions. It opens files, asks each prepared file for split ranges, and
round-robins planned morsels across the engine-requested partition count. Each
partition then runs its own `partition_work_stream`, but planned morsels from the
same file share the same `ScanExecution`, `SegmentFutureCache`, and
`FileSegmentSource`.

`DataSource::scan` is the fallback path. It yields file partitions and each file
partition creates its own `partition_work_stream`.

Limited scans force a morsel planning window of one because limit accounting is
owned by the scan runtime and must not speculatively consume rows far ahead of
the output frontier.

## Scheduler Objects

`ScanSchedulerConfig` currently has one enforced field:

- `read_byte_budget`: optional per-partition active logical segment-byte budget.

`ScanSchedulerProvider` chooses scheduler ownership:

- `Unbounded`: create an unbounded scheduler for the scan.
- `Shared`: reuse one `Arc<ScanScheduler>`.
- `PerScan`: create a fresh scheduler from the config for each logical scan.

The default `VortexSession` provider is `Unbounded`. DuckDB installs a shared
default scheduler in the extension session. The DataFusion benchmark only
installs a scheduler when `VORTEX_SCAN_SCHEDULER` is set.

There is no scheduler permit API in the V2 runtime. Task launch is admitted by
the per-partition `ScanTaskQueue` using active logical read bytes. Limited scans
still plan one active morsel at a time internally because limit accounting must
not consume rows far ahead of the output frontier, but that is not a public
tuning knob.

## Planning Morsels

`partition_work_stream` owns a `PartitionWorkSchedulerState`:

- `pending`: planned morsel ranges not yet converted to runtime state.
- `morsels`: active morsel states indexed by morsel id.
- `task_queue`: queued evidence, predicate, projection, and aggregate tasks.
- `in_flight`: launched task futures.
- `completed_morsels`: ordered-output buffer.
- `plan_window`: internal active planned-morsel cap. This is unbounded for
  normal scans and one for limited scans.

On each stream poll, the runtime:

1. Emits already-completed output if possible.
2. Plans more morsels while `active_morsels < plan_window`.
3. Launches admissible queued tasks until the task queue refuses more work.
4. Waits for one launched task to complete.
5. Updates evidence, predicate masks, projection state, and read accounting.

Planning a morsel is synchronous. It creates initial evidence work and then calls
`enqueue_ready_work`. For scans without predicates, projection work is queued
immediately. For filtered scans, the runtime queues evidence first, then residual
predicate reads, then projection once all predicates are proven for the morsel.

## Task Lanes

`ScanTaskQueue` groups queued work into lanes:

- `ScanEvidence`: scan-domain evidence shared by all morsels for one predicate.
- `Evidence`: morsel-local evidence for one predicate.
- `Predicate`: exact residual predicate evaluation.
- `Projection`: final projected values.
- `Aggregate`: aggregate reads, grouped with projection.

Admission is not FIFO across all work. The queue tries groups in this order:

1. Evidence within its byte target.
2. Predicate within its byte target.
3. Projection within its byte target.
4. Predicate ignoring group target.
5. Projection ignoring group target.
6. Evidence ignoring group target.

All groups still obey the total read-byte budget unless the task contributes no
new bytes or the runtime has no launched work at all. The empty-in-flight escape
hatch prevents deadlock when one task is larger than the configured budget.

Within a group, lower priority wins, then lower incremental read bytes, then
lower total read bytes, then lower morsel id. The incremental byte score is
important because tasks reading the same active segment can be admitted without
increasing active physical-read pressure.

There is no fixed morsel read-ahead frontier. Morsels can vary substantially in
byte size and can overlap in their segment requests, so run-ahead is governed by
incremental active read bytes rather than by a count of morsels. A later morsel
with small or already-active reads may be admitted ahead of an earlier morsel
whose reads would exceed the active byte budget.

For dynamic-predicate scans there is one extra gate: speculative projection is
suppressed while completed output is backlogged, except when there are no
launched tasks and one projection is needed to keep an ordered stream moving.
Evidence and predicate tasks are still admissible while projection is gated. This
favors avoiding wasted projection I/O over maximizing object-store request depth.

## Read-Byte Budget

`read_byte_budget` is per partition stream. It counts active logical segment
bytes for admitted tasks, deduped by `SegmentRequestKey`. If two launched tasks
await the same segment, only the first contributes bytes; the active entry keeps
a reference count until both tasks complete.

When the budget is finite, the queue divides target bytes by group:

```text
predicate:  6/8 of budget
projection: 1/8 of budget
evidence:   1/8 of budget
```

These are soft group targets. The second pass can use any remaining total budget
for predicate, projection, or evidence, but no task can exceed the total budget
unless it is the only way to make progress.

The default bounded config uses:

```text
DEFAULT_READ_BYTE_BUDGET = 256 MiB
```

`ScanSchedulerConfig::unbounded()` leaves this unset, which becomes `u64::MAX`
inside `partition_work_stream`.

## Segment Requests

Prepared reads and evidence providers expose segment requests before task launch.
The runtime turns those requests into `ScanRead` values with:

```rust
register_segment_reads_cached(cache, source, requests)
```

This call is synchronous. For cache misses, it calls the underlying
`SegmentSource::request(segment)` immediately and stores a shared future in the
scan-local `SegmentFutureCache`. That means simply planning work registers the
logical reads with the file segment source before the task future is polled.

The cache key is currently the logical `SegmentId`. That is sufficient inside one
`ScanExecution` because each execution has one bound file segment source. It is
not a cross-file or cross-scan cache key.

`SegmentInfo` contains only logical payload `bytes`, which the task scheduler
uses for read-budget admission. Segment-cache policy is owned by the
`SegmentCacheSourceAdapter`; it is not expressed through scheduler-visible
segment metadata.

## Physical I/O

`FileSegmentSource` bridges logical segment requests to a `VortexReadAt` backend.
It has an internal event stream with these request states:

- registered: a segment future exists, but has not been polled;
- requested: the segment future has been polled;
- in-flight: the physical backend read has been submitted;
- resolved: the future has completed.

Registered but unpolled requests are still visible to coalescing. When one
request is polled, `IoRequestStream` picks the earliest polled request and may
coalesce nearby registered or polled requests by physical offset.

Physical coalescing is controlled by `VortexReadAt::coalesce_config()`:

```text
in-memory:       8 KiB distance,  8 KiB max
local file:      1 MiB distance,  4 MiB max
object storage:  1 MiB distance, 16 MiB max
```

Physical request concurrency is controlled by `VortexReadAt::concurrency()`:

```text
ObjectStoreReadAt default concurrency = 192
```

This concurrency is below the scan task queue. The object-store layer can only
use that depth if the scan runtime has registered and polled enough segment
futures.

## Object Store Behavior

The current object-store path has good physical defaults but no automatic scan
scheduler preset:

- `ObjectStoreReadAt` uses object-store coalescing and high physical request
  concurrency.
- DataFusion remote benchmarks create the `VortexSession` before registering the
  object store URL, so the Vortex scheduler provider cannot infer S3/GCS from
  the source URL.
- DuckDB uses a shared scheduler with the default active read-byte budget.
- DataFusion uses an unbounded scheduler unless benchmark environment variables
  opt into a scheduler.

For object stores, the main risk is not the `ObjectStoreReadAt` queue depth. It
is failing to expose enough useful segment futures early enough, or exposing far
too many tiny/sparse reads without a workload-specific budget. The important
knobs are:

- `read_byte_budget`: how many active logical segment bytes may be polled;
- physical coalescing distance/max size on the object-store reader;
- physical object-store request concurrency;
- DataFusion output partition count, which controls how many partition streams
  run at once.

## Benchmark Knobs

The DataFusion benchmark supports:

```text
VORTEX_SCAN_SCHEDULER=unbounded|shared|per-query
VORTEX_SCAN_MAX_READ_BYTES=...
```

Useful S3 sweeps should compare:

```text
# Current compatibility behavior.
VORTEX_SCAN_SCHEDULER=unbounded

# Bounded read pressure, one scheduler per query.
VORTEX_SCAN_SCHEDULER=per-query
VORTEX_SCAN_MAX_READ_BYTES=268435456

# Larger remote-storage byte window.
VORTEX_SCAN_SCHEDULER=per-query
VORTEX_SCAN_MAX_READ_BYTES=1073741824
```

An active-logical-read target was tested as an I/O-depth proxy and rejected: it
improved some FineWeb cases, but regressed local PolarSignals enough that it was
too indirect to use as a scheduler knob.

## Tuning Guidance

For local NVMe, keep the read budget moderate and rely on local filesystem
coalescing. Excessive read-ahead can increase memory pressure without hiding much
latency.

For S3/GCS, prefer a larger byte budget so the file segment source can keep more
useful logical reads active and coalesce adjacent registered requests.
If a query is highly selective and projection reads are sparse, validate the
coalesced-byte metrics before increasing the object-store coalescing max size.
If dynamic predicates are active, also compare projection-gated behavior against
object-store request depth: the gate is intended to avoid wasted projection I/O,
but it can reduce S3 latency hiding for projection-light queries.

Use scan metrics to separate three failure modes:

- low object-store request concurrency: not enough futures are being polled;
- low coalescing: not enough adjacent futures are registered before polling;
- excessive over-read: coalesced requests are much larger than useful projected
  segment bytes.

The scheduler today cannot distinguish those automatically. The next practical
tuning step is to expose byte-based controls for physical object-store
coalescing/request pressure if logical read-byte budgeting is not enough.

## Known Gaps

- The benchmark can configure scheduler mode and read-byte budget, but not
  physical object-store coalescing or request concurrency.
- There is no automatic object-store scheduler preset.
- The scan runtime accounts logical segment bytes, not physical coalesced bytes.
- Output Arrow conversion is outside the scan task queue and has separate
  buffering in the DataFusion adapter.
