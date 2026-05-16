# Current scaffold

> **Status:** Current implementation-state page.
> **Progress:** Code map for the engine implementation. Target
> behaviour belongs in the design docs; the forward plan lives in
> [Implementation plan](roadmap.md). This page is a "where the
> code is" pointer.
> **Open questions:** items called out under each module.

The engine implementation lives under `src/physical_plan/`.

## Layout

### `src/physical_plan/`

- `plan.rs` — `Operator` trait and `PhysicalPlan`.
- `lowering.rs` — `LoweringCtx`, `PipelineTail`,
  `PipelineBuilder`, `LoweredPlan`, `Pipeline`,
  `PipelineSource` / `PipelineTransform` / `PipelineSink`.
- `abi.rs` — runtime traits and types: `Batch`, `Parallelism`,
  `SourceNode`, `TransformNode`, `SinkNode`, `OperatorPoll`,
  `PendingSend`, `TransformOutput`, ctx types
  (`SourceCtx`/`TransformCtx`/`SinkCtx`), plus the `Dyn*` /
  `*Driver` type-erasure wrappers.
- `ids.rs` — `PipelineId`, `PipelineBarrier`.
- `spawn.rs` — `SpawnRuntime`, `WorkHandle<T>`, `IoCost`,
  `Priority`.
- `driver_io.rs` — `DriverIo`, the smol-based async I/O substrate.
- `pool.rs` — work pool (`Runtime`) that hosts pipeline drivers.
- `runtime.rs` — pipeline driver loop, `BarrierRegistry`,
  `LatchBarrier`, `run_plan_blocking[_with_io]`.
- `submitter.rs` — dispatches each pipeline as a sync closure
  onto the work pool.
- `error.rs` — `BuildError`, `PlanValidationError`, `BuildResult`.

### Operator inventory

Each is one `Operator::lower` plus its `Source/Transform/Sink`
nodes.

- `operators.rs` — `IntSource`, `CollectSink`, `ProjectOne`.
- `limit.rs` — `Limit` (transform).
- `parent_child_min.rs` — `ParentChildMin` (pipeline-breaker:
  consumes a child stream, publishes per-parent min into a
  resource, then sources from it).
- `gather.rs` — `Gather` (one output pipeline draining multiple
  input pipelines).
- `merge_join.rs` + `merge_join_resource.rs` — `SortedMergeJoin`
  (multi-input: left + right build pipelines + output pipeline).
- `sum_aggregate.rs` — `SumCountAggregate` (pipeline-breaker for
  a streaming sum/count).
- `vortex_scan.rs` — `VortexScanSource` (Vortex file source
  reading one column projection).
- `vortex_aggregate.rs` — `VortexAggregate` (Vortex `AggregateFn`
  driven via the engine).

### Worked binaries

Under `src/bin/`:

- `v2_q3` — ClickBench-style avg using `VortexScanSource` +
  `VortexAggregate`.
- `v2_spiral` — three-level parent/child/grandchild query, scan
  everything path (no filter).
- `v2_spiral_pushdown` — same query with plan-time push-down on
  the grandchild range.
- `v2_avg`, `v2_sum_event_diff`, `v2_bench` — smaller harnesses.

(The `v2_` prefix is historical — these are the engine binaries.)

## How to update this page

When a file lands or an operator gains conformance, add the file
and link the operator's spec page (once those exist). Performance
numbers should reference a reproducible binary, not a one-off
measurement.
