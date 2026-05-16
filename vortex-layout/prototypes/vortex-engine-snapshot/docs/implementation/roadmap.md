# Implementation plan

> **Status:** Current forward plan.
> **Progress:** Tracks the path from the current prototype to the
> first usable single-node release. Milestones are large and may
> overlap when overlap reduces risk; each exits on a concrete
> validation, not a date.
> **Open questions:** none at the milestone level — open questions
> per milestone are recorded in the milestone notes.

Status legend:

- `Validated`: exit criteria met by tests, benches, or worked
  examples.
- `In progress`: scaffolding exists, exit criteria not yet met.
- `Planned`: design is understood, work has not started.

## Milestone 1 — Lowering and pipeline runtime

**Status:** Validated.

The lowering API, runtime traits, barrier registry, and the
single-threaded pipeline executor are in tree and exercised by
the worked examples under `src/bin/`.

Exit criteria — met:

- `Operator::lower` returns a `LoweredPlan` for streaming,
  pipeline-breaker, and multi-input shapes.
- `run_plan_blocking` runs a plan to completion using
  `SpawnRuntime` + `DriverIo` + the work pool.
- Barriers correctly serialize build-then-probe across pipelines.

## Milestone 2 — Filter API and SIP catalogue

**Status:** In progress.

Land the typed-filter mechanism described in
[Runtime filters](../concepts/runtime-filters.md) and
[Runtime filters reference](../reference/runtime-filters.md).

Scope:

1. **Resource API for filters.** Define `FilterResource<F>` with
   `publish` / `tighten` / `try_snapshot` /
   `wait_for_publication`. Publication wakes consumers via the
   existing barrier infrastructure.
2. **`RangeFilter<K>`.** Initial filter type. Published by
   `Limit`; consumed by `IntSource` and `VortexScanSource`.
3. **Translator hook.** `ParentChildMin` translates a
   `RangeFilter<ParentIdx>` into a `RangeFilter<ChildIdx>` using
   its offset table. The translator runs as a small transform
   placed between the parent's filter resource and the child
   source's subscription.
4. **`BloomFilter<K>` and `KeyListFilter<K>`.** Add once the first
   hash join operator lands (see Milestone 4).

Exit criteria:

- A `Limit`-bearing query has its source clamp reads when the
  filter is published, without per-plan rewrites at the source.
- Filter publication and consumption appear in traces.
- A no-filter execution path is unchanged in cost.

## Milestone 3 — Pipeline runtime hardening

**Status:** In progress.

Move the runtime from "single-threaded `block_on` per pipeline"
toward production-grade throughput.

Scope:

- per-thread pinned executors (N = physical cores) with the work
  pool dispatching pipelines to them;
- bounded MPMC work-stealing channel exchanges between pipelines,
  built on `crossbeam-deque::Worker` + `Stealer` per
  [ADR 0008](../decisions/0008-mpmc-work-stealing-channels.md);
- back-pressure on `ChannelExchangeSink::poll_send`;
- pipeline-level parallelism: lanes per node, intersected by the
  pipeline's `Parallelism::LaneSafe { max_lanes }` declaration;
- structured error paths from pipelines back to the runtime entry
  point.

Exit criteria:

- Multi-pipeline plans run on multiple worker threads
  concurrently.
- Channel back-pressure is visible in traces and tested under
  producer/consumer mismatch.
- A lane-scaling test where a `LaneSafe { max: None }` pipeline
  matches single-lane throughput at one lane and exceeds it
  proportionally at N lanes.

## Milestone 4 — Operator inventory expansion

**Status:** Planned.

Bring the operator inventory up to coverage parity for the target
workloads. Order targets workloads, not difficulty.

Wave A — read path:

- `Filter` (predicate transform);
- `Project` (multi-column projection transform);
- `HashAggregate` build + finalize as one operator emitting two
  pipelines linked by a barrier;
- `TopK` (sorted heap sink, publishes `RangeFilter<RowIdx>`).

Wave B — join path:

- `Sort` decomposed as `SortRunSink` + `MergeKSource`;
- `HashJoin` (build + probe) emitting two pipelines; publishes a
  `BloomFilter` for probe-side push-down;
- `PreSortedMergeJoin` reading two `Resource`-backed run streams.

Exit criteria:

- ClickBench Q1, Q3, Q5, Q20 run end-to-end at competitive
  throughput on the same shards.
- One TPC-H join query (Q1 or Q6) runs end-to-end on the engine.
- Every new operator has a worked example under
  `src/physical_plan/` or `tests/`.

## Milestone 5 — Memory and admission

**Status:** Planned.

Once Milestones 3 and 4 are in place, the runtime needs visibility
into memory across pipelines.

Scope:

- task-wide memory arbiter accounting for channel buffers,
  resource payloads, and operator-local state;
- byte-budgeted channel exchanges with the arbiter granting and
  shrinking capacity under pressure;
- cooperative spill protocol for `Sort`, `HashJoin` build, and
  `HashAggregate` build: each spill-capable operator implements
  flush-and-resume on a pressure signal;
- I/O broker layer on top of `SpawnRuntime`: priority heap,
  coalescing of overlapping reads, admission keyed by backend /
  host / route.

Exit criteria:

- A query whose working set exceeds memory completes correctly
  via spill.
- I/O traces show coalescing and admission ordering.
- Arbiter releases buffers under memory pressure visible in the
  trace viewer.

## Milestone 6 — Host adapter surface

**Status:** Planned.

Make the engine embeddable.

Scope:

- minimal external entry point: submit a `PhysicalPlan`, receive
  a `Stream<Batch>` of results;
- runtime configuration: worker count, memory limit, I/O
  substrate selection;
- cancellation through plan teardown (operators don't cancel;
  teardown releases resources);
- metrics export.

Exit criteria:

- An external Rust host wraps the engine and runs one of the
  worked queries.
- Runtime config knobs exercised in tests.

## What is intentionally out of scope

- Distributed execution / shuffle. The engine is local; an
  exchange operator is the boundary at which a future shuffle
  layer would compose task-local fragments.
- Logical optimization and SQL parsing. Frontends produce
  `PhysicalPlan`.
- Operator cancellation. Dropping a `WorkHandle` abandons the
  result; there is no per-row cancellation.
- An expression runtime. Compute is Vortex's job.

## Tracking

Per-milestone open questions and concrete TODOs live in
[Documentation TODOs](todo.md). The current code map is in
[Current scaffold](current-scaffold.md).
