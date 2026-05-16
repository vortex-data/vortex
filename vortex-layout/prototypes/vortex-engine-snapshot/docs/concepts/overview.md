# Engine overview

> **Status:** Accepted.
> **Progress:** Entry point for an advanced developer who wants to
> understand what `vortex-engine` is and how its pieces fit
> together. Concept-level only; precise APIs live in
> `docs/reference/`.
> **Open questions:** filter catalogue extensions and spill protocol
> are tracked in [Implementation plan](../implementation/roadmap.md).

`vortex-engine` is a single-node, scheduler-driven query engine for
ordered Vortex-native data. It targets workloads where sort-merge,
top-K, ordered aggregation, and offset-driven structural joins
dominate, and where most query time is spent on I/O or decode that
can be avoided once the plan is small enough.

The docs are organised in three reading levels:

- **Concepts** (this directory): the mental model.
- **Architecture** (`docs/architecture/`): how the runtime is built.
- **Reference** (`docs/reference/`): exact APIs, ABIs, and
  operator contracts.

Start here, then descend.

## What the engine does

A planner submits a `PhysicalPlan` rooted at an `Operator`. The
engine **lowers** the plan into a DAG of single-input **pipelines**
linked by **barriers** and shared **resources**, then runs the
pipelines on a pool of worker threads.

```text
PhysicalPlan ──lower──▶ Pipeline DAG ──run──▶ Result
                  │
                  └─ source → transform → … → sink
                     (linear, direct method calls within a pipeline)
```

Each pipeline is a flat linear chain:

```
SourceNode → TransformNode → … → SinkNode
```

Inside a pipeline the source, transforms, and sink talk through
direct method calls — no channels, no async, no wakers. Pipelines
exchange data with each other through two mechanisms:

- **Barriers**: a downstream pipeline cannot start until the
  upstream barrier publishes. Used for build-then-probe patterns.
- **Resources**: typed shared state captured by closures in a sink
  on one side and a source or transform on the other side. Used
  for hash tables, sorted runs, runtime filters, finalized
  aggregates.

Multi-input operators emit multiple pipelines from a single
`lower` call. The runtime never sees the multi-input concept; it
sees a DAG of single-input pipelines linked by barriers.

## What the engine deliberately does not do

The engine keeps its scope narrow:

- **No row demand.** The batch type carries no demand mask. Work
  reduction upstream is expressed by static plan push-down or by
  typed [runtime filters](runtime-filters.md).
- **No SQL parsing, no logical optimization.** A planning frontend
  produces operators.
- **No distributed execution.** The local pipeline DAG is the unit
  a future shuffle layer would compose.
- **No expression runtime.** Compute is delegated to Vortex's
  array kernels; the engine only orchestrates batches between
  operators.

## Where work goes when it has to wait

Inside `poll_next` / `push_input` / `poll_send`, an operator may
need to wait for I/O, a downstream resource, or a future. The
runtime exposes two offload primitives on every operator's ctx
(`SourceCtx`, `TransformCtx`, `SinkCtx`):

- `ctx.spawn(future)` — generic async work.
- `ctx.spawn_io(future, IoCost::bytes(N))` — async I/O routed
  through `DriverIo`.

Both return a `WorkHandle<T>`. The operator stores the handle on
its `LocalState` and polls it on later ticks. The pipeline driver
itself never blocks the executor: when an operator returns
`Poll::Pending`, its driver suspends and is re-polled when the
relevant waker fires. CPU-heavy work runs inline in the operator's
poll body — the pipeline driver runs on a dedicated worker thread,
so synchronous CPU bursts do not block any cooperative executor.

Dropping a `WorkHandle` abandons the result. The spawned task
still runs to completion — the engine has no cancellation.

See [I/O and spawn](io-and-spawn.md).

## The three planes

The engine separates three concerns:

```text
Plan plane          — Operator + lower(): the submitted plan shape.

Pipeline plane      — Source/Transform/Sink driver state machines:
                      what the runtime actually runs.

Materialization     — Lazy Vortex array compute: deferred decode,
plane                 encoded kernels, late materialization. Runs
                      inside one operator's poll when an array is
                      finally read.
```

Keeping these distinct is the load-bearing modeling decision. The
plan plane is what a planner produces. The pipeline plane is what
the worker pool drives. The materialization plane is Vortex's job
and runs inside operator code without engine involvement.

## Why this shape

The engine targets two characteristics of Vortex workloads:

1. **Per-batch overhead matters.** At Arrow/DuckDB batch sizes
   (~8k rows) per-batch compute is microsecond-scale. Async
   channel transitions and waker registration at hundreds of
   nanoseconds per hop add up over millions of batches.
   Direct-call pipelines eliminate that overhead inside a pipeline;
   channels appear only at pipeline boundaries.

2. **Ordered, structural data rewards static plan rewrites.**
   Predicate pushdown, limit-aware pruning, top-K thresholds, and
   sort-merge cursor exchange are expressible at plan time or as
   typed runtime filters published as `Resource`s. The engine does
   not need a per-batch demand mechanism on top.

## Where to read next

- [Execution model](execution-model.md): pipelines, lowering,
  barriers, resources.
- [Runtime filters](runtime-filters.md): the typed-filter
  mechanism for sideways information passing.
- [I/O and spawn](io-and-spawn.md): how operators wait for async
  work.
- [Runtime architecture](../architecture/runtime.md): how the
  runtime is built.
- [Lowering API](../reference/lowering-api.md),
  [Runtime traits](../reference/runtime-traits.md),
  [Spawn primitives](../reference/spawn-primitives.md): exact APIs.
- [Implementation plan](../implementation/roadmap.md): what is
  built, what is next.
