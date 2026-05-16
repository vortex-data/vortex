# Glossary

> **Status:** Accepted vocabulary.
> **Progress:** Defines the engine's terms and the runtime types
> they correspond to. Add a term here when a subsystem doc
> introduces a new accepted name.
> **Open questions:** none.

## Operator

A plan-time description of one logical processing step. `Operator`
is a `Send + Sync` trait with one method:

```rust
fn lower(&self, ctx: &mut dyn LoweringCtx, tail: PipelineTail) -> BuildResult<()>;
```

An operator either prepends a transform onto the tail, completes
the tail at a source via `ctx.emit_pipeline`, or splits it
(multi-input or pipeline-breaker by emitting more pipelines linked
by barriers).

## PhysicalPlan

A submitted plan, owning a root `Operator` and an `OutputContract`
describing the plan's overall output stream.

## LoweringCtx

The plan-time context handed to `Operator::lower`. Exposes
`register_domain`, `new_pipeline_barrier`, and `emit_pipeline`.

## PipelineTail

A continuation that flows from the sink toward the sources during
lowering. Carries the expected input domain and contract, the
accumulated transforms, declared barrier dependencies
(`depends_on`), and declared barrier publications (`publishes`).

## Pipeline

A flat linear chain emitted by `ctx.emit_pipeline`:
`source → transform₁ → … → sink`. Inside a pipeline there are no
channels — operators talk through direct method calls. Pipelines
form a DAG linked by barriers and channel exchanges.

## PipelineBarrier

A sticky one-shot event keyed by id. A sink that declares
`publishes(barrier)` fires the barrier when `poll_finish`
completes. A pipeline that declares `depends_on(barrier)` does
not start until the barrier fires.

## SourceNode / TransformNode / SinkNode

The three runtime trait types. Each carries an associated
`LocalState: Send + 'static`. The runtime type-erases them into
`DynSourceNode` / `DynTransformNode` / `DynSinkNode` for the
driver wrappers.

- `SourceNode::poll_next` is poll-style and may return `Pending`.
- `TransformNode` is push-pull: `can_accept_input`, `push_input`,
  `finish_input`, `poll_next_output`.
- `SinkNode::poll_send` and `poll_finish` are poll-style.

## Batch

A vectorized unit of execution carrying a Vortex `ArrayRef` and a
`DomainSpan` over the endpoint domain. Every row a batch carries
is real.

## OperatorPoll

```rust
pub type OperatorPoll<T> = Poll<EngineResult<T>>;
```

The poll result type returned by `poll_next`, `poll_next_output`,
`poll_send`, and `poll_finish`.

## TransformOutput

```rust
pub enum TransformOutput { Batch(Batch), NeedInput, Finished }
```

Returned by `TransformNode::poll_next_output`. `Batch(b)` yields a
batch downstream; `NeedInput` asks the driver for more input;
`Finished` ends the transform's output.

## Resource

Typed shared state captured by closures in a sink (writer) and a
downstream source or transform (reader). Plain `Arc<…>` with
internal synchronization. Used for hash tables, sorted runs,
finalized aggregates, and **runtime filters**.

## Runtime filter

A typed `Resource` published by one operator and consumed by
another for sideways information passing. Catalogue:
`RangeFilter<K>`, `BloomFilter<K>`, `KeyListFilter<K>`. See
[Runtime filters](concepts/runtime-filters.md).

## SpawnRuntime

The offload primitives. `ctx.spawn(future)` and
`ctx.spawn_io(future, cost)` return a `WorkHandle<T>` the operator
polls on later ticks.

## WorkHandle

A handle to spawned async work. The owning operator polls the
handle each tick until it returns `Ready`. Dropping a handle
abandons the result; the spawned work still runs to completion.

## DriverIo

The per-process async I/O substrate, backed by a smol executor
with N pinned worker threads. Reached through `SpawnRuntime::io()`.

## LoweredPlan

The artifact produced by lowering: a vector of `Pipeline`s and a
map of registered `Domain`s. Run via
`runtime::run_plan_blocking(plan)`.

## PipelineBuilder

The concrete `LoweringCtx` implementation that captures emitted
pipelines and produces a `LoweredPlan`.

## Domain / DomainId / DomainSpan

A stable ordinal row address space. Batches refer to rows by
`DomainSpan` (half-open `[start, end)`) within a `Domain`.

## OutputContract

Declared stream shape of a domain endpoint: schema (`DType`) and
ordering.

## Channel exchange

A bounded MPMC work-stealing channel between two pipelines, built
on `crossbeam-deque::Worker` per producer lane and `Stealer` per
consumer lane. Byte-counter-bounded. Surfaces in operator code as
a `ChannelExchangeSink` / `ChannelExchangeSource` pair. See
[ADR 0008](decisions/0008-mpmc-work-stealing-channels.md).

## Parallelism

A pipeline node's lane declaration:

```rust
pub enum Parallelism {
    Serial,
    LaneSafe { max_lanes: Option<usize> },
}
```

`Serial` means one lane only. `LaneSafe { max_lanes }` means the
runtime may spawn up to `max_lanes` lanes (or unbounded when
`None`). The pipeline's parallelism is the intersection of its
nodes' declarations.

## Task

The local owner of one execution. Owns the lowered plan, the
worker pool, the `BarrierRegistry`, the `SpawnRuntime`,
cancellation state, and metrics.
