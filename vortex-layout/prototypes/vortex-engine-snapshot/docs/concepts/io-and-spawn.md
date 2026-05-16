# I/O and spawn primitives

> **Status:** Accepted.
> **Progress:** Concept-level explanation of how operators wait for
> async I/O without blocking the pipeline driver. Exact APIs live in
> [Spawn primitives](../reference/spawn-primitives.md).
> **Open questions:** admission control for `spawn_io` cost hints;
> hierarchical I/O brokers for object-store key scope.

## The shape of work

The pipeline driver inside a worker runs synchronously: the
source's `poll_next` → each transform's `push_input` /
`poll_next_output` → the sink's `poll_send`, all on one thread for
the pipeline's lifetime. Inside this loop an operator must never
block.

When work has to wait — for an object-store read, a downstream
resource, a custom future — the operator offloads it and returns
to the driver. The runtime exposes two offload primitives on
every operator's ctx (`SourceCtx`, `TransformCtx`, `SinkCtx`):

```rust
let handle: WorkHandle<T> = ctx.spawn(future);
let handle: WorkHandle<T> = ctx.spawn_io(future, IoCost::bytes(N));
```

The operator stores `handle` on its `LocalState` and polls it on
later ticks. When the handle is `Pending`, the operator returns
`Poll::Pending` and registers the standard Rust `Context` /
`Waker` plumbing on `ctx.cx()`; the driver suspends; the
executor re-polls when the waker fires.

Dropping a `WorkHandle` abandons the result. The spawned task
still runs to completion — there is no cancellation in the
engine.

## When to use which primitive

- **`spawn`** — any async work that isn't specifically I/O-bound.
  Waiting on a barrier, a downstream channel, a custom future.
- **`spawn_io`** — async I/O. Routed through `DriverIo`, the
  per-process I/O substrate. The runtime may eventually use the
  `IoCost` hint to admit work against a memory or bandwidth
  budget; today the hint is recorded but unused.

CPU-heavy work that the operator wants to perform should run
inline in its `poll_*` body. The pipeline driver runs on a
dedicated worker thread, so synchronous CPU bursts don't block
any cooperative executor.

## Thread-per-core

The runtime pins one OS thread per physical core. Each thread
runs a `smol::LocalExecutor` that drives the pipeline futures.
`DriverIo` is the per-process async I/O handle; operators reach
it through `ctx.spawn().spawn_io(...)`.

```text
core 0          core 1          core 2          core 3
┌────────┐      ┌────────┐      ┌────────┐      ┌────────┐
│Executor│      │Executor│      │Executor│      │Executor│
└────────┘      └────────┘      └────────┘      └────────┘
   │               │               │               │
   └───────────────┴───────────────┴───────────────┘
                   shared work pool
                          │
                       DriverIo
```

Pipelines are sync closures dispatched to the work pool. Each
closure runs `block_on(<pipeline future>)` on the worker that
picks it up; the worker is dedicated to that pipeline for its
lifetime. Async waits inside the pipeline park the `block_on`.

## Send discipline

The runtime is `Send`-strict. Operator state must be `Send` so
the worker pool can pick up any pipeline on any thread. `!Send`
substrates (libraries that need thread affinity) stay inside
`DriverIo`, not inside operator code.

Spawned futures must also be `Send`; they may migrate across
`DriverIo` worker threads.

## Where to read next

- [Spawn primitives](../reference/spawn-primitives.md): exact
  `SpawnRuntime`, `WorkHandle`, `IoCost` signatures.
- [Runtime architecture](../architecture/runtime.md): how the
  worker pool, executor, and barrier registry compose.
- [Execution model](execution-model.md): how pipelines run on top
  of these primitives.
