# Spawn primitives

> **Status:** Accepted.
> **Progress:** Authoritative reference for `SpawnRuntime`,
> `WorkHandle<T>`, `IoCost`, and `Priority`. Implementing code lives
> in `src/physical_plan/spawn.rs`.
> **Open questions:** admission control sized by `IoCost` is recorded
> but not yet acted on; a future I/O substrate will use it.

`SpawnRuntime` is the offload surface every operator ctx exposes.
Operators that need to wait for async I/O or arbitrary futures
spawn the work, store the returned handle on their `LocalState`,
and poll it on later ticks.

## `Priority`

```rust
#[derive(Clone, Copy, Default, PartialEq, Eq, PartialOrd, Ord)]
pub enum Priority {
    Low,
    #[default]
    Normal,
    High,
}
```

Priority hint attached to spawned work. Recorded but not yet used
to rank work — a future scheduler will.

## `IoCost`

```rust
pub struct IoCost {
    pub estimated_bytes: u64,
    pub priority: Priority,
}

impl IoCost {
    pub const fn bytes(estimated_bytes: u64) -> Self;
}
```

Cost hint for `spawn_io`. The runtime may use this to back-pressure
I/O against a memory or bandwidth budget; today the hint is
recorded but unused.

## `WorkHandle<T>`

```rust
pub struct WorkHandle<T> { /* rx: oneshot::Receiver<EngineResult<T>> */ }

impl<T> WorkHandle<T> {
    pub fn poll(&mut self, cx: &mut Context<'_>) -> Poll<EngineResult<T>>;
    pub fn try_take(&mut self) -> Option<EngineResult<T>>;
}
```

The owning operator polls `handle.poll(ctx.cx())` on each tick
until it returns `Ready`. Dropping a `WorkHandle` without polling
is fine — the spawned work runs to completion and the result is
discarded.

## `SpawnRuntime`

```rust
#[derive(Clone)]
pub struct SpawnRuntime { /* io: Arc<DriverIo> */ }

impl SpawnRuntime {
    pub fn new(io: Arc<DriverIo>) -> Self;
    pub fn io(&self) -> &Arc<DriverIo>;

    pub fn spawn<F, T>(&self, future: F) -> WorkHandle<T>
    where
        F: Future<Output = EngineResult<T>> + Send + 'static,
        T: Send + 'static;

    pub fn spawn_io<F, T>(&self, future: F, cost: IoCost) -> WorkHandle<T>
    where
        F: Future<Output = EngineResult<T>> + Send + 'static,
        T: Send + 'static;
}
```

Cheap to clone (one `Arc` bump). The runtime constructs one per
plan execution and threads `&SpawnRuntime` through every operator
ctx.

`spawn` runs the future on the `DriverIo` smol executor and
returns a handle. `spawn_io` is the same path today; the
distinction exists so a future I/O substrate (io_uring or
equivalent) can route I/O through a different path with admission
control sized by `IoCost`.

Spawned futures must be `Send`; they may migrate across `DriverIo`
worker threads.

## When to use which

- **`ctx.spawn(future)`** — any async work that isn't specifically
  I/O-bound. Waiting on a barrier, a downstream channel, a custom
  future.
- **`ctx.spawn_io(future, IoCost::bytes(N))`** — async I/O. Routed
  through `DriverIo`. Pass the estimated payload size in `cost` so
  the runtime can use it once admission control is wired up.

CPU-heavy work that the operator wants to perform should run
inline in its `poll_*` body. The pipeline driver runs on a
dedicated worker thread (see `Runtime` in `src/physical_plan/pool.rs`),
so synchronous CPU bursts don't block any cooperative executor.

## `DriverIo`

```rust
pub struct DriverIo { /* … */ }

impl DriverIo {
    pub fn new(workers: usize) -> Arc<Self>;
    pub fn executor(&self) -> &smol::Executor<'static>;
    pub fn handle(&self) -> &vortex_io::Handle;
}
```

The per-process async I/O substrate, backed by smol with `workers`
pinned threads. Reached through `SpawnRuntime::io()`. Operators
that talk to Vortex's async file API obtain a `vortex_io::Handle`
via `spawn.io().handle()`.

## Worked pattern

```rust
struct State {
    pending: Option<WorkHandle<MyResult>>,
}

fn poll_next(
    &self,
    state: &mut State,
    ctx:   &mut SourceCtx<'_, '_>,
) -> OperatorPoll<Option<Batch>> {
    if let Some(handle) = &mut state.pending {
        match handle.poll(ctx.cx()) {
            Poll::Ready(Ok(value)) => {
                state.pending = None;
                Poll::Ready(Ok(Some(make_batch(value))))
            }
            Poll::Ready(Err(e)) => Poll::Ready(Err(e)),
            Poll::Pending       => Poll::Pending,
        }
    } else {
        state.pending = Some(ctx.spawn().spawn_io(load_next(), IoCost::bytes(1 << 20)));
        // The waker is registered when we re-enter poll_next; for the
        // very first call after spawning, return Pending immediately.
        Poll::Pending
    }
}
```

## See also

- [Runtime traits](runtime-traits.md) — how operators receive
  ctxs and `LocalState`.
- [I/O and spawn (concept)](../concepts/io-and-spawn.md) — when
  to use which primitive at the design level.
- [Runtime architecture](../architecture/runtime.md) — how
  `SpawnRuntime` and `DriverIo` slot into the runtime.
