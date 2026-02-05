# Runtime and Threading Model

Vortex defines its own async runtime abstraction in the `vortex-io` crate. This abstraction
decouples the rest of the ecosystem from any particular async runtime, allowing different
integrations to use the threading model that best fits their host engine.

## Runtime Abstraction

The core abstraction is the `Handle` type, a cloneable weak reference to an active runtime. All
async work in Vortex -- I/O, compute, background tasks -- is spawned through a handle. The handle
is stored in the session via `RuntimeSession` and threaded through the API alongside other session
state.

Internally, a handle wraps a trait object implementing the `Executor` trait, which defines three
spawn methods:

- **`spawn`** -- schedule an async future onto the runtime.
- **`spawn_cpu`** -- schedule a CPU-bound closure.
- **`spawn_blocking`** -- schedule a blocking I/O closure onto a dedicated blocking pool.

All spawned tasks must be `Send + 'static`. Tasks are not pinned to a particular thread and can
be picked up by any thread in the pool.

## Tokio

The `TokioRuntime` adapter wraps a `tokio::runtime::Handle` and delegates all spawning to Tokio's
thread pool. It is intended for use from applications that already run inside a Tokio context, such
as DataFusion.

```rust
let session = VortexSession::default().with_tokio();
```

When `with_tokio()` is called, the adapter captures the current Tokio runtime handle. If no Tokio
context is active, it panics. For applications that do not already use Tokio, the
`CurrentThreadRuntime` described below is preferred.

## CurrentThreadRuntime

The `CurrentThreadRuntime` (CRT) is built on [smol](https://github.com/smol-rs/smol) and provides
a more flexible threading model. Unlike Tokio, the CRT does not spawn its own background threads
by default. Instead, it relies on the calling thread to drive the executor by calling `block_on`.

This design integrates well with thread-per-core engines like DuckDB. When DuckDB calls into a
Vortex scan on one of its worker threads, that thread blocks on a future and drives the entire
smol executor for the duration of the call. No separate I/O thread pool is required, and the
engine retains full control over its threading model.

Note that in order to continue processing background I/O, thread-per-core engines may wish to spawn
additional low priority worker threads using the `CurrentThreadWorkerPool` described below.

### Worker Pool

The CRT can optionally be paired with a `CurrentThreadWorkerPool` to add background threads that
continuously drive the executor. Workers can be scaled up and down dynamically at runtime:

```rust
let runtime = CurrentThreadRuntime::new();
let pool = runtime.new_pool();

// Scale up to match available cores
pool.set_workers_to_available_parallelism();

// Or set an explicit count
pool.set_workers(4);
```

Each worker is a standard OS thread running `block_on(executor.run(...))` in a loop. When the
worker count is reduced, excess workers are signalled to shut down gracefully.

### Pitfalls

The CRT model has known pitfalls that are not yet fully resolved. The most significant is
ensuring that background I/O continues to be processed when the calling thread is occupied
with CPU-bound work. Because the CRT relies on explicit driving, a thread that is busy
evaluating a compute kernel is not polling the executor, which can stall in-flight I/O
requests. Spawning a worker pool mitigates this, but introduces its own trade-offs around
thread count and coordination overhead.

These problems require further investigation into alternate designs, such as separating I/O
polling from task execution, or using different thread pools for CPU-bound and I/O-bound work.

## Experimentation

The runtime abstraction exists precisely to enable experimentation with new threading models
without changing the rest of the Vortex stack. Some directions under consideration include:

- Separate thread pools for CPU-intensive work (decompression, compute kernels) and I/O work
  (disk reads, network calls).
- Cooperative yielding within long-running compute tasks to allow I/O progress.
- Runtime-aware I/O scheduling that batches and coalesces reads based on the capabilities of
  the underlying storage (local disk vs. object store vs. network).
