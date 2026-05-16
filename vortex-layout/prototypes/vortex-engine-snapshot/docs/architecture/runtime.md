# Runtime architecture

> **Status:** Accepted.
> **Progress:** Describes how the pipeline runtime is assembled —
> work pool, executor, barrier registry, spawn primitives. Detailed
> trait shapes live in `docs/reference/`; algorithmic background
> lives in `docs/concepts/`.
> **Open questions:** memory arbiter, cooperative spill protocol,
> hierarchical I/O broker layer.

## Layers

```text
┌─────────────────────────────────────────────────────────────┐
│  PhysicalPlan  +  Operator::lower                           │  plan
├─────────────────────────────────────────────────────────────┤
│  LoweredPlan = { Pipeline*, Barrier* }                      │  lowered
├─────────────────────────────────────────────────────────────┤
│  PipelineSubmitter                                          │  dispatch
│    pushes one sync closure per pipeline onto the work pool  │
├─────────────────────────────────────────────────────────────┤
│  Runtime (work pool)                          BarrierRegistry │  exec
│  Per-thread smol::LocalExecutor               LatchBarrier   │
│  DriverIo (smol)                              SpawnRuntime   │
└─────────────────────────────────────────────────────────────┘
```

Each layer has one job. Crossing a layer is a tool, not a side
effect — operators don't reach down through the executor into
`DriverIo` directly; they go through `ctx.spawn` / `ctx.spawn_io`.

## Pipeline driver loop

A pipeline driver runs the source → transforms → sink loop on one
worker thread:

```text
loop {
  // 1. Pull a batch from the source.
  match source.poll_next(&mut cx, &mut source_ctx) {
    Ready(Some(batch)) => batch,
    Ready(None)        => break,           // source exhausted
    Pending            => return Suspended  // waker stored, re-poll later
  };
  // 2. Push it through transforms.
  for t in &mut transforms {
    if !t.can_accept_input() { drain_next_output(t); }
    t.push_input(batch, &mut t_ctx)?;
    match t.poll_next_output(&mut t_ctx)? {
      Ready(Batch(b))     => batch = b,
      Ready(NeedInput)    => continue 'outer,
      Ready(Finished)     => break 'outer,
      Pending             => return Suspended,
    }
  }
  // 3. Send to the sink.
  loop {
    match sink.poll_send(&mut cx, &mut sink_ctx, &mut pending) {
      Ready(())    => break,
      Pending      => return Suspended,
    }
  }
}
// 4. Finalise.
sink.poll_finish(&mut cx, &mut sink_ctx);
```

Suspension happens when `poll_next`, `poll_next_output`, or
`poll_send` returns `Pending`. Transforms have no waker of their
own — they suspend by returning `Pending` from
`poll_next_output`, which the driver propagates one level up.

## Work pool

`physical_plan::pool::Runtime` is the engine's compute work pool:
N pinned OS threads, N = compute parallelism. Pipelines are sync
closures dispatched onto these threads via `Runtime::spawn`. Each
closure does `block_on(<pipeline future>)` for the lifetime of its
pipeline.

`DriverIo` is a separate per-process smol executor for async I/O.
Its threads are distinct from the compute pool. `SpawnRuntime`
exposes `spawn` and `spawn_io`; both route through `DriverIo`'s
executor today.

## Barrier registry

Cross-pipeline coordination uses **sticky one-shot barriers**
(`LatchBarrier`) keyed by `PipelineBarrier` id and stored in a
per-task `BarrierRegistry`. The barrier wraps an
`event_listener::Event` plus a `fired: bool` flag:

- `fire()` flips `fired` and notifies all listeners.
- `wait()` registers a listener; if `fired` is already set when
  the listener registers, it returns immediately.

A pipeline that `depends_on(barrier)` calls `wait()` before its
source starts. A pipeline whose sink `publishes(barrier)` calls
`fire()` when the sink's `poll_finish` returns Ready. Listeners
registered between construction and `fire()` are notified.

## Spawn integration

Every operator ctx (`SourceCtx`, `TransformCtx`, `SinkCtx`)
carries a `&SpawnRuntime`. The runtime allocates one
`SpawnRuntime` per plan execution and threads it through every
ctx. Cloning a `SpawnRuntime` is one `Arc::clone`.

A typical async source:

```rust
struct ReadState {
    pending: Option<WorkHandle<Buffer>>,
    cursor:  usize,
}

impl SourceNode for VortexSource {
    type LocalState = ReadState;

    fn label(&self) -> &str { "vortex.scan" }

    fn init_local(&self, _: &mut LocalInitRuntime<'_>) -> EngineResult<ReadState> {
        Ok(ReadState { pending: None, cursor: 0 })
    }

    fn poll_next(
        &self,
        state: &mut ReadState,
        ctx:   &mut SourceCtx<'_, '_>,
    ) -> OperatorPoll<Option<Batch>> {
        loop {
            if let Some(handle) = &mut state.pending {
                match handle.poll(ctx.cx()) {
                    Poll::Ready(Ok(buf))  => { state.pending = None; return Poll::Ready(Ok(Some(decode(buf)))) }
                    Poll::Ready(Err(e))   => return Poll::Ready(Err(e)),
                    Poll::Pending         => return Poll::Pending,
                }
            }
            if state.cursor >= self.ranges.len() { return Poll::Ready(Ok(None)); }
            state.pending = Some(ctx.spawn().spawn_io(
                read_range(self.ranges[state.cursor].clone()),
                IoCost::bytes(self.ranges[state.cursor].len())));
            state.cursor += 1;
        }
    }
}
```

The driver suspends on `Pending`; the executor re-polls when the
`WorkHandle`'s waker fires (i.e., the spawned I/O completes).

## What lives where

| Concern | Module |
|---|---|
| Plan-time trait, `PhysicalPlan` | `src/physical_plan/plan.rs` |
| `Operator::lower`, `PipelineTail`, `LoweringCtx`, `PipelineBuilder`, `LoweredPlan` | `src/physical_plan/lowering.rs` |
| `SourceNode` / `TransformNode` / `SinkNode`, `Batch`, ctx types | `src/physical_plan/abi.rs` |
| `SpawnRuntime`, `WorkHandle`, `IoCost`, `Priority` | `src/physical_plan/spawn.rs` |
| `DriverIo` | `src/physical_plan/driver_io.rs` |
| Worker pool | `src/physical_plan/pool.rs` |
| Barrier registry, pipeline driver loop, `run_plan_blocking[_with_io]` | `src/physical_plan/runtime.rs` |
| Pipeline submitter (pool dispatch) | `src/physical_plan/submitter.rs` |
| `PipelineId`, `PipelineBarrier` | `src/physical_plan/ids.rs` |
| Operator inventory | `src/physical_plan/operators.rs`, `vortex_scan.rs`, `merge_join.rs`, `parent_child_min.rs`, `limit.rs`, `gather.rs`, `sum_aggregate.rs`, `vortex_aggregate.rs` |

## What is deliberately missing

- **Memory arbiter.** Channels are byte-counter-bounded
  (ADR 0008) but there is no task-wide arbiter that grows or
  shrinks bounds under pressure.
- **Spill protocol.** Cooperative spill via a per-thread pressure
  signal that operators consult and flush state.
- **Hierarchical I/O brokers.** `SpawnRuntime` exposes `spawn_io`
  directly; a broker layer that owns priority heaps, coalescing,
  and admission keys keyed by backend / host / route is on the
  plan.
- **Custom executor with priority wakers.** The executor today is
  `smol::LocalExecutor`.

These are tracked in [Implementation plan](../implementation/roadmap.md).
