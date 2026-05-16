# Execution model

> **Status:** Accepted.
> **Progress:** Concept-level description of the lowering API,
> pipeline shape, barrier semantics, and resource handoff. Exact
> trait signatures live in [Lowering API](../reference/lowering-api.md)
> and [Runtime traits](../reference/runtime-traits.md).
> **Open questions:** spill protocol, lane-affinity policy when
> producer and consumer lane counts disagree.

## Plan, lower, run

```text
PhysicalPlan
    │
    │ .lower(ctx, sink)
    ▼
LoweredPlan = { Pipeline*, Barrier* }
    │
    │ runtime::run_plan_blocking(plan)
    ▼
Result
```

Three stages:

1. **Plan stage.** A planner constructs `Operator` instances and
   wraps the root in a `PhysicalPlan`. `Operator` is one method:
   `fn lower(&self, ctx: &mut dyn LoweringCtx, tail: PipelineTail)
   -> BuildResult<()>`.
2. **Lower stage.** Calling `plan.lower(ctx, sink)` walks the
   operator tree, accumulating a `PipelineTail` continuation and
   emitting concrete pipelines through `ctx.emit_pipeline`.
3. **Run stage.** The runtime spawns one driver per pipeline lane,
   each pumping the source → transforms → sink loop until the
   source is exhausted.

## The lowering pattern

`PipelineTail` is a continuation that flows from the sink toward
the sources. Each operator does one of three things with it:

- **Streaming operator** (Filter, Project, Cast): prepend its
  transform onto the tail, recurse into its child.
- **Source (leaf)**: complete the tail by attaching its source
  via `ctx.emit_pipeline(tail, ..., source)`. End of recursion.
- **Pipeline-breaker or multi-input** (Sort, HashJoin,
  PreSortedMergeJoin): build per-input shared state; for each
  input, call `child.lower(ctx, PipelineTail::new(...,
  SinkForThisInput).publishes(barrier))`; then complete the outer
  tail with a source that reads the shared state and depends on
  the barriers.

A streaming filter:

```rust
impl Operator for ArrayPredicate {
    fn lower(&self, ctx: &mut dyn LoweringCtx, tail: PipelineTail) -> BuildResult<()> {
        let tail = tail.prepend_transform(self.input_domain.clone(),
                                          self.input_contract.clone(),
                                          ArrayPredicateTransform::new(self.predicate.clone()));
        self.input.lower(ctx, tail)
    }
}
```

A pipeline-breaker:

```rust
impl Operator for Sort {
    fn lower(&self, ctx: &mut dyn LoweringCtx, tail: PipelineTail) -> BuildResult<()> {
        let runs = Arc::new(SortedRuns::default());
        let build_done = ctx.new_pipeline_barrier();

        // Build pipeline: input → ... → SortSink writes into `runs`
        self.input.lower(ctx,
            PipelineTail::new(self.input_domain.clone(),
                              self.input_contract.clone(),
                              SortSink::new(runs.clone(), self.keys.clone()))
                .publishes(build_done))?;

        // Output pipeline: MergeKSource(runs) → tail
        ctx.emit_pipeline(
            tail.depends_on(build_done),
            self.output_domain.clone(),
            self.output_contract.clone(),
            MergeKSource::new(runs, self.keys.clone()))?;
        Ok(())
    }
}
```

The tail flows down toward the sources instead of a builder being
mutated.

## Pipeline shape

Every pipeline emitted by `ctx.emit_pipeline` is:

```
source → transform₁ → transform₂ → … → sink
```

Flat. Single-input. Multi-input operators emit multiple pipelines
from one `lower` call and link them by barriers; the multi-input
concept does not exist at runtime.

Each pipeline declares:

- one `SourceNode`;
- zero or more `TransformNode`s;
- one `SinkNode`;
- a set of barriers it `depends_on`;
- a set of barriers it `publishes` (the sink fires them on
  `poll_finish`).

The runtime walks the barrier DAG and runs pipelines concurrently
subject to dependencies.

## The three runtime traits

Lowering produces nodes of three concrete trait types:

- **`SourceNode`** owns its data and is polled to produce batches.
  `poll_next` is poll-style: returns `Pending` and registers a
  waker via `ctx.cx()` when blocked on I/O, a resource, or an
  upstream barrier.
- **`TransformNode`** is push-pull: the driver checks
  `can_accept_input`, pushes via `push_input`, drains via
  `poll_next_output`. `poll_next_output` may also return
  `Pending` while waiting on spawned work between batches.
- **`SinkNode`** is poll-style on the send side: `poll_send` may
  return `Pending` when the destination is full (channel
  back-pressure, resource publish blocked). `poll_finish` for the
  terminal flush.

All three carry an associated `LocalState: Send + 'static` that
the runtime type-erases via `DynSourceNode` / `DynTransformNode` /
`DynSinkNode` wrappers.

Exact signatures: [Runtime traits](../reference/runtime-traits.md).

## Cross-pipeline handoff: barriers and resources

Two complementary mechanisms link pipelines:

**Barriers** are sticky one-shot events keyed by `PipelineBarrier`
id. A sink declares `publishes(barrier)`; the barrier fires when
the sink's `poll_finish` returns Ready. A pipeline with
`depends_on(barrier)` waits on the barrier registry before its
source starts pulling.

**Resources** are typed `Arc<…>` data structures captured by the
sink (writer) and a downstream source or transform (reader). The
runtime knows nothing about them: they are plain Rust values with
their own internal synchronization. The build-then-probe pattern is
just:

```
build sink writes into  Arc<HashTable>   ← captured by both
probe transform reads from same Arc<HashTable>
probe pipeline depends_on build_barrier
```

When the build barrier fires, the probe pipeline's source starts;
the probe transform looks up keys in the (now-ready) hash table.

This is the standard cross-pipeline pattern. Sort, HashJoin,
HashAggregate, runtime filters, and finalized aggregates all use
it.

## Channels between pipelines

Concurrent producer/consumer pipelines exchange data through
**bounded MPMC work-stealing channels** built on
`crossbeam-deque::Worker` (one per producer lane) and `Stealer`
(one per consumer lane). Byte-counter bound enforced at admission.
See [ADR 0008](../decisions/0008-mpmc-work-stealing-channels.md).

A channel exchange appears as a `ChannelExchangeSink` →
`ChannelExchangeSource` pair between two pipelines. Inside a
pipeline there are no channels — only direct calls.

## What an operator never does

To keep the per-batch cost low, operators must not:

- Loop on `block_on` inside `poll_*`. Wait via `Poll::Pending`
  plus a stored waker, or offload via `ctx.spawn` /
  `ctx.spawn_io`.
- Touch global mutable state. All cross-pipeline state goes
  through a typed resource.
- Carry per-batch row demand. Reduction of work upstream is
  expressed by [runtime filters](runtime-filters.md) or by
  plan-time push-down before lowering.

## Worked example: sort-merge join

The most multi-input case. `SortedMergeJoin::lower` emits five
pipelines and four barriers:

```
Scan(L) → Filter(L) → SortSink         → [left_runs_done]
Scan(R) → Filter(R) → SortSink         → [right_runs_done]
[deps left_runs_done]  MergeKSource(L) → MergeJoinSink(Left)  → [left_ready]
[deps right_runs_done] MergeKSource(R) → MergeJoinSink(Right) → [right_ready]
[deps left_ready, right_ready] MergeJoinSource → Project → Sink
```

The two `MergeJoinSink`s share an `Arc<MergeJoinState>` captured
by closure; the output `MergeJoinSource` reads from the same
state. Symmetric sinks, no primary-vs-side asymmetry.

## Where to read next

- [Runtime filters](runtime-filters.md): the typed-filter
  mechanism for sideways information passing.
- [I/O and spawn](io-and-spawn.md): how operators wait for async
  work.
- [Runtime architecture](../architecture/runtime.md): how the
  pipeline executor and barrier registry are built.
- [Lowering API](../reference/lowering-api.md): exact `Operator`,
  `PipelineTail`, and `LoweringCtx` signatures.
- [Runtime traits](../reference/runtime-traits.md): exact
  `SourceNode`, `TransformNode`, `SinkNode` signatures.
