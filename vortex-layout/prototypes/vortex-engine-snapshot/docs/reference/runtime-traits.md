# Runtime traits

> **Status:** Accepted.
> **Progress:** Authoritative reference for the three runtime node
> traits, the batch type, the poll result, the ctx types, and the
> type-erasure layer. Implementing code lives in
> `src/physical_plan/abi.rs`.
> **Open questions:** none.

Lowering produces pipelines composed of three concrete trait types:
`SourceNode`, `TransformNode`, `SinkNode`. Each carries an
associated `LocalState: Send + 'static`. The runtime type-erases
them through `Dyn*Node` and `*Driver` wrappers.

## `Batch`

```rust
pub struct Batch { /* array: ArrayRef, span: DomainSpan */ }

impl Batch {
    pub fn new(array: ArrayRef, span: DomainSpan) -> Self;
    pub fn from_values(values: Vec<i64>) -> Self;
    pub fn from_values_with_span(values: Vec<i64>, span: DomainSpan) -> Self;

    pub fn rows(&self) -> usize;
    pub const fn span(&self) -> DomainSpan;
    pub fn array(&self) -> &ArrayRef;
    pub fn into_array(self) -> ArrayRef;
    pub fn values(&self) -> Vec<i64>;
}
```

Every row a `Batch` carries is real. The batch type has no demand
mask.

## `OperatorPoll`

```rust
pub type OperatorPoll<T> = Poll<EngineResult<T>>;
```

`Poll::Pending` parks the pipeline until the waker registered via
the ctx fires. `Poll::Ready(Ok(t))` returns a value;
`Poll::Ready(Err(e))` propagates a fatal error.

## `Parallelism`

```rust
pub enum Parallelism {
    Serial,
    LaneSafe { max_lanes: Option<usize> },
}

impl Parallelism {
    pub const fn serial() -> Self;
    pub const fn lane_safe(max_lanes: Option<usize>) -> Self;
    pub fn intersect(self, other: Self) -> Self;
}
```

`Serial` means one lane only. `LaneSafe { max_lanes }` means the
runtime may spawn up to `max_lanes` lanes (or unbounded when
`None`). A pipeline's parallelism is the intersection of its
source's, transforms', and sink's `parallelism()`.

## `SourceNode`

```rust
pub trait SourceNode: Send + Sync + 'static {
    type LocalState: Send + 'static;
    fn label(&self) -> &str;
    fn parallelism(&self) -> Parallelism { Parallelism::Serial }
    fn init_local(&self, runtime: &mut LocalInitRuntime<'_>) -> EngineResult<Self::LocalState>;
    fn poll_next(
        &self,
        local: &mut Self::LocalState,
        ctx: &mut SourceCtx<'_, '_>,
    ) -> OperatorPoll<Option<Batch>>;
}
```

`poll_next` returns `Ready(Ok(Some(batch)))` to produce, `Ready(Ok(None))` to
signal end of stream, or `Pending` after registering the waker on
`ctx.cx()`.

## `TransformNode`

```rust
pub enum TransformOutput {
    Batch(Batch),
    NeedInput,
    Finished,
}

pub trait TransformNode: Send + Sync + 'static {
    type LocalState: Send + 'static;
    fn label(&self) -> &str;
    fn parallelism(&self) -> Parallelism { Parallelism::Serial }
    fn init_local(&self, runtime: &mut LocalInitRuntime<'_>) -> EngineResult<Self::LocalState>;
    fn can_accept_input(&self, local: &Self::LocalState) -> bool;
    fn push_input(
        &self,
        local: &mut Self::LocalState,
        batch: Batch,
        ctx: &mut TransformCtx<'_, '_>,
    ) -> EngineResult<()>;
    fn finish_input(
        &self,
        local: &mut Self::LocalState,
        ctx: &mut TransformCtx<'_, '_>,
    ) -> EngineResult<()>;
    fn poll_next_output(
        &self,
        local: &mut Self::LocalState,
        ctx: &mut TransformCtx<'_, '_>,
    ) -> OperatorPoll<TransformOutput>;
}
```

Push-pull. The driver checks `can_accept_input`, pushes via
`push_input`, and pulls outputs via `poll_next_output`.
`finish_input` signals end of input; transforms may still emit
buffered output afterward and finally return
`TransformOutput::Finished`.

`poll_next_output` may return `Pending` while waiting on spawned
work between batches.

## `SinkNode`

```rust
pub struct PendingSend { /* batch: Option<Batch> */ }
impl PendingSend {
    pub fn new(batch: Batch) -> Self;
    pub fn take(&mut self) -> Option<Batch>;
    pub fn peek(&self) -> Option<&Batch>;
    pub fn is_consumed(&self) -> bool;
}

pub trait SinkNode: Send + Sync + 'static {
    type LocalState: Send + 'static;
    fn label(&self) -> &str;
    fn parallelism(&self) -> Parallelism { Parallelism::Serial }
    fn init_local(&self, runtime: &mut LocalInitRuntime<'_>) -> EngineResult<Self::LocalState>;
    fn poll_send(
        &self,
        local: &mut Self::LocalState,
        ctx: &mut SinkCtx<'_, '_>,
        send: &mut PendingSend,
    ) -> OperatorPoll<()>;
    fn poll_finish(
        &self,
        local: &mut Self::LocalState,
        ctx: &mut SinkCtx<'_, '_>,
    ) -> OperatorPoll<()>;
}
```

`poll_send` consumes one batch from `send` (via `send.take()`) on
successful completion; returning `Pending` without taking the
batch parks the pipeline until the destination has capacity.
`poll_finish` is the terminal flush — once it returns `Ready`, any
barriers the pipeline publishes fire.

## Ctx types

```rust
pub struct SourceCtx<'a, 'cx>   { /* cx, output_domain, output_contract, spawn */ }
pub struct TransformCtx<'a, 'cx> { /* cx, spawn */ }
pub struct SinkCtx<'a, 'cx>     { /* cx, input_domain, input_contract, spawn */ }
```

Each ctx exposes:

- `cx(&mut self) -> &mut Context<'cx>` — the standard Rust async
  context. Use it to clone wakers for spawned `WorkHandle`s.
- `spawn(&self) -> &SpawnRuntime` — the offload primitives.

Source and sink ctxs also expose their endpoint domain and
contract for operators that need to consult them at runtime.

## `LocalInitRuntime`

```rust
pub struct LocalInitRuntime<'a> { /* pipeline, spawn, submitter */ }

impl<'a> LocalInitRuntime<'a> {
    pub fn new(pipeline: PipelineId, spawn: &'a SpawnRuntime) -> Self;
    pub fn detached() -> Self;
    pub fn detached_with_spawn(spawn: &'a SpawnRuntime) -> Self;
    pub fn with_submitter(self, submitter: &'a PipelineSubmitter) -> Self;

    pub fn pipeline(&self) -> Option<PipelineId>;
    pub fn spawn(&self) -> Option<&SpawnRuntime>;
    pub fn submitter(&self) -> Option<&PipelineSubmitter>;
}
```

`init_local` receives a `&mut LocalInitRuntime`. The runtime owns
its lifetime; the lane state it returns lives until the lane
completes.

`with_submitter` is used by dynamic-expansion operators (Gather,
recursive CTE, etc.) that lower child operator subtrees and spawn
them onto the runtime mid-flight.

## Type erasure

```rust
pub trait DynSourceNode: Send + Sync { /* label, parallelism, init_local -> Box<dyn SourceDriver> */ }
pub trait DynTransformNode: Send + Sync { /* … -> Box<dyn TransformDriver> */ }
pub trait DynSinkNode: Send + Sync { /* … -> Box<dyn SinkDriver> */ }

pub trait SourceDriver: Send { /* poll_next */ }
pub trait TransformDriver: Send { /* can_accept_input, push_input, finish_input, poll_next_output */ }
pub trait SinkDriver: Send { /* poll_send, poll_finish */ }

pub struct TypedSourceNode<E: SourceNode> { /* … */ }
pub struct TypedTransformNode<E: TransformNode> { /* … */ }
pub struct TypedSinkNode<E: SinkNode> { /* … */ }
```

The blanket impls wrap a typed node, box its `LocalState` at
`init_local` time, and expose a `*Driver` with vtable dispatch
per poll call. One `Box::pin`-equivalent per lane spawn; one
vtable hop per poll.

## Worked example: an async source

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
            if state.cursor >= self.ranges.len() {
                return Poll::Ready(Ok(None));
            }
            state.pending = Some(ctx.spawn().spawn_io(
                read_range(self.ranges[state.cursor].clone()),
                IoCost::bytes(self.ranges[state.cursor].len())));
            state.cursor += 1;
        }
    }
}
```

## See also

- [Lowering API](lowering-api.md) for how nodes get attached to a
  pipeline.
- [Spawn primitives](spawn-primitives.md) for `SpawnRuntime`,
  `WorkHandle`, and `IoCost`.
- [Execution model](../concepts/execution-model.md) for the
  concept-level overview.
