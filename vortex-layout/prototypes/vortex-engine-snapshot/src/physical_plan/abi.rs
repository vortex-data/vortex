//! Runtime ABI: `Batch`, `Parallelism`, and the three core node
//! traits (`SourceNode`, `TransformNode`, `SinkNode`) plus their
//! type-erased duals (`DynSourceNode`/`SourceDriver` etc.).
//!
//! All three node roles are poll-based synchronous state machines.
//! When an operator needs to wait — for I/O completion, a barrier,
//! a CPU-offloaded computation — it calls one of the spawn
//! primitives on its ctx (`spawn`, `spawn_io`, `spawn_cpu`),
//! receives a `WorkHandle<T>`, and polls that handle on subsequent
//! ticks. The runtime owns the spawned work; operators just hold
//! handles to it.
//!
//! The v2 batch type has no row demand. Demand is intentionally not
//! part of the new model.

use std::sync::Arc;
use std::task::Context;
use std::task::Poll;

use vortex_array::ArrayRef;
use vortex_array::Canonical;
use vortex_array::IntoArray;
use vortex_array::arrays::Primitive;
use vortex_array::arrays::PrimitiveArray;

use crate::Domain;
use crate::DomainSpan;
use crate::EngineResult;
use crate::OutputContract;
use crate::physical_plan::ids::PipelineId;
use crate::physical_plan::spawn::SpawnRuntime;

/// Poll result returned by source/transform/sink runtime nodes.
pub type OperatorPoll<T> = Poll<EngineResult<T>>;

/// Lane-instantiation support declared by a pipeline node.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Parallelism {
    Serial,
    LaneSafe { max_lanes: Option<usize> },
}

impl Parallelism {
    pub const fn serial() -> Self {
        Self::Serial
    }

    pub const fn lane_safe(max_lanes: Option<usize>) -> Self {
        Self::LaneSafe { max_lanes }
    }

    pub fn intersect(self, other: Self) -> Self {
        match (self, other) {
            (Self::Serial, _) | (_, Self::Serial) => Self::Serial,
            (Self::LaneSafe { max_lanes: left }, Self::LaneSafe { max_lanes: right }) => {
                Self::LaneSafe {
                    max_lanes: min_optional_lane_count(left, right),
                }
            }
        }
    }
}

fn min_optional_lane_count(left: Option<usize>, right: Option<usize>) -> Option<usize> {
    match (left, right) {
        (Some(left), Some(right)) => Some(left.min(right)),
        (Some(value), None) | (None, Some(value)) => Some(value),
        (None, None) => None,
    }
}

/// Runtime batch payload.
///
/// `array` carries the Vortex data for the batch. `span` identifies
/// the half-open row range represented by the batch in its endpoint
/// domain.
///
/// **No demand mask.** The v2 model has no row demand or
/// cancellation. Every row a batch carries is fully real.
#[derive(Clone, Debug)]
pub struct Batch {
    array: ArrayRef,
    span: DomainSpan,
}

impl Batch {
    pub fn new(array: ArrayRef, span: DomainSpan) -> Self {
        debug_assert_eq!(array.len() as u64, span.len());
        Self { array, span }
    }

    pub fn from_values(values: Vec<i64>) -> Self {
        let span = DomainSpan::from_len(values.len() as u64);
        Self::from_values_with_span(values, span)
    }

    pub fn from_values_with_span(values: Vec<i64>, span: DomainSpan) -> Self {
        debug_assert_eq!(values.len() as u64, span.len());
        let array = PrimitiveArray::from_iter(values).into_array();
        Self::new(array, span)
    }

    pub fn rows(&self) -> usize {
        self.array.len()
    }

    pub const fn span(&self) -> DomainSpan {
        self.span
    }

    pub fn array(&self) -> &ArrayRef {
        &self.array
    }

    pub fn into_array(self) -> ArrayRef {
        self.array
    }

    pub fn values(&self) -> Vec<i64> {
        primitive_i64_values(self.array.clone())
    }
}

fn primitive_i64_values(array: ArrayRef) -> Vec<i64> {
    #[expect(deprecated)]
    let canonical = array.to_canonical().expect("vortex primitive canonical");
    let primitive: PrimitiveArray = match canonical {
        Canonical::Primitive(p) => p,
        other => other
            .into_array()
            .try_downcast::<Primitive>()
            .expect("primitive array"),
    };
    let buffer = primitive.to_buffer::<i64>();
    buffer.iter().copied().collect()
}

/// Runtime services available while initialising lane-local state.
pub struct LocalInitRuntime<'a> {
    pipeline: Option<PipelineId>,
    spawn: Option<&'a SpawnRuntime>,
    submitter: Option<&'a crate::physical_plan::submitter::PipelineSubmitter>,
}

impl<'a> LocalInitRuntime<'a> {
    pub fn new(pipeline: PipelineId, spawn: &'a SpawnRuntime) -> Self {
        Self {
            pipeline: Some(pipeline),
            spawn: Some(spawn),
            submitter: None,
        }
    }

    pub fn detached() -> Self {
        Self {
            pipeline: None,
            spawn: None,
            submitter: None,
        }
    }

    /// Variant used by the runtime when there's no specific pipeline
    /// context but operators may still need access to `SpawnRuntime`
    /// (e.g. to grab the DriverIo).
    pub fn detached_with_spawn(spawn: &'a SpawnRuntime) -> Self {
        Self {
            pipeline: None,
            spawn: Some(spawn),
            submitter: None,
        }
    }

    /// Attach a `PipelineSubmitter` — used by dynamic-expansion
    /// operators (Gather, recursive CTE, etc.) to lower child
    /// operator subtrees and spawn them onto the runtime mid-flight.
    pub fn with_submitter(
        mut self,
        submitter: &'a crate::physical_plan::submitter::PipelineSubmitter,
    ) -> Self {
        self.submitter = Some(submitter);
        self
    }

    pub fn pipeline(&self) -> Option<PipelineId> {
        self.pipeline
    }

    pub fn spawn(&self) -> Option<&SpawnRuntime> {
        self.spawn
    }

    pub fn submitter(&self) -> Option<&crate::physical_plan::submitter::PipelineSubmitter> {
        self.submitter
    }
}

impl Default for LocalInitRuntime<'_> {
    fn default() -> Self {
        Self::detached()
    }
}

/// Pending send token passed to `SinkNode::poll_send`. Wraps the
/// batch in a `take()`-once container so the operator can move it
/// out only on successful completion.
pub struct PendingSend {
    batch: Option<Batch>,
}

impl PendingSend {
    pub fn new(batch: Batch) -> Self {
        Self { batch: Some(batch) }
    }

    pub fn take(&mut self) -> Option<Batch> {
        self.batch.take()
    }

    pub fn peek(&self) -> Option<&Batch> {
        self.batch.as_ref()
    }

    pub fn is_consumed(&self) -> bool {
        self.batch.is_none()
    }
}

/// Polling ctx for a `SourceNode`. Carries the waker context (for
/// registering on async work or barriers), endpoint metadata, and
/// the spawn primitives.
pub struct SourceCtx<'a, 'cx> {
    cx: &'a mut Context<'cx>,
    output_domain: &'a Domain,
    output_contract: &'a OutputContract,
    spawn: &'a SpawnRuntime,
}

impl<'a, 'cx> SourceCtx<'a, 'cx> {
    pub fn new(
        cx: &'a mut Context<'cx>,
        output_domain: &'a Domain,
        output_contract: &'a OutputContract,
        spawn: &'a SpawnRuntime,
    ) -> Self {
        Self {
            cx,
            output_domain,
            output_contract,
            spawn,
        }
    }

    pub fn cx(&mut self) -> &mut Context<'cx> {
        self.cx
    }

    pub fn output_domain(&self) -> &Domain {
        self.output_domain
    }

    pub fn output_contract(&self) -> &OutputContract {
        self.output_contract
    }

    pub fn spawn(&self) -> &SpawnRuntime {
        self.spawn
    }
}

/// Polling ctx for a `TransformNode`. Transforms get the same
/// access to spawn primitives that sources/sinks do — they can
/// suspend on async work between batches.
pub struct TransformCtx<'a, 'cx> {
    cx: &'a mut Context<'cx>,
    spawn: &'a SpawnRuntime,
}

impl<'a, 'cx> TransformCtx<'a, 'cx> {
    pub fn new(cx: &'a mut Context<'cx>, spawn: &'a SpawnRuntime) -> Self {
        Self { cx, spawn }
    }

    pub fn cx(&mut self) -> &mut Context<'cx> {
        self.cx
    }

    pub fn spawn(&self) -> &SpawnRuntime {
        self.spawn
    }
}

/// Polling ctx for a `SinkNode`.
pub struct SinkCtx<'a, 'cx> {
    cx: &'a mut Context<'cx>,
    input_domain: &'a Domain,
    input_contract: &'a OutputContract,
    spawn: &'a SpawnRuntime,
}

impl<'a, 'cx> SinkCtx<'a, 'cx> {
    pub fn new(
        cx: &'a mut Context<'cx>,
        input_domain: &'a Domain,
        input_contract: &'a OutputContract,
        spawn: &'a SpawnRuntime,
    ) -> Self {
        Self {
            cx,
            input_domain,
            input_contract,
            spawn,
        }
    }

    pub fn cx(&mut self) -> &mut Context<'cx> {
        self.cx
    }

    pub fn input_domain(&self) -> &Domain {
        self.input_domain
    }

    pub fn input_contract(&self) -> &OutputContract {
        self.input_contract
    }

    pub fn spawn(&self) -> &SpawnRuntime {
        self.spawn
    }
}

/// Typed user-facing source node.
pub trait SourceNode: Send + Sync + 'static {
    type LocalState: Send + 'static;

    fn label(&self) -> &str;

    fn parallelism(&self) -> Parallelism {
        Parallelism::Serial
    }

    fn init_local(&self, runtime: &mut LocalInitRuntime<'_>) -> EngineResult<Self::LocalState>;

    /// Produce the next batch. Returning `Poll::Pending` parks the
    /// pipeline until the waker registered via `ctx.cx()` fires.
    /// `Ready(Ok(None))` signals end-of-stream.
    fn poll_next(
        &self,
        local: &mut Self::LocalState,
        ctx: &mut SourceCtx<'_, '_>,
    ) -> OperatorPoll<Option<Batch>>;
}

/// Output returned when draining a transform node.
pub enum TransformOutput {
    Batch(Batch),
    NeedInput,
    Finished,
}

/// Typed user-facing transform node.
///
/// Transforms can suspend on spawned work via `poll_next_output`
/// returning `Pending`. `push_input` and `finish_input` are
/// synchronous: a transform either accepts input or it doesn't
/// (gated by `can_accept_input`).
pub trait TransformNode: Send + Sync + 'static {
    type LocalState: Send + 'static;

    fn label(&self) -> &str;

    fn parallelism(&self) -> Parallelism {
        Parallelism::Serial
    }

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

    /// Pull the next output batch. May return `Pending` while
    /// waiting on spawned work.
    fn poll_next_output(
        &self,
        local: &mut Self::LocalState,
        ctx: &mut TransformCtx<'_, '_>,
    ) -> OperatorPoll<TransformOutput>;
}

/// Typed user-facing sink node.
pub trait SinkNode: Send + Sync + 'static {
    type LocalState: Send + 'static;

    fn label(&self) -> &str;

    fn parallelism(&self) -> Parallelism {
        Parallelism::Serial
    }

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

// -- Type erasure: Dyn{Source,Transform,Sink}Node + *Driver --------

pub trait DynSourceNode: Send + Sync {
    fn label(&self) -> &str;
    fn parallelism(&self) -> Parallelism;
    fn init_local(
        &self,
        runtime: &mut LocalInitRuntime<'_>,
    ) -> EngineResult<Box<dyn SourceDriver>>;
}

pub trait SourceDriver: Send {
    fn poll_next(
        &mut self,
        ctx: &mut SourceCtx<'_, '_>,
    ) -> OperatorPoll<Option<Batch>>;
}

pub struct TypedSourceNode<E: SourceNode> {
    node: Arc<E>,
}

impl<E: SourceNode> TypedSourceNode<E> {
    pub fn new(node: E) -> Self {
        Self {
            node: Arc::new(node),
        }
    }
}

struct TypedSourceDriver<E: SourceNode> {
    node: Arc<E>,
    local: E::LocalState,
}

impl<E> DynSourceNode for TypedSourceNode<E>
where
    E: SourceNode,
{
    fn label(&self) -> &str {
        self.node.label()
    }

    fn parallelism(&self) -> Parallelism {
        self.node.parallelism()
    }

    fn init_local(
        &self,
        runtime: &mut LocalInitRuntime<'_>,
    ) -> EngineResult<Box<dyn SourceDriver>> {
        let local = self.node.init_local(runtime)?;
        Ok(Box::new(TypedSourceDriver {
            node: Arc::clone(&self.node),
            local,
        }))
    }
}

impl<E> SourceDriver for TypedSourceDriver<E>
where
    E: SourceNode,
{
    fn poll_next(
        &mut self,
        ctx: &mut SourceCtx<'_, '_>,
    ) -> OperatorPoll<Option<Batch>> {
        self.node.poll_next(&mut self.local, ctx)
    }
}

pub trait DynTransformNode: Send + Sync {
    fn label(&self) -> &str;
    fn parallelism(&self) -> Parallelism;
    fn init_local(
        &self,
        runtime: &mut LocalInitRuntime<'_>,
    ) -> EngineResult<Box<dyn TransformDriver>>;
}

pub trait TransformDriver: Send {
    fn can_accept_input(&self) -> bool;

    fn push_input(
        &mut self,
        batch: Batch,
        ctx: &mut TransformCtx<'_, '_>,
    ) -> EngineResult<()>;

    fn finish_input(&mut self, ctx: &mut TransformCtx<'_, '_>) -> EngineResult<()>;

    fn poll_next_output(
        &mut self,
        ctx: &mut TransformCtx<'_, '_>,
    ) -> OperatorPoll<TransformOutput>;
}

pub struct TypedTransformNode<E: TransformNode> {
    node: Arc<E>,
}

impl<E: TransformNode> TypedTransformNode<E> {
    pub fn new(node: E) -> Self {
        Self {
            node: Arc::new(node),
        }
    }
}

struct TypedTransformDriver<E: TransformNode> {
    node: Arc<E>,
    local: E::LocalState,
}

impl<E> DynTransformNode for TypedTransformNode<E>
where
    E: TransformNode,
{
    fn label(&self) -> &str {
        self.node.label()
    }

    fn parallelism(&self) -> Parallelism {
        self.node.parallelism()
    }

    fn init_local(
        &self,
        runtime: &mut LocalInitRuntime<'_>,
    ) -> EngineResult<Box<dyn TransformDriver>> {
        let local = self.node.init_local(runtime)?;
        Ok(Box::new(TypedTransformDriver {
            node: Arc::clone(&self.node),
            local,
        }))
    }
}

impl<E> TransformDriver for TypedTransformDriver<E>
where
    E: TransformNode,
{
    fn can_accept_input(&self) -> bool {
        self.node.can_accept_input(&self.local)
    }

    fn push_input(
        &mut self,
        batch: Batch,
        ctx: &mut TransformCtx<'_, '_>,
    ) -> EngineResult<()> {
        self.node.push_input(&mut self.local, batch, ctx)
    }

    fn finish_input(&mut self, ctx: &mut TransformCtx<'_, '_>) -> EngineResult<()> {
        self.node.finish_input(&mut self.local, ctx)
    }

    fn poll_next_output(
        &mut self,
        ctx: &mut TransformCtx<'_, '_>,
    ) -> OperatorPoll<TransformOutput> {
        self.node.poll_next_output(&mut self.local, ctx)
    }
}

pub trait DynSinkNode: Send + Sync {
    fn label(&self) -> &str;
    fn parallelism(&self) -> Parallelism;
    fn init_local(&self, runtime: &mut LocalInitRuntime<'_>) -> EngineResult<Box<dyn SinkDriver>>;
}

pub trait SinkDriver: Send {
    fn poll_send(
        &mut self,
        ctx: &mut SinkCtx<'_, '_>,
        send: &mut PendingSend,
    ) -> OperatorPoll<()>;

    fn poll_finish(&mut self, ctx: &mut SinkCtx<'_, '_>) -> OperatorPoll<()>;
}

pub struct TypedSinkNode<E: SinkNode> {
    node: Arc<E>,
}

impl<E: SinkNode> TypedSinkNode<E> {
    pub fn new(node: E) -> Self {
        Self {
            node: Arc::new(node),
        }
    }
}

struct TypedSinkDriver<E: SinkNode> {
    node: Arc<E>,
    local: E::LocalState,
}

impl<E> DynSinkNode for TypedSinkNode<E>
where
    E: SinkNode,
{
    fn label(&self) -> &str {
        self.node.label()
    }

    fn parallelism(&self) -> Parallelism {
        self.node.parallelism()
    }

    fn init_local(&self, runtime: &mut LocalInitRuntime<'_>) -> EngineResult<Box<dyn SinkDriver>> {
        let local = self.node.init_local(runtime)?;
        Ok(Box::new(TypedSinkDriver {
            node: Arc::clone(&self.node),
            local,
        }))
    }
}

impl<E> SinkDriver for TypedSinkDriver<E>
where
    E: SinkNode,
{
    fn poll_send(
        &mut self,
        ctx: &mut SinkCtx<'_, '_>,
        send: &mut PendingSend,
    ) -> OperatorPoll<()> {
        self.node.poll_send(&mut self.local, ctx, send)
    }

    fn poll_finish(&mut self, ctx: &mut SinkCtx<'_, '_>) -> OperatorPoll<()> {
        self.node.poll_finish(&mut self.local, ctx)
    }
}
