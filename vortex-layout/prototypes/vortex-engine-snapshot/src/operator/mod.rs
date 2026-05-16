use std::any::Any;
use std::sync::atomic::AtomicBool;
use std::sync::atomic::Ordering;
use std::task::Context;

use crate::AsyncWorkId;
use crate::Batch;
use crate::BrokerId;
use crate::CompletedInterest;
use crate::Domain;
use crate::EngineError;
use crate::EngineResult;
use crate::scheduler::FakeIoRequest;
use crate::InputPortId;
use crate::InputPortRef;
use crate::InterestId;
use crate::InterestSpec;
use crate::RequirementSet;
use crate::ResourceValue;
use crate::WorkKey;
use crate::WorkProposal;
use crate::WorkStatus;

/// Sort discipline a port declares. `None` on an output port means
/// the producer makes no sort claim; `None` on an input port means
/// the consumer accepts any (or no) sort.
///
/// Two declared keys are compatible if they're structurally equal:
/// same variant, same fields. `RowIndex` is sorted by row position
/// within the producer's domain; bind compares it against the
/// matching consumer position. `Natural` compares on column path +
/// direction.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum SortKey {
    /// Rows are in ascending row-position order within the
    /// producer's output domain. The most common shape — what a
    /// flat scan, a chunked layout drained in chunk order, or a
    /// dict-decode of in-order codes naturally produces.
    RowIndex,
    /// Rows are sorted on the listed columns in the listed
    /// directions. `columns.len() == directions.len()`.
    Natural {
        columns: Vec<String>,
        directions: Vec<SortDirection>,
    },
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SortDirection {
    Ascending,
    Descending,
}

/// Channel contract an input port declares. Three mutually-exclusive
/// variants. The bind layer reads this once at prepare time to decide
/// (a) whether to fold the consumer into the producer's channel as a
/// `Transform`, and (b) whether to refuse a sort-mismatched
/// connection.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum InputShape {
    /// No constraints. Operator runs as a normal scheduler-admitted
    /// node fed by a real channel.
    Unsorted,
    /// Operator requires its input sorted on `K`. Bind validates
    /// every connected producer's output `sort_key` matches and
    /// refuses the query otherwise.
    Sorted(SortKey),
    /// Operator opts into fusion. Promise: no broker registration,
    /// no resource handles, no `request_propagation`, no per-batch
    /// state. Bind layer may call `Operator::into_transform` and
    /// fold this operator into the producer's emit pipeline,
    /// eliminating the inter-operator channel entirely.
    Fused,
}

impl Default for InputShape {
    fn default() -> Self {
        Self::Unsorted
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct InputPortSpec {
    pub name: String,
    pub domain: Domain,
    pub columns: usize,
    pub shape: InputShape,
}

impl InputPortSpec {
    pub fn new(name: impl Into<String>, domain: Domain, columns: usize) -> Self {
        Self {
            name: name.into(),
            domain,
            columns,
            shape: InputShape::Unsorted,
        }
    }

    pub fn with_required_sort(mut self, sort: SortKey) -> Self {
        self.shape = InputShape::Sorted(sort);
        self
    }

    pub fn with_fusion(mut self) -> Self {
        self.shape = InputShape::Fused;
        self
    }

    /// Convenience accessor for `Sorted(_)`.
    pub fn required_sort(&self) -> Option<&SortKey> {
        match &self.shape {
            InputShape::Sorted(k) => Some(k),
            _ => None,
        }
    }

    pub fn accepts_fusion(&self) -> bool {
        matches!(self.shape, InputShape::Fused)
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct OutputPortSpec {
    pub name: String,
    pub domain: Domain,
    pub columns: usize,
    /// What the rows on this output are sorted by, if anything.
    pub sort_key: Option<SortKey>,
}

impl OutputPortSpec {
    pub fn new(name: impl Into<String>, domain: Domain, columns: usize) -> Self {
        Self {
            name: name.into(),
            domain,
            columns,
            sort_key: None,
        }
    }

    pub fn with_sort_key(mut self, sort: SortKey) -> Self {
        self.sort_key = Some(sort);
        self
    }
}

/// Operator parallelism contract.
///
/// A **lane** is one parallel execution slot of the operator: one
/// `LocalState` driven by one worker. Operators declare how many
/// lanes they want. The scheduler picks the actual count clamped by
/// host parallelism (`TaskOptions::worker_count`).
///
/// Work distribution among lanes is the operator's concern: lanes
/// typically pull work units (chunks, hash buckets, batches) from a
/// shared queue in `GlobalState`. This means lane state stays small
/// (a current-work-unit cursor) and work-stealing across lanes
/// happens via atomics on the shared queue.
///
/// "Shard" in the docs refers to one work unit the operator
/// processes (e.g. one chunk of a chunked layout). Shards are an
/// operator-internal concept; the engine doesn't know about them.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Parallelism {
    /// One lane only.
    #[default]
    Serial,
    /// Operator scales to host parallelism. `max` caps the lane
    /// count (`None` means "as many as the host gives me"). The
    /// scheduler picks `lane_count = min(max.unwrap_or(host),
    /// host).max(1)`.
    Lanes { max: Option<usize> },
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct OperatorSpec {
    pub label: String,
    pub inputs: Vec<InputPortSpec>,
    /// At most one output port. `None` for sinks. Multiple-output
    /// operators don't exist in this engine — fan-out happens at
    /// the channel level (an SPMC channel can have multiple
    /// downstream consumers).
    pub output: Option<OutputPortSpec>,
    pub parallelism: Parallelism,
}

impl OperatorSpec {
    pub fn new(
        label: impl Into<String>,
        inputs: Vec<InputPortSpec>,
        output: Option<OutputPortSpec>,
    ) -> Self {
        Self {
            label: label.into(),
            inputs,
            output,
            parallelism: Parallelism::Serial,
        }
    }

    /// Declare this operator scales to host parallelism, with an
    /// optional cap on lane count. `None` means "use as many lanes
    /// as the host has workers." `Some(N)` caps at `N`; the
    /// scheduler picks `min(N, host_workers)`.
    pub fn lanes(mut self, max: Option<usize>) -> Self {
        self.parallelism = Parallelism::Lanes { max };
        self
    }
}

/// Context passed to `init_global` during preparation. Empty for now;
/// production will expose resource and broker registration here.
pub struct GlobalInitCtx<'a> {
    pub(crate) operator: super::OperatorId,
    pub(crate) _phantom: std::marker::PhantomData<&'a ()>,
}

impl GlobalInitCtx<'_> {
    pub fn operator(&self) -> super::OperatorId {
        self.operator
    }
}

/// Context passed to `init_local` during preparation, once per lane.
///
/// `lane.index` is in `0..lane_count`. For `Serial` operators
/// `lane_count == 1`. For `Lanes { max }` operators `lane_count` is
/// whatever the scheduler discovered (capped by `max` and by
/// `TaskOptions::worker_count`).
pub struct LocalInitCtx<'a> {
    pub(crate) operator: super::OperatorId,
    pub(crate) lane: LaneId,
    pub(crate) lane_count: usize,
    pub(crate) _phantom: std::marker::PhantomData<&'a ()>,
}

impl LocalInitCtx<'_> {
    pub fn operator(&self) -> super::OperatorId {
        self.operator
    }
    pub fn lane(&self) -> LaneId {
        self.lane
    }
    pub fn lane_count(&self) -> usize {
        self.lane_count
    }
}

/// Identifies one runtime lane of an operator. `Serial` operators
/// always use `LaneId(0)`; `Lanes { max }` operators use
/// `0..lane_count`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct LaneId {
    pub index: usize,
}

impl LaneId {
    pub const fn new(index: usize) -> Self {
        Self { index }
    }
}

/// Public authoring trait for one physical operator type.
///
/// The scheduler holds two pieces of state per node:
///
/// - `GlobalState`: built once during preparation, shared by `&`
///   reference across every lane. `Send + Sync` so the lanes
///   running on different workers can borrow it concurrently. Holds
///   immutable prepared artifacts, broker handles, shared resource
///   handles, or thread-safe interior state.
/// - `LocalState`: built once per lane, owned by exactly one worker
///   at execution time. `Send` so the scheduler can reassign a lane
///   between workers during preparation or shutdown. Holds
///   lane-private witness slices, hash-table shards, in-flight
///   futures, output cursors, and reorder buffers.
///
/// `Serial` operators have one `LocalState` per node. `Lanes { max }`
/// operators have one `LocalState` per discovered lane.
///
/// ### Bounds
///
/// `Operator` is `Send` so a worker can move an operator's owning
/// node between threads. It is **not** `Sync`: cross-worker sharing
/// of mutable runtime data is `GlobalState`'s job, not `&self`'s.
/// After preparation `&self` is read-only operator configuration —
/// any concurrent mutable state belongs in `GlobalState` instead.
pub trait Operator: Send + Sync + 'static {
    /// Shared per-node state, built once and visible to every lane
    /// by `&`. Most operators have `type GlobalState = ();` and put
    /// everything in `LocalState`; declare a non-trivial
    /// `GlobalState` when shared resources actually need to live
    /// once per node (compiled expressions, resource handles,
    /// broker registrations, finalized lookup tables).
    type GlobalState: Send + Sync + 'static;

    /// Per-lane state owned by exactly one worker.
    type LocalState: Send + 'static;

    fn spec(&self) -> OperatorSpec;

    /// Build shared per-node state once during task preparation.
    /// The scheduler stores the result for the node's lifetime and
    /// hands it out by `&` to every lane and every driving method.
    fn init_global(&self, ctx: &mut GlobalInitCtx<'_>) -> EngineResult<Self::GlobalState>;

    /// Build one per-lane state. Called once for `Serial` operators
    /// and once per discovered lane for `Lanes { max }` operators.
    fn init_local(
        &self,
        global: &Self::GlobalState,
        ctx: &mut LocalInitCtx<'_>,
    ) -> EngineResult<Self::LocalState>;

    /// Translate the merged downstream `outputs` into upstream
    /// `inputs`. The scheduler sizes both slices from the operator's
    /// `OperatorSpec` and pre-fills `inputs` with empty
    /// `RequirementSet`s (semantically `Unknown` for every row).
    /// Operators write only the inputs they actually translate; the
    /// rest stay as `Unknown`.
    ///
    /// Backed by a reusable per-node buffer in the scheduler, so this
    /// call is allocation-free in the common case once warmed up.
    fn propagate_requirements(
        &self,
        global: &Self::GlobalState,
        local: &mut Self::LocalState,
        output: &RequirementSet,
        inputs: &mut [RequirementSet],
        ctx: &RequirementCtx<'_>,
    ) -> EngineResult<()>;

    /// Whether this operator's `propagate_requirements` reads
    /// state mutated outside translation — operator-private
    /// `LocalState`, an `Arc`-shared resource not registered with
    /// the scheduler, or any source whose updates are not visible
    /// as a `T2` (downstream merged-requirement change) or a
    /// `T3` (`request_propagation` self-call). When this returns
    /// `true`, the scheduler re-arms the operator's propagation
    /// flag after every successful `run`, restoring the per-batch
    /// retranslation behavior that is otherwise removed.
    ///
    /// Default: `false` (translation depends only on `output` and
    /// static spec — pure-of-output, the most common case).
    /// Override and return `true` only when retranslation is
    /// genuinely needed at every batch, and consider migrating to
    /// an explicit `request_propagation` call from `update` or
    /// `run` when the trigger event is well-defined.
    fn propagation_depends_on_state(&self) -> bool {
        false
    }

    /// Maintenance and proposal generation. Always called for the
    /// operator each scheduler turn before EV-ranked work runs.
    /// May absorb completed async work, update internal state, read
    /// requirements, and emit `WorkProposal`s.
    ///
    /// `cx` is the lane's `std::task::Context`. Operator-owned
    /// Operator-owned futures stored in `local` should be polled with
    /// `ctx.cx()` so their wakes re-mark this lane dirty with
    /// `DirtyCause::ExternalWake`. The lane's causes for this turn are
    /// available via `ctx.causes()`.
    fn update(
        &self,
        global: &Self::GlobalState,
        local: &mut Self::LocalState,
        ctx: &mut UpdateCtx<'_>,
    ) -> EngineResult<()>;

    /// Execute one selected proposal. The scheduler chose this
    /// `WorkKey` after ranking proposals across operators.
    fn run(
        &self,
        global: &Self::GlobalState,
        local: &mut Self::LocalState,
        work: WorkKey,
        ctx: &mut WorkCtx<'_>,
    ) -> EngineResult<WorkStatus>;
}

/// Erased global state. `Sync` because lanes on different workers
/// borrow it concurrently in production; the prototype never actually
/// shares it across threads, but the bound is what the design says.
pub type ErasedGlobalState = Box<dyn Any + Send + Sync>;

/// Erased per-lane state.
pub type ErasedLocalState = Box<dyn Any + Send>;

pub trait DynOperator: Send + Sync {
    fn spec(&self) -> &OperatorSpec;
    fn propagation_depends_on_state(&self) -> bool;
    fn init_global(&self, ctx: &mut GlobalInitCtx<'_>) -> EngineResult<ErasedGlobalState>;
    fn init_local(
        &self,
        global: &(dyn Any + Send + Sync),
        ctx: &mut LocalInitCtx<'_>,
    ) -> EngineResult<ErasedLocalState>;
    fn propagate_requirements(
        &self,
        global: &(dyn Any + Send + Sync),
        local: &mut dyn Any,
        output: &RequirementSet,
        inputs: &mut [RequirementSet],
        ctx: &RequirementCtx<'_>,
    ) -> EngineResult<()>;
    fn update(
        &self,
        global: &(dyn Any + Send + Sync),
        local: &mut dyn Any,
        ctx: &mut UpdateCtx<'_>,
    ) -> EngineResult<()>;
    fn run(
        &self,
        global: &(dyn Any + Send + Sync),
        local: &mut dyn Any,
        work: WorkKey,
        ctx: &mut WorkCtx<'_>,
    ) -> EngineResult<WorkStatus>;
}

pub struct OperatorNode {
    spec: OperatorSpec,
    erased: Box<dyn DynOperator>,
}

impl OperatorNode {
    pub fn new<O>(operator: O) -> Self
    where
        O: Operator,
    {
        let spec = operator.spec();
        Self {
            spec: spec.clone(),
            erased: Box::new(TypedOperator { operator, spec }),
        }
    }

    pub fn spec(&self) -> &OperatorSpec {
        &self.spec
    }

    pub(crate) fn erased(&self) -> &dyn DynOperator {
        self.erased.as_ref()
    }
}

struct TypedOperator<O> {
    operator: O,
    spec: OperatorSpec,
}

impl<O> DynOperator for TypedOperator<O>
where
    O: Operator,
{
    fn spec(&self) -> &OperatorSpec {
        &self.spec
    }

    fn propagation_depends_on_state(&self) -> bool {
        self.operator.propagation_depends_on_state()
    }

    fn init_global(&self, ctx: &mut GlobalInitCtx<'_>) -> EngineResult<ErasedGlobalState> {
        Ok(Box::new(self.operator.init_global(ctx)?))
    }

    fn init_local(
        &self,
        global: &(dyn Any + Send + Sync),
        ctx: &mut LocalInitCtx<'_>,
    ) -> EngineResult<ErasedLocalState> {
        let Some(global) = global.downcast_ref::<O::GlobalState>() else {
            return Err(EngineError::message("operator global state type mismatch"));
        };
        Ok(Box::new(self.operator.init_local(global, ctx)?))
    }

    fn propagate_requirements(
        &self,
        global: &(dyn Any + Send + Sync),
        local: &mut dyn Any,
        output: &RequirementSet,
        inputs: &mut [RequirementSet],
        ctx: &RequirementCtx<'_>,
    ) -> EngineResult<()> {
        let (global, local) = downcast_states::<O>(global, local)?;
        self.operator
            .propagate_requirements(global, local, output, inputs, ctx)
    }

    fn update(
        &self,
        global: &(dyn Any + Send + Sync),
        local: &mut dyn Any,
        ctx: &mut UpdateCtx<'_>,
    ) -> EngineResult<()> {
        let (global, local) = downcast_states::<O>(global, local)?;
        self.operator.update(global, local, ctx)
    }

    fn run(
        &self,
        global: &(dyn Any + Send + Sync),
        local: &mut dyn Any,
        work: WorkKey,
        ctx: &mut WorkCtx<'_>,
    ) -> EngineResult<WorkStatus> {
        let (global, local) = downcast_states::<O>(global, local)?;
        self.operator.run(global, local, work, ctx)
    }
}

fn downcast_states<'a, O: Operator>(
    global: &'a (dyn Any + Send + Sync),
    local: &'a mut dyn Any,
) -> EngineResult<(&'a O::GlobalState, &'a mut O::LocalState)> {
    let global = global
        .downcast_ref::<O::GlobalState>()
        .ok_or_else(|| EngineError::message("operator global state type mismatch"))?;
    let local = local
        .downcast_mut::<O::LocalState>()
        .ok_or_else(|| EngineError::message("operator local state type mismatch"))?;
    Ok((global, local))
}

/// Side-band lookups available during `propagate_requirements`.
///
/// Output and input requirement traffic flows through the
/// `output: &RequirementSet` and `inputs: &mut [RequirementSet]`
/// arguments. This ctx is for the things that aren't a slice — today
/// that's typed resource access (used by dynamic-filter operators
/// that consult a build-side resource to refine their input
/// requirement).
pub struct RequirementCtx<'a> {
    pub(crate) resource_reader: &'a dyn Fn(&str) -> Option<ResourceValue>,
}

impl<'a> RequirementCtx<'a> {
    pub fn resource(&self, id: &str) -> Option<ResourceValue> {
        (self.resource_reader)(id)
    }
}

/// Context for `Operator::update`.
///
/// Exposes read access to channel/resource state, async-work
/// absorption, memory reservation, tracing, and a `propose` method
/// for emitting `WorkProposal`s.
pub struct UpdateCtx<'a> {
    pub(crate) operator: super::OperatorId,
    pub(crate) inputs: &'a [InputPortRef],
    pub(crate) has_output: bool,
    /// Reasons this lane was woken for the current `update` call.
    /// Operators that recompute their proposals every turn can ignore
    /// this. Operators with expensive per-call work can branch on it
    /// and skip recomputation when no relevant cause is present.
    pub(crate) causes: &'a [super::DirtyCause],
    /// Async polling context. Operators holding `Future`s should poll
    /// against this `Context`'s `Waker`; when the future fires, the
    /// scheduler re-marks this lane dirty with `DirtyCause::ExternalWake`.
    pub(crate) cx: &'a mut Context<'a>,
    pub(crate) peek_input: &'a dyn Fn(InputPortRef) -> Option<Batch>,
    pub(crate) input_finished: &'a dyn Fn(InputPortRef) -> bool,
    pub(crate) input_requirement: &'a dyn Fn(InputPortRef) -> RequirementSet,
    pub(crate) output_requirement: &'a dyn Fn() -> RequirementSet,
    pub(crate) output_capacity: &'a dyn Fn() -> bool,
    pub(crate) resource_reader: &'a dyn Fn(&str) -> Option<ResourceValue>,
    pub(crate) take_async: &'a mut dyn FnMut(AsyncWorkId) -> Option<Batch>,
    pub(crate) cancel_async: &'a mut dyn FnMut(AsyncWorkId) -> bool,
    pub(crate) broker_register: &'a mut dyn FnMut(BrokerId, super::OperatorId, InterestSpec) -> InterestId,
    pub(crate) broker_cancel: &'a mut dyn FnMut(BrokerId, InterestId),
    pub(crate) broker_take: &'a mut dyn FnMut(BrokerId, super::OperatorId) -> Option<CompletedInterest>,
    pub(crate) trace_event: &'a mut dyn FnMut(String),
    pub(crate) memory: &'a mut dyn MemoryHandle,
    pub(crate) proposals: &'a mut Vec<WorkProposal>,
    /// Atomic flag the operator can set to request that the
    /// scheduler re-run its `propagate_requirements` on the next
    /// propagation pass. See [`UpdateCtx::request_propagation`].
    pub(crate) propagation_pending: &'a AtomicBool,
}

impl<'a> UpdateCtx<'a> {
    pub fn input_count(&self) -> usize {
        self.inputs.len()
    }

    pub fn has_output(&self) -> bool {
        self.has_output
    }

    pub fn peek(&self, input: InputPortId) -> Option<Batch> {
        self.inputs
            .get(input.index())
            .and_then(|input_ref| (self.peek_input)(*input_ref))
    }

    pub fn input_finished(&self, input: InputPortId) -> bool {
        self.inputs
            .get(input.index())
            .is_some_and(|input_ref| (self.input_finished)(*input_ref))
    }

    pub fn input_requirement(&self, input: InputPortId) -> RequirementSet {
        self.inputs
            .get(input.index())
            .map(|input_ref| (self.input_requirement)(*input_ref))
            .unwrap_or_default()
    }

    pub fn output_requirement(&self) -> RequirementSet {
        if self.has_output {
            (self.output_requirement)()
        } else {
            RequirementSet::default()
        }
    }

    pub fn has_capacity(&self) -> bool {
        self.has_output && (self.output_capacity)()
    }

    pub fn resource(&self, id: &str) -> Option<ResourceValue> {
        (self.resource_reader)(id)
    }

    pub fn take_async(&mut self, id: AsyncWorkId) -> Option<Batch> {
        (self.take_async)(id)
    }

    pub fn cancel_async(&mut self, id: AsyncWorkId) -> bool {
        (self.cancel_async)(id)
    }

    pub fn trace(&mut self, reason: impl Into<String>) {
        (self.trace_event)(reason.into());
    }

    pub fn memory(&mut self) -> &mut dyn MemoryHandle {
        self.memory
    }

    /// Emit a work proposal to the scheduler.
    pub fn propose(&mut self, proposal: WorkProposal) {
        self.proposals.push(proposal);
    }

    /// Register an interest with a broker. Returns the interest id
    /// the operator should store to take the result later.
    pub fn broker_register(&mut self, broker: BrokerId, spec: InterestSpec) -> InterestId {
        (self.broker_register)(broker, self.operator, spec)
    }

    /// Cancel a previously registered broker interest.
    pub fn broker_cancel(&mut self, broker: BrokerId, interest: InterestId) {
        (self.broker_cancel)(broker, interest)
    }

    /// Take a completed broker result destined for this operator.
    pub fn broker_take(&mut self, broker: BrokerId) -> Option<CompletedInterest> {
        (self.broker_take)(broker, self.operator)
    }

    /// Mark this operator's `propagate_requirements` for re-run on
    /// the next propagation pass. Use when local state — or a
    /// resource the operator reads during translation — has just
    /// changed in a way that would alter the translation output.
    /// Idempotent and cheap (single relaxed atomic store); no-op if
    /// the flag is already set.
    pub fn request_propagation(&self) {
        self.propagation_pending.store(true, Ordering::Release);
    }
}

/// Context for `Operator::run`.
///
/// Provides full mutate access to inputs, outputs, resources, and
/// broker submission. Memory reservation is constrained to what the
/// proposal's `WorkConstraints` declared.
pub struct WorkCtx<'a> {
    pub(crate) inputs: &'a [InputPortRef],
    pub(crate) has_output: bool,
    pub(crate) peek_input: &'a dyn Fn(InputPortRef) -> Option<Batch>,
    pub(crate) pop_input: &'a mut dyn FnMut(InputPortRef) -> Option<Batch>,
    pub(crate) input_finished: &'a dyn Fn(InputPortRef) -> bool,
    pub(crate) input_requirement: &'a dyn Fn(InputPortRef) -> RequirementSet,
    pub(crate) output_requirement: &'a dyn Fn() -> RequirementSet,
    pub(crate) output_capacity: &'a dyn Fn() -> bool,
    pub(crate) push_output: &'a mut dyn FnMut(Batch) -> EngineResult<()>,
    pub(crate) seal_output: &'a mut dyn FnMut() -> EngineResult<()>,
    pub(crate) resource_reader: &'a dyn Fn(&str) -> Option<ResourceValue>,
    pub(crate) resource_writer: &'a mut dyn FnMut(&str, ResourceValue) -> EngineResult<()>,
    pub(crate) spawn_fake_io: &'a mut dyn FnMut(FakeIoRequest) -> EngineResult<AsyncWorkId>,
    pub(crate) take_async: &'a mut dyn FnMut(AsyncWorkId) -> Option<Batch>,
    pub(crate) cancel_async: &'a mut dyn FnMut(AsyncWorkId) -> bool,
    pub(crate) trace_event: &'a mut dyn FnMut(String),
    pub(crate) memory: &'a mut dyn MemoryHandle,
    /// See [`WorkCtx::request_propagation`].
    pub(crate) propagation_pending: &'a AtomicBool,
}

impl<'a> WorkCtx<'a> {
    pub fn peek(&self, input: InputPortId) -> Option<Batch> {
        self.inputs
            .get(input.index())
            .and_then(|input_ref| (self.peek_input)(*input_ref))
    }

    pub fn pop(&mut self, input: InputPortId) -> Option<Batch> {
        self.inputs
            .get(input.index())
            .and_then(|input_ref| (self.pop_input)(*input_ref))
    }

    pub fn input_finished(&self, input: InputPortId) -> bool {
        self.inputs
            .get(input.index())
            .is_some_and(|input_ref| (self.input_finished)(*input_ref))
    }

    pub fn input_requirement(&self, input: InputPortId) -> RequirementSet {
        self.inputs
            .get(input.index())
            .map(|input_ref| (self.input_requirement)(*input_ref))
            .unwrap_or_default()
    }

    pub fn has_output(&self) -> bool {
        self.has_output
    }

    pub fn output_requirement(&self) -> RequirementSet {
        if self.has_output {
            (self.output_requirement)()
        } else {
            RequirementSet::default()
        }
    }

    pub fn has_capacity(&self) -> bool {
        self.has_output && (self.output_capacity)()
    }

    pub fn push(&mut self, batch: Batch) -> EngineResult<()> {
        if !self.has_output {
            return Err(EngineError::message("operator has no output port"));
        }
        (self.push_output)(batch)
    }

    pub fn seal(&mut self) -> EngineResult<()> {
        if !self.has_output {
            return Err(EngineError::message("operator has no output port"));
        }
        (self.seal_output)()
    }

    pub fn resource(&self, id: &str) -> Option<ResourceValue> {
        (self.resource_reader)(id)
    }

    pub fn publish_resource(&mut self, id: &str, value: ResourceValue) -> EngineResult<()> {
        (self.resource_writer)(id, value)
    }

    pub fn spawn_fake_io(&mut self, request: FakeIoRequest) -> EngineResult<AsyncWorkId> {
        (self.spawn_fake_io)(request)
    }

    pub fn take_async(&mut self, id: AsyncWorkId) -> Option<Batch> {
        (self.take_async)(id)
    }

    pub fn cancel_async(&mut self, id: AsyncWorkId) -> bool {
        (self.cancel_async)(id)
    }

    pub fn trace(&mut self, reason: impl Into<String>) {
        (self.trace_event)(reason.into());
    }

    pub fn memory(&mut self) -> &mut dyn MemoryHandle {
        self.memory
    }

    /// Mark this operator's `propagate_requirements` for re-run on
    /// the next propagation pass. See
    /// [`UpdateCtx::request_propagation`] for semantics.
    pub fn request_propagation(&self) {
        self.propagation_pending.store(true, Ordering::Release);
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MemoryReason {
    OutputBatch,
    RetainedWitness,
    ReorderBuffer,
    SourceIo,
    ResourcePayload,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MemoryReservation {
    bytes: usize,
}

impl MemoryReservation {
    pub const fn new(bytes: usize) -> Self {
        Self { bytes }
    }

    pub const fn bytes(&self) -> usize {
        self.bytes
    }
}

pub trait MemoryHandle {
    fn try_reserve(
        &mut self,
        bytes: usize,
        reason: MemoryReason,
    ) -> EngineResult<Option<MemoryReservation>>;
}

#[derive(Default)]
pub struct NoopMemoryHandle;

impl MemoryHandle for NoopMemoryHandle {
    fn try_reserve(
        &mut self,
        bytes: usize,
        _reason: MemoryReason,
    ) -> EngineResult<Option<MemoryReservation>> {
        Ok(Some(MemoryReservation::new(bytes)))
    }
}
