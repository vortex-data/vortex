mod async_work;
mod trace;
mod turn;
mod worker;

pub use async_work::*;
pub use trace::*;
pub use worker::WorkerId;
pub(crate) use worker::{LaneRuntime, WorkerRuntime};

use std::collections::BTreeMap;
use std::collections::BTreeSet;
use std::collections::VecDeque;
use std::sync::Arc;
use std::sync::atomic::AtomicBool;
use std::sync::atomic::AtomicUsize;
use std::sync::atomic::Ordering;
use std::task::RawWaker;
use std::task::RawWakerVTable;
use std::task::Waker;

use parking_lot::Mutex;
use crate::AsyncWorkId;
use crate::Batch;
use crate::Broker;
use crate::BrokerGrant;
use crate::BrokerId;
use crate::BrokerProposal;
use crate::Channel;
use crate::CompletedInterest;
use crate::EngineError;
use crate::EngineResult;
use crate::InputPortRef;
use crate::InterestId;
use crate::InterestSpec;
use crate::OperatorGraph;
use crate::OperatorId;
use crate::OperatorNode;
use crate::RequirementSet;
use crate::Resource;
use crate::WorkProposal;
use crate::WorkStatus;

/// Reason a lane was marked dirty for the next scheduler turn.
///
/// Multiple causes may be queued between turns; the operator reads
/// the slice via `UpdateCtx::causes()` and either ignores it
/// (recompute everything) or branches on it (touch only what
/// changed). On the very first turn each lane is seeded with
/// `Initial` so operators get exactly one wake to lay down their
/// starting proposals.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum DirtyCause {
    /// First wake after preparation. Operators should populate
    /// initial proposals here.
    Initial,
    /// A batch landed on this input port.
    InputArrived { port: crate::InputPortId },
    /// This input port observed its sealed signal.
    InputSealed { port: crate::InputPortId },
    /// Downstream consumed from one of our output channels — there's
    /// likely capacity to push again.
    OutputCapacityFreed,
    /// A downstream consumer changed the requirement on one of our
    /// output channels. Re-run propagation; possibly re-run update.
    OutputRequirementChanged,
    /// The named resource published a new value.
    ResourceUpdated { id: String },
    /// An async work item we registered completed.
    AsyncCompleted { id: AsyncWorkId },
    /// An external `Waker` we handed out fired (see
    /// `UpdateCtx::waker()`). Whoever fired it is responsible for
    /// having actually moved state forward; the cause itself carries
    /// no detail.
    ExternalWake,
}

/// Per-lane wake signal: a fast `pending` flag for the scheduler's
/// scan plus a list of causes the lane will see on its next call.
struct DirtySignal {
    pending: AtomicBool,
    causes: Mutex<Vec<DirtyCause>>,
}

impl DirtySignal {
    fn new_initial() -> Self {
        Self {
            pending: AtomicBool::new(true),
            causes: Mutex::new(vec![DirtyCause::Initial]),
        }
    }

    /// Append a cause and mark pending. Cheap: brief mutex hold.
    fn push(&self, cause: DirtyCause) {
        self.causes.lock().push(cause);
        self.pending.store(true, Ordering::Release);
    }

    fn is_pending(&self) -> bool {
        self.pending.load(Ordering::Acquire)
    }

    /// Atomically clear the pending flag and take all queued causes.
    /// Returns an empty Vec if the lane wasn't pending.
    fn drain(&self) -> Vec<DirtyCause> {
        if !self.pending.swap(false, Ordering::AcqRel) {
            return Vec::new();
        }
        std::mem::take(&mut *self.causes.lock())
    }

    /// Return true if any queued cause matches `predicate`. Does
    /// not drain — leaves causes for the eventual `update` call.
    /// Used by `propagate_requirements_with_ctx` to skip operators
    /// whose downstream requirements haven't changed.
    fn has_cause(&self, predicate: impl Fn(&DirtyCause) -> bool) -> bool {
        if !self.is_pending() {
            return false;
        }
        self.causes.lock().iter().any(predicate)
    }
}

// ---------------------------------------------------------------------
// Lane waker: builds a `std::task::Waker` that pushes
// `DirtyCause::ExternalWake` to a specific lane's `DirtySignal` when
// fired. Used by `update`/`run` to integrate with external `Future`s.
// ---------------------------------------------------------------------

fn lane_waker(signal: Arc<DirtySignal>) -> Waker {
    let raw = lane_waker_raw(Arc::into_raw(signal));
    // SAFETY: the vtable below correctly clone/wake/drop the
    // `Arc<DirtySignal>` whose pointer we just leaked.
    unsafe { Waker::from_raw(raw) }
}

fn lane_waker_raw(ptr: *const DirtySignal) -> RawWaker {
    RawWaker::new(ptr.cast(), &LANE_WAKER_VTABLE)
}

const LANE_WAKER_VTABLE: RawWakerVTable = RawWakerVTable::new(
    lane_waker_clone,
    lane_waker_wake,
    lane_waker_wake_by_ref,
    lane_waker_drop,
);

unsafe fn lane_waker_clone(data: *const ()) -> RawWaker {
    let arc = unsafe { Arc::from_raw(data.cast::<DirtySignal>()) };
    let cloned = Arc::clone(&arc);
    // Avoid dropping the original.
    std::mem::forget(arc);
    lane_waker_raw(Arc::into_raw(cloned))
}

unsafe fn lane_waker_wake(data: *const ()) {
    let arc = unsafe { Arc::from_raw(data.cast::<DirtySignal>()) };
    arc.push(DirtyCause::ExternalWake);
    // arc dropped here.
}

unsafe fn lane_waker_wake_by_ref(data: *const ()) {
    let arc = unsafe { Arc::from_raw(data.cast::<DirtySignal>()) };
    arc.push(DirtyCause::ExternalWake);
    // Don't drop — the caller still owns its ref.
    std::mem::forget(arc);
}

unsafe fn lane_waker_drop(data: *const ()) {
    drop(unsafe { Arc::from_raw(data.cast::<DirtySignal>()) });
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct ExecutionMetrics {
    source_rows_read: BTreeMap<String, usize>,
    source_rows_skipped: BTreeMap<String, usize>,
    source_value_reads: BTreeMap<(String, usize), usize>,
    source_value_dontcare: BTreeMap<(String, usize), usize>,
    lazy_materialized_rows: BTreeMap<String, usize>,
    async_started: BTreeMap<String, usize>,
    async_completed: BTreeMap<String, usize>,
    async_cancelled: BTreeMap<String, usize>,
    async_wakeups: BTreeMap<String, usize>,
    channel_topologies: BTreeMap<String, &'static str>,
    lazy_column_materialized: BTreeMap<(String, usize), usize>,
    lazy_batches_emitted: BTreeMap<String, usize>,
    peak_memory_bytes: usize,
}

impl ExecutionMetrics {
    pub fn source_rows_read(&self, name: &str) -> usize {
        self.source_rows_read.get(name).copied().unwrap_or_default()
    }

    pub fn source_rows_skipped(&self, name: &str) -> usize {
        self.source_rows_skipped
            .get(name)
            .copied()
            .unwrap_or_default()
    }

    pub fn source_value_reads(&self, name: &str, column: usize) -> usize {
        self.source_value_reads
            .get(&(name.to_owned(), column))
            .copied()
            .unwrap_or_default()
    }

    pub fn source_value_dontcare(&self, name: &str, column: usize) -> usize {
        self.source_value_dontcare
            .get(&(name.to_owned(), column))
            .copied()
            .unwrap_or_default()
    }

    pub fn lazy_materialized_rows(&self, name: &str) -> usize {
        self.lazy_materialized_rows
            .get(name)
            .copied()
            .unwrap_or_default()
    }

    pub fn async_started(&self, name: &str) -> usize {
        self.async_started.get(name).copied().unwrap_or_default()
    }

    pub fn async_completed(&self, name: &str) -> usize {
        self.async_completed.get(name).copied().unwrap_or_default()
    }

    pub fn async_cancelled(&self, name: &str) -> usize {
        self.async_cancelled.get(name).copied().unwrap_or_default()
    }

    pub fn async_wakeups(&self, name: &str) -> usize {
        self.async_wakeups.get(name).copied().unwrap_or_default()
    }

    pub fn channel_topology(&self, name: &str) -> Option<&'static str> {
        self.channel_topologies.get(name).copied()
    }

    pub fn lazy_column_materialized(&self, name: &str, column: usize) -> usize {
        self.lazy_column_materialized
            .get(&(name.to_owned(), column))
            .copied()
            .unwrap_or_default()
    }

    pub fn lazy_batches_emitted(&self, name: &str) -> usize {
        self.lazy_batches_emitted
            .get(name)
            .copied()
            .unwrap_or_default()
    }

    pub const fn peak_memory_bytes(&self) -> usize {
        self.peak_memory_bytes
    }

    pub fn add_source_rows_read(&mut self, name: &str, rows: usize) {
        *self.source_rows_read.entry(name.to_owned()).or_default() += rows;
    }

    pub fn add_source_rows_skipped(&mut self, name: &str, rows: usize) {
        *self.source_rows_skipped.entry(name.to_owned()).or_default() += rows;
    }

    pub fn add_source_value_read(&mut self, name: &str, column: usize, rows: usize) {
        *self
            .source_value_reads
            .entry((name.to_owned(), column))
            .or_default() += rows;
    }

    pub fn add_source_value_dontcare(&mut self, name: &str, column: usize, rows: usize) {
        *self
            .source_value_dontcare
            .entry((name.to_owned(), column))
            .or_default() += rows;
    }

    pub fn add_lazy_materialized_rows(&mut self, name: &str, rows: usize) {
        *self
            .lazy_materialized_rows
            .entry(name.to_owned())
            .or_default() += rows;
    }

    pub fn add_async_started(&mut self, name: &str) {
        *self.async_started.entry(name.to_owned()).or_default() += 1;
    }

    pub fn add_async_completed(&mut self, name: &str) {
        *self.async_completed.entry(name.to_owned()).or_default() += 1;
    }

    pub fn add_async_cancelled(&mut self, name: &str) {
        *self.async_cancelled.entry(name.to_owned()).or_default() += 1;
    }

    pub fn add_async_wakeup(&mut self, name: &str) {
        *self.async_wakeups.entry(name.to_owned()).or_default() += 1;
    }

    pub fn set_channel_topology(&mut self, name: &str, topology: &'static str) {
        self.channel_topologies.insert(name.to_owned(), topology);
    }

    pub fn add_lazy_column_materialized(&mut self, name: &str, column: usize, rows: usize) {
        *self
            .lazy_column_materialized
            .entry((name.to_owned(), column))
            .or_default() += rows;
    }

    pub fn add_lazy_batch_emitted(&mut self, name: &str) {
        *self
            .lazy_batches_emitted
            .entry(name.to_owned())
            .or_default() += 1;
    }

    pub fn observe_memory_bytes(&mut self, bytes: usize) {
        self.peak_memory_bytes = self.peak_memory_bytes.max(bytes);
    }
}

/// Side-effects observed during phase-1 maintenance, used by
/// outcome classification.
#[derive(Clone, Copy, Debug, Default)]
pub struct TurnPhase1 {
    pub async_wake: bool,
    pub substrate_activity: bool,
    pub propagated: bool,
    pub grants_changed: bool,
}

/// Driver-side handle to one worker's turn primitive. Held while a
/// driver dispatches per-worker turns concurrently (e.g. via a
/// thread pool). All fields are references with the lifetime of the
/// originating `&PreparedTask` borrow.
pub struct WorkerHandle<'a> {
    ctx: WorkerCtx<'a>,
    worker_id: WorkerId,
}

impl<'a> WorkerHandle<'a> {
    /// Identifier of this worker in the pool.
    pub fn worker_id(&self) -> WorkerId {
        self.worker_id
    }

    /// Run one full turn of this worker's autonomous loop: drain
    /// `incoming`, update woken lanes, pop EV-ranked work, run, and
    /// try stealing from peers when local work runs dry. Returns
    /// `true` if the worker made forward progress.
    pub fn turn(&mut self) -> EngineResult<bool> {
        PreparedTask::worker_turn(self.ctx, self.worker_id)
    }
}

/// Result of one scheduler turn. Drivers consume this to decide
/// whether to run another turn immediately, yield to peers, wait on
/// a wake, or wrap up.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TurnOutcome {
    /// Forward progress happened: an action ran, requirements
    /// changed, async completion was absorbed, or memory was
    /// rebalanced. The driver should run another turn soon.
    Made,
    /// No forward progress this turn, but external work is in
    /// flight (pending async I/O or a broker waiting on completion).
    /// The driver should wait on a wake before running another turn.
    Idle,
    /// Every operator has reached terminal. The driver may now
    /// consume the task via `finish_report()`.
    Done,
}

#[derive(Clone, Debug)]
pub struct TaskOptions {
    pub max_turns: usize,
    pub memory_limit_bytes: usize,
    /// Number of worker threads the driver intends to run this task
    /// on. Lane discovery uses this to clamp `Parallelism::Workers`
    /// and to inform per-shard placement. Drivers should set this
    /// to their pool size (or 1 for a single-thread caller).
    pub worker_count: usize,
}

impl Default for TaskOptions {
    fn default() -> Self {
        Self {
            max_turns: 10_000,
            memory_limit_bytes: 4096,
            worker_count: 1,
        }
    }
}

pub struct PreparedTask {
    nodes: Vec<OperatorRuntime>,
    /// Per-channel mutex-protected state. Each channel has its own
    /// lock so two operators using different channels don't
    /// serialise. parking_lot's uncontended cost is ~10ns; for our
    /// per-turn channel-op count this is invisible single-threaded.
    /// Multi-threaded paths (per-shard scheduler) need locking
    /// because operators on different shards may push/pop the same
    /// channel concurrently.
    channels: Vec<Mutex<Channel>>,
    /// Shared resource registry. Each `Resource` is mutex-protected
    /// individually so writers and readers on different shards
    /// don't serialise on a global resource lock.
    resources: BTreeMap<String, Mutex<Resource>>,
    /// One mutex per broker. Per-broker locks are short and brokers
    /// rarely contend (each shard typically has its own broker
    /// interest set).
    brokers: Vec<Mutex<Box<dyn Broker>>>,
    /// Refreshed at start of each turn; scanned during admission.
    broker_proposals: Mutex<Vec<BrokerProposal>>,
    async_work: Mutex<AsyncWorkSet>,
    trace: Mutex<ScheduleTrace>,
    metrics: Arc<Mutex<ExecutionMetrics>>,
    options: TaskOptions,
    last_run_node: OperatorId,
    /// Flat lane storage: one entry per `(op, lane)` pair, indexed by
    /// `op_lane_offset[op_idx] + lane.index`. Wrapped in `Mutex` so
    /// the current owning worker can hold it through the operator's
    /// `update`/`run` (potentially milliseconds) without blocking
    /// peers' wake-routing or admission decisions. Peers touch only
    /// the worker's `shared` Mutex, never the `LaneRuntime` itself.
    lanes: Vec<Mutex<LaneRuntime>>,
    /// `op_lane_offset[op_idx]` = base index into `lanes` for op.
    op_lane_offset: Vec<usize>,
    /// `op_lane_count[op_idx]` = number of lanes for op.
    op_lane_count: Vec<usize>,
    /// Current owner of each lane. Index into `workers`. Stealing
    /// CAS-updates this; channel push wake-routing reads it.
    lane_owner: Vec<AtomicUsize>,
    /// Lock-free finished flag per lane. Mirrors
    /// `LaneRuntime::finished` but allows `is_complete()` to scan
    /// without taking lane locks.
    lane_finished: Vec<AtomicBool>,
    /// Per-lane wake signal. Replaces the old per-(shard, lane)
    /// dirty signal grid.
    lane_dirty: Vec<Arc<DirtySignal>>,
    /// Per-operator "propagation needed" flag. Set by
    /// `set_requirement` when a downstream channel's merged
    /// requirement changes; survives `update_lane`'s drain so it
    /// correctly persists across multi-pass turns.
    propagation_pending: Vec<AtomicBool>,
    /// Reverse-topological iteration order for
    /// `propagate_requirements`: sinks first, sources last. A single
    /// pass following this order completes the demand cascade
    /// because every consumer is processed before any of its
    /// producers, so flags set on producers during the pass are
    /// picked up later in the same pass.
    propagation_order: Vec<usize>,
    /// Per-worker runtime state: EV heap, incoming wake queue,
    /// deferred entries. Indexed by `WorkerId`.
    workers: Vec<WorkerRuntime>,
}

struct OperatorRuntime {
    id: OperatorId,
    node: OperatorNode,
    /// Per-node global state. Built once via `init_global`. Lives in
    /// `PreparedTask` and is borrowed by `&` across all shards
    /// during a turn (multi-thread access is fine because the trait
    /// requires `GlobalState: Send + Sync`).
    global: super::ErasedGlobalState,
    /// Number of lanes derived from `OperatorSpec::parallelism` at
    /// preparation, possibly clamped by `TaskOptions::worker_count`
    /// for `Workers { max }` operators.
    lane_count: usize,
    inputs: Vec<InputPortRef>,
    /// True if this operator has an output port; false for sinks.
    has_output: bool,
    /// Channel index for each input port (parallel to `inputs`).
    /// `None` means the port is unconnected. Indices into
    /// `PreparedTask::channels`.
    input_channels: Vec<Option<usize>>,
    /// Channel indices for the single output port (fanout for SPMC).
    /// Empty vec means no output or unconnected. Indices into
    /// `PreparedTask::channels`.
    output_channels: Vec<usize>,
}

impl OperatorRuntime {
    fn input_channel(&self, port: super::InputPortId) -> Option<usize> {
        self.input_channels.get(port.index()).copied().flatten()
    }

    fn output_channel_indices(&self) -> &[usize] {
        &self.output_channels
    }

}

/// Bundle of shared (immutable, mutex-protected) references a
/// worker thread needs to drive its turn loop. All fields are
/// references — the struct is `Copy` and `Send + Sync`, so it
/// crosses thread boundaries trivially.
///
/// Note: `LaneRuntime`, `WorkerRuntime`, `HeapEntry` are defined in
/// `worker.rs`. This struct names them via the `pub(crate) use` at
/// the top of `mod.rs`.
#[derive(Clone, Copy)]
pub(crate) struct WorkerCtx<'a> {
    pub nodes: &'a [OperatorRuntime],
    pub channels: &'a [Mutex<Channel>],
    pub resources: &'a BTreeMap<String, Mutex<Resource>>,
    pub brokers: &'a [Mutex<Box<dyn Broker>>],
    pub broker_proposals: &'a Mutex<Vec<BrokerProposal>>,
    pub async_work: &'a Mutex<AsyncWorkSet>,
    pub trace: &'a Mutex<ScheduleTrace>,
    pub metrics: &'a Arc<Mutex<ExecutionMetrics>>,
    /// Flat `(op, lane)` storage: index by `op_lane_offset[op] + lane.index`.
    pub lanes: &'a [Mutex<LaneRuntime>],
    /// `op_lane_offset[op_idx]` = base index into `lanes` for op.
    pub op_lane_offset: &'a [usize],
    /// `op_lane_count[op_idx]` = number of lanes for op.
    pub op_lane_count: &'a [usize],
    /// `lane_owner[lane_addr]` = current owning worker id (atomic).
    pub lane_owner: &'a [AtomicUsize],
    /// `lane_finished[lane_addr]` = atomic finished flag (lock-free
    /// for `is_complete`-style scans).
    pub lane_finished: &'a [AtomicBool],
    /// Per-lane wake signal (replaces the per-shard dirty signals).
    pub lane_dirty: &'a [Arc<DirtySignal>],
    /// Per-op propagation-pending flag.
    pub propagation_pending: &'a [AtomicBool],
    /// Reverse-topological op iteration order used by
    /// `propagate_requirements` so the demand cascade completes in
    /// one pass.
    pub propagation_order: &'a [usize],
    /// Per-worker runtime state (heap + incoming queue + deferred).
    pub workers: &'a [WorkerRuntime],
    pub memory_limit_bytes: usize,
}

impl<'a> WorkerCtx<'a> {
    /// Push one event onto the shared trace. Replaces the verbose
    /// `ctx.trace.lock().push(...)` call sites — same semantics,
    /// scoped lock so callers don't accidentally hold it across an
    /// unrelated op call.
    pub fn trace(&self, event: TraceEvent) {
        self.trace.lock().push(event);
    }

    /// Wake every lane of `op` by routing the dirty signal to the
    /// current owner of each lane (via `lane_owner`).
    pub fn mark_op_dirty(&self, op: OperatorId, cause: DirtyCause) {
        let op_idx = op.index();
        let start = self.op_lane_offset[op_idx];
        let count = self.op_lane_count[op_idx];
        for i in 0..count {
            let lane_addr = start + i;
            self.lane_dirty[lane_addr].push(cause.clone());
            let owner = self.lane_owner[lane_addr].load(Ordering::Acquire);
            self.workers[owner]
                .shared
                .lock()
                .incoming
                .push_back(lane_addr);
        }
    }

    /// Wake every lane (across all ops). Conservative fallback when
    /// the precise lane targets aren't tracked yet.
    pub fn mark_all_dirty(&self, cause: DirtyCause) {
        for (lane_addr, signal) in self.lane_dirty.iter().enumerate() {
            signal.push(cause.clone());
            let owner = self.lane_owner[lane_addr].load(Ordering::Acquire);
            self.workers[owner]
                .shared
                .lock()
                .incoming
                .push_back(lane_addr);
        }
    }
}

impl PreparedTask {
    pub fn prepare(
        graph: OperatorGraph,
        metrics: Arc<Mutex<ExecutionMetrics>>,
        options: TaskOptions,
    ) -> EngineResult<Self> {
        let (nodes, channel_specs, resource_specs) = graph.into_parts();

        // Shard count: one per worker thread, at least 1.
        let shard_count = options.worker_count.max(1);

        // First pass: build OperatorRuntime metadata + build all
        // (op, lane) LaneRuntimes flat. We'll distribute them into
        // shards in the next pass.
        let mut runtime_nodes = Vec::with_capacity(nodes.len());
        let mut all_lanes: Vec<LaneRuntime> = Vec::new();
        for (index, node) in nodes.into_iter().enumerate() {
            let id = OperatorId::from_index(index);
            let mut global_ctx = super::GlobalInitCtx {
                operator: id,
                _phantom: std::marker::PhantomData,
            };
            let global = node.erased().init_global(&mut global_ctx)?;

            // Lane discovery:
            // - `Serial`: 1 lane.
            // - `Lanes(n)`: `n` lanes (data-determined; each lane is
            //   one work unit).
            // - `Workers { max }`: clamped to the driver's worker
            //   count (each lane is a worker, not a work unit).
            let lane_count = match node.spec().parallelism {
                super::Parallelism::Serial => 1,
                super::Parallelism::Lanes { max } => {
                    let cap = max.unwrap_or(usize::MAX);
                    cap.min(options.worker_count).max(1)
                }
            };

            let n_inputs = node.spec().inputs.len();
            let inputs: Vec<InputPortRef> = (0..n_inputs)
                .map(|port| InputPortRef::new(id, super::InputPortId::from_index(port)))
                .collect();
            let has_output = node.spec().output.is_some();

            for lane_idx in 0..lane_count {
                let mut local_ctx = super::LocalInitCtx {
                    operator: id,
                    lane: super::LaneId::new(lane_idx),
                    lane_count,
                    _phantom: std::marker::PhantomData,
                };
                let local = node
                    .erased()
                    .init_local(global.as_ref(), &mut local_ctx)?;
                all_lanes.push(LaneRuntime {
                    op: id,
                    lane: super::LaneId::new(lane_idx),
                    local,
                    finished: false,
                    proposals: Vec::new(),
                    propagate_inputs_buffer: vec![RequirementSet::default(); n_inputs],
                    epoch: 0,
                });
            }

            let input_channels_for_op = vec![None; n_inputs];
            runtime_nodes.push(OperatorRuntime {
                id,
                node,
                global,
                lane_count,
                inputs,
                has_output,
                input_channels: input_channels_for_op,
                output_channels: Vec::new(),
            });
        }

        // Bind-time sort validation. For each channel spec, compare
        // the producer's output `sort_key` against every consumer
        // input port's `required_sort`. A consumer requiring a sort
        // the producer doesn't claim is a fatal bind error — we
        // can't auto-insert a Sort operator (task-graph machinery
        // is not yet implemented), so the only honest response is
        // to refuse the query at prepare time.
        for spec in &channel_specs {
            for input_ref in &spec.to {
                let consumer = &runtime_nodes[input_ref.operator().index()];
                let consumer_spec = consumer.node.spec();
                let input_port = consumer_spec
                    .inputs
                    .get(input_ref.port().index())
                    .ok_or_else(|| {
                        EngineError::message(format!(
                            "bind: channel '{}' references invalid input port {} on op#{}",
                            spec.label,
                            input_ref.port().index(),
                            input_ref.operator().index(),
                        ))
                    })?;
                let Some(required) = input_port.required_sort() else {
                    continue;
                };
                // Every producer on this channel must claim a matching
                // sort. (Today's auto-derived case: single-producer
                // channels — the one producer must satisfy. Multi-
                // producer: every connected producer's sort must
                // match; the planner is also responsible for wiring
                // producers in key-monotonic order so that connection-
                // add-order drain preserves K, but this validation only
                // catches sort declarations.)
                for producer_op in &spec.from {
                    let producer = &runtime_nodes[producer_op.index()];
                    let producer_spec = producer.node.spec();
                    let producer_sort = producer_spec
                        .output
                        .as_ref()
                        .and_then(|p| p.sort_key.as_ref());
                    if producer_sort != Some(required) {
                        return Err(EngineError::message(format!(
                            "bind: sort mismatch on channel '{}': consumer '{}' input port \
                             '{}' requires sort {:?} but producer '{}' output declares {:?}. \
                             Insert a Sort/exchange to bridge (not yet supported).",
                            spec.label,
                            consumer_spec.label,
                            input_port.name,
                            required,
                            producer_spec.label,
                            producer_sort,
                        )));
                    }
                }
            }
        }

        // Build channels and populate per-operator channel-index
        // arrays in one pass. Each `channel_spec.from` is an output
        // port of some operator; each `channel_spec.to[i]` is an
        // input port. We index directly into the runtime nodes —
        // no intermediate maps.
        let channels: Vec<Mutex<Channel>> = channel_specs
            .into_iter()
            .enumerate()
            .map(|(index, spec)| {
                for input in &spec.to {
                    let op = input.operator().index();
                    let port = input.port().index();
                    runtime_nodes[op].input_channels[port] = Some(index);
                }
                for producer_op in &spec.from {
                    runtime_nodes[producer_op.index()]
                        .output_channels
                        .push(index);
                }
                metrics
                    .lock()
                    .set_channel_topology(&spec.label, spec.topology.as_str());
                Mutex::new(Channel::new(spec))
            })
            .collect();

        // Reverse-topological propagation order. Sinks first, then
        // their producers, then theirs, etc., so that one
        // `propagate_requirements` pass walks the demand cascade
        // sink-to-source in a single traversal: any flag a consumer
        // sets on its producer is picked up later in the same pass,
        // not deferred to a future propagate call. Computed once at
        // preparation time; immutable for the rest of the task's
        // life since the operator graph and channel topology are
        // fixed after `prepare`.
        let n_ops_for_order = runtime_nodes.len();
        let propagation_order: Vec<usize> = {
            let mut out_degree: Vec<usize> = (0..n_ops_for_order)
                .map(|op_idx| {
                    runtime_nodes[op_idx]
                        .output_channels
                        .iter()
                        .map(|ci| channels[*ci].lock().spec().to.len())
                        .sum::<usize>()
                })
                .collect();
            let mut queue: VecDeque<usize> = VecDeque::new();
            for (op_idx, &deg) in out_degree.iter().enumerate() {
                if deg == 0 {
                    queue.push_back(op_idx);
                }
            }
            let mut order: Vec<usize> = Vec::with_capacity(n_ops_for_order);
            while let Some(op_idx) = queue.pop_front() {
                order.push(op_idx);
                for input_ch_opt in &runtime_nodes[op_idx].input_channels {
                    if let Some(ci) = input_ch_opt {
                        let producers: Vec<usize> = channels[*ci]
                            .lock()
                            .spec()
                            .from
                            .iter()
                            .map(|o| o.index())
                            .collect();
                        for producer_idx in producers {
                            out_degree[producer_idx] =
                                out_degree[producer_idx].saturating_sub(1);
                            if out_degree[producer_idx] == 0 {
                                queue.push_back(producer_idx);
                            }
                        }
                    }
                }
            }
            // If the graph has cycles or stranded ops, fall back to
            // index order for the remainder. Cycles aren't supported
            // by the engine but this keeps the code defensive.
            if order.len() != n_ops_for_order {
                let visited: BTreeSet<usize> = order.iter().copied().collect();
                for op_idx in 0..n_ops_for_order {
                    if !visited.contains(&op_idx) {
                        order.push(op_idx);
                    }
                }
            }
            order
        };

        // Build flat lane storage. Lanes are appended in op-id order
        // so `op_lane_offset[op] + lane_idx` is a simple add. For
        // each op, lanes 0..lane_count are pushed in order.
        let n_ops = runtime_nodes.len();
        let mut op_lane_offset: Vec<usize> = vec![0; n_ops];
        let mut op_lane_count_v: Vec<usize> = vec![0; n_ops];
        let mut lanes_by_op: Vec<Vec<LaneRuntime>> = (0..n_ops).map(|_| Vec::new()).collect();
        for lane in all_lanes {
            lanes_by_op[lane.op.index()].push(lane);
        }
        let mut flat_lanes: Vec<Mutex<LaneRuntime>> = Vec::new();
        for (op_idx, op_lanes) in lanes_by_op.into_iter().enumerate() {
            op_lane_offset[op_idx] = flat_lanes.len();
            op_lane_count_v[op_idx] = op_lanes.len();
            for lane in op_lanes {
                flat_lanes.push(Mutex::new(lane));
            }
        }
        let total_lanes = flat_lanes.len();

        // Worker count: requested by options, at least 1.
        let worker_count = options.worker_count.max(1);

        // Initial lane→worker placement. Distribute lanes across workers
        // to balance the *initial* set; work-stealing rebalances at run
        // time. Same intuition as the old shard placement:
        // - Serial ops (lane_count = 1): round-robin by op_idx so they
        //   don't all pile on worker 0.
        // - Lanes ops (lane_count > 1): spread by lane.index.
        let mut lane_owner: Vec<AtomicUsize> = Vec::with_capacity(total_lanes);
        for (op_idx, node) in runtime_nodes.iter().enumerate() {
            let count = node.lane_count;
            for lane_idx in 0..count {
                let worker = if count == 1 {
                    op_idx % worker_count
                } else {
                    lane_idx % worker_count
                };
                lane_owner.push(AtomicUsize::new(worker));
            }
        }

        // Per-lane finished flags + dirty signals. Every lane starts
        // dirty with `Initial` so the first turn updates everything.
        let lane_finished: Vec<AtomicBool> =
            (0..total_lanes).map(|_| AtomicBool::new(false)).collect();
        let lane_dirty: Vec<Arc<DirtySignal>> = (0..total_lanes)
            .map(|_| Arc::new(DirtySignal::new_initial()))
            .collect();

        // Build workers. Seed each worker's `incoming` with its
        // initially-owned lanes so the first turn drains them into the
        // heap.
        let mut workers: Vec<WorkerRuntime> = (0..worker_count)
            .map(|i| WorkerRuntime::new(WorkerId(i)))
            .collect();
        for lane_addr in 0..total_lanes {
            let owner = lane_owner[lane_addr].load(Ordering::Relaxed);
            workers[owner].shared.lock().incoming.push_back(lane_addr);
        }

        let resources = resource_specs
            .into_iter()
            .map(|spec| (spec.id.clone(), Mutex::new(Resource::new(spec))))
            .collect();

        // T1 — every operator runs propagate once at startup.
        //
        // A "sinks only" init was tried and rejected: many
        // demand-originating operators (Filter, Aggregate, Union with
        // Exact-cardinality inputs) are not sinks but still
        // unconditionally publish a static seed on their inputs.
        // With sinks-only init those seeds never fire and the cascade
        // dies at the first non-sink boundary that doesn't translate
        // its output forward (e.g. Union over `Cardinality::Unknown`
        // inputs writes nothing through). Per-batch re-translation
        // is still avoided because the post-`run` re-arm is gated on
        // `propagation_depends_on_state` — see `turn.rs`.
        let propagation_pending: Vec<AtomicBool> =
            (0..runtime_nodes.len()).map(|_| AtomicBool::new(true)).collect();
        Ok(Self {
            nodes: runtime_nodes,
            channels,
            resources,
            brokers: Vec::new(),
            broker_proposals: Mutex::new(Vec::new()),
            async_work: Mutex::new(AsyncWorkSet::default()),
            trace: Mutex::new(ScheduleTrace::default()),
            metrics,
            options,
            last_run_node: OperatorId::from_index(usize::MAX),
            lanes: flat_lanes,
            op_lane_offset,
            op_lane_count: op_lane_count_v,
            lane_owner,
            lane_finished,
            lane_dirty,
            propagation_pending,
            propagation_order,
            workers,
        })
    }

    /// Number of worker threads the scheduler is configured for.
    /// Drivers that want to run workers on their own runtime use
    /// this with `worker_handles` to get per-worker turn handles.
    pub fn worker_count(&self) -> usize {
        self.workers.len()
    }

    /// Back-compat alias — old name for `worker_count`. Some callers
    /// outside the engine still use it.
    pub fn shard_count(&self) -> usize {
        self.worker_count()
    }

    /// Build a `WorkerCtx` snapshot. Cheap (just reference packing).
    /// Used internally and by the per-worker turn helpers in
    /// `turn.rs`.
    pub(crate) fn worker_ctx(&self) -> WorkerCtx<'_> {
        WorkerCtx {
            nodes: &self.nodes,
            channels: &self.channels,
            resources: &self.resources,
            brokers: &self.brokers,
            broker_proposals: &self.broker_proposals,
            async_work: &self.async_work,
            trace: &self.trace,
            metrics: &self.metrics,
            lanes: &self.lanes,
            op_lane_offset: &self.op_lane_offset,
            op_lane_count: &self.op_lane_count,
            lane_owner: &self.lane_owner,
            lane_finished: &self.lane_finished,
            lane_dirty: &self.lane_dirty,
            propagation_pending: &self.propagation_pending,
            propagation_order: &self.propagation_order,
            workers: &self.workers,
            memory_limit_bytes: self.options.memory_limit_bytes,
        }
    }

    /// Phase-1 maintenance: drain async completions, broker
    /// upkeep, drive the driver-supplied I/O substrate, propagate
    /// requirements, rebalance memory. Single-thread, mutates
    /// `self`. Drivers run this before letting shards turn.
    ///
    /// `io` is the driver-owned I/O capability — passed in (not
    /// stored) so engine state stays `Send` even when the
    /// substrate is `!Send`.
    ///
    /// Returns `(activity, any_propagated_or_grants_changed)` so
    /// callers know whether to dirty-mark.
    pub fn turn_phase1(
        &mut self,
        io: &mut dyn super::DriverIo,
    ) -> EngineResult<TurnPhase1> {
        let async_wake = self.advance_async_work();
        self.maintain_brokers();
        let substrate_activity = self.drive_substrate(io)?;
        let propagated = self.propagate_requirements()?;
        let grants_changed = self.rebalance_memory();
        if async_wake || propagated || grants_changed || substrate_activity {
            // Conservative: any of these signals may invalidate proposals.
            // The precise per-cause routing is plumbed elsewhere
            // (e.g. propagation marks `OutputRequirementChanged` per-op);
            // this fallback covers the still-coarse paths.
            self.mark_all_dirty(DirtyCause::ExternalWake);
        }
        if !self.brokers.is_empty() {
            self.mark_all_dirty(DirtyCause::ExternalWake);
        }
        Ok(TurnPhase1 {
            async_wake,
            substrate_activity,
            propagated,
            grants_changed,
        })
    }

    /// Phase-2 broker admission: pick admissible broker proposals
    /// in EV order and enqueue them onto broker submittable heaps.
    /// Sequential, no concurrency: brokers are global state.
    /// `io` is currently unused at this phase (the actual `submit`
    /// happens inside `drive_substrate` during phase 1) but kept on
    /// the signature so drivers always see the I/O capability flow
    /// through every entry point uniformly.
    /// Returns `true` if any admission happened.
    pub fn turn_phase2_admit_brokers(
        &mut self,
        _io: &mut dyn super::DriverIo,
    ) -> EngineResult<bool> {
        self.collect_broker_proposals();
        let mut any = false;
        let mut skipped: BTreeSet<(BrokerId, u64)> = BTreeSet::new();
        loop {
            let proposal = {
                let props = self.broker_proposals.lock();
                let mut best: Option<(i64, BrokerProposal)> = None;
                for proposal in props.iter() {
                    if skipped.contains(&(proposal.broker, proposal.key)) {
                        continue;
                    }
                    let composite = score_broker_proposal(proposal);
                    match &best {
                        None => best = Some((composite, proposal.clone())),
                        Some((s, _)) if composite > *s => {
                            best = Some((composite, proposal.clone()));
                        }
                        _ => {}
                    }
                }
                best.map(|(_, p)| p)
            };
            let Some(proposal) = proposal else { break };
            let label: Arc<str> = format!("broker:{}", proposal.broker.index()).into();
            let score = score_broker_proposal(&proposal);
            self.trace.lock().push(TraceEvent::BrokerSubmit {
                broker: proposal.broker,
                label,
                latency: proposal.latency_class,
                required_rows: proposal.value.required_rows,
                score,
            });
            self.commit_broker_action(proposal)?;
            any = true;
            self.collect_broker_proposals();
        }
        drop(skipped);
        Ok(any)
    }

    /// Build per-worker turn handles. The returned `WorkerHandle`s
    /// share an immutable `WorkerCtx` view of the task; each runs its
    /// own work-stealing loop. Drivers can hand them to a thread
    /// pool / `thread::scope` / Tokio executor.
    pub fn worker_handles(&self) -> Vec<WorkerHandle<'_>> {
        let ctx = self.worker_ctx();
        (0..self.workers.len())
            .map(|i| WorkerHandle {
                ctx,
                worker_id: WorkerId(i),
            })
            .collect()
    }

    /// Back-compat alias for `worker_handles`. Returned values still
    /// run via `WorkerHandle::turn()`.
    pub fn shard_workers(&self) -> Vec<WorkerHandle<'_>> {
        self.worker_handles()
    }

    /// Internal entry point used by `WorkerHandle::turn()`. Forwards
    /// to `turn::worker_turn`.
    pub(crate) fn worker_turn(
        ctx: WorkerCtx<'_>,
        worker_id: WorkerId,
    ) -> EngineResult<bool> {
        turn::worker_turn(ctx, worker_id)
    }

    /// Phase-3 classification: decide whether the task is Made,
    /// Idle, or Done given any-progress this turn.
    pub fn turn_phase3_classify(
        &self,
        any_progress: bool,
        phase1: TurnPhase1,
        io: &dyn super::DriverIo,
    ) -> EngineResult<TurnOutcome> {
        if any_progress
            || phase1.propagated
            || phase1.grants_changed
            || phase1.async_wake
            || phase1.substrate_activity
        {
            return Ok(TurnOutcome::Made);
        }
        if self.async_work.lock().has_pending() {
            return Ok(TurnOutcome::Idle);
        }
        if self.brokers.iter().any(|b| b.lock().has_pending()) {
            return Ok(TurnOutcome::Idle);
        }
        if io.in_flight() > 0 {
            return Ok(TurnOutcome::Idle);
        }
        if self.is_complete() {
            return Ok(TurnOutcome::Done);
        }
        Err(EngineError::message(format!(
            "operator graph quiesced before completion; unfinished={:?}",
            self.unfinished_labels()
        )))
    }

    /// Register a broker before execution. Returns the assigned
    /// `BrokerId` which operators reference from `update`/`run`.
    pub fn register_broker(&mut self, broker: Box<dyn Broker>) -> BrokerId {
        let id = broker.id();
        while self.brokers.len() <= id.index() {
            self.brokers.push(Mutex::new(Box::new(NoopBroker {
                id: BrokerId::from_index(self.brokers.len()),
            })));
        }
        self.brokers[id.index()] = Mutex::new(broker);
        id
    }

    /// Run one scheduler turn. Drivers call this in their own loop
    /// (yielding, waiting, or stealing between calls); `run()` is
    /// the convenience that loops on `turn()` synchronously up to
    /// `options.max_turns`.
    ///
    /// Phases each turn:
    ///   1. drain async completions and broker maintenance;
    ///   2. propagate requirements + rebalance memory + collect
    ///      proposals;
    ///   3. inner admission loop: pick best action, run it, refresh
    ///      proposals, repeat until no admissible action remains;
    ///   4. classify outcome.
    pub fn turn(&mut self, io: &mut dyn super::DriverIo) -> EngineResult<TurnOutcome> {
        let async_wake = self.advance_async_work();
        self.maintain_brokers();
        // Driver step: pull from broker heaps into the substrate,
        // and drain substrate completions back to brokers. With this
        // split, broker.commit no longer submits — admission and
        // submission are separated, the substrate is the only thing
        // that touches I/O.
        let substrate_activity = self.drive_substrate(io)?;
        let propagated = self.propagate_requirements()?;
        let grants_changed = self.rebalance_memory();

        // Turn-level fallbacks for state changes whose precise lane
        // targets aren't tracked yet: any of these mark every
        // non-finished node dirty so its proposals get refreshed
        // this turn. Per-channel/per-resource targeting is wired
        // through the run-time closures (push/pop/seal/publish).
        if async_wake || propagated || grants_changed || substrate_activity {
            self.mark_all_dirty(DirtyCause::ExternalWake);
        }
        if !self.brokers.is_empty() {
            // Brokers don't yet expose "what completed this turn";
            // until they do, any maintenance turn re-marks every
            // node that subscribed to broker work. The cost is
            // amortised across the turn.
            self.mark_all_dirty(DirtyCause::ExternalWake);
        }

        // Phase 2a: broker admissions on the main thread. Broker
        // proposals are global (not per-shard), so we admit them
        // sequentially before launching shard workers.
        self.collect_broker_proposals();
        let mut any_progress = false;
        let mut skipped_brokers: BTreeSet<(BrokerId, u64)> = BTreeSet::new();
        loop {
            let proposal = {
                let props = self.broker_proposals.lock();
                let mut best: Option<(i64, BrokerProposal)> = None;
                for proposal in props.iter() {
                    if skipped_brokers.contains(&(proposal.broker, proposal.key)) {
                        continue;
                    }
                    let composite = score_broker_proposal(proposal);
                    match &best {
                        None => best = Some((composite, proposal.clone())),
                        Some((s, _)) if composite > *s => {
                            best = Some((composite, proposal.clone()));
                        }
                        _ => {}
                    }
                }
                best.map(|(_, p)| p)
            };
            let Some(proposal) = proposal else { break };
            let label: Arc<str> = format!("broker:{}", proposal.broker.index()).into();
            let score = score_broker_proposal(&proposal);
            self.trace.lock().push(TraceEvent::BrokerSubmit {
                broker: proposal.broker,
                label,
                latency: proposal.latency_class,
                required_rows: proposal.value.required_rows,
                score,
            });
            self.commit_broker_action(proposal)?;
            any_progress = true;
            self.collect_broker_proposals();
        }
        drop(skipped_brokers);

        // Phase 2b: per-worker turns, looped with main-thread
        // maintenance (propagate, rebalance) between iterations. The
        // scheduler stays runtime-neutral here: workers run sequentially
        // on the calling thread. Drivers that want parallelism use
        // `worker_handles()` and a thread pool / Tokio executor.
        loop {
            let ctx = self.worker_ctx();
            let mut any_made = false;
            for w in 0..self.workers.len() {
                if turn::worker_turn(ctx, WorkerId(w))? {
                    any_made = true;
                }
            }
            if !any_made {
                break;
            }
            any_progress = true;
            self.propagate_requirements()?;
            if self.rebalance_memory() {
                self.mark_all_dirty(DirtyCause::ExternalWake);
            }
        }

        if any_progress || propagated || grants_changed || async_wake {
            return Ok(TurnOutcome::Made);
        }
        // Pending async work will produce a wake later; don't
        // quiesce while it's still in flight.
        if self.async_work.lock().has_pending() {
            return Ok(TurnOutcome::Idle);
        }
        if self.brokers.iter().any(|b| b.lock().has_pending()) {
            return Ok(TurnOutcome::Idle);
        }
        if io.in_flight() > 0 {
            return Ok(TurnOutcome::Idle);
        }
        if self.is_complete() {
            return Ok(TurnOutcome::Done);
        }
        Err(EngineError::message(format!(
            "operator graph quiesced before completion; unfinished={:?}",
            self.unfinished_labels()
        )))
    }

    /// True once every operator has reached terminal. Drivers call
    /// this after `turn()` returns `Done` to decide whether to
    /// consume the task.
    pub fn is_done(&self) -> bool {
        self.is_complete()
    }

    /// Consume the task and produce its final report. Valid only
    /// after `turn()` returns `Done` (or equivalently `is_done()`
    /// returns true).
    pub fn into_report(self) -> TaskReport {
        self.finish_report()
    }

    /// Synchronous loop over `turn()` using a default
    /// `FakeDriverIo` for the I/O capability — convenience for
    /// tests and callers that don't actually exercise broker I/O.
    /// Cooperative drivers should loop over `turn(io)` directly with
    /// their own substrate.
    pub fn run(mut self) -> EngineResult<TaskReport> {
        let mut io = super::FakeDriverIo::new();
        self.run_with_io(&mut io)
    }

    /// Synchronous loop over `turn(io)` with a caller-provided I/O
    /// capability. Used by drivers that don't need cooperative
    /// yielding (e.g. tests, callers that just want the answer);
    /// cooperative drivers loop over `turn(io)` directly and yield
    /// to peers between calls.
    pub fn run_with_io(
        mut self,
        io: &mut dyn super::DriverIo,
    ) -> EngineResult<TaskReport> {
        let max_turns = self.options.max_turns;
        for _ in 0..max_turns {
            match self.turn(io)? {
                TurnOutcome::Made | TurnOutcome::Idle => continue,
                TurnOutcome::Done => return Ok(self.finish_report()),
            }
        }
        Err(EngineError::message(format!(
            "operator graph exceeded scheduler turn limit; unfinished={:?}",
            self.unfinished_labels()
        )))
    }

    pub fn metrics_handle(&self) -> Arc<Mutex<ExecutionMetrics>> {
        Arc::clone(&self.metrics)
    }

    fn finish_report(self) -> TaskReport {
        TaskReport {
            trace: self.trace.into_inner(),
            metrics: self.metrics.lock().clone(),
        }
    }

    fn is_complete(&self) -> bool {
        // Lock-free scan via the per-lane atomic finished flag.
        self.lane_finished
            .iter()
            .all(|f| f.load(Ordering::Acquire))
    }

    fn op_finished(&self, op: OperatorId) -> bool {
        let op_idx = op.index();
        let start = self.op_lane_offset[op_idx];
        let count = self.op_lane_count[op_idx];
        (0..count).all(|i| self.lane_finished[start + i].load(Ordering::Acquire))
    }

    fn unfinished_labels(&self) -> Vec<String> {
        self.nodes
            .iter()
            .filter(|node| !self.op_finished(node.id))
            .map(|node| node.node.spec().label.clone())
            .collect()
    }

    pub(crate) fn propagate_requirements(&mut self) -> EngineResult<bool> {
        turn::propagate_requirements(self.worker_ctx())
    }

    /// Wake every lane (across all ops) by routing the dirty signal
    /// to the current owner of each lane. Used when a state change
    /// (memory grant, async wake, broker maint) doesn't have a
    /// precise lane target.
    pub(crate) fn mark_all_dirty(&mut self, cause: DirtyCause) {
        for lane_addr in 0..self.lanes.len() {
            self.lane_dirty[lane_addr].push(cause.clone());
            let owner = self.lane_owner[lane_addr].load(Ordering::Acquire);
            self.workers[owner]
                .shared
                .lock()
                .incoming
                .push_back(lane_addr);
        }
    }

    /// Wake every lane of `op` by routing each lane's signal to its
    /// current owner.
    fn mark_op_dirty(&self, op: OperatorId, cause: DirtyCause) {
        let op_idx = op.index();
        let start = self.op_lane_offset[op_idx];
        let count = self.op_lane_count[op_idx];
        for i in 0..count {
            let lane_addr = start + i;
            self.lane_dirty[lane_addr].push(cause.clone());
            let owner = self.lane_owner[lane_addr].load(Ordering::Acquire);
            self.workers[owner]
                .shared
                .lock()
                .incoming
                .push_back(lane_addr);
        }
    }

    fn collect_broker_proposals(&mut self) {
        let mut props = self.broker_proposals.lock();
        props.clear();
        for broker in &self.brokers {
            broker.lock().proposals(&mut props);
        }
    }

    fn maintain_brokers(&mut self) {
        for broker in &self.brokers {
            broker.lock().maintain();
        }
    }

    fn commit_broker_action(&mut self, proposal: BrokerProposal) -> EngineResult<()> {
        let broker_index = proposal.broker.index();
        if let Some(broker) = self.brokers.get(broker_index) {
            broker.lock().enqueue(proposal, BrokerGrant)?;
        }
        Ok(())
    }

    fn drive_substrate(&mut self, io: &mut dyn super::DriverIo) -> EngineResult<bool> {
        let mut activity = false;
        for broker_idx in 0..self.brokers.len() {
            let free = io.capacity_hint().free_slots;
            if free == 0 {
                break;
            }
            let max_n = free.min(64);
            let submitted = self.brokers[broker_idx].lock().pull(max_n, io)?;
            if submitted > 0 {
                activity = true;
            }
        }
        let completions = io.drain_completions();
        if !completions.is_empty() {
            activity = true;
        }
        for completion in completions {
            let idx = completion.broker.index();
            if let Some(broker) = self.brokers.get(idx) {
                broker
                    .lock()
                    .complete(completion.request, completion.result)?;
            }
        }
        Ok(activity)
    }

    pub(crate) fn rebalance_memory(&mut self) -> bool {
        let used = self
            .channels
            .iter()
            .map(|c| c.lock().retained_bytes())
            .sum::<usize>()
            .saturating_add(self.async_work.lock().retained_bytes());
        self.metrics.lock().observe_memory_bytes(used);
        let mut changed = false;
        if used >= self.options.memory_limit_bytes {
            for channel_mutex in &self.channels {
                let mut channel = channel_mutex.lock();
                let min_bytes = channel.spec().buffer.min_bytes();
                if channel.set_current_capacity(min_bytes) {
                    changed = true;
                }
            }
            if changed {
                self.trace.lock().push(TraceEvent::MemoryGrantShrink);
            }
            return changed;
        }
        if used <= self.options.memory_limit_bytes / 2 {
            for channel_mutex in &self.channels {
                let mut channel = channel_mutex.lock();
                let target_bytes = channel.spec().buffer.target_bytes();
                if channel.set_current_capacity(target_bytes) {
                    changed = true;
                }
            }
            if changed {
                self.trace.lock().push(TraceEvent::MemoryGrantGrow);
            }
        }
        changed
    }

    fn advance_async_work(&mut self) -> bool {
        let events = self.async_work.lock().advance();
        for event in &events {
            self.metrics.lock().add_async_completed(&event.label);
            self.metrics.lock().add_async_wakeup(&event.label);
            self.trace.lock().push(TraceEvent::AsyncWake {
                label: event.label.clone().into(),
                span: event.span,
            });
        }
        !events.is_empty()
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TaskReport {
    pub trace: ScheduleTrace,
    pub metrics: ExecutionMetrics,
}

/// EV ranking for an operator forward-work proposal.
fn score_operator_proposal(p: &WorkProposal) -> i64 {
    let row_band = if p.value.required_rows > 0 {
        1_000_000_000
    } else if p.value.candidate_rows > 0 {
        0
    } else {
        0
    };
    let value = (p.value.required_rows as i64) * 256
        + (p.value.candidate_rows as i64) * (p.value.p_needed_x256 as i64)
        + (p.value.memory_release_bytes as i64) / 64;
    let cost = p.cost.cpu_micros as i64 + p.cost.memory_delta_bytes.max(0) / 64;
    let class_tiebreak = p.class.priority();
    row_band + value - cost + class_tiebreak
}

/// EV ranking for a broker proposal. Includes the latency overlap
/// bonus so launching async work pre-emptively can rank above
/// already-loaded CPU work.
fn score_broker_proposal(p: &BrokerProposal) -> i64 {
    let row_band = if p.value.required_rows > 0 {
        1_000_000_000
    } else if p.value.candidate_rows > 0 {
        0
    } else {
        0
    };
    let row_value = (p.value.required_rows as i64) * 256
        + (p.value.candidate_rows as i64) * (p.value.p_needed_x256 as i64);
    let memory_release = (p.value.memory_release_bytes as i64) / 64;
    // Overlap bonus: launching async work now lets the host runtime
    // run it concurrently with later admissions in this turn.
    let unblock = p.value.required_rows.max(1) as i64;
    let latency_overlap = (p.latency_class.expected_turns() as i64) * unblock / 8;
    let value = row_value + memory_release + latency_overlap;
    let cost = p.cost.cpu_micros as i64 + p.cost.memory_delta_bytes.max(0) / 64;
    // SubmitBroker class is the lowest tiebreaker but row band + EV
    // dominate.
    row_band + value - cost
}

/// Placeholder broker used to fill gaps in `PreparedTask::brokers`
/// before user-supplied brokers are registered.
struct NoopBroker {
    id: BrokerId,
}

impl Broker for NoopBroker {
    fn id(&self) -> BrokerId {
        self.id
    }
    fn maintain(&mut self) {}
    fn proposals(&self, _out: &mut Vec<BrokerProposal>) {}
    fn enqueue(&mut self, _proposal: BrokerProposal, _grant: BrokerGrant) -> EngineResult<()> {
        Ok(())
    }
    fn pull(
        &mut self,
        _max_n: usize,
        _substrate: &mut dyn super::DriverIo,
    ) -> EngineResult<usize> {
        Ok(0)
    }
    fn complete(
        &mut self,
        _request: super::SubmittedRequestId,
        _result: super::IoResult,
    ) -> EngineResult<()> {
        Ok(())
    }
    fn register(&mut self, _owner: OperatorId, _spec: InterestSpec) -> InterestId {
        InterestId::from_index(0)
    }
    fn cancel(&mut self, _interest: InterestId) {}
    fn take_completed(&mut self, _owner: OperatorId) -> Option<CompletedInterest> {
        None
    }
    fn has_pending(&self) -> bool {
        false
    }
}

pub fn collect_first_column(rows: &[Batch]) -> Vec<i64> {
    rows.iter()
        .flat_map(Batch::first_column_values)
        .collect::<Vec<_>>()
}

fn contiguous_presence(
    requirement: &RequirementSet,
    presence: super::RowDemand,
) -> Option<(u64, u64)> {
    // Find the first interval matching `presence`, then extend across
    // any directly-adjacent intervals that also match (the interval
    // representation already coalesces equal-requirement neighbours,
    // so this loop runs at most twice in practice).
    let intervals = requirement.intervals();
    let first = intervals
        .iter()
        .position(|iv| iv.demand == presence)?;
    let start = intervals[first].start;
    let mut end = intervals[first].end;
    for iv in &intervals[first + 1..] {
        if iv.start == end && iv.demand == presence {
            end = iv.end;
        } else {
            break;
        }
    }
    Some((start, end))
}

#[cfg(test)]
mod prepared_task_send_check {
    /// Assert at compile time that `PreparedTask` is `Send`. This is
    /// the load-bearing guarantee from the driver-owned-I/O design:
    /// the engine's mutable state must be movable across threads,
    /// which means it cannot own a `!Send` substrate. The driver
    /// passes `&mut dyn DriverIo` in at drive time instead.
    fn _prepared_task_is_send() {
        fn assert_send<T: Send>() {}
        assert_send::<super::PreparedTask>();
    }
}
