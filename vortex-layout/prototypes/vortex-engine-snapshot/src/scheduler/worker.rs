//! Per-worker runtime state for the work-stealing scheduler.
//!
//! Each worker thread owns a slice of `(operator, lane)` pairs and
//! drives them through their turn loop. The unit of ownership is the
//! lane (one `LaneRuntime`); ownership is recorded in
//! `PreparedTask::lane_owner` (a per-lane `AtomicUsize` indexing the
//! workers Vec) and migrates between workers via the work-stealing
//! protocol.
//!
//! Why this shape:
//!
//! - Each `LaneRuntime` is wrapped in a `Mutex` so the owning worker
//!   can run the operator's `update`/`run` for milliseconds without
//!   blocking peers' wake-routing or admission decisions. Peers
//!   touch only the worker's `shared` Mutex (a thin metadata cell),
//!   never the `LaneRuntime` itself.
//!
//! - The owner's `work_heap` is local to the worker — no global
//!   contention on the EV heap. Peers steal whole lanes, not heap
//!   entries: stealing changes `lane_owner` and pushes the stolen
//!   lane onto the new owner's `incoming` queue. Stale heap entries
//!   on the old owner are filtered at pop time via an owner check.
//!
//! - Cross-worker wakes (e.g. a channel push waking a consumer lane
//!   on another worker) are O(1): look up the consumer's current
//!   owner, lock that worker's `shared`, push lane addr to its
//!   `incoming`. The actual work happens on the consumer's thread on
//!   its next iteration.

use std::collections::BinaryHeap;
use std::collections::VecDeque;
use std::sync::Arc;
use std::sync::atomic::AtomicBool;
use std::sync::atomic::AtomicUsize;
use std::sync::atomic::Ordering;

use parking_lot::Mutex;

use super::DirtySignal;
use crate::OperatorId;
use crate::RequirementSet;
use crate::WorkProposal;

/// Identifier for a worker thread in the scheduler's pool. Index into
/// `PreparedTask::workers`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct WorkerId(pub usize);

impl WorkerId {
    pub fn index(self) -> usize {
        self.0
    }
}

/// Flat lane address: `op_lane_offset[op_idx] + lane_idx`.
pub type LaneAddr = usize;

/// One pending work item in a worker's `work_heap`. `Ord` is by `score`
/// so the default `BinaryHeap` (max-heap) yields the highest-priority
/// item on each pop.
#[derive(Clone, Debug, Eq)]
pub(crate) struct HeapEntry {
    pub score: i64,
    pub lane_addr: LaneAddr,
    pub proposal_idx: usize,
    /// Bumped on every `update_lane` call; entries with a stale epoch
    /// are skipped on pop because the proposal vector has been
    /// replaced.
    pub epoch: u32,
}

impl PartialEq for HeapEntry {
    fn eq(&self, other: &Self) -> bool {
        self.score == other.score
    }
}
impl PartialOrd for HeapEntry {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}
impl Ord for HeapEntry {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        // Primary: score (max-heap).
        // Tiebreaker: lower lane_addr pops first (mirrors the
        // pre-WorkerRuntime behaviour of iterating in op-id /
        // lane-index order).
        self.score
            .cmp(&other.score)
            .then_with(|| other.lane_addr.cmp(&self.lane_addr))
            .then_with(|| other.proposal_idx.cmp(&self.proposal_idx))
    }
}

/// Per-`(operator, lane)` runtime state. Lives in `PreparedTask::lanes`
/// flat (one entry per `(op, lane)` pair). Wrapped in `Mutex` so the
/// current owning worker holds it through `update` and `run`, while
/// peers touch only the worker's metadata.
pub(crate) struct LaneRuntime {
    pub op: OperatorId,
    pub lane: crate::LaneId,
    pub local: crate::ErasedLocalState,
    pub finished: bool,
    pub proposals: Vec<WorkProposal>,
    /// Reusable buffer for `propagate_requirements`'s input slots.
    pub propagate_inputs_buffer: Vec<RequirementSet>,
    /// Bumped each time `update` re-fills `proposals`. Heap entries
    /// from earlier updates carry the old epoch; the heap pop loop
    /// discards them.
    pub epoch: u32,
}

/// One worker's per-thread state. Mutex-protected metadata so peers
/// can wake / steal-request without contending the lane data.
pub(crate) struct WorkerRuntime {
    pub id: WorkerId,
    pub shared: Mutex<WorkerShared>,
}

/// Worker metadata, locked briefly by both the owning worker (per
/// pop / push) and peers (per wake / steal).
pub(crate) struct WorkerShared {
    /// EV-sorted heap of admitted proposals for lanes I currently own.
    /// Entries are stale if `lane_owner[entry.lane_addr] != self.id`
    /// or `entry.epoch != lane.epoch` — filtered at pop time.
    pub work_heap: BinaryHeap<HeapEntry>,
    /// Lanes that need to be re-`update`d on this worker's next
    /// iteration. Pushed by:
    /// - peer workers when a channel push wakes a lane I own;
    /// - peer workers when they grant me a lane via stealing;
    /// - my own action loop after a `run` woke another of my own
    ///   lanes;
    /// - `prepare()` to seed every lane initially.
    pub incoming: VecDeque<LaneAddr>,
    /// Heap entries deferred this turn (constraint not yet satisfied,
    /// e.g. waiting on output capacity). Re-armed at the start of the
    /// next iteration once channel/resource state has changed.
    pub deferred: Vec<HeapEntry>,
}

impl WorkerRuntime {
    pub fn new(id: WorkerId) -> Self {
        Self {
            id,
            shared: Mutex::new(WorkerShared {
                work_heap: BinaryHeap::new(),
                incoming: VecDeque::new(),
                deferred: Vec::new(),
            }),
        }
    }
}

/// Routing helper: enqueue `lane_addr` onto whichever worker currently
/// owns it. Called by `mark_op_dirty` and by `propagate_requirements`
/// when an upstream's requirement publish wakes a consumer.
pub(crate) fn wake_lane(
    workers: &[WorkerRuntime],
    lane_owner: &[AtomicUsize],
    lane_dirty: &[Arc<DirtySignal>],
    lane_addr: LaneAddr,
    cause: super::DirtyCause,
) {
    lane_dirty[lane_addr].push(cause);
    let owner = lane_owner[lane_addr].load(Ordering::Acquire);
    let mut shared = workers[owner].shared.lock();
    shared.incoming.push_back(lane_addr);
}

/// `true` iff every entry in `lane_finished` is set. Used to detect
/// graph completion.
pub(crate) fn all_lanes_finished(lane_finished: &[AtomicBool]) -> bool {
    lane_finished.iter().all(|f| f.load(Ordering::Acquire))
}
