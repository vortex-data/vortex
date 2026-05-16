//! External work brokers.
//!
//! A broker is the queue manager for one constrained external resource
//! subsystem. Operators register interests; the broker coalesces and
//! emits admissible `BrokerProposal`s to the scheduler. The scheduler
//! ranks broker proposals against operator `WorkProposal`s using one
//! EV ranking. When the scheduler commits a broker proposal, the
//! broker submits the concrete async work to the host runtime and
//! returns immediately — the work runs concurrently with later
//! admissions in the same scheduler turn. That overlap is what
//! enables pipelining.

use crate::Batch;
use crate::DomainSpan;
use crate::EngineResult;
use crate::OperatorId;
use crate::WorkCost;
use crate::WorkValue;

/// Identifier for one broker instance in the task. Stable for the
/// lifetime of the task.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct BrokerId(usize);

impl BrokerId {
    pub const fn from_index(index: usize) -> Self {
        Self(index)
    }

    pub const fn index(self) -> usize {
        self.0
    }
}

/// Identifier for one registered interest within a broker.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct InterestId(usize);

impl InterestId {
    pub const fn from_index(index: usize) -> Self {
        Self(index)
    }

    pub const fn index(self) -> usize {
        self.0
    }
}

/// Row-presence class for an interest. Drives the EV row band.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum RowClass {
    Required,
    Candidate,
}

/// Latency expectation for the request the broker would submit.
/// Drives the `latency_overlap_bonus` EV term.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum LatencyClass {
    FastLocal,
    LocalSSD,
    NetworkRtt,
    LongTail,
}

impl LatencyClass {
    /// Approximate scheduler turns the request takes to complete.
    pub const fn expected_turns(self) -> u32 {
        match self {
            Self::FastLocal => 1,
            Self::LocalSSD => 4,
            Self::NetworkRtt => 16,
            Self::LongTail => 64,
        }
    }
}

/// Operator-supplied description of one external-work interest. The
/// broker uses the metadata to coalesce, rank, and route completions.
#[derive(Clone, Debug)]
pub struct InterestSpec {
    pub label: String,
    pub span: DomainSpan,
    /// The batch the broker should deliver on completion. In a real
    /// system the broker would fetch bytes and decode; the prototype
    /// carries the pre-built batch through the broker queue.
    pub batch: Batch,
    pub bytes: usize,
    /// Number of scheduler turns until the host runtime delivers the
    /// completion. Stand-in for real wall-clock latency.
    pub delay_turns: usize,
    pub row_class: RowClass,
    /// Selectivity for `Candidate` interests, fixed-point in [0, 256].
    pub p_needed_x256: u32,
    pub latency_class: LatencyClass,
    /// Number of rows this interest unblocks downstream.
    pub unblock_rows: u64,
}

/// Result delivered to the registering operator after the broker
/// commits a proposal that satisfied this interest.
#[derive(Clone, Debug)]
pub struct CompletedInterest {
    pub interest: InterestId,
    pub label: String,
    pub batch: Batch,
}

/// One coalesced admissible request the broker has computed. The
/// scheduler ranks this against operator proposals using EV.
#[derive(Clone, Debug)]
pub struct BrokerProposal {
    pub broker: BrokerId,
    /// Broker-internal key identifying this proposal. The broker
    /// receives it back in `commit()`.
    pub key: u64,
    /// Interests this commit will fulfill in one shot.
    pub satisfies: Vec<InterestId>,
    pub cost: WorkCost,
    pub value: WorkValue,
    pub latency_class: LatencyClass,
}

/// Bounded-budget grant the scheduler gives a broker on admission.
/// For V1 prototypes this is just an opaque marker; production would
/// carry concurrency tokens here.
#[derive(Clone, Copy, Debug, Default)]
pub struct BrokerGrant;

/// One coalesced physical request the broker hands a substrate at
/// pull time. Carries enough metadata for the substrate to schedule
/// it (bytes, latency class) and for `complete()` to route the
/// result back to the registered interests.
#[derive(Clone, Debug)]
pub struct PhysicalRequest {
    pub broker: BrokerId,
    pub label: String,
    pub bytes: usize,
    pub delay_turns: usize,
    pub latency_class: LatencyClass,
    /// Pre-built batch the prototype carries through; in production
    /// the substrate would deliver bytes and the broker would decode.
    pub batch: Batch,
    /// Interests this request fulfills when the substrate completes.
    pub satisfies: Vec<InterestId>,
}

/// Substrate-issued id for one submitted request. The broker stores
/// this to route the eventual completion back to the right interest
/// slots.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct SubmittedRequestId(pub u64);

/// Substrate's response to `submit`. Either the request was accepted
/// and bound to an id, or the substrate refused (full, throttled);
/// the broker re-queues refused requests on its heap.
pub enum IoAdmission {
    Submitted(SubmittedRequestId),
    Refused(PhysicalRequest),
}

/// Hint the broker may consult before pulling. `free_slots` is the
/// substrate's best estimate of how many requests it can absorb
/// right now without blocking. `total_slots` is the substrate's
/// nominal queue depth.
#[derive(Clone, Copy, Debug, Default)]
pub struct IoCapacity {
    pub free_slots: usize,
    pub total_slots: usize,
}

/// Result the substrate hands the broker on completion.
#[derive(Clone, Debug)]
pub struct IoResult {
    pub batch: Batch,
}

/// Driver-supplied handle to the !Send substrate the broker submits
/// to. The driver owns the substrate (e.g. io_uring SQ); brokers own
/// the queue of work to submit. This split is what makes I/O
/// pluggable: a TPC driver hands brokers a per-core io_uring; a
/// Tokio driver hands brokers a tokio I/O substrate; a test driver
/// hands brokers a fake.
pub trait DriverIo {
    /// Submit one physical request. Returns either an id (substrate
    /// accepted) or the original request (substrate refused).
    fn submit(&mut self, request: PhysicalRequest) -> IoAdmission;

    /// Inspect substrate capacity. Brokers may consult this before
    /// coalescing decisions — submitting one large coalesced request
    /// is preferable when substrate slots are scarce.
    fn capacity_hint(&self) -> IoCapacity {
        IoCapacity {
            free_slots: usize::MAX,
            total_slots: usize::MAX,
        }
    }

    /// Advance the substrate one driver tick: drain any newly
    /// completed requests so the driver can route them back to the
    /// originating broker via `Broker::complete`. Real substrates
    /// (io_uring) use their event source (cqe ring) here; the
    /// prototype's `FakeDriverIo` decrements per-turn counters.
    /// Returns the broker id and request id pairs the driver should
    /// then forward — keyed so that one substrate can serve many
    /// brokers if the driver chooses.
    fn drain_completions(&mut self) -> Vec<IoCompletion>;

    /// True if any submission is in flight inside the substrate. The
    /// scheduler uses this with `Broker::has_pending` to decide
    /// whether to declare the task quiesced.
    fn in_flight(&self) -> usize;

    /// Approximate bytes retained inside the substrate (queued
    /// buffers, in-flight reads). Drivers report this so the
    /// scheduler's memory arbiter sees substrate-side memory.
    fn retained_bytes(&self) -> usize {
        0
    }
}

/// One substrate completion the driver routes to a broker.
#[derive(Clone, Debug)]
pub struct IoCompletion {
    pub broker: BrokerId,
    pub request: SubmittedRequestId,
    pub result: IoResult,
}

/// Scheduler-facing trait. The runtime stores brokers as
/// `Box<dyn Broker>`; operators see only `BrokerHandle`s through
/// `UpdateCtx::broker(id)`.
pub trait Broker: Send + 'static {
    /// Bounded scheduling work. Drains completions into per-interest
    /// result slots. Refreshes the proposal queue. Releases any
    /// consumed concurrency tokens.
    fn maintain(&mut self);

    /// Read current admissible proposals. Pure read of broker state.
    fn proposals(&self, out: &mut Vec<BrokerProposal>);

    /// Admit a proposal: enqueue the corresponding physical request
    /// onto the broker's submittable heap. *Does not submit* — the
    /// driver pulls from the heap separately.
    fn enqueue(&mut self, proposal: BrokerProposal, grant: BrokerGrant) -> EngineResult<()>;

    /// Pull up to `max_n` highest-EV submittable requests, performing
    /// any final coalescing, and submit each through `substrate`.
    /// Returns the number actually submitted (may be less than
    /// `max_n` if the heap is smaller, the substrate refused, or
    /// coalescing combined entries).
    fn pull(
        &mut self,
        max_n: usize,
        substrate: &mut dyn DriverIo,
    ) -> EngineResult<usize>;

    /// Substrate completed a request previously returned by `pull`.
    /// The broker routes the result to the registered interests
    /// bundled in that request.
    fn complete(
        &mut self,
        request: SubmittedRequestId,
        result: IoResult,
    ) -> EngineResult<()>;

    /// Operator-driven: register a new interest. Returns the id the
    /// operator stores to take completions or cancel.
    fn register(&mut self, owner: OperatorId, spec: InterestSpec) -> InterestId;

    /// Operator-driven: cancel a previously registered interest.
    /// Best effort — may drop the unsubmitted interest, request
    /// backend cancellation for in-flight, or release a completed
    /// result without delivering it.
    fn cancel(&mut self, interest: InterestId);

    /// Operator-driven: take the next completed result destined for
    /// the given operator. Returns None if no completion is ready.
    fn take_completed(&mut self, owner: OperatorId) -> Option<CompletedInterest>;

    /// True if any work is registered, on the submittable heap,
    /// in-flight, or completed but not yet taken. The scheduler
    /// must not declare quiesce while this is true.
    fn has_pending(&self) -> bool;

    /// Identifier for tracing.
    fn id(&self) -> BrokerId;
}

/// A simple delay-based broker for prototype validation.
///
/// One interest = one proposal. No coalescing in this implementation;
/// real brokers would coalesce overlapping byte ranges. Admitted
/// proposals are pushed onto a submittable Vec (heap behaviour for
/// the prototype: pull-time sort by interest order; production would
/// use a real heap keyed on EV). Pull builds `PhysicalRequest`s and
/// hands them to the substrate.
#[derive(Debug)]
pub struct SimpleDelayBroker {
    id: BrokerId,
    label: String,
    interests: Vec<InterestSlot>,
    /// Heap of admitted-but-not-yet-submitted entries. For the
    /// prototype this is just a Vec; pull pops in registration order.
    submittable: Vec<SubmittableEntry>,
    /// Map of substrate-issued ids → interest indices, so
    /// `complete` can route results back to the right slot.
    in_flight: Vec<(SubmittedRequestId, usize)>,
}

#[derive(Debug, Clone)]
struct SubmittableEntry {
    interest_idx: usize,
}

#[derive(Debug, Clone)]
struct InterestSlot {
    spec: InterestSpec,
    owner: OperatorId,
    state: SlotState,
}

#[derive(Debug, Clone)]
enum SlotState {
    Registered,
    /// Proposal admitted; physical request is on the submittable heap.
    Submittable,
    /// Driver pulled and substrate accepted; awaiting completion.
    /// Reverse-lookup of substrate id → interest idx lives in the
    /// parallel `in_flight` Vec on the broker; no need to duplicate
    /// the id here.
    InFlight,
    Completed {
        batch: Batch,
    },
    Cancelled,
    Taken,
}

impl SimpleDelayBroker {
    pub fn new(id: BrokerId, label: impl Into<String>) -> Self {
        Self {
            id,
            label: label.into(),
            interests: Vec::new(),
            submittable: Vec::new(),
            in_flight: Vec::new(),
        }
    }

    pub fn label(&self) -> &str {
        &self.label
    }
}

impl Broker for SimpleDelayBroker {
    fn id(&self) -> BrokerId {
        self.id
    }

    fn maintain(&mut self) {
        // The broker no longer advances delay timers — the substrate
        // does. Maintain stays as a hook for future per-turn cleanup
        // (retry-backoff updates, EV recompute on the heap, etc.).
    }

    fn proposals(&self, out: &mut Vec<BrokerProposal>) {
        for (idx, slot) in self.interests.iter().enumerate() {
            if !matches!(slot.state, SlotState::Registered) {
                continue;
            }
            let interest_id = InterestId::from_index(idx);
            let value = match slot.spec.row_class {
                RowClass::Required => WorkValue {
                    required_rows: slot.spec.unblock_rows,
                    candidate_rows: 0,
                    p_needed_x256: 256,
                    memory_release_bytes: 0,
                },
                RowClass::Candidate => WorkValue {
                    required_rows: 0,
                    candidate_rows: slot.spec.unblock_rows,
                    p_needed_x256: slot.spec.p_needed_x256,
                    memory_release_bytes: 0,
                },
            };
            let cost = WorkCost {
                cpu_micros: 1,
                memory_delta_bytes: i64::try_from(slot.spec.bytes).unwrap_or(i64::MAX),
            };
            out.push(BrokerProposal {
                broker: self.id,
                key: idx as u64,
                satisfies: vec![interest_id],
                cost,
                value,
                latency_class: slot.spec.latency_class,
            });
        }
    }

    fn enqueue(&mut self, proposal: BrokerProposal, _grant: BrokerGrant) -> EngineResult<()> {
        let idx = proposal.key as usize;
        if let Some(slot) = self.interests.get_mut(idx) {
            if matches!(slot.state, SlotState::Registered) {
                slot.state = SlotState::Submittable;
                self.submittable.push(SubmittableEntry { interest_idx: idx });
            }
        }
        Ok(())
    }

    fn pull(
        &mut self,
        max_n: usize,
        substrate: &mut dyn DriverIo,
    ) -> EngineResult<usize> {
        if max_n == 0 || self.submittable.is_empty() {
            return Ok(0);
        }
        let mut submitted = 0;
        let take = max_n.min(self.submittable.len());
        // Drain `take` entries; refused entries get pushed back.
        let mut refused: Vec<SubmittableEntry> = Vec::new();
        for entry in self.submittable.drain(..take) {
            let slot = match self.interests.get(entry.interest_idx) {
                Some(s) => s,
                None => continue,
            };
            // Skip if cancelled between enqueue and pull.
            if !matches!(slot.state, SlotState::Submittable) {
                continue;
            }
            let request = PhysicalRequest {
                broker: self.id,
                label: slot.spec.label.clone(),
                bytes: slot.spec.bytes,
                delay_turns: slot.spec.delay_turns,
                latency_class: slot.spec.latency_class,
                batch: slot.spec.batch.clone(),
                satisfies: vec![InterestId::from_index(entry.interest_idx)],
            };
            match substrate.submit(request) {
                IoAdmission::Submitted(id) => {
                    self.in_flight.push((id, entry.interest_idx));
                    if let Some(slot) = self.interests.get_mut(entry.interest_idx) {
                        slot.state = SlotState::InFlight;
                    }
                    submitted += 1;
                }
                IoAdmission::Refused(_) => {
                    refused.push(entry);
                }
            }
        }
        // Refused entries go back on the heap front for the next pull.
        for entry in refused.into_iter().rev() {
            self.submittable.insert(0, entry);
        }
        Ok(submitted)
    }

    fn complete(
        &mut self,
        request: SubmittedRequestId,
        result: IoResult,
    ) -> EngineResult<()> {
        let pos = self.in_flight.iter().position(|(id, _)| *id == request);
        if let Some(pos) = pos {
            let (_, interest_idx) = self.in_flight.swap_remove(pos);
            if let Some(slot) = self.interests.get_mut(interest_idx) {
                if matches!(slot.state, SlotState::InFlight) {
                    slot.state = SlotState::Completed { batch: result.batch };
                }
            }
        }
        Ok(())
    }

    fn register(&mut self, owner: OperatorId, spec: InterestSpec) -> InterestId {
        let id = InterestId::from_index(self.interests.len());
        self.interests.push(InterestSlot {
            spec,
            owner,
            state: SlotState::Registered,
        });
        id
    }

    fn cancel(&mut self, interest: InterestId) {
        if let Some(slot) = self.interests.get_mut(interest.index()) {
            if !matches!(slot.state, SlotState::Taken) {
                slot.state = SlotState::Cancelled;
            }
        }
    }

    fn take_completed(&mut self, owner: OperatorId) -> Option<CompletedInterest> {
        for (idx, slot) in self.interests.iter_mut().enumerate() {
            if slot.owner != owner {
                continue;
            }
            if let SlotState::Completed { batch } = slot.state.clone() {
                slot.state = SlotState::Taken;
                return Some(CompletedInterest {
                    interest: InterestId::from_index(idx),
                    label: slot.spec.label.clone(),
                    batch,
                });
            }
        }
        None
    }

    fn has_pending(&self) -> bool {
        self.interests.iter().any(|slot| {
            matches!(
                slot.state,
                SlotState::Registered
                    | SlotState::Submittable
                    | SlotState::InFlight
                    | SlotState::Completed { .. }
            )
        })
    }
}

/// Reference substrate implementation for the prototype: a Vec of
/// in-flight requests with per-turn delay counters. Substitutes for
/// io_uring/kqueue/etc. while we validate the broker/driver
/// architecture. Drivers that own this substrate call `tick()` once
/// per turn to advance counters; completed requests are returned to
/// the driver, which forwards them to `broker.complete`.
#[derive(Debug, Default)]
pub struct FakeDriverIo {
    /// Capacity in slots. `usize::MAX` means unbounded.
    capacity: usize,
    next_id: u64,
    in_flight: Vec<DelayInFlight>,
}

#[derive(Debug)]
struct DelayInFlight {
    broker: BrokerId,
    id: SubmittedRequestId,
    remaining_turns: usize,
    batch: Batch,
    bytes: usize,
}

impl FakeDriverIo {
    pub fn new() -> Self {
        Self {
            capacity: usize::MAX,
            next_id: 0,
            in_flight: Vec::new(),
        }
    }

    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            capacity,
            next_id: 0,
            in_flight: Vec::new(),
        }
    }

    pub fn in_flight_count(&self) -> usize {
        self.in_flight.len()
    }
}

impl DriverIo for FakeDriverIo {
    fn submit(&mut self, request: PhysicalRequest) -> IoAdmission {
        if self.in_flight.len() >= self.capacity {
            return IoAdmission::Refused(request);
        }
        let id = SubmittedRequestId(self.next_id);
        self.next_id += 1;
        self.in_flight.push(DelayInFlight {
            broker: request.broker,
            id,
            remaining_turns: request.delay_turns.max(1),
            batch: request.batch,
            bytes: request.bytes,
        });
        IoAdmission::Submitted(id)
    }

    fn capacity_hint(&self) -> IoCapacity {
        IoCapacity {
            free_slots: self.capacity.saturating_sub(self.in_flight.len()),
            total_slots: self.capacity,
        }
    }

    fn drain_completions(&mut self) -> Vec<IoCompletion> {
        let mut completed = Vec::new();
        let mut keep = Vec::with_capacity(self.in_flight.len());
        for entry in self.in_flight.drain(..) {
            if entry.remaining_turns > 1 {
                keep.push(DelayInFlight {
                    broker: entry.broker,
                    id: entry.id,
                    remaining_turns: entry.remaining_turns - 1,
                    batch: entry.batch,
                    bytes: entry.bytes,
                });
            } else {
                completed.push(IoCompletion {
                    broker: entry.broker,
                    request: entry.id,
                    result: IoResult { batch: entry.batch },
                });
            }
        }
        self.in_flight = keep;
        completed
    }

    fn in_flight(&self) -> usize {
        self.in_flight.len()
    }

    fn retained_bytes(&self) -> usize {
        self.in_flight.iter().map(|e| e.bytes).sum()
    }
}
