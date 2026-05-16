//! Reconstructed scheduler state — what the cursor exposes at any
//! timeline position.
//!
//! Mirrors `TurnSnapshot` plus mid-turn fields (current phase, the
//! per-worker step in progress).

use serde::{Deserialize, Serialize};

use vortex_trace_format::record::{
    BrokerId, ChannelId, OperatorId, PhaseKind, StepKind, WorkerId,
};
use vortex_trace_format::serialized::{SerializedDomainSpan, SerializedRequirementSet};
use vortex_trace_format::snapshot::{
    AsyncSnap, BrokerSnapshot, ChannelSnapshot, HeapEntrySnap, PortRequirementSnap, TurnSnapshot,
    WorkerSnapshot,
};

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct WorkerState {
    pub worker_id: u32,
    pub heap: Vec<HeapEntrySnap>,
    pub last_popped: Option<(OperatorId, u32, f32)>,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct ChannelState {
    pub channel: u32,
    pub buffered: Vec<SerializedDomainSpan>,
    pub buffered_bytes: u64,
    pub capacity_bytes: u64,
    pub output_requirement: SerializedRequirementSet,
    pub sealed: bool,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct BrokerState {
    pub broker: u32,
    pub in_flight: Vec<InFlightState>,
    pub pending_proposals: Vec<PendingProposal>,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct InFlightState {
    pub request_id: u64,
    pub label: String,
    pub since_turn: u32,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct PendingProposal {
    pub score: f32,
    pub rows: u64,
    pub bytes: u64,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct AsyncState {
    pub async_id: u64,
    pub label: String,
    pub span: SerializedDomainSpan,
    pub since_turn: u32,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct PortRequirementState {
    pub op: u32,
    pub port: u32,
    pub requirement: SerializedRequirementSet,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct WorkerStepInProgress {
    pub op: OperatorId,
    pub lane: u32,
    pub kind: StepKind,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct SchedulerState {
    pub turn: u32,
    pub workers: Vec<WorkerState>,
    pub channels: Vec<ChannelState>,
    pub brokers: Vec<BrokerState>,
    pub async_in_flight: Vec<AsyncState>,
    pub requirements: Vec<PortRequirementState>,
    pub lane_finished: Vec<u8>,
    pub lane_owner: Vec<u8>,
    pub memory_used_bytes: u64,
    pub current_phase: Option<PhaseKind>,
    pub worker_step: Vec<Option<WorkerStepInProgress>>,
    pub last_event_summary: Option<String>,
}

impl SchedulerState {
    /// Initialise an empty state from the trace header. Containers
    /// are sized so per-event applies are O(1) without realloc.
    pub fn from_header(header: &vortex_trace_format::TraceHeader) -> Self {
        let workers = (0..header.task_options.worker_count)
            .map(|i| WorkerState {
                worker_id: i,
                ..Default::default()
            })
            .collect();
        let channels = header
            .channels
            .iter()
            .map(|c| ChannelState {
                channel: c.id.0,
                capacity_bytes: c.initial_capacity_bytes,
                ..Default::default()
            })
            .collect();
        let brokers = header
            .brokers
            .iter()
            .map(|b| BrokerState {
                broker: b.id.0,
                ..Default::default()
            })
            .collect();
        let worker_step = (0..header.task_options.worker_count).map(|_| None).collect();
        Self {
            workers,
            channels,
            brokers,
            worker_step,
            ..Default::default()
        }
    }

    /// Reset the state from a `TurnSnapshot`. After this call, the
    /// state corresponds to the *start* of the next turn (i.e. the
    /// snapshot is the close of the recorded turn; all events
    /// between this turn-end and the next TurnBegin are absorbed).
    pub fn load_snapshot(&mut self, snap: &TurnSnapshot) {
        self.turn = snap.turn;
        self.workers = snap
            .workers
            .iter()
            .map(|w: &WorkerSnapshot| WorkerState {
                worker_id: w.worker_id.0,
                heap: w.heap.clone(),
                last_popped: None,
            })
            .collect();
        self.channels = snap
            .channels
            .iter()
            .map(|c: &ChannelSnapshot| ChannelState {
                channel: c.channel.0,
                buffered: c.buffered.clone(),
                buffered_bytes: c.buffered_bytes,
                capacity_bytes: c.capacity_bytes,
                output_requirement: c.output_requirement.clone(),
                sealed: false,
            })
            .collect();
        self.brokers = snap
            .brokers
            .iter()
            .map(|b: &BrokerSnapshot| BrokerState {
                broker: b.broker.0,
                in_flight: b
                    .in_flight
                    .iter()
                    .map(|r| InFlightState {
                        request_id: r.request_id,
                        label: r.label.clone(),
                        since_turn: r.since_turn,
                    })
                    .collect(),
                pending_proposals: b
                    .pending_proposals
                    .iter()
                    .map(|p| PendingProposal {
                        score: p.score,
                        rows: p.value.rows,
                        bytes: p.value.bytes,
                    })
                    .collect(),
            })
            .collect();
        self.async_in_flight = snap
            .async_in_flight
            .iter()
            .map(|a: &AsyncSnap| AsyncState {
                async_id: a.async_id.0,
                label: a.label.clone(),
                span: a.span.clone(),
                since_turn: a.since_turn,
            })
            .collect();
        self.requirements = snap
            .requirements
            .iter()
            .map(|r: &PortRequirementSnap| PortRequirementState {
                op: r.port.op.0,
                port: r.port.port.0,
                requirement: r.requirement.clone(),
            })
            .collect();
        self.lane_finished = snap.lane_finished.clone();
        self.lane_owner = snap.lane_owner.clone();
        self.memory_used_bytes = snap.memory_used_bytes;
        self.current_phase = None;
        for slot in self.worker_step.iter_mut() {
            *slot = None;
        }
        self.last_event_summary = None;
    }

    fn worker_mut(&mut self, id: WorkerId) -> Option<&mut WorkerState> {
        if id.is_main() {
            return None;
        }
        self.workers.iter_mut().find(|w| w.worker_id == id.0)
    }

    fn worker_step_slot(&mut self, id: WorkerId) -> Option<&mut Option<WorkerStepInProgress>> {
        if id.is_main() {
            return None;
        }
        self.worker_step.get_mut(id.0 as usize)
    }

    fn channel_mut(&mut self, id: ChannelId) -> Option<&mut ChannelState> {
        self.channels.iter_mut().find(|c| c.channel == id.0)
    }

    fn broker_mut(&mut self, id: BrokerId) -> Option<&mut BrokerState> {
        self.brokers.iter_mut().find(|b| b.broker == id.0)
    }

    fn require_port_mut(&mut self, op: OperatorId, port: u32) -> &mut PortRequirementState {
        let idx = self
            .requirements
            .iter()
            .position(|r| r.op == op.0 && r.port == port);
        match idx {
            Some(i) => &mut self.requirements[i],
            None => {
                self.requirements.push(PortRequirementState {
                    op: op.0,
                    port,
                    requirement: Default::default(),
                });
                self.requirements.last_mut().unwrap()
            }
        }
    }

    /// Apply one recorded event as a delta. The state is mutated in
    /// place. Returns a short human label suitable for the event
    /// log.
    pub fn apply(&mut self, record: &vortex_trace_format::TraceRecord) {
        use vortex_trace_format::record::TracePayload as P;
        let summary = match &record.payload {
            P::TurnBegin => {
                self.turn = record.turn;
                self.current_phase = None;
                "TurnBegin".to_string()
            }
            P::PhaseBegin { phase } => {
                self.current_phase = Some(*phase);
                format!("PhaseBegin {:?}", phase)
            }
            P::PhaseEnd { phase } => {
                if Some(*phase) == self.current_phase {
                    self.current_phase = None;
                }
                format!("PhaseEnd {:?}", phase)
            }
            P::TurnEnd { outcome } => {
                self.current_phase = None;
                format!("TurnEnd {:?}", outcome)
            }
            P::WorkerStepBegin { op, lane, kind } => {
                if let Some(slot) = self.worker_step_slot(record.worker_id) {
                    *slot = Some(WorkerStepInProgress {
                        op: *op,
                        lane: *lane,
                        kind: *kind,
                    });
                }
                format!("WorkerStepBegin op#{}/{}", op.0, lane)
            }
            P::WorkerStepEnd { op, lane } => {
                if let Some(slot) = self.worker_step_slot(record.worker_id) {
                    *slot = None;
                }
                format!("WorkerStepEnd op#{}/{}", op.0, lane)
            }
            P::RequirementChanged { port, requirement } => {
                let r = self.require_port_mut(port.op, port.port.0);
                r.requirement = requirement.clone();
                format!("RequirementChanged op#{}/p{}", port.op.0, port.port.0)
            }
            P::RequirementSpmcMerged { channel, merged } => {
                if let Some(ch) = self.channel_mut(*channel) {
                    ch.output_requirement = merged.clone();
                }
                format!("RequirementSpmcMerged ch#{}", channel.0)
            }
            P::RequirementRootRequired { rows } => {
                format!("RequirementRootRequired rows={}", rows)
            }
            P::RequirementNotNeeded { op, start, end } => {
                format!("RequirementNotNeeded op#{} [{}, {})", op.0, start, end)
            }
            P::AggregateLimitSealed { op } => format!("AggregateLimitSealed op#{}", op.0),
            P::LateFilterMarkedSuffix { op } => format!("LateFilterMarkedSuffix op#{}", op.0),
            P::ProposalsCleared { op, lane } => {
                if let Some(w) = self.worker_mut(record.worker_id) {
                    w.heap.retain(|e| !(e.op == *op && e.lane == *lane));
                }
                format!("ProposalsCleared op#{}/{}", op.0, lane)
            }
            P::ProposalEnqueued {
                op,
                lane,
                class,
                score,
                value,
                cost,
            } => {
                if let Some(w) = self.worker_mut(record.worker_id) {
                    w.heap.push(HeapEntrySnap {
                        op: *op,
                        lane: *lane,
                        class: *class,
                        score: *score,
                        value: value.clone(),
                        cost: cost.clone(),
                    });
                    // Keep the heap sorted by score descending so
                    // panels can render top-K cheaply.
                    w.heap.sort_by(|a, b| {
                        b.score
                            .partial_cmp(&a.score)
                            .unwrap_or(std::cmp::Ordering::Equal)
                    });
                }
                format!("ProposalEnqueued op#{}/{} score={:.3}", op.0, lane, score)
            }
            P::ProposalPopped { op, lane, score } => {
                if let Some(w) = self.worker_mut(record.worker_id) {
                    if let Some(pos) = w.heap.iter().position(|e| {
                        e.op == *op && e.lane == *lane && (e.score - *score).abs() < 1e-6
                    }) {
                        w.heap.remove(pos);
                    } else if let Some(pos) =
                        w.heap.iter().position(|e| e.op == *op && e.lane == *lane)
                    {
                        w.heap.remove(pos);
                    }
                    w.last_popped = Some((*op, *lane, *score));
                }
                format!("ProposalPopped op#{}/{} score={:.3}", op.0, lane, score)
            }
            P::OperatorRun {
                op,
                lane,
                label,
                class: _,
                score,
            } => {
                format!("OperatorRun op#{}/{} \"{}\" score={:.3}", op.0, lane, label, score)
            }
            P::ChannelPush {
                channel,
                op: _,
                lane: _,
                span,
                rows: _,
                bytes,
            } => {
                if let Some(c) = self.channel_mut(*channel) {
                    c.buffered.push(span.clone());
                    c.buffered_bytes = c.buffered_bytes.saturating_add(*bytes);
                }
                format!("ChannelPush ch#{} bytes={}", channel.0, bytes)
            }
            P::ChannelPop {
                channel,
                op: _,
                lane: _,
                port: _,
                rows: _,
            } => {
                if let Some(c) = self.channel_mut(*channel) {
                    if !c.buffered.is_empty() {
                        let span = c.buffered.remove(0);
                        let approx = (span.end - span.start).max(1);
                        c.buffered_bytes = c.buffered_bytes.saturating_sub(approx);
                    }
                }
                format!("ChannelPop ch#{}", channel.0)
            }
            P::ChannelSeal { channel } => {
                if let Some(c) = self.channel_mut(*channel) {
                    c.sealed = true;
                }
                format!("ChannelSeal ch#{}", channel.0)
            }
            P::AsyncSubmitted {
                op,
                lane: _,
                async_id,
                label,
                span,
                latency_class: _,
            } => {
                self.async_in_flight.push(AsyncState {
                    async_id: async_id.0,
                    label: label.clone(),
                    span: span.clone(),
                    since_turn: record.turn,
                });
                format!("AsyncSubmitted op#{} async={}", op.0, async_id.0)
            }
            P::AsyncWake {
                async_id,
                label: _,
                span: _,
            } => {
                self.async_in_flight.retain(|a| a.async_id != async_id.0);
                format!("AsyncWake async={}", async_id.0)
            }
            P::AsyncCancelled {
                async_id,
                label: _,
                span: _,
            } => {
                self.async_in_flight.retain(|a| a.async_id != async_id.0);
                format!("AsyncCancelled async={}", async_id.0)
            }
            P::BrokerProposalEnqueued {
                broker,
                score,
                value,
                cost: _,
            } => {
                if let Some(b) = self.broker_mut(*broker) {
                    b.pending_proposals.push(PendingProposal {
                        score: *score,
                        rows: value.rows,
                        bytes: value.bytes,
                    });
                }
                format!("BrokerProposalEnqueued broker#{} score={:.3}", broker.0, score)
            }
            P::BrokerSubmit {
                broker,
                label,
                latency: _,
                required_rows: _,
                score: _,
            } => {
                if let Some(b) = self.broker_mut(*broker) {
                    if !b.pending_proposals.is_empty() {
                        b.pending_proposals.remove(0);
                    }
                }
                format!("BrokerSubmit broker#{} \"{}\"", broker.0, label)
            }
            P::BrokerPull {
                broker,
                request_id,
                count,
            } => {
                if let Some(b) = self.broker_mut(*broker) {
                    b.in_flight.push(InFlightState {
                        request_id: *request_id,
                        label: format!("pull#{}", request_id),
                        since_turn: record.turn,
                    });
                }
                format!("BrokerPull broker#{} req={} count={}", broker.0, request_id, count)
            }
            P::SubstrateComplete {
                broker,
                request_id,
                result_summary: _,
            } => {
                if let Some(b) = self.broker_mut(*broker) {
                    b.in_flight.retain(|r| r.request_id != *request_id);
                }
                format!(
                    "SubstrateComplete broker#{} req={}",
                    broker.0, request_id
                )
            }
            P::ResourcePublished { id } => format!("ResourcePublished id={}", id),
            P::MemoryGrantChanged { changes } => {
                for change in changes {
                    if let Some(c) = self.channel_mut(change.channel) {
                        c.capacity_bytes = change.new_capacity_bytes;
                    }
                }
                format!("MemoryGrantChanged ({} channels)", changes.len())
            }
            P::OperatorMessage { op, message } => {
                format!("OperatorMessage op#{}: {}", op.0, message)
            }
        };
        self.last_event_summary = Some(summary);
    }
}
