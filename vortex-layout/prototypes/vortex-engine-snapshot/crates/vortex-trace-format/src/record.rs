//! Per-event records and the `TracePayload` enum.

use alloc::string::String;
use alloc::vec::Vec;
use serde::{Deserialize, Serialize};

use crate::serialized::{SerializedDomainSpan, SerializedRequirementSet};

/// Stable identifier for an operator within one trace.
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct OperatorId(pub u32);

/// Stable identifier for a channel.
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct ChannelId(pub u32);

/// Stable identifier for a broker.
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct BrokerId(pub u32);

/// Identifier for an in-flight async operation.
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct AsyncId(pub u64);

/// Identifier for a worker; the sentinel `MAIN` covers main-thread
/// (phase-1 / admit) events.
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct WorkerId(pub u32);

impl WorkerId {
    pub const MAIN: Self = Self(u32::MAX);

    pub fn is_main(self) -> bool {
        self == Self::MAIN
    }
}

/// Identifier for one input port on an operator.
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct InputPortId(pub u32);

/// Reference to a specific operator's input port.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct InputPortRef {
    pub op: OperatorId,
    pub port: InputPortId,
}

#[derive(Copy, Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum PhaseKind {
    MaintainAsync,
    DriveSubstrate,
    Propagate,
    RebalanceMemory,
    AdmitBrokers,
    WorkerSteps,
    Classify,
}

#[derive(Copy, Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum StepKind {
    Update,
    Run,
}

#[derive(Copy, Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum WorkClass {
    Cpu,
    Mixed,
    Io,
}

#[derive(Copy, Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum LatencyClass {
    Inline,
    Short,
    Long,
}

#[derive(Copy, Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum TurnOutcome {
    Progress,
    Quiesced,
    Done,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ProposalValueSnap {
    pub rows: u64,
    pub bytes: u64,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ProposalCostSnap {
    pub class: WorkClass,
    pub estimated_micros: u32,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ChannelGrantChange {
    pub channel: ChannelId,
    pub old_capacity_bytes: u64,
    pub new_capacity_bytes: u64,
}

/// The envelope wrapping every recorded event.
///
/// `worker_id` is `WorkerId::MAIN` for main-thread events
/// (phase-1 maintenance, admission, classify); otherwise it
/// identifies which worker produced the event.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct TraceRecord {
    pub worker_id: WorkerId,
    pub turn: u32,
    pub payload: TracePayload,
}

/// The full event vocabulary. Each variant corresponds to a single
/// observable state change in the engine.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum TracePayload {
    // Turn and phase markers
    TurnBegin,
    PhaseBegin {
        phase: PhaseKind,
    },
    PhaseEnd {
        phase: PhaseKind,
    },
    TurnEnd {
        outcome: TurnOutcome,
    },
    WorkerStepBegin {
        op: OperatorId,
        lane: u32,
        kind: StepKind,
    },
    WorkerStepEnd {
        op: OperatorId,
        lane: u32,
    },

    // Demand signals
    RequirementChanged {
        port: InputPortRef,
        requirement: SerializedRequirementSet,
    },
    RequirementSpmcMerged {
        channel: ChannelId,
        merged: SerializedRequirementSet,
    },
    RequirementRootRequired {
        rows: u64,
    },
    RequirementNotNeeded {
        op: OperatorId,
        start: u64,
        end: u64,
    },
    AggregateLimitSealed {
        op: OperatorId,
    },
    LateFilterMarkedSuffix {
        op: OperatorId,
    },

    // Heap and proposal events
    ProposalsCleared {
        op: OperatorId,
        lane: u32,
    },
    ProposalEnqueued {
        op: OperatorId,
        lane: u32,
        class: WorkClass,
        score: f32,
        value: ProposalValueSnap,
        cost: ProposalCostSnap,
    },
    ProposalPopped {
        op: OperatorId,
        lane: u32,
        score: f32,
    },
    OperatorRun {
        op: OperatorId,
        lane: u32,
        label: String,
        class: WorkClass,
        score: f32,
    },

    // Channel events
    ChannelPush {
        channel: ChannelId,
        op: OperatorId,
        lane: u32,
        span: SerializedDomainSpan,
        rows: u64,
        bytes: u64,
    },
    ChannelPop {
        channel: ChannelId,
        op: OperatorId,
        lane: u32,
        port: InputPortId,
        rows: u64,
    },
    ChannelSeal {
        channel: ChannelId,
    },

    // In-flight work and I/O
    AsyncSubmitted {
        op: OperatorId,
        lane: u32,
        async_id: AsyncId,
        label: String,
        span: SerializedDomainSpan,
        latency_class: LatencyClass,
    },
    AsyncWake {
        async_id: AsyncId,
        label: String,
        span: SerializedDomainSpan,
    },
    AsyncCancelled {
        async_id: AsyncId,
        label: String,
        span: SerializedDomainSpan,
    },

    BrokerProposalEnqueued {
        broker: BrokerId,
        score: f32,
        value: ProposalValueSnap,
        cost: ProposalCostSnap,
    },
    BrokerSubmit {
        broker: BrokerId,
        label: String,
        latency: LatencyClass,
        required_rows: u64,
        score: f32,
    },
    BrokerPull {
        broker: BrokerId,
        request_id: u64,
        count: u64,
    },
    SubstrateComplete {
        broker: BrokerId,
        request_id: u64,
        result_summary: String,
    },

    // Resources, memory, free-form
    ResourcePublished {
        id: u32,
    },
    MemoryGrantChanged {
        changes: Vec<ChannelGrantChange>,
    },
    OperatorMessage {
        op: OperatorId,
        message: String,
    },
}

impl TracePayload {
    /// Stable, low-cardinality variant name for filter UIs and
    /// timeline summaries.
    pub fn variant(&self) -> &'static str {
        match self {
            TracePayload::TurnBegin => "TurnBegin",
            TracePayload::PhaseBegin { .. } => "PhaseBegin",
            TracePayload::PhaseEnd { .. } => "PhaseEnd",
            TracePayload::TurnEnd { .. } => "TurnEnd",
            TracePayload::WorkerStepBegin { .. } => "WorkerStepBegin",
            TracePayload::WorkerStepEnd { .. } => "WorkerStepEnd",
            TracePayload::RequirementChanged { .. } => "RequirementChanged",
            TracePayload::RequirementSpmcMerged { .. } => "RequirementSpmcMerged",
            TracePayload::RequirementRootRequired { .. } => "RequirementRootRequired",
            TracePayload::RequirementNotNeeded { .. } => "RequirementNotNeeded",
            TracePayload::AggregateLimitSealed { .. } => "AggregateLimitSealed",
            TracePayload::LateFilterMarkedSuffix { .. } => "LateFilterMarkedSuffix",
            TracePayload::ProposalsCleared { .. } => "ProposalsCleared",
            TracePayload::ProposalEnqueued { .. } => "ProposalEnqueued",
            TracePayload::ProposalPopped { .. } => "ProposalPopped",
            TracePayload::OperatorRun { .. } => "OperatorRun",
            TracePayload::ChannelPush { .. } => "ChannelPush",
            TracePayload::ChannelPop { .. } => "ChannelPop",
            TracePayload::ChannelSeal { .. } => "ChannelSeal",
            TracePayload::AsyncSubmitted { .. } => "AsyncSubmitted",
            TracePayload::AsyncWake { .. } => "AsyncWake",
            TracePayload::AsyncCancelled { .. } => "AsyncCancelled",
            TracePayload::BrokerProposalEnqueued { .. } => "BrokerProposalEnqueued",
            TracePayload::BrokerSubmit { .. } => "BrokerSubmit",
            TracePayload::BrokerPull { .. } => "BrokerPull",
            TracePayload::SubstrateComplete { .. } => "SubstrateComplete",
            TracePayload::ResourcePublished { .. } => "ResourcePublished",
            TracePayload::MemoryGrantChanged { .. } => "MemoryGrantChanged",
            TracePayload::OperatorMessage { .. } => "OperatorMessage",
        }
    }

    pub fn is_turn_end(&self) -> bool {
        matches!(self, TracePayload::TurnEnd { .. })
    }

    pub fn is_turn_begin(&self) -> bool {
        matches!(self, TracePayload::TurnBegin)
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct BrokerProposalSnap {
    pub broker: BrokerId,
    pub score: f32,
    pub value: ProposalValueSnap,
    pub cost: ProposalCostSnap,
}
