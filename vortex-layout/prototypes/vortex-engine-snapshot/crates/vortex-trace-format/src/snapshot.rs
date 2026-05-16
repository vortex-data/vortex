//! Per-turn keyframe snapshots.

use alloc::string::String;
use alloc::vec::Vec;
use serde::{Deserialize, Serialize};

use crate::record::{
    AsyncId, BrokerId, BrokerProposalSnap, ChannelId, InputPortRef, LatencyClass, OperatorId,
    ProposalCostSnap, ProposalValueSnap, WorkClass, WorkerId,
};
use crate::serialized::{SerializedDomainSpan, SerializedRequirementSet};

/// One entry in a worker's EV heap at a turn boundary.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct HeapEntrySnap {
    pub op: OperatorId,
    pub lane: u32,
    pub class: WorkClass,
    pub score: f32,
    pub value: ProposalValueSnap,
    pub cost: ProposalCostSnap,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct WorkerSnapshot {
    pub worker_id: WorkerId,
    pub heap: Vec<HeapEntrySnap>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ChannelSnapshot {
    pub channel: ChannelId,
    pub buffered: Vec<SerializedDomainSpan>,
    pub buffered_bytes: u64,
    pub capacity_bytes: u64,
    pub output_requirement: SerializedRequirementSet,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct InFlightRequest {
    pub request_id: u64,
    pub label: String,
    pub since_turn: u32,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct BrokerSnapshot {
    pub broker: BrokerId,
    pub in_flight: Vec<InFlightRequest>,
    pub pending_proposals: Vec<BrokerProposalSnap>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct AsyncSnap {
    pub async_id: AsyncId,
    pub label: String,
    pub span: SerializedDomainSpan,
    pub since_turn: u32,
    pub latency_class: LatencyClass,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct PortRequirementSnap {
    pub port: InputPortRef,
    pub requirement: SerializedRequirementSet,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct TurnSnapshot {
    pub turn: u32,
    pub workers: Vec<WorkerSnapshot>,
    pub channels: Vec<ChannelSnapshot>,
    pub brokers: Vec<BrokerSnapshot>,
    pub async_in_flight: Vec<AsyncSnap>,
    pub requirements: Vec<PortRequirementSnap>,
    /// Bitset of finished `(op, lane)` pairs. Layout: a flat Vec
    /// indexed by `op_id * lane_count + lane`. Lane count comes
    /// from the operator's header.
    pub lane_finished: Vec<u8>,
    pub lane_owner: Vec<u8>,
    pub memory_used_bytes: u64,
}
