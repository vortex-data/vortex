//! Static topology header written once at the start of a `*.vtrx`
//! file.

use alloc::string::String;
use alloc::vec::Vec;
use serde::{Deserialize, Serialize};

use crate::record::{BrokerId, ChannelId, InputPortRef, OperatorId};

/// Channel cardinality topology. Mirrors the engine's
/// `ChannelTopology` (`src/channels/mod.rs`).
#[derive(Copy, Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum ChannelTopology {
    Spsc,
    Mpsc,
    Spmc,
    Mpmc,
}

impl ChannelTopology {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Spsc => "spsc",
            Self::Mpsc => "mpsc",
            Self::Spmc => "spmc",
            Self::Mpmc => "mpmc",
        }
    }
}

/// Recorded `TaskOptions` at the time the trace was started.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct TaskOptionsSnap {
    pub max_turns: u64,
    pub memory_limit_bytes: u64,
    pub worker_count: u32,
}

/// Static description of one operator in the recorded graph.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct OperatorInfo {
    pub id: OperatorId,
    pub name: String,
    pub kind: String,
    pub input_ports: Vec<String>,
    pub output_ports: Vec<String>,
    pub lane_count: u32,
}

/// Static description of one channel between operators.
///
/// `producers` and `consumers` are vectors so the four
/// `ChannelTopology` variants can all be represented:
/// SPSC has 1+1, SPMC has 1+N, MPSC has N+1, MPMC has N+M.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ChannelInfo {
    pub id: ChannelId,
    pub name: String,
    pub topology: ChannelTopology,
    pub producers: Vec<OperatorId>,
    pub consumers: Vec<InputPortRef>,
    pub initial_capacity_bytes: u64,
}

/// Static description of one external work broker.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct BrokerInfo {
    pub id: BrokerId,
    pub name: String,
    pub label: String,
}

/// A resource produced by a producing operator and consumed by
/// peers via the resource bus.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ResourceInfo {
    pub id: u32,
    pub name: String,
    pub producer: OperatorId,
}

/// The full topology header. Written once at trace start.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct TraceHeader {
    pub format_version: u32,
    pub recorder_version: String,
    pub task_options: TaskOptionsSnap,
    pub operators: Vec<OperatorInfo>,
    pub channels: Vec<ChannelInfo>,
    pub brokers: Vec<BrokerInfo>,
    pub resources: Vec<ResourceInfo>,
    pub recorded_at_unix_secs: u64,
}
