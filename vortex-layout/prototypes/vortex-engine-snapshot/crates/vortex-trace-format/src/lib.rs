//! On-disk schema and framing for `*.vtrx` engine trace files.
//!
//! This crate is the contract between the recorder (in
//! `vortex-engine`) and the replayer (`vortex-trace-replay`). It is
//! `no_std`-friendly so the replayer can compile to wasm without
//! pulling the engine.
//!
//! See `docs/design/trace-recording.md` for the rationale.

#![no_std]

extern crate alloc;

pub mod framing;
pub mod header;
pub mod record;
pub mod serialized;
pub mod snapshot;

pub use framing::{FORMAT_VERSION, MAGIC, RecordKind, decode_record_header, encode_record_header};
pub use header::{
    BrokerInfo, ChannelInfo, ChannelTopology, OperatorInfo, ResourceInfo, TaskOptionsSnap,
    TraceHeader,
};
pub use record::{
    AsyncId, BrokerId, BrokerProposalSnap, ChannelGrantChange, ChannelId, InputPortId,
    InputPortRef, LatencyClass, OperatorId, PhaseKind, ProposalCostSnap, ProposalValueSnap,
    StepKind, TracePayload, TraceRecord, TurnOutcome, WorkClass, WorkerId,
};
pub use serialized::{SerializedDomainSpan, SerializedRequirementSet, SerializedRequirementSpan};
pub use snapshot::{
    AsyncSnap, BrokerSnapshot, ChannelSnapshot, HeapEntrySnap, InFlightRequest,
    PortRequirementSnap, TurnSnapshot, WorkerSnapshot,
};

pub use postcard;
