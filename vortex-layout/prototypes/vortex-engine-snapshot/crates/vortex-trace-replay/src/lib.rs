//! Replay engine for `*.vtrx` files.
//!
//! Open a trace file with [`TraceFile::open`] (native) or
//! [`TraceFile::from_bytes`] (wasm), then drive a [`ReplayCursor`]
//! to reconstruct scheduler state at any timeline position.
//!
//! See `docs/design/trace-replay.md` for the spec.

pub mod cursor;
pub mod error;
pub mod file;
pub mod fixture;
pub mod index;
pub mod state;
pub mod writer;

pub use cursor::{ReplayCursor, TimelinePos};
pub use error::ReplayError;
pub use file::TraceFile;
pub use index::{TimelineSummary, TurnCategoryCounts};
pub use state::{
    AsyncState, BrokerState, ChannelState, InFlightState, PendingProposal, PortRequirementState,
    SchedulerState, WorkerState, WorkerStepInProgress,
};
pub use writer::{TraceWriter, write_event, write_snapshot};

pub use vortex_trace_format as format;
