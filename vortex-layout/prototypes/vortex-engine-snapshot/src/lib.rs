//! Operator-graph execution prototype for the Vortex engine.
//!
//! Top-level modules:
//!
//! - [`operator`] — authoring ABI: `Operator` trait, `OperatorSpec`,
//!   lane / parallelism declarations, the `*Ctx` types passed to
//!   `update`/`run`/`propagate_requirements`.
//! - [`graph`] — `OperatorGraph`, port specs, and `OperatorId` /
//!   `BrokerId` / port id types.
//! - [`channels`] — point-to-point channel between operator ports
//!   (`Channel`, `ChannelBuffer`, `ChannelSpec`) plus `Batch`.
//! - [`domain`] — `Domain` / `DomainSpan` / `Cardinality` and the
//!   row-coverage `RequirementSet` / `RowDemand` types.
//! - [`work`] — work proposal vocabulary: `WorkProposal`,
//!   `WorkClass`, `WorkValue`, `WorkCost`, `WorkConstraints`,
//!   `WorkKey`, `WorkStatus`.
//! - [`resources`] — shared side-state objects published between
//!   operators (bloom filters, dynamic filters, finalised tables).
//! - [`brokers`] — scheduler-visible queues for constrained
//!   external work (`Broker`, `BrokerProposal`, `SimpleDelayBroker`).
//! - [`scheduler`] — `PreparedTask`, per-shard scheduler, lane
//!   runtime, turn pipeline, async work, execution metrics.
//! - [`drivers`] — spawning policy: `CurrentThreadDriver` today,
//!   thread-per-core / Tokio drivers planned.
//! - [`error`] — `EngineError` / `EngineResult`.
//! - [`examples`] — worked end-to-end examples that demonstrate
//!   scheduler behaviours (used by integration tests).
//! - [`operators`] — concrete operators (`Aggregate`, sinks).
//! - [`layouts`] — Vortex layout binding: every layout encoding
//!   decomposes into engine-visible operators via
//!   `bind_into_graph`.
//! - [`queries`] — application-level query helpers (ClickBench
//!   Q5, Q20, …).
//!
//! Most internal modules cross-reference each other through the
//! crate-root re-exports below, which keep import paths short
//! inside the crate without committing every internal module to
//! one specific path.

pub mod brokers;
pub mod channels;
pub mod domain;
pub mod drivers;
pub mod error;
pub mod examples;
pub mod graph;
pub mod kernels;
pub mod layouts;
pub mod operator;
pub mod operators;
pub mod physical_plan;
pub mod queries;
pub mod resources;
pub mod scheduler;
pub mod work;

// Crate-root re-exports for ergonomic intra-crate imports.
// Public users may also import from these modules directly if they
// prefer fully-qualified paths.
pub use brokers::*;
pub use channels::*;
pub use domain::*;
pub use drivers::*;
pub use error::*;
pub use graph::*;
pub use operator::*;
pub use resources::*;
pub use scheduler::*;
pub use work::*;
