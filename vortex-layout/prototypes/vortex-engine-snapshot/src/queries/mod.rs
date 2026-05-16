//! End-to-end query implementations driven by the operator-graph
//! engine and the layout binding layer.
//!
//! Each submodule wires a specific (small) SQL-shaped query as a graph
//! over `crate::layouts` source operators, ending in a sink that
//! produces the final scalar/aggregate result. These are not yet a
//! frontend — there's no parser or planner. They exist so the engine
//! can be measured against established benchmark queries.

pub mod bench;
pub mod clickbench;

pub use bench::*;
