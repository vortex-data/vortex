// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

#![deny(missing_docs)]

//! An experimental morsel-driven scan executor for Vortex layouts.
//!
//! This crate is the P1 spine of the design recorded in
//! `docs/developer-guide/internals/scan-execution-models/morsel-based-plan-execution.md`: the scan
//! is cut into *morsels* (contiguous root row ranges), and each morsel is driven by a tree of
//! stateful [`ExecNode`] state machines that pull values from their children.
//!
//! The two halves of the contract are:
//!
//! * [`ExecNode::next_plan`] — planning. A node *names* the IO it will need by registering
//!   [`IoUse`](io::IoUse)s against the [`IoPlane`](io::IoPlane), which hands back tickets. Nodes
//!   do not read during planning. Planning is budget-bounded and resumable: a node that exhausts
//!   its quantum yields [`PlanItem::Plan`] and resumes from its own cursor on the next call.
//! * [`ExecNode::execute`] — value production. When a named required cell is still unissued,
//!   [`ExecCx::ready`](node::ExecCx::ready) may attempt one source-provided read guaranteed not to
//!   wait on storage (Linux files use `preadv2(RWF_NOWAIT)`). A hit is consumed inline. A miss
//!   suspends on the exact ticket and the scheduler submits its batch to the shared urgent IO
//!   queue. Execution never polls a background future or waits for IO on the worker thread.
//!
//! Compared to the V1 `LayoutReader` path this executor differs in two measurable ways:
//!
//! 1. There is no async task per evaluation. Planning, IO polling, and execution continuations
//!    share one bounded worker pool; pending IO never parks a worker.
//! 2. Each worker owns one arena and one active morsel. Arenas never migrate, and emission order
//!    is restored by morsel index.
//!
//! Raw request cells are shared for the lifetime of a scan, deduplicating both pending and
//! completed segment reads. Decoded chunks use leased shared cells ([`cells::SharedCells`]): a
//! decoded chunk lives exactly while some not-yet-retired morsel holds a lease computed from the
//! morsel cut, and is dropped at the last release. Decoded sharing can be disabled independently
//! as a differential-test and benchmark mode.
//!
//! Only the FLAT, CHUNKED and STRUCT layout nodes are supported, plus the FILTER and
//! CONJUNCT operators. Anything else is rejected at build time by [`build::build_plan`].

pub mod build;
pub mod cells;
pub mod driver;
pub mod executor;
#[cfg(any(test, feature = "_test-harness"))]
pub mod fixtures;
#[cfg(any(test, feature = "_test-harness"))]
pub mod harness;
pub mod io;
pub mod node;
pub mod nodes;
pub mod stats;
#[cfg(any(test, feature = "_test-harness"))]
pub mod tpch;
#[cfg(any(test, feature = "_test-harness"))]
pub mod workloads;

pub use build::ExecPlan;
pub use build::build_plan;
pub use driver::MorselScan;
pub use driver::SharedMorselWorkerPool;
pub use driver::morsels;
pub use executor::MorselScanExecutor;
pub use node::ExecCx;
pub use node::ExecNode;
pub use node::ExecPoll;
pub use node::PlanCx;
pub use node::PlanItem;
pub use node::PlanPoll;
pub use node::Value;
pub use node::ValueBatch;
pub use stats::ScanStats;

#[cfg(test)]
mod tests;
