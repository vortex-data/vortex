// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Push-based execution of physical plans as pipelines.
//!
//! A [`Scan`] runs a [`PlanRef`](crate::plan::physical::PlanRef) over a list of splits, row ranges
//! of the plan's domain. Each split is compiled into pipelines: a pipeline is one
//! [`Source`](super::operator::Source), zero or more [`Operator`](super::operator::Operator)s,
//! and the ports its last stage writes. The driver asks the source for a batch and pushes it
//! through the stages into the ports. Every port runs from one pipeline's sink,
//! the end that pushes into its outlets, to another's source, which reads it as an inlet, and
//! the plan joining them declares how many batches each inlet may hold.
//!
//! A source checks its readiness before computing, waiting for a segment read or an inlet.
//! The driver also blocks a pipeline when an outlet is full. The scheduler re-runs a blocked
//! pipeline when that one condition clears.
//!
//! The owner of a scan answers reads: [`Scan::step`] returns each read as a [`Turn::Read`], and
//! [`Scan::deliver`] hands its bytes back. Nothing in a scan awaits.
//!
//! # Masks
//!
//! Every mask a scan applies is known when the pipelines that apply it are compiled: the split's
//! selection, and the rows the earlier conjuncts kept. The
//! compiler places the mask in the stage that reads the rows, so chunks a mask selects nothing
//! of are never built and never read.
//!
//! # Sharing
//!
//! A segment several readers decode is decoded once. Before the scan starts, one pass over the
//! plans every split's stages run records the rows each segment is read for, so the splits that
//! will read it are found by binary search over the splits. The first reader of a segment read
//! more than once builds one pipeline that decodes it and fans the decoded array out into one
//! single-use port per reader, tagged with the reader's split: a reader in the same stage, a
//! later stage of a query, or a later split claims its own port. A split's ports it never
//! claims, because its rows were pruned or a stage never ran, leave the sharing pipeline's
//! outlets and are freed when it finishes, so nothing is ever pushed into a port nobody will
//! read, and a claimed port is dropped once its source has drained it, so a decoded segment
//! lives only as long as a split that may still read it.

mod compile;
mod port;
mod predicate;
mod scan;
pub mod synthetic;
mod values;

pub use self::compile::Chain;
pub use self::compile::Compiler;
pub(crate) use self::compile::overlapping;
pub(super) use self::port::Arena;
pub use self::port::Inlet;
pub(super) use self::port::PortId;
pub(crate) use self::predicate::PredicateCache;
pub use self::scan::ReadId;
pub use self::scan::ReadRequest;
pub use self::scan::Scan;
pub use self::scan::Split;
pub use self::scan::Turn;
pub use self::values::ValueCache;

/// Batches an inlet holds before its sink is blocked, unless the plan joining it says
/// otherwise; see [`Compiler::join_with`].
pub const DEFAULT_CAPACITY: usize = 8;

/// How a scan accesses rows within encoded segments.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum AccessMode {
    /// Stream slices and filter batches, retaining whole-value conversions for reuse.
    #[default]
    Scan,
    /// Prefer encoded gathers for sparse selections before materializing values.
    RandomAccess,
}

/// Leave dense selections to bulk kernels; sparse gathers avoid decoding unused values.
pub(crate) fn sparse_selection(selected: usize, total: usize) -> bool {
    selected > 0 && selected.saturating_mul(16) < total
}

#[cfg(test)]
mod scheduling_tests;
#[cfg(test)]
mod tests;
