// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Push-based execution of physical plans as pipelines.
//!
//! A [`Scan`] runs a [`PlanRef`](crate::plan::PlanRef) over a list of splits, row ranges
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
//! selection, the rows the earlier conjuncts kept, and the rows a running source computed for a
//! compile it asked for. The compiler places the mask in the stage that reads the rows, so chunks
//! a mask selects nothing of are never built and never read.
//!
//! # Compiling at run time
//!
//! A split's pipelines are compiled before any of them runs, except where the rows to read depend
//! on data: a list's elements depend on its offsets. A source then asks for more of the plan to
//! be compiled with [`Cx::compile`](super::operator::Cx::compile), over rows and a mask it
//! computed. Between runs, the scan compiles it into ordinary pipelines writing a new inlet of
//! that source, so the source reads the result like any other inlet. Only the scan changes the
//! pipeline graph, and only between runs.
//!
//! # Segments
//!
//! Each source reads the segments it needs: a segment two sources need is asked for by each.
//! Reading a segment twice is left to the scan's owner to make cheap, as by keeping recently read
//! segments; decoded arrays are shared only as whole values, through the [`ValueCache`].

mod compile;
pub(crate) mod port;
mod scan;
pub mod synthetic;
mod values;

pub use self::compile::Chain;
pub use self::compile::Compiler;
pub(crate) use self::compile::overlapping;
pub(crate) use self::port::Arena;
pub use self::port::Inlet;
pub(super) use self::port::PortId;
pub use self::scan::ReadId;
pub use self::scan::ReadRequest;
pub use self::scan::Scan;
pub use self::scan::Split;
pub use self::scan::Turn;
pub use self::values::ValueCache;

/// Batches an inlet holds before its sink is blocked, unless the plan joining it says
/// otherwise; see [`Compiler::join_with`].
pub const DEFAULT_CAPACITY: usize = 8;

#[cfg(test)]
mod deferred_tests;

#[cfg(test)]
mod scheduling_tests;

#[cfg(test)]
mod tests;
