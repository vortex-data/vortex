// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Push-based execution of physical plans as pipelines.
//!
//! A [`Scan`] runs a [`PlanRef`](crate::plan::PlanRef) over a list of splits, row ranges of the
//! plan's domain. Each split is compiled into pipelines: a pipeline is one [`Source`], zero or
//! more [`Operator`] stages, and the ports its last stage writes. The driver asks the source for
//! a batch and pushes it through the stages into the ports. Every port has exactly one writer and
//! one reader, and the reader's source declares how many batches it may hold.
//!
//! Only a source blocks, and only on one of three things: a segment read it requested, an empty
//! inlet, or a full outlet, which the driver detects. The scheduler re-runs a blocked pipeline
//! when that one condition clears.
//!
//! The owner of a scan answers reads: [`Scan::step`] returns each read as a [`Turn::Read`], and
//! [`Scan::deliver`] hands its bytes back. Nothing in a scan awaits.
//!
//! # Masks
//!
//! Every mask a scan applies is known when the pipelines that apply it are compiled: the split's
//! selection, and in a [`Query`](crate::plan::Query) the rows the earlier conjuncts kept. The
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
//! claims, because its rows were pruned or a stage never ran, are dropped when it finishes, and
//! a claimed port is dropped once its reader has drained it, so a decoded segment lives only as
//! long as a split that may still read it.

mod compile;
mod ops;
mod port;
mod query;
mod scan;
mod stream;

use vortex_array::ArrayRef;
use vortex_array::ExecutionCtx;
use vortex_array::buffer::BufferHandle;
use vortex_error::VortexResult;
use vortex_mask::Mask;
use vortex_session::VortexSession;

use self::port::Arena;
pub use self::port::Inlet;
use self::port::PortId;
pub use self::scan::ReadId;
pub use self::scan::ReadRequest;
pub use self::scan::Scan;
pub use self::scan::Split;
pub use self::scan::Turn;
pub use self::stream::ScanStream;
pub use self::stream::execute;
use crate::plan::PlanRef;
use crate::segments::SegmentId;

/// Batches an inlet holds before its writer is blocked, unless its reader says otherwise.
pub const DEFAULT_CAPACITY: usize = 2;

/// What a pipeline stage is handed on each call.
#[derive(Debug)]
pub enum Input {
    /// One batch from the stage below.
    Chunk(ArrayRef),
    /// The stage below has finished. Flush whatever is held.
    End,
    /// Nothing new. Sent to a source on every call, and to a stage that returned
    /// [`Step::More`].
    None,
}

/// What a stage says back.
#[derive(Debug)]
pub enum Step {
    /// A batch, and more is ready now without further input. The driver pushes the batch on
    /// and calls again with [`Input::None`].
    More(ArrayRef),
    /// A batch, and the input is spent.
    Last(ArrayRef),
    /// The input was absorbed and nothing is ready.
    Consumed,
    /// Nothing can happen until the condition clears. Sources only.
    Blocked(Blocked),
    /// No more output will ever come. The driver sends [`Input::End`] to the next stage.
    Finished,
}

/// Why a pipeline cannot run.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Blocked {
    /// A read was requested and has not been delivered.
    Io,
    /// The inlet is empty and its writer has not closed it.
    Inlet(usize),
    /// An outlet is full. Produced by the driver, never by a stage.
    Outlet,
}

/// A stage of a pipeline: takes one input at a time and pushes what it produces on.
///
/// A stage never blocks. A stage returning [`Step::More`] must eventually return something
/// else when called again with [`Input::None`]. A stage that has returned [`Step::Finished`] is
/// not called again. A stage receiving [`Input::End`] flushes, returning [`Step::More`] while it
/// has more to flush.
pub trait Operator: Send {
    /// Takes `input` and says what happens next.
    fn compute(&mut self, input: Input, cx: &mut Cx<'_>) -> VortexResult<Step>;
}

/// The start of a pipeline. Reads its inlets and requests segments rather than taking input
/// from a stage below. Always called with [`Input::None`].
pub trait Source: Operator {
    /// The inlets this source reads, numbered from zero.
    fn inlet_count(&self) -> usize {
        0
    }

    /// How many batches inlet `inlet` may hold before its writer is blocked. Asked once, when
    /// the port is made. An empty port always has room.
    fn capacity(&self, inlet: usize) -> usize {
        let _ = inlet;
        DEFAULT_CAPACITY
    }

    /// The segment this source wants read next, asked before every
    /// [`compute`](Operator::compute). A source that returns one is not computed until its
    /// bytes have arrived through [`Cx::take_bytes`].
    fn request(&mut self) -> Option<SegmentId> {
        None
    }
}

/// A plan a source asks to have compiled into a new inlet.
pub(crate) struct Spawn {
    plan: PlanRef,
    rows: std::ops::Range<u64>,
    mask: Mask,
}

/// What a stage may touch during one call: its pipeline's inlets, the bytes it asked for, and
/// the session.
pub struct Cx<'a> {
    arena: &'a mut Arena,
    inlets: &'a [PortId],
    bytes: &'a mut Option<BufferHandle>,
    session: &'a VortexSession,
    exec: &'a mut ExecutionCtx,
    spawns: &'a mut Vec<Spawn>,
}

impl Cx<'_> {
    /// The reading side of inlet `index` of this pipeline's source.
    pub fn inlet(&mut self, index: usize) -> Inlet<'_> {
        self.arena.inlet(self.inlets[index])
    }

    /// The bytes of the segment the source requested, once they have arrived.
    pub fn take_bytes(&mut self) -> Option<BufferHandle> {
        self.bytes.take()
    }

    /// The session used for decoding and evaluation.
    pub fn session(&self) -> &VortexSession {
        self.session
    }

    /// The execution context for executing arrays.
    pub fn exec(&mut self) -> &mut ExecutionCtx {
        self.exec
    }

    /// Asks for `plan` over `rows`, restricted to `mask`, to be compiled into a new inlet of this
    /// source once the call returns, and returns the inlet's index.
    ///
    /// For a source that learns what it must read only from what it has read, as a list learns
    /// its elements' range from its offsets.
    pub(crate) fn spawn(&mut self, plan: PlanRef, rows: std::ops::Range<u64>, mask: Mask) -> usize {
        self.spawns.push(Spawn { plan, rows, mask });
        self.inlets.len() + self.spawns.len() - 1
    }
}

#[cfg(test)]
mod tests;
