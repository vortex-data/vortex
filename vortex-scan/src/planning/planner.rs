// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! The planner side of the protocol: work scopes, object state, and planner outputs.

use std::fmt;
use std::ops::Range;

use vortex_error::VortexResult;
use vortex_io::request::IoBatch;
use vortex_io::request::IoConsumer;

use crate::planning::morsel::Morsel;

/// Logical position carried with work and output for future driver ordering policies.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct WorkScope {
    /// Ordinal of the file the work belongs to.
    pub file_ordinal: u64,
    /// File-local row domain covered by the work.
    pub rows: Range<u64>,
}

/// What an object needs next. Cheap and side-effect free.
///
/// `state()` is called before every `compute()`, and `compute()` only when `state()` is
/// `NeedsCompute`. It never lists requests: every request is published once, from `compute()`,
/// and the driver keeps each `Fetch` until it is delivered.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum State {
    /// The object has finished and will be dropped without further calls.
    Done,
    /// The object can make progress on the CPU.
    NeedsCompute,
    /// The object waits for a delivery of one of the `Fetch` requests it published; any delivery
    /// may make it runnable. Waiting with no fetch outstanding is a protocol error.
    Waiting,
}

/// What one `compute()` produced.
///
/// Returning `NeedsIO` registers the batch immediately, and each request in it exactly once. The
/// next `state()` determines whether the object waits or has more CPU work. Optional requests
/// never require a wait or delivery. After `Done` the object is dropped without further calls.
pub enum PlannerOutput {
    /// The planner has finished.
    Done,
    /// A CPU-only checkpoint; the driver requeues the planner.
    Continue,
    /// Requests the planner publishes: fetches it needs, and optional hints.
    NeedsIO(IoBatch),
    /// A child planner, with the scope it is responsible for.
    Planner(WorkScope, Box<dyn Planner>),
    /// A morsel authorised to produce arrays for the scope.
    Morsel(WorkScope, Box<dyn Morsel>),
}

/// A planner decides which work is needed.
///
/// It privately owns cursors, selections, and pending inputs. `compute()` advances CPU work
/// when `state()` reports `NeedsCompute` and may produce one child, one IO batch, a checkpoint,
/// or finish. A planner moves between threads with the run that owns it.
pub trait Planner: IoConsumer + Send {
    /// Reports what the planner needs next without doing work.
    fn state(&self) -> State;

    /// Advances CPU work. Only called when [`state`](Self::state) is `NeedsCompute`.
    fn compute(&mut self) -> VortexResult<PlannerOutput>;
}

impl fmt::Debug for PlannerOutput {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Done => f.write_str("Done"),
            Self::Continue => f.write_str("Continue"),
            Self::NeedsIO(batch) => f.debug_tuple("NeedsIO").field(batch).finish(),
            Self::Planner(scope, _) => f.debug_tuple("Planner").field(scope).finish(),
            Self::Morsel(scope, _) => f.debug_tuple("Morsel").field(scope).finish(),
        }
    }
}
