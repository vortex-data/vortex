// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! The morsel side of the protocol: array-producing work and its outputs.

use std::fmt;

use vortex_array::ArrayRef;
use vortex_error::VortexResult;
use vortex_io::segments::IoKey;
use vortex_io::segments::IoRequest;
use vortex_io::segments::SourceResult;

pub use crate::planning::planner::State;

/// What one morsel `compute()` produced.
///
/// A `Batch` is never empty; a morsel that finds no rows returns `Done`. Returning `NeedsIO`
/// registers the batch immediately, and each request in it exactly once; the next `state()`
/// determines whether the morsel waits. Optional requests need no delivery. After `Done` the
/// morsel is dropped without further calls.
pub enum MorselOutput {
    /// The morsel has finished.
    Done,
    /// A CPU-only checkpoint; the driver requeues the morsel.
    Continue,
    /// Requests the morsel publishes: fetches it needs, and optional hints.
    NeedsIO(Vec<IoRequest>),
    /// One non-empty array of selected rows, in input-row order.
    Batch(ArrayRef),
}

/// A morsel produces arrays for an authorised piece of work.
///
/// `state()` is called before every `compute()`, and `compute()` only when `state()` is
/// `NeedsCompute`. Batches preserve selected input-row order within and across calls. A morsel
/// moves between threads with the run that owns it.
pub trait Morsel: Send {
    /// Reports what the morsel needs next without doing work.
    fn state(&self) -> State;

    /// Advances CPU work. Only called when [`state`](Self::state) is `NeedsCompute`.
    fn compute(&mut self) -> VortexResult<MorselOutput>;

    /// The result of a read it fetched, once per fetch: a segment's bytes or a source's size. It
    /// never has two fetches of one key outstanding. A failed fetch is never delivered: whatever
    /// drives it fails the scan with the read's error instead.
    fn deliver(&mut self, key: IoKey, result: SourceResult);
}

impl fmt::Debug for MorselOutput {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Done => f.write_str("Done"),
            Self::Continue => f.write_str("Continue"),
            Self::NeedsIO(batch) => f.debug_tuple("NeedsIO").field(batch).finish(),
            Self::Batch(array) => f.debug_tuple("Batch").field(&array.len()).finish(),
        }
    }
}
