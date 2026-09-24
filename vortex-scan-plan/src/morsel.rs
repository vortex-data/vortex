// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! The morsel side of the protocol: array-producing work and its outputs.

use std::fmt;

use vortex_array::ArrayRef;
use vortex_error::VortexResult;

use crate::io::IoBatch;
use crate::io::IoConsumer;
pub use crate::planner::State;

/// What one morsel `compute()` produced.
///
/// A `Batch` is never empty; a morsel that finds no rows returns `Done`. Returning `NeedsIO`
/// publishes the batch, but the driver acts on the next `state()`, so the morsel must report
/// the same batch there. After `Done` the morsel is dropped without further calls.
pub enum MorselOutput {
    /// The morsel has finished.
    Done,
    /// A CPU-only checkpoint; the driver requeues the morsel.
    Continue,
    /// The morsel discovered a batch of IO it needs.
    NeedsIO(IoBatch),
    /// One non-empty array of selected rows, in input-row order.
    Batch(ArrayRef),
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

/// A morsel produces arrays for an authorised piece of work.
///
/// `state()` is called before every `compute()`, and `compute()` only when `state()` is
/// `NeedsCompute`. Batches preserve selected input-row order within and across calls.
pub trait Morsel: IoConsumer {
    /// Reports what the morsel needs next without doing work.
    fn state(&self) -> State;

    /// Advances CPU work. Only called when [`state`](Self::state) is `NeedsCompute`.
    fn compute(&mut self) -> VortexResult<MorselOutput>;
}
