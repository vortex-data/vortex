// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_error::VortexResult;

use crate::ArrayRef;
use crate::arrays::ListViewArray;
use crate::scalar::Scalar;

/// The execution interface for all aggregation.
///
/// An accumulator processes one group at a time: the caller feeds element batches via
/// [`accumulate`](Accumulator::accumulate), then calls [`flush`](Accumulator::flush) to finalize
/// the group and begin the next. The accumulator owns an output buffer and returns all results
/// via [`finish`](Accumulator::finish).
pub trait Accumulator: Send + Sync {
    /// Feed a batch of elements for the currently open group.
    ///
    /// May be called multiple times per group (e.g., chunked elements).
    fn accumulate(&mut self, batch: &ArrayRef) -> VortexResult<()>;

    /// Accumulate all groups defined by a [`ListViewArray`] in one call.
    ///
    /// Default: for each group, accumulate its elements then flush.
    /// Override for vectorized fast paths (e.g., segmented sum over the flat
    /// elements + offsets without per-group slicing).
    fn accumulate_list(&mut self, list: &ListViewArray) -> VortexResult<()> {
        for i in 0..list.len() {
            self.accumulate(&list.list_elements_at(i)?)?;
            self.flush()?;
        }
        Ok(())
    }

    /// Merge pre-computed partial state into the currently open group.
    ///
    /// The scalar's dtype must match the aggregate's `state_dtype`.
    /// This is equivalent to having processed raw elements that would produce
    /// this state — used by encoding-specific optimizations.
    fn merge(&mut self, state: &Scalar) -> VortexResult<()>;

    /// Merge an array of pre-computed states, one per group, flushing each.
    ///
    /// The array's dtype must match the aggregate's `state_dtype`.
    /// Default: merge + flush for each element.
    fn merge_list(&mut self, states: &ArrayRef) -> VortexResult<()> {
        for i in 0..states.len() {
            self.merge(&states.scalar_at(i)?)?;
            self.flush()?;
        }
        Ok(())
    }

    /// Whether the currently open group's result is fully determined.
    ///
    /// When true, callers may skip further accumulate/merge calls and proceed
    /// directly to [`flush`](Accumulator::flush). Resets to false after flush.
    fn is_saturated(&self) -> bool {
        false
    }

    /// Finalize the currently open group: push its result to the output buffer
    /// and reset internal state for the next group.
    ///
    /// Flushing a group with zero accumulated elements produces the aggregate's
    /// identity value (e.g., 0 for Sum, u64::MAX for Min) or null if no identity
    /// exists.
    fn flush(&mut self) -> VortexResult<()>;

    /// Return all flushed results as a single array.
    ///
    /// Length equals the number of [`flush`](Accumulator::flush) calls made over the
    /// accumulator's lifetime.
    fn finish(self: Box<Self>) -> VortexResult<ArrayRef>;
}
