// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Explicit ownership of aggregate results and producer guarantees for one input.
//!
//! An [`ArrayInput`] retains its array and the same aggregate store used by array-owned caching.
//! Its clones share both. Subsets get fresh stores: positive integer sortedness survives stable
//! selection, while exact extrema become bounds. Facts belong to the exact array handle and do
//! not travel implicitly through execution rewrites or array serialization.

mod bounds;
use self::bounds::VerifiedIntegerBounds;

mod execution;
pub(crate) use self::execution::AggregateInputFrame;

mod guarantees;
mod owner;
mod partial;

use std::fmt::Debug;
use std::fmt::Formatter;
use std::sync::Arc;
use std::sync::OnceLock;

use crate::ArrayRef;
use crate::stats::Aggregations;

/// Selects result ownership for the input experiment.
///
/// This changes lookup and population for the exercised aggregate paths. All modes retain the
/// array's existing cache storage; disabling access does not remove that storage allocation.
/// Explicitly retained typed partial states always belong to the input owner.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum AggregateCacheMode {
    /// Retain finalized results on the underlying array.
    Array,
    /// Retain finalized results on this input owner and bypass the array cache.
    #[default]
    Input,
    /// Bypass both result stores, including producer guarantees.
    Disabled,
}

struct ArrayInputInner {
    array: ArrayRef,
    aggregations: Aggregations,
    // Generic aggregate plugins are safe to implement. Their results cannot mint the proof used
    // to bypass an unsafe decimal constructor's value-precision precondition.
    verified_bounds: OnceLock<VerifiedIntegerBounds>,
}

/// An input array and its explicitly retained aggregate results.
///
/// Construction accepts any encoding. Clones share one store. Independently wrapping the same
/// array creates a new store, and cloning out an [`ArrayRef`] does not retain this owner.
///
/// ```
/// use vortex_array::{ArrayInput, IntoArray, VortexSessionExecute};
/// use vortex_array::aggregate_fn::{AggregateFnVTableExt, NumericalAggregateOpts};
/// use vortex_array::aggregate_fn::fns::sum::Sum;
/// use vortex_array::arrays::PrimitiveArray;
/// use vortex_array::expr::stats::Precision;
///
/// let input = ArrayInput::new(PrimitiveArray::from_iter([1i32, 2, 3]).into_array());
/// let sum = Sum.bind(NumericalAggregateOpts::skip_nans());
/// assert_eq!(input.get_result(&sum), Precision::Absent);
/// let mut ctx = vortex_array::array_session().create_execution_ctx();
/// let result = input.compute_result(&sum, &mut ctx)?;
/// assert_eq!(i64::try_from(&result)?, 6);
/// assert_eq!(input.clone().get_result(&sum), Precision::Exact(result));
/// assert_eq!(input.array().aggregations().get_result(&sum), Precision::Absent);
/// # Ok::<(), vortex_error::VortexError>(())
/// ```
#[derive(Clone)]
pub struct ArrayInput {
    inner: Arc<ArrayInputInner>,
    cache_mode: AggregateCacheMode,
}

impl Debug for ArrayInput {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ArrayInput")
            .field("array", &self.inner.array)
            .field("cache_mode", &self.cache_mode)
            .finish_non_exhaustive()
    }
}

#[cfg(test)]
mod tests;
