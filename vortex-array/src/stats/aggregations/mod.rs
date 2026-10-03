// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Finalized aggregate results cached on one immutable array.
//!
//! The bound view supplies the input dtype for computation. The private store retains finalized
//! scalars and functions, rather than partial states or an input handle. The fixed statistics
//! facade projects this same store during migration.

use parking_lot::RwLock;
use vortex_error::VortexError;
use vortex_error::VortexResult;

use crate::ArrayRef;
use crate::ExecutionCtx;
use crate::aggregate_fn::AggregateFnRef;
use crate::aggregate_fn::DynAccumulator;
use crate::aggregate_fn::fns::min_max::MinMax;
use crate::aggregate_fn::fns::min_max::MinMaxResult;
use crate::aggregate_fn::fns::min_max::cache_min_max;
use crate::dtype::DType;
use crate::expr::stats::Precision;
use crate::scalar::Scalar;
use crate::stats::AggregateResults;

#[derive(Debug, Default)]
pub(crate) struct Aggregations {
    entries: RwLock<Vec<(AggregateFnRef, Precision<Scalar>)>>,
}

/// Finalized aggregate results bound to the array they describe.
///
/// A cloned array handle shares its cache. A new representation has its own cache. Functions and
/// scalars remain retained until the cache is cleared or the array is released. Custom vtables
/// should avoid retaining the owning array, which would form a reference cycle.
#[derive(Clone, Copy)]
pub struct AggregationsRef<'a> {
    array: &'a ArrayRef,
    aggregations: &'a Aggregations,
}

impl Aggregations {
    pub(crate) fn to_ref<'a>(&'a self, array: &'a ArrayRef) -> AggregationsRef<'a> {
        AggregationsRef {
            array,
            aggregations: self,
        }
    }

    pub(crate) fn snapshot_results(&self) -> AggregateResults {
        AggregateResults::from_validated(self.entries.read().to_vec())
    }
}

impl AggregationsRef<'_> {
    /// Look up a finalized result without executing the input.
    pub fn get_result(&self, aggregate: &AggregateFnRef) -> Precision<Scalar> {
        self.aggregations
            .entries
            .read()
            .iter()
            .find(|(key, _)| key == aggregate)
            .map(|(_, result)| result.clone())
            .unwrap_or_default()
    }

    /// Read a result as a scalar-compatible Rust type, preserving its precision.
    pub fn get_result_as<T>(&self, aggregate: &AggregateFnRef) -> VortexResult<Precision<T>>
    where
        T: for<'a> TryFrom<&'a Scalar, Error = VortexError>,
    {
        self.get_result(aggregate)
            .map(|result| T::try_from(&result))
            .transpose()
    }

    /// Return the exact result, computing and caching it when needed.
    ///
    /// Computation holds no cache lock. Concurrent requests can compute the same result. Failed
    /// target computations are not retained, although successful nested requests can be retained.
    pub fn compute_result(
        &self,
        aggregate: &AggregateFnRef,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<Scalar> {
        if let Precision::Exact(result) = self.get_result(aggregate) {
            return Ok(result);
        }

        let mut accumulator = aggregate.accumulator(self.array.dtype())?;
        accumulator.accumulate(self.array, ctx)?;
        let result = accumulator.finish()?;
        // The accumulator validates its finalized dtype before returning it.
        self.insert_result(aggregate.clone(), Precision::Exact(result.clone()));
        Ok(result)
    }

    /// Compute this array into a reusable accumulator and cache its exact non-null result.
    ///
    /// The accumulator must be a core [`crate::aggregate_fn::Accumulator`] for the requested
    /// function, options, and input dtype. Its previous state is reset internally. The returned
    /// final scalar does not drain the new partial state, so the caller can merge that state.
    /// Cached finals are reused only when the aggregate can recover a complete partial from them.
    ///
    /// Null finals are returned with their partial state retained, but are not newly cached. This
    /// preserves the file writer's chunk hint presence. [`Self::compute_result`] also caches nulls.
    /// [`MinMax`] computation additionally caches compatible non-null Min and Max results.
    /// Failed target computations are not published, although successful nested requests can be.
    /// No cache lock is held during computation.
    ///
    /// Returns an error if the accumulator has a different vtable, function, options, or resolved
    /// dtype, or if computation fails. A mismatch leaves the accumulator state unchanged.
    pub fn compute_into(
        &self,
        aggregate: &AggregateFnRef,
        accumulator: &mut dyn DynAccumulator,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<Scalar> {
        let result = aggregate.compute_into(self.array, accumulator, ctx)?;
        if !result.is_null() {
            if let Some(options) = aggregate.as_opt::<MinMax>() {
                let extrema = MinMaxResult::from_scalar(result.clone())?;
                cache_min_max(self.array, *options, extrema.as_ref())?;
            }
            self.insert_result(aggregate.clone(), Precision::Exact(result.clone()));
        }
        Ok(result)
    }

    /// Compute the exact result and convert it to a scalar-compatible Rust type.
    pub fn compute_as<T>(
        &self,
        aggregate: &AggregateFnRef,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<T>
    where
        T: for<'a> TryFrom<&'a Scalar, Error = VortexError>,
    {
        T::try_from(&self.compute_result(aggregate, ctx)?)
    }

    /// Remove all cached results without changing the input.
    pub fn clear(&self) {
        // Custom function destructors can reenter the cache, so release the lock before dropping.
        let entries = std::mem::take(&mut *self.aggregations.entries.write());
        drop(entries);
    }

    /// Snapshot results without retaining a lock or an input reference.
    pub fn snapshot_results(&self) -> AggregateResults {
        self.aggregations.snapshot_results()
    }

    /// Publish a finalized result whose dtype and meaning the caller has established.
    pub(crate) fn insert_result(&self, aggregate: AggregateFnRef, result: Precision<Scalar>) {
        if result.is_absent() {
            return;
        }
        let mut entries = self.aggregations.entries.write();
        if let Some((_, existing)) = entries.iter_mut().find(|(key, _)| key == &aggregate) {
            if !existing.is_exact() && result.is_exact() {
                *existing = result;
            }
        } else {
            entries.push((aggregate, result));
        }
    }

    /// Replace a legacy producer fact at the temporary fixed facade boundary.
    pub(crate) fn set_result(&self, aggregate: AggregateFnRef, result: Precision<Scalar>) {
        if result.is_absent() {
            self.clear_result(&aggregate);
            return;
        }
        let mut entries = self.aggregations.entries.write();
        if let Some((_, existing)) = entries.iter_mut().find(|(key, _)| key == &aggregate) {
            *existing = result;
        } else {
            entries.push((aggregate, result));
        }
    }

    pub(crate) fn clear_result(&self, aggregate: &AggregateFnRef) {
        let removed = {
            let mut entries = self.aggregations.entries.write();
            entries
                .iter()
                .position(|(key, _)| key == aggregate)
                .map(|index| entries.remove(index))
        };
        drop(removed);
    }

    /// Transfer detached results after consuming the array they describe.
    ///
    /// The caller must associate the snapshot with its source dtype and length, and preserve logical
    /// values, validity, and order. The guards and portability filter match [`Self::inherit_from`].
    pub(crate) fn inherit_from_snapshot(
        &self,
        source: &AggregateResults,
        source_dtype: &DType,
        source_len: usize,
    ) {
        if self.array.dtype() != source_dtype || self.array.len() != source_len {
            return;
        }

        for (aggregate, result) in source.iter() {
            if aggregate.is_representation_invariant() {
                self.insert_result(aggregate.clone(), result.clone());
            }
        }
    }

    /// Reuse finalized results after changing only the input representation.
    ///
    /// The caller must preserve the source's logical values, validity, order, and dtype. Matching
    /// dtype and length alone does not establish this relationship. Transferring results between
    /// different logical inputs can produce incorrect answers. Slices, filters, and gathers need
    /// their own propagation rules.
    ///
    /// Only results whose bound aggregate opts into representation reuse are transferred. Function
    /// options and precision are preserved, and existing exact destination results take precedence.
    /// A different dtype or length leaves the destination unchanged. Partial states are not retained
    /// or transferred.
    pub fn inherit_from(&self, source: AggregationsRef<'_>) {
        if std::ptr::eq(self.aggregations, source.aggregations)
            || self.array.dtype() != source.array.dtype()
            || self.array.len() != source.array.len()
        {
            return;
        }
        // Release the source lock before publishing to the destination.
        let results = source.aggregations.entries.read().clone();
        for (aggregate, result) in results {
            if aggregate.is_representation_invariant() {
                self.insert_result(aggregate, result);
            }
        }
    }
}

#[cfg(test)]
mod tests;
