// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Cache finalized aggregate results for one immutable input.
//!
//! [`Aggregations`] stores results without owning its input. [`AggregationsRef`] binds the store
//! to the array that owns it. Streaming states remain owned by accumulators; a finalized result
//! can be reused as a partial only when its aggregate explicitly supports that conversion.

use std::sync::Arc;

use parking_lot::RwLock;
use vortex_error::VortexError;
use vortex_error::VortexResult;
use vortex_error::vortex_ensure;
use vortex_error::vortex_err;

use crate::ArrayRef;
use crate::ExecutionCtx;
use crate::aggregate_fn::AggregateFnRef;
use crate::aggregate_fn::fns::is_constant::IsConstant;
use crate::aggregate_fn::fns::is_sorted::IsSorted;
use crate::aggregate_fn::fns::max::Max;
use crate::aggregate_fn::fns::min::Min;
use crate::dtype::DType;
use crate::dtype::Nullability;
use crate::expr::stats::Precision;
use crate::scalar::Scalar;
use crate::stats::AggregateResults;

type CachedResults = Vec<(AggregateFnRef, Precision<Scalar>)>;

/// Shared finalized results for one immutable input.
///
/// Keys include the function and all of its options. Cloning this handle shares its cache; only
/// owners of the same input may share a handle. A new physical representation receives a fresh
/// store and inherits only representation-invariant results.
#[derive(Clone, Debug, Default)]
pub struct Aggregations {
    entries: Arc<RwLock<CachedResults>>,
}

/// Borrowed access to an input and its finalized aggregate cache.
///
/// Missing entries are unknown. Exact nulls are known results, including an overflowing sum.
/// Inexact entries are bounds defined by the aggregate and cannot answer an exact computation.
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
        AggregateResults::from_validated(self.entries.read().clone())
    }

    pub(crate) fn inherit_results(&self, results: &AggregateResults) {
        let inherited = results
            .iter()
            .filter(|(aggregate, _)| aggregate.is_representation_invariant())
            .map(|(aggregate, result)| (aggregate.clone(), result.clone()))
            .collect::<Vec<_>>();
        let mut entries = self.entries.write();
        for (aggregate, result) in inherited {
            insert(&mut entries, aggregate, result);
        }
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

    /// Read a cached result as a scalar-compatible Rust type, preserving its precision.
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
    /// Computation runs without holding a cache lock. Concurrent requests may compute the same
    /// result. A failed target result is not retained, although successful nested computations can
    /// retain their own results.
    pub fn compute_result(
        &self,
        aggregate: &AggregateFnRef,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<Scalar> {
        if let Precision::Exact(result) = self.get_result(aggregate) {
            return Ok(result);
        }

        let mut accumulator = aggregate.accumulator(self.array.dtype())?;
        accumulator.accumulate_uncached(self.array, ctx)?;
        let result = accumulator.finish()?;
        self.insert_result(aggregate.clone(), Precision::Exact(result.clone()))?;

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
        self.aggregations.entries.write().clear();
    }

    /// Snapshot finalized results without retaining a lock or an input reference.
    pub fn snapshot_results(&self) -> AggregateResults {
        self.aggregations.snapshot_results()
    }

    /// Seed a producer-known result without executing the input.
    ///
    /// # Safety
    /// The caller must prove that `result` describes this exact immutable input for `aggregate`,
    /// including its options. An inexact value must obey the aggregate's bound direction. False
    /// facts can cause incorrect results. Scalar dtype validation does not establish these facts.
    /// Seeded results remain generic metadata and do not establish the independent proof required
    /// by unchecked constructors.
    #[doc(hidden)]
    pub unsafe fn seed_result(
        &self,
        aggregate: AggregateFnRef,
        result: Precision<Scalar>,
    ) -> VortexResult<()> {
        self.insert_result(aggregate, result)
    }

    pub(crate) fn insert_result(
        &self,
        aggregate: AggregateFnRef,
        result: Precision<Scalar>,
    ) -> VortexResult<()> {
        if let Some(result) = result.as_ref().into_inner() {
            let dtype = aggregate
                .return_dtype(self.array.dtype())
                .or_else(|| {
                    // The helper APIs also define constantness and sortedness for empty, null, and
                    // structurally unsupported inputs. Those known booleans do not form partials.
                    (aggregate.is::<IsConstant>() || aggregate.is::<IsSorted>())
                        .then_some(DType::Bool(Nullability::NonNullable))
                })
                .ok_or_else(|| {
                    vortex_err!(
                        "Aggregate {aggregate} does not support {}",
                        self.array.dtype()
                    )
                })?;
            vortex_ensure!(
                result.dtype() == &dtype,
                "Aggregate {aggregate} requires result dtype {dtype}, got {}",
                result.dtype()
            );
        }

        insert(&mut self.aggregations.entries.write(), aggregate, result);
        Ok(())
    }

    pub(crate) fn inherit_results(&self, results: &AggregateResults) {
        self.aggregations.inherit_results(results);
    }

    pub(crate) fn inherit_from(&self, source: AggregationsRef<'_>) -> VortexResult<()> {
        vortex_ensure!(
            self.array
                .dtype()
                .eq_ignore_nullability(source.array.dtype())
                && self.array.len() == source.array.len(),
            "Aggregate inheritance requires matching logical input dtype and length"
        );
        if !Arc::ptr_eq(&self.aggregations.entries, &source.aggregations.entries) {
            // Release the source lock before locking the destination. Opposite concurrent
            // inherit operations must not acquire both stores in opposite orders.
            // An aggregate may depend on input nullability even when its result dtype does not.
            // A dtype-changing reduction can execute, but it cannot inherit generic results.
            if self.array.dtype() == source.array.dtype() {
                self.aggregations
                    .inherit_results(&source.snapshot_results());
            }
        }
        Ok(())
    }

    pub(crate) fn load_historical_results(&self, results: AggregateResults) {
        // The compatibility codec validates historical field types independently of the current
        // kernels. This path preserves known metadata for inputs those kernels cannot compute.
        let mut entries = self.aggregations.entries.write();
        for (aggregate, result) in results.iter() {
            insert(&mut entries, aggregate.clone(), result.clone());
        }
    }
}

fn insert(entries: &mut CachedResults, aggregate: AggregateFnRef, result: Precision<Scalar>) {
    if result.is_absent() {
        return;
    }
    let Some((_, existing)) = entries.iter_mut().find(|(key, _)| key == &aggregate) else {
        entries.push((aggregate, result));
        return;
    };
    if existing.is_exact() {
        return;
    }
    if let (Precision::Inexact(current), Precision::Inexact(incoming)) = (&*existing, &result) {
        let keep_existing = if aggregate.is::<Min>() {
            current
                .partial_cmp(incoming)
                .is_some_and(|order| order.is_ge())
        } else if aggregate.is::<Max>() {
            current
                .partial_cmp(incoming)
                .is_some_and(|order| order.is_le())
        } else {
            true
        };
        if keep_existing {
            return;
        }
    }
    *existing = result;
}

#[cfg(test)]
mod tests;
