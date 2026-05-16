//! Shared helpers for the three aggregate operators
//! ([`Aggregate`], [`PartialAggregate`], [`MergeAggregate`]).
//!
//! Specifically:
//!
//! - [`is_lane_safe`] gates whether the planner is allowed to split
//!   an aggregate into the partial-then-merge shape.
//! - [`merge_partials`] folds a `Vec<Scalar>` of per-shard partial
//!   states into a single finalised scalar — used by
//!   [`MergeAggregate`] on seal.
//! - [`scalar_to_array`] wraps a scalar in a 1-row constant array,
//!   the format every aggregate operator emits its result in.
//!
//! [`Aggregate`]: super::Aggregate
//! [`PartialAggregate`]: super::PartialAggregate
//! [`MergeAggregate`]: super::MergeAggregate

use vortex_array::IntoArray;
use vortex_array::aggregate_fn::AggregateFnRef;
use vortex_array::aggregate_fn::AggregateFnVTable;
use vortex_array::aggregate_fn::combined::Combined;
use vortex_array::aggregate_fn::fns::count::Count;
use vortex_array::aggregate_fn::fns::mean::Mean;
use vortex_array::aggregate_fn::fns::min_max::MinMax;
use vortex_array::aggregate_fn::fns::nan_count::NanCount;
use vortex_array::aggregate_fn::fns::sum::Sum;
use vortex_array::dtype::DType;
use vortex_array::scalar::Scalar;

use crate::EngineError;
use crate::EngineResult;

/// Returns true if `agg` is one of the aggregate functions whose
/// per-lane partial state can be merged commutatively/associatively
/// into a final result.
///
/// Hard-coded by downcasting the [`AggregateFnRef`] to known vtable
/// types. Stopgap until Vortex exposes a `lane_safe()` property on
/// `AggregateFn` directly.
pub fn is_lane_safe(agg: &AggregateFnRef) -> bool {
    agg.is::<Sum>()
        || agg.is::<Count>()
        || agg.is::<MinMax>()
        || agg.is::<Combined<Mean>>()
        || agg.is::<NanCount>()
}

/// Merge per-shard partial scalars into the final aggregate result.
///
/// Used by [`super::MergeAggregate`] on seal. Hard-coded per known
/// vtable type; the supported set must mirror [`is_lane_safe`]. Each
/// branch grabs the typed vtable + options via
/// [`AggregateFnRef::vtable_ref`] / [`AggregateFnRef::as_opt`],
/// builds an empty partial via
/// [`AggregateFnVTable::empty_partial`], folds every per-lane scalar
/// in via [`AggregateFnVTable::combine_partials`], and finalises
/// with [`AggregateFnVTable::finalize_scalar`].
pub(super) fn merge_partials(
    agg: &AggregateFnRef,
    accumulator_dtype: &DType,
    partials: Vec<Scalar>,
) -> EngineResult<Scalar> {
    fn merge_via<V: AggregateFnVTable>(
        vtable: &V,
        options: &V::Options,
        accumulator_dtype: &DType,
        partials: Vec<Scalar>,
    ) -> EngineResult<Scalar> {
        let mut partial = vtable
            .empty_partial(options, accumulator_dtype)
            .map_err(|e| EngineError::message(format!("merge empty_partial: {e}")))?;
        for s in partials {
            vtable
                .combine_partials(&mut partial, s)
                .map_err(|e| EngineError::message(format!("merge combine_partials: {e}")))?;
        }
        vtable
            .finalize_scalar(&partial)
            .map_err(|e| EngineError::message(format!("merge finalize_scalar: {e}")))
    }

    if let (Some(v), Some(opts)) = (agg.vtable_ref::<Sum>(), agg.as_opt::<Sum>()) {
        merge_via(v, opts, accumulator_dtype, partials)
    } else if let (Some(v), Some(opts)) = (agg.vtable_ref::<Count>(), agg.as_opt::<Count>()) {
        merge_via(v, opts, accumulator_dtype, partials)
    } else if let (Some(v), Some(opts)) = (agg.vtable_ref::<MinMax>(), agg.as_opt::<MinMax>()) {
        merge_via(v, opts, accumulator_dtype, partials)
    } else if let (Some(v), Some(opts)) = (
        agg.vtable_ref::<Combined<Mean>>(),
        agg.as_opt::<Combined<Mean>>(),
    ) {
        merge_via(v, opts, accumulator_dtype, partials)
    } else if let (Some(v), Some(opts)) = (agg.vtable_ref::<NanCount>(), agg.as_opt::<NanCount>())
    {
        merge_via(v, opts, accumulator_dtype, partials)
    } else {
        Err(EngineError::message(format!(
            "aggregate {agg} is not lane-safe; no merge strategy",
        )))
    }
}

/// Wrap a scalar in a single-row constant array.
pub(super) fn scalar_to_array(scalar: &Scalar) -> vortex_array::ArrayRef {
    use vortex_array::arrays::ConstantArray;
    ConstantArray::new(scalar.clone(), 1).into_array()
}

#[cfg(test)]
mod tests {
    use super::*;
    use vortex_array::aggregate_fn::AggregateFnVTableExt;
    use vortex_array::aggregate_fn::EmptyOptions;
    use vortex_array::aggregate_fn::combined::PairOptions;
    use vortex_array::aggregate_fn::fns::first::First;
    use vortex_array::aggregate_fn::fns::is_constant::IsConstant;
    use vortex_array::aggregate_fn::fns::last::Last;

    #[test]
    fn lane_safety_matches_known_vtables() {
        assert!(is_lane_safe(&Sum.bind(EmptyOptions)));
        assert!(is_lane_safe(&Count.bind(EmptyOptions)));
        assert!(is_lane_safe(&MinMax.bind(EmptyOptions)));
        assert!(is_lane_safe(&NanCount.bind(EmptyOptions)));
        assert!(is_lane_safe(
            &Mean::combined().bind(PairOptions(EmptyOptions, EmptyOptions))
        ));

        assert!(!is_lane_safe(&First.bind(EmptyOptions)));
        assert!(!is_lane_safe(&Last.bind(EmptyOptions)));
        assert!(!is_lane_safe(&IsConstant.bind(EmptyOptions)));
    }
}
