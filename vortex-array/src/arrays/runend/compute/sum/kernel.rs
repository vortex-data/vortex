// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Whole-array sum dispatch and scalar partial construction.
//!
//! Run traversal and arithmetic are implemented in [`super::runs`].

use std::ops::Range;

use vortex_array::ArrayRef;
use vortex_array::ExecutionCtx;
use vortex_array::aggregate_fn::AggregateFnRef;
use vortex_array::aggregate_fn::NumericalAggregateOpts;
use vortex_array::aggregate_fn::fns::sum::Sum;
use vortex_array::aggregate_fn::fns::sum_v2::SumV2;
use vortex_array::aggregate_fn::kernels::DynAggregateKernel;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::dtype::DType;
use vortex_array::dtype::NativePType;
use vortex_array::dtype::Nullability::Nullable;
use vortex_array::dtype::UnsignedPType;
use vortex_array::match_each_native_ptype;
use vortex_array::match_each_unsigned_integer_ptype;
use vortex_array::scalar::PValue;
use vortex_array::scalar::Scalar;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_mask::Mask;

use super::runs::add_float_run;
use super::runs::add_signed_run;
use super::runs::add_unsigned_run;
use super::runs::sum_range;
use crate::RunEnd;
use crate::RunEndArrayExt;
use crate::RunEndArraySlotsExt;

/// Whole-array primitive sum kernel for [`RunEnd`].
#[derive(Debug)]
pub(crate) struct RunEndSumKernel;

impl DynAggregateKernel for RunEndSumKernel {
    fn aggregate(
        &self,
        aggregate_fn: &AggregateFnRef,
        batch: &ArrayRef,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<Option<Scalar>> {
        if !batch.dtype().is_primitive() {
            return Ok(None);
        }

        let Some(options) = aggregate_fn
            .as_opt::<Sum>()
            .or_else(|| aggregate_fn.as_opt::<SumV2>())
            .copied()
        else {
            return Ok(None);
        };

        let Some(array) = batch.as_opt::<RunEnd>() else {
            return Ok(None);
        };

        if array.is_empty() {
            return Ok(Some(empty_partial(aggregate_fn, batch.dtype())?));
        }

        let validity = array
            .values()
            .validity()?
            .execute_mask(array.values().len(), ctx)?;
        if validity.all_false() {
            return Ok(Some(empty_partial(aggregate_fn, batch.dtype())?));
        }

        let ends = array.ends().clone().execute::<PrimitiveArray>(ctx)?;
        let values = array.values().clone().execute::<PrimitiveArray>(ctx)?;
        let range = array.offset()..array.offset() + batch.len();
        let (sum, is_empty) = sum_primitive_runs(&ends, &values, &validity, range, options);

        Ok(Some(partial_scalar(aggregate_fn, sum, is_empty)?))
    }
}

/// Select run-end and value types, widening sums to `u64`, `i64`, or `f64`.
///
/// Returns `(sum, is_empty)` with overflow represented by a null scalar. Input requirements and
/// empty-input semantics are defined in [`super::runs`].
fn sum_primitive_runs(
    ends: &PrimitiveArray,
    values: &PrimitiveArray,
    validity: &Mask,
    range: Range<usize>,
    options: NumericalAggregateOpts,
) -> (Scalar, bool) {
    match_each_unsigned_integer_ptype!(ends.ptype(), |E| {
        let ends = ends.as_slice::<E>();

        match_each_native_ptype!(values.ptype(),
            unsigned: |T| {
                sum_typed_runs(ends, values.as_slice::<T>(), validity, range, add_unsigned_run::<T>)
            },
            signed: |T| {
                sum_typed_runs(ends, values.as_slice::<T>(), validity, range, add_signed_run::<T>)
            },
            floating: |T| {
                sum_typed_runs(ends, values.as_slice::<T>(), validity, range, |sum, value, len| {
                    add_float_run(sum, value, len, options.skip_nans)
                })
            }
        )
    })
}

/// Convert the reduction's sum to a nullable scalar, preserving its empty-input flag.
fn sum_typed_runs<E: UnsignedPType, T: NativePType, A: NativePType + Into<PValue>>(
    ends: &[E],
    values: &[T],
    validity: &Mask,
    range: Range<usize>,
    add_run: impl Fn(A, T, usize) -> Option<A>,
) -> (Scalar, bool) {
    let (sum, is_empty) = sum_range(ends, values, validity, range, add_run);
    let sum = match sum {
        Some(sum) => Scalar::primitive(sum, Nullable),
        None => Scalar::null(DType::Primitive(A::PTYPE, Nullable)),
    };

    (sum, is_empty)
}

fn empty_partial(aggregate_fn: &AggregateFnRef, dtype: &DType) -> VortexResult<Scalar> {
    let sum_dtype = aggregate_fn
        .return_dtype(dtype)
        .vortex_expect("The primitive sum kernel accepts only supported dtypes");

    partial_scalar(aggregate_fn, Scalar::zero_value(&sum_dtype), true)
}

/// Wrap the sum in the aggregate's partial representation without finalizing empty inputs.
fn partial_scalar(
    aggregate_fn: &AggregateFnRef,
    sum: Scalar,
    is_empty: bool,
) -> VortexResult<Scalar> {
    if aggregate_fn.is::<SumV2>() {
        SumV2::partial_from_sum(sum, is_empty)
    } else {
        Ok(sum)
    }
}
