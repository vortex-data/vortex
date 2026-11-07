// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Whole-array sum dispatch and scalar partial construction.
//!
//! Run traversal and arithmetic are implemented in [`super::runs`].

use std::ops::Range;

use vortex_buffer::BitBufferMut;
use vortex_buffer::Buffer;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_mask::Mask;

use super::runs::add_float_run;
use super::runs::add_signed_run;
use super::runs::add_unsigned_run;
use super::runs::sum_range;
use crate::ArrayRef;
use crate::ExecutionCtx;
use crate::IntoArray;
use crate::aggregate_fn::AggregateFnRef;
use crate::aggregate_fn::GroupRanges;
use crate::aggregate_fn::GroupedArray;
use crate::aggregate_fn::NumericalAggregateOpts;
use crate::aggregate_fn::fns::sum::Sum;
use crate::aggregate_fn::fns::sum_v2::SumV2;
use crate::aggregate_fn::fns::sum_v2::grouped_partials;
use crate::aggregate_fn::kernels::DynAggregateKernel;
use crate::aggregate_fn::kernels::DynGroupedAggregateKernel;
use crate::arrays::PrimitiveArray;
use crate::arrays::runend::RunEnd;
use crate::arrays::runend::RunEndArrayExt;
use crate::arrays::runend::RunEndArraySlotsExt;
use crate::builtins::ArrayBuiltins;
use crate::dtype::DType;
use crate::dtype::NativePType;
use crate::dtype::Nullability::Nullable;
use crate::dtype::PType;
use crate::dtype::UnsignedPType;
use crate::match_each_native_ptype;
use crate::match_each_unsigned_integer_ptype;
use crate::scalar::PValue;
use crate::scalar::Scalar;

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

/// Grouped [`SumV2`] kernel for run-end elements: each group sums the runs it covers, each value
/// times the length of its overlap with the group, without decoding the runs.
#[derive(Debug)]
pub(crate) struct RunEndGroupedSumV2Kernel;

impl DynGroupedAggregateKernel for RunEndGroupedSumV2Kernel {
    fn grouped_aggregate(
        &self,
        aggregate_fn: &AggregateFnRef,
        groups: &GroupedArray,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<Option<ArrayRef>> {
        let Some(options) = aggregate_fn.as_opt::<SumV2>().copied() else {
            return Ok(None);
        };
        let Some(array) = groups.elements().as_opt::<RunEnd>() else {
            return Ok(None);
        };
        let values = match array.values().dtype() {
            DType::Primitive(..) => array.values().clone(),
            DType::Bool(nullability) => array
                .values()
                .cast(DType::Primitive(PType::U8, *nullability))?,
            _ => return Ok(None),
        };
        let values = values.execute::<PrimitiveArray>(ctx)?;
        let validity = values
            .as_ref()
            .validity()?
            .execute_mask(values.len(), ctx)?;
        let ends = array.ends().clone().execute::<PrimitiveArray>(ctx)?;
        let group_ranges = groups.group_ranges(ctx)?;
        let group_validity = groups.group_validity(ctx)?;
        let offset = array.offset();

        Ok(Some(match_each_unsigned_integer_ptype!(
            ends.ptype(),
            |E| {
                let ends = ends.as_slice::<E>();
                let groups = (offset, &group_ranges, &group_validity);
                match_each_native_ptype!(values.ptype(),
                    unsigned: |T| {
                        grouped_runs(ends, values.as_slice::<T>(), &validity, groups, add_unsigned_run::<T>)
                    },
                    signed: |T| {
                        grouped_runs(ends, values.as_slice::<T>(), &validity, groups, add_signed_run::<T>)
                    },
                    floating: |T| {
                        grouped_runs(ends, values.as_slice::<T>(), &validity, groups, |sum, value, len| {
                            add_float_run(sum, value, len, options.skip_nans)
                        })
                    }
                )
            }
        )))
    }
}

/// Sum each group's range of runs, one [`SumV2`] partial per group. `groups` is the array's offset
/// with the groups' ranges and validity.
fn grouped_runs<E: UnsignedPType, T: NativePType, A: NativePType>(
    ends: &[E],
    values: &[T],
    validity: &Mask,
    (offset, group_ranges, group_validity): (usize, &GroupRanges, &Mask),
    add_run: impl Fn(A, T, usize) -> Option<A>,
) -> ArrayRef {
    let mut is_overflow = BitBufferMut::new_unset(group_ranges.len());
    let mut is_empty = BitBufferMut::new_unset(group_ranges.len());
    let sums: Buffer<A> = group_ranges
        .iter()
        .enumerate()
        .map(|(group, (start, size))| {
            if !group_validity.value(group) {
                return A::default();
            }
            let range = offset + start..offset + start + size;
            match sum_range(ends, values, validity, range, &add_run) {
                (Some(sum), empty) => {
                    if empty {
                        is_empty.set(group);
                    }
                    sum
                }
                (None, _) => {
                    is_overflow.set(group);
                    A::default()
                }
            }
        })
        .collect();
    grouped_partials(
        sums.into_array(),
        is_overflow.freeze(),
        is_empty.freeze(),
        group_validity,
    )
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
