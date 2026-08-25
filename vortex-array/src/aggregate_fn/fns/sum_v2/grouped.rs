// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_buffer::BitBuffer;
use vortex_buffer::BitBufferMut;
use vortex_error::VortexResult;
use vortex_mask::Mask;

use super::SumV2;
use super::sum_v2_partial_fields;
use crate::ArrayRef;
use crate::ExecutionCtx;
use crate::IntoArray;
use crate::aggregate_fn::AggregateFnRef;
use crate::aggregate_fn::GroupRanges;
use crate::aggregate_fn::GroupedArray;
use crate::aggregate_fn::fns::sum::sum_float_all;
use crate::aggregate_fn::fns::sum::sum_signed_all;
use crate::aggregate_fn::fns::sum::sum_unsigned_all;
use crate::aggregate_fn::kernels::DynGroupedAggregateKernel;
use crate::arrays::BoolArray;
use crate::arrays::Primitive;
use crate::arrays::PrimitiveArray;
use crate::arrays::StructArray;
use crate::dtype::NativePType;
use crate::dtype::Nullability;
use crate::match_each_native_ptype;
use crate::validity::Validity;

/// Encoding-specific grouped [`SumV2`] kernel for primitive element arrays.
#[derive(Debug)]
pub(crate) struct PrimitiveGroupedSumV2EncodingKernel;

impl DynGroupedAggregateKernel for PrimitiveGroupedSumV2EncodingKernel {
    fn grouped_aggregate(
        &self,
        aggregate_fn: &AggregateFnRef,
        groups: &GroupedArray,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<Option<ArrayRef>> {
        let Some(options) = aggregate_fn.as_opt::<SumV2>() else {
            return Ok(None);
        };
        try_grouped_sum(groups, ctx, options.skip_nans)
    }
}

/// Grouped [`SumV2`] implementation for canonical primitive elements.
///
/// Reuses the scalar primitive-sum reductions ([`sum_unsigned_all`]/[`sum_signed_all`]/
/// [`sum_float_all`]) so the per-group semantics match scalar `sum_v2` exactly (overflow saturates
/// to a null sum, NaNs are skipped). The element validity mask is materialized once and sliced
/// per group, rather than the per-group accumulator setup of the generic fallback path.
pub(super) fn try_grouped_sum(
    groups: &GroupedArray,
    ctx: &mut ExecutionCtx,
    skip_nans: bool,
) -> VortexResult<Option<ArrayRef>> {
    if !groups.elements().is::<Primitive>() {
        return Ok(None);
    }
    let elements = groups.elements().clone().downcast::<Primitive>();
    let group_ranges = groups.group_ranges(ctx)?;
    let group_validity = groups.group_validity(ctx)?;

    Ok(Some(grouped_sum(
        &elements,
        &group_ranges,
        &group_validity,
        ctx,
        skip_nans,
    )?))
}

/// Sum each group described by `group_ranges` (element `(offset, size)` pairs), one sum per group.
fn grouped_sum(
    elements: &PrimitiveArray,
    group_ranges: &GroupRanges,
    group_validity: &Mask,
    ctx: &mut ExecutionCtx,
    skip_nans: bool,
) -> VortexResult<ArrayRef> {
    let elem_mask = elements
        .as_ref()
        .validity()?
        .execute_mask(elements.as_ref().len(), ctx)?;
    let all_valid = elem_mask.all_true();

    let (sums, is_overflow, is_empty) = match_each_native_ptype!(elements.ptype(),
        unsigned: |T| {
            let values = elements.as_slice::<T>();
            collect_sums::<T, u64>(
                values, group_ranges, group_validity, &elem_mask, all_valid, sum_unsigned_all)
        },
        signed: |T| {
            let values = elements.as_slice::<T>();
            collect_sums::<T, i64>(
                values, group_ranges, group_validity, &elem_mask, all_valid, sum_signed_all)
        },
        floating: |T| {
            let values = elements.as_slice::<T>();
            collect_sums::<T, f64>(
                values, group_ranges, group_validity, &elem_mask, all_valid,
                |acc, slice| { sum_float_all(acc, slice, skip_nans); false })
        }
    );

    let partial_fields = sum_v2_partial_fields(sums.dtype().clone());

    // SAFETY: all three children have one value per group and match `partial_fields`; the struct
    // validity is derived from the same group count.
    Ok(unsafe {
        StructArray::new_unchecked(
            vec![
                sums.into_array(),
                BoolArray::new(is_overflow, Validity::NonNullable).into_array(),
                BoolArray::new(is_empty, Validity::NonNullable).into_array(),
            ],
            partial_fields,
            group_validity.len(),
            Validity::from_mask(group_validity.clone(), Nullability::Nullable),
        )
    }
    .into_array())
}

/// Reduce each group's element slice into a non-null sum, overflow bitmap, and empty bitmap.
fn collect_sums<T: NativePType, A: NativePType + Default>(
    values: &[T],
    group_ranges: &GroupRanges,
    group_validity: &Mask,
    elem_mask: &Mask,
    all_valid: bool,
    sum_run: impl Fn(&mut A, &[T]) -> bool,
) -> (PrimitiveArray, BitBuffer, BitBuffer) {
    let group_count = group_ranges.len();
    let mut is_overflow = BitBufferMut::new_unset(group_count);
    let mut is_empty = BitBufferMut::new_unset(group_count);
    let sums = group_ranges.iter().enumerate().map(|(i, (offset, size))| {
        if !group_validity.value(i) {
            return A::default();
        }
        let mut acc = A::default();
        let (overflow, any_valid) = if all_valid {
            (sum_run(&mut acc, &values[offset..offset + size]), size > 0)
        } else {
            sum_masked_group(&mut acc, values, offset, size, elem_mask, &sum_run)
        };
        if overflow {
            // SAFETY: `i` comes from enumerating `group_ranges`, and the bitmap has one bit per
            // group.
            unsafe { is_overflow.set_unchecked(i) };
        }
        if !any_valid {
            // SAFETY: `i` comes from enumerating `group_ranges`, and the bitmap has one bit per
            // group.
            unsafe { is_empty.set_unchecked(i) };
        }
        acc
    });
    let sums = PrimitiveArray::from_iter(sums);
    (sums, is_overflow.freeze(), is_empty.freeze())
}

/// Sum valid runs in one group, returning `(overflow, any_valid)`.
fn sum_masked_group<T: NativePType, A>(
    acc: &mut A,
    values: &[T],
    offset: usize,
    size: usize,
    elem_mask: &Mask,
    sum_run: &impl Fn(&mut A, &[T]) -> bool,
) -> (bool, bool) {
    match elem_mask {
        Mask::AllTrue(_) => (sum_run(acc, &values[offset..offset + size]), size > 0),
        Mask::AllFalse(_) => (false, false),
        Mask::Values(mask_values) => {
            let validity = mask_values
                .bit_buffer()
                .as_view()
                .slice(offset..offset + size);
            let mut any_valid = false;
            for (start, end) in validity.set_slices() {
                any_valid = true;
                if sum_run(acc, &values[offset + start..offset + end]) {
                    return (true, true);
                }
            }
            (false, any_valid)
        }
    }
}
