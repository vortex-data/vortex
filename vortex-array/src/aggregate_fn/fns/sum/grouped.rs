// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_error::VortexResult;
use vortex_mask::AllOr;
use vortex_mask::Mask;

use super::Sum;
use super::primitive::sum_float_all;
use super::primitive::sum_signed_all;
use super::primitive::sum_unsigned_all;
use crate::ArrayRef;
use crate::ExecutionCtx;
use crate::IntoArray;
use crate::aggregate_fn::AggregateFnRef;
use crate::aggregate_fn::GroupRanges;
use crate::aggregate_fn::GroupedArray;
use crate::aggregate_fn::kernels::DynGroupedAggregateKernel;
use crate::arrays::Primitive;
use crate::arrays::PrimitiveArray;
use crate::dtype::NativePType;
use crate::match_each_native_ptype;

/// Encoding-specific grouped [`Sum`] kernel for primitive element arrays.
#[derive(Debug)]
pub(crate) struct PrimitiveGroupedSumEncodingKernel;

impl DynGroupedAggregateKernel for PrimitiveGroupedSumEncodingKernel {
    fn grouped_aggregate(
        &self,
        aggregate_fn: &AggregateFnRef,
        groups: &GroupedArray,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<Option<ArrayRef>> {
        let Some(options) = aggregate_fn.as_opt::<Sum>() else {
            return Ok(None);
        };
        try_grouped_sum(groups, ctx, options.skip_nans)
    }
}

/// Grouped [`Sum`] implementation for canonical primitive elements.
///
/// Reuses the scalar primitive-sum reductions ([`sum_unsigned_all`]/[`sum_signed_all`]/
/// [`sum_float_all`]) so the per-group semantics match scalar `sum` exactly (overflow saturates to
/// a null sum, NaNs are skipped). The element validity mask is materialized once and sliced per
/// group, rather than the per-group accumulator setup of the generic fallback path.
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
    let all_valid = matches!(elem_mask.slices(), AllOr::All);

    let result = match_each_native_ptype!(elements.ptype(),
        unsigned: |T| {
            let values = elements.as_slice::<T>();
            collect_sums::<T, u64>(values, group_ranges, group_validity, &elem_mask, all_valid,
                sum_unsigned_all)
        },
        signed: |T| {
            let values = elements.as_slice::<T>();
            collect_sums::<T, i64>(values, group_ranges, group_validity, &elem_mask, all_valid,
                sum_signed_all)
        },
        floating: |T| {
            let values = elements.as_slice::<T>();
            collect_sums::<T, f64>(values, group_ranges, group_validity, &elem_mask, all_valid,
                |acc, slice| { sum_float_all(acc, slice, skip_nans); false })
        }
    );

    Ok(result.into_array())
}

/// Reduce each group's element slice into a nullable sum. A group is null when the group
/// itself is invalid, or when summing it overflows (`sum_run` returns `true`).
fn collect_sums<T: NativePType, A: NativePType + Default>(
    values: &[T],
    group_ranges: &GroupRanges,
    group_validity: &Mask,
    elem_mask: &Mask,
    all_valid: bool,
    sum_run: impl Fn(&mut A, &[T]) -> bool,
) -> PrimitiveArray {
    let sums = group_ranges.iter().enumerate().map(|(i, (offset, size))| {
        if !group_validity.value(i) {
            return None;
        }
        let mut acc = A::default();
        let overflow = if all_valid {
            sum_run(&mut acc, &values[offset..offset + size])
        } else {
            sum_masked_group(&mut acc, values, offset, size, elem_mask, &sum_run)
        };
        (!overflow).then_some(acc)
    });
    PrimitiveArray::from_option_iter(sums)
}

/// Sum the valid elements of a single group, using the contiguous valid runs of the element mask
/// intersected with the group's `[offset, offset + size)` range.
fn sum_masked_group<T: NativePType, A>(
    acc: &mut A,
    values: &[T],
    offset: usize,
    size: usize,
    elem_mask: &Mask,
    sum_run: &impl Fn(&mut A, &[T]) -> bool,
) -> bool {
    match elem_mask.slice(offset..offset + size).slices() {
        AllOr::All => sum_run(acc, &values[offset..offset + size]),
        AllOr::None => false,
        AllOr::Some(runs) => {
            for &(start, end) in runs {
                if sum_run(acc, &values[offset + start..offset + end]) {
                    return true;
                }
            }
            false
        }
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::cast_possible_truncation)]

    use vortex_buffer::buffer;
    use vortex_error::VortexResult;

    use crate::ArrayRef;
    use crate::IntoArray;
    use crate::VortexSessionExecute;
    use crate::aggregate_fn::DynGroupedAccumulator;
    use crate::aggregate_fn::GroupedAccumulator;
    use crate::aggregate_fn::NumericalAggregateOpts;
    use crate::aggregate_fn::fns::sum::Sum;
    use crate::aggregate_fn::fns::sum::sum;
    use crate::array_session;
    use crate::arrays::FixedSizeListArray;
    use crate::arrays::ListViewArray;
    use crate::arrays::PrimitiveArray;
    use crate::assert_arrays_eq;
    use crate::builders::builder_with_capacity;
    use crate::dtype::DType;
    use crate::dtype::Nullability::NonNullable;
    use crate::dtype::Nullability::Nullable;
    use crate::dtype::PType;
    use crate::validity::Validity;

    /// Run a grouped sum through the accumulator.
    fn grouped_sum_actual(groups: &ArrayRef, elem_dtype: &DType) -> VortexResult<ArrayRef> {
        let mut acc = GroupedAccumulator::try_new(
            Sum,
            NumericalAggregateOpts::default(),
            elem_dtype.clone(),
        )?;
        acc.accumulate_list(groups, &mut array_session().create_execution_ctx())?;
        acc.finish()
    }

    /// Reference sums computed exactly like the generic slow path: per-group scalar [`sum`] for
    /// valid groups, a null sum for invalid groups.
    fn grouped_sum_reference(
        elements: &ArrayRef,
        ranges: &[(usize, usize)],
        group_valid: &[bool],
        elem_dtype: &DType,
    ) -> VortexResult<ArrayRef> {
        use crate::aggregate_fn::AggregateFnVTable;

        let mut ctx = array_session().create_execution_ctx();
        let sum_dtype = Sum
            .partial_dtype(&NumericalAggregateOpts::default(), elem_dtype)
            .expect("sum partial dtype");
        let mut builder = builder_with_capacity(&sum_dtype, ranges.len());
        for (i, &(offset, size)) in ranges.iter().enumerate() {
            if group_valid[i] {
                let slice = elements.slice(offset..offset + size)?;
                builder.append_scalar(&sum(&slice, &mut ctx)?)?;
            } else {
                builder.append_null();
            }
        }
        Ok(builder.finish())
    }

    fn offsets_sizes(ranges: &[(usize, usize)]) -> (ArrayRef, ArrayRef) {
        let offsets = PrimitiveArray::from_iter(ranges.iter().map(|&(o, _)| o as i32));
        let sizes = PrimitiveArray::from_iter(ranges.iter().map(|&(_, s)| s as i32));
        (offsets.into_array(), sizes.into_array())
    }

    fn listview(
        elements: ArrayRef,
        ranges: &[(usize, usize)],
        group_valid: &[bool],
    ) -> VortexResult<ArrayRef> {
        let (offsets, sizes) = offsets_sizes(ranges);
        let validity = if group_valid.iter().all(|&v| v) {
            Validity::NonNullable
        } else {
            Validity::from_iter(group_valid.iter().copied())
        };
        Ok(ListViewArray::try_new(elements, offsets, sizes, validity)?.into_array())
    }

    #[test]
    fn listview_matches_reference_unsigned() -> VortexResult<()> {
        let mut ctx = array_session().create_execution_ctx();
        let elements =
            PrimitiveArray::new(buffer![1u32, 2, 3, 4, 5, 6], Validity::NonNullable).into_array();
        let elem_dtype = DType::Primitive(PType::U32, NonNullable);
        let ranges = [(0, 2), (2, 1), (3, 3)];
        let valid = [true, true, true];

        let groups = listview(elements.clone(), &ranges, &valid)?;
        let actual = grouped_sum_actual(&groups, &elem_dtype)?;
        let expected = grouped_sum_reference(&elements, &ranges, &valid, &elem_dtype)?;

        // Unsigned input sums to U64.
        let direct = PrimitiveArray::from_option_iter([Some(3u64), Some(3u64), Some(15u64)]);
        assert_arrays_eq!(&actual, &direct.into_array(), &mut ctx);
        assert_arrays_eq!(&actual, &expected, &mut ctx);
        Ok(())
    }

    #[test]
    fn listview_out_of_order_offsets_with_null_group() -> VortexResult<()> {
        let mut ctx = array_session().create_execution_ctx();
        // Offsets are not in group order and a group is null: the group validity must be indexed by
        // group index, not by element offset.
        let elements =
            PrimitiveArray::new(buffer![10i32, 20, 30, 40, 50, 60], Validity::NonNullable)
                .into_array();
        let elem_dtype = DType::Primitive(PType::I32, NonNullable);
        let ranges = [(4, 2), (0, 2), (2, 2)];
        let valid = [true, false, true];

        let groups = listview(elements.clone(), &ranges, &valid)?;
        let actual = grouped_sum_actual(&groups, &elem_dtype)?;
        let expected = grouped_sum_reference(&elements, &ranges, &valid, &elem_dtype)?;

        let direct = PrimitiveArray::from_option_iter([Some(110i64), None, Some(70i64)]);
        assert_arrays_eq!(&actual, &direct.into_array(), &mut ctx);
        assert_arrays_eq!(&actual, &expected, &mut ctx);
        Ok(())
    }

    #[test]
    fn listview_interior_and_full_nulls() -> VortexResult<()> {
        let mut ctx = array_session().create_execution_ctx();
        // Group 1 has an interior null, group 2 is entirely null, group 3 is empty.
        let elements =
            PrimitiveArray::from_option_iter([Some(1i32), None, Some(3), None, None, Some(9)])
                .into_array();
        let elem_dtype = DType::Primitive(PType::I32, Nullable);
        let ranges = [(0, 3), (3, 2), (5, 0), (5, 1)];
        let valid = [true, true, true, true];

        let groups = listview(elements.clone(), &ranges, &valid)?;
        let actual = grouped_sum_actual(&groups, &elem_dtype)?;
        let expected = grouped_sum_reference(&elements, &ranges, &valid, &elem_dtype)?;

        let direct =
            PrimitiveArray::from_option_iter([Some(4i64), Some(0i64), Some(0i64), Some(9i64)]);
        assert_arrays_eq!(&actual, &direct.into_array(), &mut ctx);
        assert_arrays_eq!(&actual, &expected, &mut ctx);
        Ok(())
    }

    #[test]
    fn listview_overflow_group_is_null() -> VortexResult<()> {
        let mut ctx = array_session().create_execution_ctx();
        let elements =
            PrimitiveArray::new(buffer![i64::MAX, 1, 2, 3], Validity::NonNullable).into_array();
        let elem_dtype = DType::Primitive(PType::I64, NonNullable);
        let ranges = [(0, 2), (2, 2)];
        let valid = [true, true];

        let groups = listview(elements.clone(), &ranges, &valid)?;
        let actual = grouped_sum_actual(&groups, &elem_dtype)?;
        let expected = grouped_sum_reference(&elements, &ranges, &valid, &elem_dtype)?;

        // First group overflows -> null sum; second group sums normally.
        let direct = PrimitiveArray::from_option_iter([None, Some(5i64)]);
        assert_arrays_eq!(&actual, &direct.into_array(), &mut ctx);
        assert_arrays_eq!(&actual, &expected, &mut ctx);
        Ok(())
    }

    #[test]
    fn listview_float_nan_and_inf() -> VortexResult<()> {
        let elements = PrimitiveArray::new(
            buffer![1.0f64, f64::NAN, 2.0, f64::INFINITY, f64::NEG_INFINITY, 4.0],
            Validity::NonNullable,
        )
        .into_array();
        let elem_dtype = DType::Primitive(PType::F64, NonNullable);
        let ranges = [(0, 3), (3, 3)];
        let valid = [true, true];

        let groups = listview(elements.clone(), &ranges, &valid)?;
        let actual = grouped_sum_actual(&groups, &elem_dtype)?;

        // Group 0: NaN skipped -> 3.0. Group 1: INF + -INF = NaN. (Avoid array equality here since
        // NaN != NaN; compare element scalars against the reference path instead.)
        let mut ctx = array_session().create_execution_ctx();
        let expected = grouped_sum_reference(&elements, &ranges, &valid, &elem_dtype)?;
        let g0 = actual.execute_scalar(0, &mut ctx)?;
        assert_eq!(g0.as_primitive().typed_value::<f64>(), Some(3.0));
        assert_eq!(
            g0.as_primitive().typed_value::<f64>(),
            expected
                .execute_scalar(0, &mut ctx)?
                .as_primitive()
                .typed_value::<f64>()
        );
        let g1 = actual.execute_scalar(1, &mut ctx)?;
        assert!(g1.as_primitive().typed_value::<f64>().unwrap().is_nan());
        assert!(
            expected
                .execute_scalar(1, &mut ctx)?
                .as_primitive()
                .typed_value::<f64>()
                .unwrap()
                .is_nan()
        );
        Ok(())
    }

    #[test]
    fn listview_float_nan_not_skipping() -> VortexResult<()> {
        let elements = PrimitiveArray::new(
            buffer![1.0f64, f64::NAN, 2.0, 3.0, 4.0],
            Validity::NonNullable,
        )
        .into_array();
        let elem_dtype = DType::Primitive(PType::F64, NonNullable);
        let groups = listview(elements, &[(0, 3), (3, 2)], &[true, true])?;

        let mut acc =
            GroupedAccumulator::try_new(Sum, NumericalAggregateOpts::include_nans(), elem_dtype)?;
        acc.accumulate_list(&groups, &mut array_session().create_execution_ctx())?;
        let actual = acc.finish()?;

        let mut ctx = array_session().create_execution_ctx();
        // Group 0 contains a NaN -> NaN sum; group 1 sums normally.
        let g0 = actual.execute_scalar(0, &mut ctx)?;
        assert!(g0.as_primitive().typed_value::<f64>().unwrap().is_nan());
        let g1 = actual.execute_scalar(1, &mut ctx)?;
        assert_eq!(g1.as_primitive().typed_value::<f64>(), Some(7.0));
        Ok(())
    }

    #[test]
    fn fixed_size_overflow_and_nan() -> VortexResult<()> {
        let mut ctx = array_session().create_execution_ctx();
        // FixedSize path: first group overflows -> null sum, second sums normally.
        let elements =
            PrimitiveArray::new(buffer![i64::MAX, 1, 2, 3], Validity::NonNullable).into_array();
        let elem_dtype = DType::Primitive(PType::I64, NonNullable);
        let groups = FixedSizeListArray::try_new(elements.clone(), 2, Validity::NonNullable, 2)?
            .into_array();

        let actual = grouped_sum_actual(&groups, &elem_dtype)?;
        let expected =
            grouped_sum_reference(&elements, &[(0, 2), (2, 2)], &[true, true], &elem_dtype)?;
        let direct = PrimitiveArray::from_option_iter([None, Some(5i64)]);
        assert_arrays_eq!(&actual, &direct.into_array(), &mut ctx);
        assert_arrays_eq!(&actual, &expected, &mut ctx);
        Ok(())
    }
}
