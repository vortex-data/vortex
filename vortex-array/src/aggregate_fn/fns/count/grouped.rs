// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_buffer::Buffer;
use vortex_error::VortexResult;
use vortex_mask::Mask;

use super::Count;
use crate::ArrayRef;
use crate::ExecutionCtx;
use crate::IntoArray;
use crate::aggregate_fn::AggregateFnRef;
use crate::aggregate_fn::GroupRanges;
use crate::aggregate_fn::GroupedArray;
use crate::aggregate_fn::kernels::DynGroupedAggregateKernel;
use crate::arrays::PrimitiveArray;
use crate::validity::Validity;

/// Encoding-independent grouped [`Count`] kernel.
#[derive(Debug)]
pub(crate) struct CountGroupedKernel;

impl DynGroupedAggregateKernel for CountGroupedKernel {
    fn grouped_aggregate(
        &self,
        aggregate_fn: &AggregateFnRef,
        groups: &GroupedArray,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<Option<ArrayRef>> {
        let Some(options) = aggregate_fn.as_opt::<Count>() else {
            return Ok(None);
        };
        // NaN-skipping counts over floats must inspect the element values, which this
        // validity-only kernel cannot do; fall back to the per-group accumulator path.
        if options.skip_nans && groups.elements().dtype().is_float() {
            return Ok(None);
        }
        try_grouped_count(groups, ctx)
    }
}

/// Count each valid group from the element validity mask.
///
/// The [`Count`] partial dtype is non-nullable `U64`, so a null outer group cannot be represented
/// as a partial state. If any outer group is invalid, this returns `Ok(None)` and lets the caller
/// use the existing fallback behavior.
pub(super) fn try_grouped_count(
    groups: &GroupedArray,
    ctx: &mut ExecutionCtx,
) -> VortexResult<Option<ArrayRef>> {
    if !groups.all_groups_valid(ctx)? {
        return Ok(None);
    }
    let group_ranges = groups.group_ranges(ctx)?;

    Ok(Some(grouped_count(groups.elements(), &group_ranges, ctx)?))
}

/// Count the valid elements of each group described by `group_ranges` (element `(offset, size)`
/// pairs) into a non-nullable `U64` array, one entry per group.
fn grouped_count(
    elements: &ArrayRef,
    group_ranges: &GroupRanges,
    ctx: &mut ExecutionCtx,
) -> VortexResult<ArrayRef> {
    let elem_mask = elements.validity()?.execute_mask(elements.len(), ctx)?;

    let counts: Buffer<u64> = if elem_mask.all_true() {
        group_ranges.iter().map(|(_, size)| size as u64).collect()
    } else {
        group_ranges
            .iter()
            .map(|(offset, size)| valid_count(&elem_mask, offset, size) as u64)
            .collect()
    };

    Ok(PrimitiveArray::new(counts, Validity::NonNullable).into_array())
}

/// Number of valid elements in the `[offset, offset + size)` range of the element mask.
fn valid_count(elem_mask: &Mask, offset: usize, size: usize) -> usize {
    elem_mask.slice(offset..offset + size).true_count()
}

#[cfg(test)]
mod tests {
    #![allow(clippy::cast_possible_truncation)]

    use vortex_buffer::Buffer;
    use vortex_buffer::buffer;
    use vortex_error::VortexResult;

    use crate::ArrayRef;
    use crate::IntoArray;
    use crate::LEGACY_SESSION;
    use crate::VortexSessionExecute;
    use crate::aggregate_fn::DynGroupedAccumulator;
    use crate::aggregate_fn::GroupedAccumulator;
    use crate::aggregate_fn::NumericalAggregateOpts;
    use crate::aggregate_fn::fns::count::Count;
    use crate::arrays::FixedSizeListArray;
    use crate::arrays::ListViewArray;
    use crate::arrays::PrimitiveArray;
    use crate::arrays::VarBinViewArray;
    use crate::assert_arrays_eq;
    use crate::dtype::DType;
    use crate::dtype::Nullability::NonNullable;
    use crate::dtype::Nullability::Nullable;
    use crate::dtype::PType;
    use crate::validity::Validity;

    /// Run a grouped count through the accumulator.
    fn grouped_count_actual(groups: &ArrayRef, elem_dtype: &DType) -> VortexResult<ArrayRef> {
        let mut acc = GroupedAccumulator::try_new(
            Count,
            NumericalAggregateOpts::default(),
            elem_dtype.clone(),
        )?;
        acc.accumulate_list(groups, &mut LEGACY_SESSION.create_execution_ctx())?;
        acc.finish()
    }

    /// Reference valid-counts (non-nullable `U64`), one per group.
    fn grouped_count_reference(
        elements: &ArrayRef,
        ranges: &[(usize, usize)],
    ) -> VortexResult<ArrayRef> {
        let mut ctx = LEGACY_SESSION.create_execution_ctx();
        let counts: Buffer<u64> = ranges
            .iter()
            .map(|&(offset, size)| {
                Ok(elements
                    .slice(offset..offset + size)?
                    .valid_count(&mut ctx)? as u64)
            })
            .collect::<VortexResult<_>>()?;
        Ok(PrimitiveArray::new(counts, Validity::NonNullable).into_array())
    }

    fn listview(elements: ArrayRef, ranges: &[(usize, usize)]) -> VortexResult<ArrayRef> {
        let offsets = PrimitiveArray::from_iter(ranges.iter().map(|&(o, _)| o as i32));
        let sizes = PrimitiveArray::from_iter(ranges.iter().map(|&(_, s)| s as i32));
        Ok(ListViewArray::try_new(
            elements,
            offsets.into_array(),
            sizes.into_array(),
            Validity::NonNullable,
        )?
        .into_array())
    }

    #[test]
    fn listview_counts_all_valid() -> VortexResult<()> {
        let elements =
            PrimitiveArray::new(buffer![1i32, 2, 3, 4, 5, 6], Validity::NonNullable).into_array();
        let elem_dtype = DType::Primitive(PType::I32, NonNullable);
        let ranges = [(0, 2), (2, 1), (3, 3), (6, 0)];

        let groups = listview(elements.clone(), &ranges)?;
        let actual = grouped_count_actual(&groups, &elem_dtype)?;
        let expected = grouped_count_reference(&elements, &ranges)?;

        let direct =
            PrimitiveArray::new(buffer![2u64, 1, 3, 0], Validity::NonNullable).into_array();
        assert_arrays_eq!(&actual, &direct);
        assert_arrays_eq!(&actual, &expected);
        Ok(())
    }

    #[test]
    fn listview_counts_with_nulls() -> VortexResult<()> {
        let elements =
            PrimitiveArray::from_option_iter([Some(1i32), None, Some(3), None, None, Some(9)])
                .into_array();
        let elem_dtype = DType::Primitive(PType::I32, Nullable);
        let ranges = [(0, 3), (3, 2), (5, 1)];

        let groups = listview(elements.clone(), &ranges)?;
        let actual = grouped_count_actual(&groups, &elem_dtype)?;
        let expected = grouped_count_reference(&elements, &ranges)?;

        // Group 0: {1, null, 3} -> 2. Group 1: {null, null} -> 0. Group 2: {9} -> 1.
        let direct = PrimitiveArray::new(buffer![2u64, 0, 1], Validity::NonNullable).into_array();
        assert_arrays_eq!(&actual, &direct);
        assert_arrays_eq!(&actual, &expected);
        Ok(())
    }

    #[test]
    fn listview_counts_varbinview_with_nulls() -> VortexResult<()> {
        let elements = VarBinViewArray::from_iter_nullable_str([
            Some("a"),
            None,
            Some("bbb"),
            None,
            Some("cc"),
        ])
        .into_array();
        let elem_dtype = elements.dtype().clone();
        let ranges = [(0, 2), (2, 2), (4, 1)];

        let groups = listview(elements.clone(), &ranges)?;
        let actual = grouped_count_actual(&groups, &elem_dtype)?;
        let expected = grouped_count_reference(&elements, &ranges)?;

        let direct = PrimitiveArray::new(buffer![1u64, 1, 1], Validity::NonNullable).into_array();
        assert_arrays_eq!(&actual, &direct);
        assert_arrays_eq!(&actual, &expected);
        Ok(())
    }

    #[test]
    fn fixed_size_counts_float_nans() -> VortexResult<()> {
        let elements =
            PrimitiveArray::from_option_iter([Some(1.0f64), Some(f64::NAN), None, Some(2.0)])
                .into_array();
        let elem_dtype = DType::Primitive(PType::F64, Nullable);
        let groups =
            FixedSizeListArray::try_new(elements, 2, Validity::NonNullable, 2)?.into_array();

        // NaNs are excluded by default and counted otherwise.
        let actual = grouped_count_actual(&groups, &elem_dtype)?;
        let expected = PrimitiveArray::new(buffer![1u64, 1], Validity::NonNullable).into_array();
        assert_arrays_eq!(&actual, &expected);

        let mut acc =
            GroupedAccumulator::try_new(Count, NumericalAggregateOpts::include_nans(), elem_dtype)?;
        acc.accumulate_list(&groups, &mut LEGACY_SESSION.create_execution_ctx())?;
        let actual = acc.finish()?;
        let expected = PrimitiveArray::new(buffer![2u64, 1], Validity::NonNullable).into_array();
        assert_arrays_eq!(&actual, &expected);
        Ok(())
    }

    #[test]
    fn fixed_size_counts_with_nulls() -> VortexResult<()> {
        let elements =
            PrimitiveArray::from_option_iter([Some(1i32), None, Some(3), Some(4)]).into_array();
        let elem_dtype = DType::Primitive(PType::I32, Nullable);
        let groups =
            FixedSizeListArray::try_new(elements, 2, Validity::NonNullable, 2)?.into_array();

        let actual = grouped_count_actual(&groups, &elem_dtype)?;
        let direct = PrimitiveArray::new(buffer![1u64, 2], Validity::NonNullable).into_array();
        assert_arrays_eq!(&actual, &direct);
        Ok(())
    }
}
