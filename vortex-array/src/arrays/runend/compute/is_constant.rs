// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_error::VortexResult;

use super::active_physical_range;
use crate::ArrayRef;
use crate::ExecutionCtx;
use crate::aggregate_fn::AggregateFnRef;
use crate::aggregate_fn::fns::is_constant::IsConstant;
use crate::aggregate_fn::fns::is_constant::is_constant;
use crate::aggregate_fn::kernels::DynAggregateKernel;
use crate::arrays::runend::RunEnd;
use crate::arrays::runend::array::RunEndArraySlotsExt;
use crate::scalar::Scalar;

/// RunEnd-specific is_constant kernel.
///
/// If the active slice of the values array covering `[offset..offset + len)` is constant, the
/// entire array is constant.
#[derive(Debug)]
pub(crate) struct RunEndIsConstantKernel;

impl DynAggregateKernel for RunEndIsConstantKernel {
    fn aggregate(
        &self,
        aggregate_fn: &AggregateFnRef,
        batch: &ArrayRef,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<Option<Scalar>> {
        if !aggregate_fn.is::<IsConstant>() {
            return Ok(None);
        }

        let Some(run_end) = batch.as_opt::<RunEnd>() else {
            return Ok(None);
        };

        if run_end.is_empty() {
            return Ok(Some(IsConstant::make_partial(batch, false, ctx)?));
        }

        let active_range = active_physical_range(run_end, ctx)?;
        let result = if active_range.len() == 1 {
            true
        } else {
            let values = run_end.values();
            if active_range == (0..values.len()) {
                is_constant(values, ctx)?
            } else {
                let sliced_values = values.slice(active_range)?;
                is_constant(&sliced_values, ctx)?
            }
        };

        Ok(Some(IsConstant::make_partial(batch, result, ctx)?))
    }
}

#[cfg(test)]
mod tests {
    use rstest::rstest;
    use vortex_buffer::buffer;
    use vortex_error::VortexExpect;
    use vortex_error::VortexResult;

    use super::RunEndIsConstantKernel;
    use crate::ArrayRef;
    use crate::Canonical;
    use crate::IntoArray;
    use crate::VortexSessionExecute;
    use crate::aggregate_fn::AggregateFnVTableExt;
    use crate::aggregate_fn::EmptyOptions;
    use crate::aggregate_fn::fns::is_constant::IsConstant;
    use crate::aggregate_fn::fns::is_constant::is_constant;
    use crate::aggregate_fn::kernels::DynAggregateKernel;
    use crate::arrays::PrimitiveArray;
    use crate::arrays::runend::RunEnd;
    use crate::arrays::runend::tests::SESSION;
    use crate::dtype::Nullability;
    use crate::scalar::Scalar;

    fn check_is_constant(array: ArrayRef, expected: bool) -> VortexResult<()> {
        let mut ctx = SESSION.create_execution_ctx();
        let decoded = array.clone().execute::<Canonical>(&mut ctx)?.into_array();

        let aggregate = IsConstant.bind(EmptyOptions);

        let mut reference = aggregate.accumulator(array.dtype())?;
        reference.accumulate(&decoded, &mut ctx)?;
        let expected_scalar = Scalar::bool(expected, Nullability::NonNullable);
        assert_eq!(reference.finish()?, expected_scalar);

        let partial = RunEndIsConstantKernel
            .aggregate(&aggregate, &array, &mut ctx)?
            .vortex_expect("RunEndIsConstantKernel should handle RunEnd arrays");
        let mut direct = aggregate.accumulator(array.dtype())?;
        direct.combine_partials(&partial)?;
        assert_eq!(direct.finish()?, expected_scalar);

        let mut dispatched = aggregate.accumulator(array.dtype())?;
        dispatched.accumulate(&array, &mut ctx)?;
        assert_eq!(dispatched.finish()?, expected_scalar);

        assert_eq!(is_constant(&array, &mut ctx)?, expected);
        Ok(())
    }

    #[rstest]
    #[case::leading_zero_length_run(vec![0u32, 3, 6], vec![1i32, 5, 5], 0, 6, true)]
    #[case::leading_distinct_run_trimmed(vec![3u32, 6, 10], vec![1i32, 5, 5], 3, 7, true)]
    #[case::leading_distinct_run_still_active(vec![3u32, 6, 10], vec![1i32, 5, 5], 2, 8, false)]
    #[case::trailing_distinct_run_trimmed_mid_run(vec![3u32, 6, 10], vec![5i32, 5, 9], 0, 5, true)]
    #[case::leading_and_trailing_trimmed_multi_run(
        vec![3u32, 6, 9, 12],
        vec![1i32, 5, 5, 9],
        3,
        6,
        true
    )]
    #[case::single_active_middle_run(vec![3u32, 6, 9], vec![1i32, 5, 9], 3, 3, true)]
    #[case::empty_with_retained_runs(vec![3u32, 6, 10], vec![5i32, 5, 5], 3, 0, false)]
    fn test_runend_is_constant_sliced(
        #[case] ends: Vec<u32>,
        #[case] values: Vec<i32>,
        #[case] offset: usize,
        #[case] length: usize,
        #[case] expected: bool,
    ) -> VortexResult<()> {
        let mut ctx = SESSION.create_execution_ctx();
        let array = RunEnd::try_new_offset_length(
            PrimitiveArray::from_iter(ends).into_array(),
            PrimitiveArray::from_iter(values).into_array(),
            offset,
            length,
            &mut ctx,
        )?
        .into_array();

        check_is_constant(array, expected)
    }

    #[rstest]
    #[case::active_all_null(vec![Some(1i32), None, None, Some(2)], 3, 6, true)]
    #[case::active_all_valid_same(vec![None, Some(7i32), Some(7), None], 3, 6, true)]
    fn test_runend_is_constant_nullable_sliced(
        #[case] values: Vec<Option<i32>>,
        #[case] offset: usize,
        #[case] length: usize,
        #[case] expected: bool,
    ) -> VortexResult<()> {
        let mut ctx = SESSION.create_execution_ctx();
        let array = RunEnd::try_new_offset_length(
            buffer![3u32, 6, 9, 12].into_array(),
            PrimitiveArray::from_option_iter(values).into_array(),
            offset,
            length,
            &mut ctx,
        )?
        .into_array();

        check_is_constant(array, expected)
    }

    #[test]
    fn test_runend_is_constant_leading_null_end() -> VortexResult<()> {
        let mut ctx = SESSION.create_execution_ctx();
        let array = RunEnd::try_new_offset_length(
            PrimitiveArray::from_option_iter([None, Some(3u32), Some(6)]).into_array(),
            buffer![1i32, 5, 5].into_array(),
            0,
            6,
            &mut ctx,
        )?
        .into_array();

        check_is_constant(array, true)
    }

    #[test]
    fn test_runend_is_constant_empty_batch_preserves_accumulator() -> VortexResult<()> {
        let mut ctx = SESSION.create_execution_ctx();
        let empty = RunEnd::try_new_offset_length(
            buffer![3u32, 6].into_array(),
            buffer![1i32, 2].into_array(),
            3,
            0,
            &mut ctx,
        )?
        .into_array();
        let constant = RunEnd::try_new(
            buffer![4u32].into_array(),
            buffer![42i32].into_array(),
            &mut ctx,
        )?
        .into_array();

        let aggregate = IsConstant.bind(EmptyOptions);
        let mut acc = aggregate.accumulator(empty.dtype())?;
        acc.accumulate(&empty, &mut ctx)?;
        acc.accumulate(&constant, &mut ctx)?;
        assert_eq!(acc.finish()?, Scalar::bool(true, Nullability::NonNullable));
        Ok(())
    }
}
