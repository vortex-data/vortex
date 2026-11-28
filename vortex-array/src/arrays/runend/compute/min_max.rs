// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_error::VortexResult;

use super::active_physical_range;
use crate::ArrayRef;
use crate::ExecutionCtx;
use crate::aggregate_fn::AggregateFnRef;
use crate::aggregate_fn::fns::min_max::MinMax;
use crate::aggregate_fn::fns::min_max::make_minmax_dtype;
use crate::aggregate_fn::fns::min_max::min_max;
use crate::aggregate_fn::kernels::DynAggregateKernel;
use crate::arrays::runend::RunEnd;
use crate::arrays::runend::array::RunEndArraySlotsExt;
use crate::scalar::Scalar;

/// RunEnd-specific min/max kernel.
///
/// Run-end encoded arrays store each run value once, so min/max can be computed directly on the
/// active slice of the values array covering `[offset..offset + len)` without decoding.
#[derive(Debug)]
pub(crate) struct RunEndMinMaxKernel;

impl DynAggregateKernel for RunEndMinMaxKernel {
    fn aggregate(
        &self,
        aggregate_fn: &AggregateFnRef,
        batch: &ArrayRef,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<Option<Scalar>> {
        let Some(options) = aggregate_fn.as_opt::<MinMax>() else {
            return Ok(None);
        };

        let Some(run_end) = batch.as_opt::<RunEnd>() else {
            return Ok(None);
        };

        let struct_dtype = make_minmax_dtype(batch.dtype());
        if run_end.is_empty() {
            return Ok(Some(Scalar::null(struct_dtype)));
        }

        let active_range = active_physical_range(run_end, ctx)?;
        let values = run_end.values();
        let result = if active_range == (0..values.len()) {
            min_max(values, ctx, *options)?
        } else {
            let sliced_values = values.slice(active_range)?;
            min_max(&sliced_values, ctx, *options)?
        };

        match result {
            Some(result) => Ok(Some(Scalar::struct_(
                struct_dtype,
                vec![result.min, result.max],
            ))),
            None => Ok(Some(Scalar::null(struct_dtype))),
        }
    }
}

#[cfg(test)]
mod tests {
    use rstest::rstest;
    use vortex_buffer::buffer;
    use vortex_error::VortexExpect;
    use vortex_error::VortexResult;

    use super::RunEndMinMaxKernel;
    use crate::ArrayRef;
    use crate::Canonical;
    use crate::IntoArray;
    use crate::VortexSessionExecute;
    use crate::aggregate_fn::AggregateFnVTableExt;
    use crate::aggregate_fn::NumericalAggregateOpts;
    use crate::aggregate_fn::fns::min_max::MinMax;
    use crate::aggregate_fn::fns::min_max::MinMaxResult;
    use crate::aggregate_fn::fns::min_max::min_max;
    use crate::aggregate_fn::kernels::DynAggregateKernel;
    use crate::arrays::PrimitiveArray;
    use crate::arrays::runend::RunEnd;
    use crate::arrays::runend::tests::SESSION;

    fn check_min_max(
        array: ArrayRef,
        options: NumericalAggregateOpts,
        expected: Option<MinMaxResult>,
    ) -> VortexResult<()> {
        let mut ctx = SESSION.create_execution_ctx();
        let decoded = array.clone().execute::<Canonical>(&mut ctx)?.into_array();

        let aggregate = MinMax.bind(options);

        let mut reference = aggregate.accumulator(array.dtype())?;
        reference.accumulate(&decoded, &mut ctx)?;
        let expected_scalar = reference.finish()?;
        assert_eq!(MinMaxResult::from_scalar(&expected_scalar)?, expected);

        let partial = RunEndMinMaxKernel
            .aggregate(&aggregate, &array, &mut ctx)?
            .vortex_expect("RunEndMinMaxKernel should handle RunEnd arrays");
        let mut direct = aggregate.accumulator(array.dtype())?;
        direct.combine_partials(&partial)?;
        assert_eq!(direct.finish()?, expected_scalar);

        let mut dispatched = aggregate.accumulator(array.dtype())?;
        dispatched.accumulate(&array, &mut ctx)?;
        assert_eq!(dispatched.finish()?, expected_scalar);

        assert_eq!(min_max(&array, &mut ctx, options)?, expected);
        Ok(())
    }

    #[rstest]
    #[case::leading_zero_length_run(vec![0u32, 3, 6], vec![-100i32, 5, 10], 0, 6, Some((5, 10)))]
    #[case::leading_unreferenced_extreme(vec![3u32, 6, 10], vec![-100i32, 5, 10], 3, 7, Some((5, 10)))]
    #[case::trailing_unreferenced_mid_run(vec![3u32, 6, 10], vec![5i32, 10, 999], 0, 5, Some((5, 10)))]
    #[case::leading_and_trailing_unreferenced(
        vec![3u32, 6, 9, 12],
        vec![-100i32, 5, 10, 999],
        3,
        5,
        Some((5, 10))
    )]
    #[case::single_active_middle_run(
        vec![3u32, 6, 9],
        vec![-100i32, 7, 999],
        3,
        3,
        Some((7, 7))
    )]
    #[case::empty_with_retained_runs(vec![3u32, 6, 10], vec![-100i32, 5, 999], 3, 0, None)]
    fn test_runend_min_max_sliced(
        #[case] ends: Vec<u32>,
        #[case] values: Vec<i32>,
        #[case] offset: usize,
        #[case] length: usize,
        #[case] expected: Option<(i32, i32)>,
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

        check_min_max(
            array,
            NumericalAggregateOpts::default(),
            expected.map(|(min, max)| MinMaxResult {
                min: min.into(),
                max: max.into(),
            }),
        )
    }

    #[test]
    fn test_runend_min_max_leading_null_end() -> VortexResult<()> {
        let mut ctx = SESSION.create_execution_ctx();
        let array = RunEnd::try_new_offset_length(
            PrimitiveArray::from_option_iter([None, Some(3u32), Some(6)]).into_array(),
            buffer![-100i32, 5, 10].into_array(),
            0,
            6,
            &mut ctx,
        )?
        .into_array();

        check_min_max(
            array,
            NumericalAggregateOpts::default(),
            Some(MinMaxResult {
                min: 5i32.into(),
                max: 10i32.into(),
            }),
        )
    }

    #[test]
    fn test_runend_min_max_nullable_active_all_null() -> VortexResult<()> {
        let mut ctx = SESSION.create_execution_ctx();
        let array = RunEnd::try_new_offset_length(
            buffer![3u32, 6, 10].into_array(),
            PrimitiveArray::from_option_iter([Some(1i32), None, Some(2)]).into_array(),
            3,
            3,
            &mut ctx,
        )?
        .into_array();

        check_min_max(array, NumericalAggregateOpts::default(), None)
    }

    #[test]
    fn test_runend_min_max_unreferenced_nan_not_poisoned() -> VortexResult<()> {
        let mut ctx = SESSION.create_execution_ctx();
        let array = RunEnd::try_new_offset_length(
            buffer![3u32, 6, 9, 12].into_array(),
            buffer![f32::NAN, 1.5f32, 4.5f32, f32::NAN].into_array(),
            3,
            6,
            &mut ctx,
        )?
        .into_array();

        check_min_max(
            array,
            NumericalAggregateOpts { skip_nans: false },
            Some(MinMaxResult {
                min: 1.5f32.into(),
                max: 4.5f32.into(),
            }),
        )
    }
}
