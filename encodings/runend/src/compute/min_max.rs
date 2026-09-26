// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_array::ArrayRef;
use vortex_array::ExecutionCtx;
use vortex_array::aggregate_fn::AggregateFnRef;
use vortex_array::aggregate_fn::fns::min_max::MinMax;
use vortex_array::aggregate_fn::fns::min_max::make_minmax_dtype;
use vortex_array::aggregate_fn::fns::min_max::min_max;
use vortex_array::aggregate_fn::kernels::DynAggregateKernel;
use vortex_array::scalar::Scalar;
use vortex_error::VortexResult;

use crate::RunEnd;
use crate::compute::windowed_values;

/// RunEnd-specific min/max kernel.
///
/// Run-end encoded arrays store each unique run value once, so min/max can be computed directly
/// on the run values without decoding — but only over the runs the array's logical window covers,
/// since runs outside it hold no rows. See [`windowed_values`].
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
        let values = windowed_values(&run_end, batch.len(), ctx)?;
        match min_max(&values, ctx, *options)? {
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
    use vortex_array::ArrayRef;
    use vortex_array::IntoArray;
    use vortex_array::VortexSessionExecute;
    use vortex_array::aggregate_fn::AggregateFnVTableExt;
    use vortex_array::aggregate_fn::NumericalAggregateOpts;
    use vortex_array::aggregate_fn::fns::min_max::MinMax;
    use vortex_array::aggregate_fn::kernels::DynAggregateKernel;
    use vortex_array::arrays::PrimitiveArray;
    use vortex_array::expr::stats::Stat;
    use vortex_buffer::buffer;
    use vortex_error::VortexExpect;
    use vortex_error::VortexResult;

    use super::RunEndMinMaxKernel;
    use crate::RunEnd;
    use crate::tests::SESSION;

    /// `validate_parts` only requires the runs to *cover* `offset..offset + length`, so runs may
    /// legally sit outside the window. Their values hold no rows and must not reach min/max.
    #[rstest]
    // Trailing run is entirely past the window: logical rows are [2, 2, 2].
    #[case::trailing_run_outside(buffer![7u64, 10].into_array(), buffer![2i64, 3].into_array(), 2, 3)]
    // Leading run is entirely before the window: logical rows are [3, 3].
    #[case::leading_run_outside(buffer![4u64, 9].into_array(), buffer![2i64, 3].into_array(), 4, 2)]
    // Both ends outside: logical rows are [5, 5]. `validate_parts` requires
    // `first_run_end >= offset`, so a leading run can only sit outside when it ends exactly at the
    // offset — hence 3 here rather than 2.
    #[case::both_ends_outside(
        buffer![3u64, 6, 9].into_array(),
        buffer![1i64, 5, 9].into_array(),
        3,
        2
    )]
    // The whole window, so the fast path must still be taken.
    #[case::window_spans_every_run(
        buffer![3u64, 5].into_array(),
        buffer![2i64, 3].into_array(),
        0,
        5
    )]
    fn min_max_only_sees_the_logical_window(
        #[case] ends: ArrayRef,
        #[case] values: ArrayRef,
        #[case] offset: usize,
        #[case] length: usize,
    ) -> VortexResult<()> {
        let mut ctx = SESSION.create_execution_ctx();
        let array =
            RunEnd::try_new_offset_length(ends, values, offset, length, &mut ctx)?.into_array();
        let decoded = array
            .clone()
            .execute::<PrimitiveArray>(&mut ctx)?
            .into_array();

        // The statistics path is what fills a file's zone map, so check it first and on its own:
        // it is the route by which a wrong extremum becomes wrong pruning.
        let stat_min = array.statistics().compute_stat(Stat::Min, &mut ctx)?;
        let decoded_min = decoded.statistics().compute_stat(Stat::Min, &mut ctx)?;
        assert_eq!(
            stat_min, decoded_min,
            "Stat::Min disagrees with the decoded array"
        );
        let stat_max = array.statistics().compute_stat(Stat::Max, &mut ctx)?;
        let decoded_max = decoded.statistics().compute_stat(Stat::Max, &mut ctx)?;
        assert_eq!(
            stat_max, decoded_max,
            "Stat::Max disagrees with the decoded array"
        );

        // Also drive the kernel directly, so the test cannot pass merely because some dispatch
        // path decided to decode.
        let aggregate = MinMax.bind(NumericalAggregateOpts::default());
        let partial = RunEndMinMaxKernel
            .aggregate(&aggregate, &array, &mut ctx)?
            .vortex_expect("the run-end min/max kernel handles a primitive run-end array");
        let mut direct = aggregate.accumulator(array.dtype())?;
        direct.combine_partials(partial)?;

        let mut reference = aggregate.accumulator(array.dtype())?;
        reference.accumulate(&decoded, &mut ctx)?;

        assert_eq!(
            direct.finish()?,
            reference.finish()?,
            "the kernel disagrees with the decoded array"
        );
        Ok(())
    }
}
