// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_array::ArrayRef;
use vortex_array::Canonical;
use vortex_array::ExecutionCtx;
use vortex_array::IntoArray as _;
use vortex_array::aggregate_fn::AggregateFnRef;
use vortex_array::aggregate_fn::fns::is_sorted::IsSorted;
use vortex_array::aggregate_fn::fns::is_sorted::is_sorted;
use vortex_array::aggregate_fn::fns::is_sorted::is_strict_sorted;
use vortex_array::aggregate_fn::kernels::DynAggregateKernel;
use vortex_array::scalar::Scalar;
use vortex_error::VortexResult;

use crate::RunEnd;
use crate::array::RunEndArraySlotsExt;

/// RunEnd-specific is_sorted kernel.
///
/// Non-strict: values array sorted implies the run-end array is sorted.
/// Strict: a run longer than one row repeats its value, so the array is not strictly sorted;
/// otherwise canonicalize.
#[derive(Debug)]
pub(crate) struct RunEndIsSortedKernel;

impl DynAggregateKernel for RunEndIsSortedKernel {
    fn aggregate(
        &self,
        aggregate_fn: &AggregateFnRef,
        batch: &ArrayRef,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<Option<Scalar>> {
        let Some(options) = aggregate_fn.as_opt::<IsSorted>() else {
            return Ok(None);
        };

        let Some(array) = batch.as_opt::<RunEnd>() else {
            return Ok(None);
        };

        let result = if options.strict {
            // Fewer runs than rows means some run repeats a value, which is never strictly sorted.
            if array.values().len() < batch.len() {
                return Ok(Some(IsSorted::make_partial(batch, false, true, ctx)?));
            }
            // Otherwise canonicalize, since the runs themselves are not checked here.
            is_strict_sorted(
                &array
                    .array()
                    .clone()
                    .execute::<Canonical>(ctx)?
                    .into_array(),
                ctx,
            )?
        } else {
            is_sorted(array.values(), ctx)?
        };

        Ok(Some(IsSorted::make_partial(
            batch,
            result,
            options.strict,
            ctx,
        )?))
    }
}

#[cfg(test)]
mod tests {
    use rstest::rstest;
    use vortex_array::IntoArray;
    use vortex_array::VortexSessionExecute;
    use vortex_array::aggregate_fn::fns::is_sorted::is_sorted;
    use vortex_array::aggregate_fn::fns::is_sorted::is_strict_sorted;
    use vortex_array::arrays::PrimitiveArray;
    use vortex_error::VortexResult;

    use crate::RunEnd;
    use crate::tests::SESSION;

    #[rstest]
    #[case::repeated_run(PrimitiveArray::from_iter([1i32, 1, 2, 3]), true, false)]
    #[case::single_rows(PrimitiveArray::from_iter([1i32, 2, 3]), true, true)]
    #[case::unsorted(PrimitiveArray::from_iter([3i32, 3, 1]), false, false)]
    #[case::repeated_nulls(PrimitiveArray::from_option_iter([None, None, Some(1i32)]), true, false)]
    fn runend_is_sorted(
        #[case] values: PrimitiveArray,
        #[case] sorted: bool,
        #[case] strict: bool,
    ) -> VortexResult<()> {
        let mut ctx = SESSION.create_execution_ctx();
        let array = RunEnd::encode(values.into_array(), &mut ctx)?.into_array();
        assert_eq!(is_sorted(&array, &mut ctx)?, sorted);
        assert_eq!(is_strict_sorted(&array, &mut ctx)?, strict);
        Ok(())
    }
}
