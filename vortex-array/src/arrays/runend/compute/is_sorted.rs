// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_error::VortexResult;

use crate::ArrayRef;
use crate::ExecutionCtx;
use crate::aggregate_fn::AggregateFnRef;
use crate::aggregate_fn::fns::is_sorted::IsSorted;
use crate::aggregate_fn::fns::is_sorted::is_sorted;
use crate::aggregate_fn::fns::is_sorted::is_strict_sorted;
use crate::aggregate_fn::kernels::DynAggregateKernel;
use crate::arrays::runend::RunEnd;
use crate::arrays::runend::array::RunEndArraySlotsExt;
use crate::scalar::Scalar;

/// RunEnd-specific is_sorted kernel.
///
/// Non-strict: values array sorted implies the run-end array is sorted.
/// Strict: a run longer than one element repeats its value, so the array is strictly sorted
/// exactly when every run holds one element and the run values are strictly sorted. Neither
/// check needs the array to be decoded.
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
            array.ends().len() == array.len() && is_strict_sorted(array.values(), ctx)?
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
    use vortex_buffer::buffer;
    use vortex_error::VortexResult;

    use crate::IntoArray;
    use crate::VortexSessionExecute;
    use crate::aggregate_fn::fns::is_sorted::is_sorted;
    use crate::aggregate_fn::fns::is_sorted::is_strict_sorted;
    use crate::arrays::PrimitiveArray;
    use crate::arrays::runend::RunEnd;
    use crate::arrays::runend::tests::SESSION;

    #[rstest]
    #[case::one_element_runs(vec![1, 2, 3, 4], true)]
    #[case::repeated_run(vec![1, 2, 2, 3], false)]
    #[case::descending(vec![4, 3, 2, 1], false)]
    #[case::single_long_run(vec![5, 5, 5, 5], false)]
    fn strict_sorted_matches_decoded(
        #[case] values: Vec<i32>,
        #[case] expected: bool,
    ) -> VortexResult<()> {
        let mut ctx = SESSION.create_execution_ctx();
        let decoded = PrimitiveArray::from_iter(values).into_array();
        let encoded = RunEnd::encode(decoded.clone(), &mut ctx)?.into_array();

        assert_eq!(is_strict_sorted(&encoded, &mut ctx)?, expected);
        assert_eq!(is_strict_sorted(&decoded, &mut ctx)?, expected);
        Ok(())
    }

    // Slicing trims the runs to the window, so the sliced array is strictly sorted.
    #[test]
    fn strict_sorted_after_slice() -> VortexResult<()> {
        let mut ctx = SESSION.create_execution_ctx();
        let encoded =
            RunEnd::encode(buffer![1i32, 1, 2, 3, 4, 4].into_array(), &mut ctx)?.into_array();

        assert!(!is_strict_sorted(&encoded, &mut ctx)?);
        assert!(is_strict_sorted(&encoded.slice(1..5)?, &mut ctx)?);
        assert!(is_sorted(&encoded, &mut ctx)?);
        Ok(())
    }
}
