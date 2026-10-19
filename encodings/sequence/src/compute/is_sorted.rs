// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use num_traits::zero;
use vortex_array::ArrayRef;
use vortex_array::ExecutionCtx;
use vortex_array::aggregate_fn::AggregateFnRef;
use vortex_array::aggregate_fn::fns::is_sorted::IsSorted;
use vortex_array::aggregate_fn::kernels::DynAggregateKernel;
use vortex_array::match_each_native_ptype;
use vortex_array::scalar::Scalar;
use vortex_error::VortexResult;

use crate::Sequence;

/// Sequence-specific is_sorted kernel.
///
/// A sequence `A[i] = base + i * multiplier` with at least two elements is sorted iff
/// multiplier >= 0, and strict sorted iff multiplier > 0.
#[derive(Debug)]
pub(crate) struct SequenceIsSortedKernel;

impl DynAggregateKernel for SequenceIsSortedKernel {
    fn aggregate(
        &self,
        aggregate_fn: &AggregateFnRef,
        batch: &ArrayRef,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<Option<Scalar>> {
        let Some(options) = aggregate_fn.as_opt::<IsSorted>() else {
            return Ok(None);
        };

        let Some(array) = batch.as_opt::<Sequence>() else {
            return Ok(None);
        };

        // A single element is strictly sorted whatever the multiplier.
        let result = if array.len() <= 1 {
            true
        } else {
            let m = array.multiplier();
            match_each_native_ptype!(m.ptype(), |P| {
                m.cast::<P>().map(|x| {
                    if options.strict {
                        x > zero::<P>()
                    } else {
                        x >= zero::<P>()
                    }
                })
            })?
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
    use std::sync::LazyLock;

    use rstest::rstest;
    use vortex_array::IntoArray;
    use vortex_array::VortexSessionExecute;
    use vortex_array::aggregate_fn::AggregateFnVTableExt;
    use vortex_array::aggregate_fn::fns::is_sorted::IsSorted;
    use vortex_array::aggregate_fn::fns::is_sorted::IsSortedOptions;
    use vortex_array::aggregate_fn::kernels::DynAggregateKernel;
    use vortex_array::dtype::Nullability;
    use vortex_error::VortexResult;
    use vortex_session::VortexSession;

    use super::SequenceIsSortedKernel;
    use crate::Sequence;

    static SESSION: LazyLock<VortexSession> = LazyLock::new(|| {
        let session = vortex_array::array_session();
        crate::initialize(&session);
        session
    });

    #[rstest]
    fn singleton_is_sorted_for_any_multiplier(
        #[values(-1i64, 0, 3)] multiplier: i64,
        #[values(false, true)] strict: bool,
    ) -> VortexResult<()> {
        let mut ctx = SESSION.create_execution_ctx();
        let array =
            Sequence::try_new_typed(10i64, multiplier, Nullability::NonNullable, 1)?.into_array();

        let result = SequenceIsSortedKernel.aggregate(
            &IsSorted.bind(IsSortedOptions { strict }),
            &array,
            &mut ctx,
        )?;

        let expected = IsSorted::make_partial(&array, true, strict, &mut ctx)?;
        assert_eq!(result, Some(expected));
        Ok(())
    }
}
