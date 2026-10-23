// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_array::ArrayRef;
use vortex_array::ExecutionCtx;
use vortex_array::aggregate_fn::AggregateFnRef;
use vortex_array::aggregate_fn::fns::is_sorted::IsSorted;
use vortex_array::aggregate_fn::kernels::DynAggregateKernel;
use vortex_array::scalar::Scalar;
use vortex_error::VortexResult;

use crate::Sequence;

/// Sequence-specific is_sorted kernel.
///
/// See [`Sequence::is_sorted`] for when a sequence is (strictly) sorted.
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

        // `is_sorted` short-circuits on len <= 1, but the Chunked accumulator dispatches this
        // kernel per chunk without that outer check, so the length has to be considered here too.
        let result = Sequence::is_sorted(array.multiplier(), array.len(), options.strict);

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
    use vortex_array::aggregate_fn::fns::is_sorted::is_sorted;
    use vortex_array::aggregate_fn::fns::is_sorted::is_strict_sorted;
    use vortex_array::aggregate_fn::kernels::DynAggregateKernel;
    use vortex_array::arrays::ChunkedArray;
    use vortex_array::dtype::DType;
    use vortex_array::dtype::Nullability;
    use vortex_array::dtype::PType;
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

    #[rstest]
    fn singleton_chunk_keeps_chunked_array_sorted(
        #[values(-1i64, 0)] multiplier: i64,
    ) -> VortexResult<()> {
        let mut ctx = SESSION.create_execution_ctx();
        // The Chunked accumulator hands each chunk to this kernel directly, without going through
        // `is_sorted`, so the one-row first chunk must not report unsorted.
        let chunked = ChunkedArray::try_new(
            [
                Sequence::try_new_typed(5i64, multiplier, Nullability::NonNullable, 1)?
                    .into_array(),
                Sequence::try_new_typed(10i64, 1, Nullability::NonNullable, 3)?.into_array(),
            ],
            DType::Primitive(PType::I64, Nullability::NonNullable),
        )?
        .into_array();

        assert!(is_sorted(&chunked, &mut ctx)?);
        assert!(is_strict_sorted(&chunked, &mut ctx)?);
        Ok(())
    }
}
