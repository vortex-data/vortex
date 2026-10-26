// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_array::ArrayRef;
use vortex_array::Columnar;
use vortex_array::ExecutionCtx;
use vortex_array::IntoArray;
use vortex_array::aggregate_fn::Accumulator;
use vortex_array::aggregate_fn::AggregateArgs;
use vortex_array::aggregate_fn::AggregateFnId;
use vortex_array::aggregate_fn::AggregateFnVTable;
use vortex_array::aggregate_fn::DynAccumulator;
use vortex_array::aggregate_fn::EmptyOptions;
use vortex_array::aggregate_fn::stat_aggregates;
use vortex_array::dtype::DType;
use vortex_array::dtype::Nullability::NonNullable;
use vortex_array::dtype::PType;
use vortex_array::expr::stats::Precision;
use vortex_array::expr::stats::Stat;
use vortex_array::expr::stats::StatsProvider;
use vortex_array::scalar::Scalar;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_error::vortex_err;
use vortex_session::VortexSession;

/// Return the number of null values in an array.
pub fn null_count(array: &ArrayRef, ctx: &mut ExecutionCtx) -> VortexResult<usize> {
    if let Precision::Exact(null_count_scalar) = array.statistics().get(Stat::NullCount) {
        return usize::try_from(&null_count_scalar)
            .map_err(|e| vortex_err!("Failed to convert null count stat to usize: {e}"));
    }

    let mut acc = Accumulator::try_new(NullCount, EmptyOptions, array.dtype().clone())?;
    acc.accumulate(array, ctx)?;
    let result = acc.finish()?;

    let count = result
        .as_primitive()
        .typed_value::<u64>()
        .vortex_expect("null_count result should not be null");
    // The accumulator caches the null count on `array` as a statistic.
    Ok(usize::try_from(count).vortex_expect("Cannot be more nulls than usize::MAX"))
}

/// Count the number of null values in an array.
///
/// Applies to all types and returns a non-null `u64`.
#[derive(Clone, Debug)]
pub struct NullCount;

impl AggregateFnVTable for NullCount {
    type Options = EmptyOptions;
    type Partial = u64;

    fn id(&self) -> AggregateFnId {
        *stat_aggregates::NULL_COUNT_ID
    }

    fn serialize(&self, _options: &Self::Options) -> VortexResult<Option<Vec<u8>>> {
        Ok(Some(vec![]))
    }

    fn deserialize(
        &self,
        _metadata: &[u8],
        _session: &VortexSession,
    ) -> VortexResult<Self::Options> {
        Ok(EmptyOptions)
    }

    fn return_dtype(&self, _options: &Self::Options, _input_dtype: &DType) -> Option<DType> {
        Some(DType::Primitive(PType::U64, NonNullable))
    }

    fn partial_dtype(&self, options: &Self::Options, input_dtype: &DType) -> Option<DType> {
        self.return_dtype(options, input_dtype)
    }

    fn empty_partial(
        &self,
        _args: AggregateArgs<'_, Self::Options>,
    ) -> VortexResult<Self::Partial> {
        Ok(0)
    }

    fn partial_from_scalar(
        &self,
        _args: AggregateArgs<'_, Self::Options>,
        scalar: &Scalar,
    ) -> VortexResult<Self::Partial> {
        Ok(scalar
            .as_primitive()
            .typed_value::<u64>()
            .vortex_expect("null_count partial should not be null"))
    }

    fn merge_partials(
        &self,
        _args: AggregateArgs<'_, Self::Options>,
        first: Self::Partial,
        second: Self::Partial,
    ) -> VortexResult<Self::Partial> {
        Ok(first + second)
    }

    fn to_scalar(
        &self,
        _args: AggregateArgs<'_, Self::Options>,
        partial: &Self::Partial,
    ) -> VortexResult<Scalar> {
        Ok(Scalar::primitive(*partial, NonNullable))
    }

    #[inline]
    fn is_saturated(
        &self,
        _args: AggregateArgs<'_, Self::Options>,
        _partial: &Self::Partial,
    ) -> bool {
        false
    }

    fn try_accumulate(
        &self,
        _args: AggregateArgs<'_, Self::Options>,
        state: &mut Self::Partial,
        batch: &ArrayRef,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<bool> {
        *state += batch.invalid_count(ctx)? as u64;
        Ok(true)
    }

    fn accumulate(
        &self,
        _args: AggregateArgs<'_, Self::Options>,
        partial: &mut Self::Partial,
        batch: &Columnar,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<()> {
        *partial += match batch {
            Columnar::Constant(c) => {
                if c.scalar().is_null() {
                    c.len() as u64
                } else {
                    0
                }
            }
            Columnar::Canonical(c) => c.clone().into_array().invalid_count(ctx)? as u64,
        };
        Ok(())
    }

    fn finalize(
        &self,
        _args: AggregateArgs<'_, Self::Options>,
        partials: ArrayRef,
    ) -> VortexResult<ArrayRef> {
        Ok(partials)
    }

    fn finalize_scalar(
        &self,
        args: AggregateArgs<'_, Self::Options>,
        partial: &Self::Partial,
    ) -> VortexResult<Scalar> {
        self.to_scalar(args, partial)
    }
}

#[cfg(test)]
mod tests {

    use std::sync::LazyLock;

    use vortex_array::IntoArray;
    use vortex_array::VortexSessionExecute;
    use vortex_array::aggregate_fn::Accumulator;
    use vortex_array::aggregate_fn::DynAccumulator;
    use vortex_array::aggregate_fn::EmptyOptions;
    use vortex_array::arrays::PrimitiveArray;
    use vortex_array::dtype::DType;
    use vortex_array::dtype::Nullability;
    use vortex_array::dtype::PType;
    use vortex_array::expr::stats::Precision;
    use vortex_array::expr::stats::Stat;
    use vortex_array::expr::stats::StatsProviderExt;
    use vortex_error::VortexResult;
    use vortex_session::VortexSession;

    use crate::null_count::NullCount;
    use crate::null_count::null_count;

    static SESSION: LazyLock<VortexSession> = LazyLock::new(|| {
        let session = VortexSession::empty();
        crate::initialize(&session);
        session
    });

    #[test]
    fn null_count_with_nulls() -> VortexResult<()> {
        let array =
            PrimitiveArray::from_option_iter([Some(1i32), None, Some(3), None]).into_array();
        let mut ctx = SESSION.create_execution_ctx();

        assert_eq!(null_count(&array, &mut ctx)?, 2);
        assert_eq!(
            array.statistics().get_as::<u64>(Stat::NullCount),
            Precision::exact(2u64)
        );
        Ok(())
    }

    #[test]
    fn null_count_multi_batch() -> VortexResult<()> {
        let mut ctx = SESSION.create_execution_ctx();
        let dtype = DType::Primitive(PType::I32, Nullability::Nullable);
        let mut acc = Accumulator::try_new(NullCount, EmptyOptions, dtype)?;

        let batch1 = PrimitiveArray::from_option_iter([Some(1i32), None, Some(3)]).into_array();
        acc.accumulate(&batch1, &mut ctx)?;

        let batch2 = PrimitiveArray::from_option_iter([None, Some(5i32), None]).into_array();
        acc.accumulate(&batch2, &mut ctx)?;

        let result = acc.finish()?;
        assert_eq!(result.as_primitive().typed_value::<u64>(), Some(3));
        Ok(())
    }
}
