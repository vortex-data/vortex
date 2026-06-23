// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_error::vortex_err;
use vortex_session::VortexSession;

use crate::ArrayRef;
use crate::Columnar;
use crate::ExecutionCtx;
use crate::IntoArray;
use crate::aggregate_fn::Accumulator;
use crate::aggregate_fn::AggregateFnId;
use crate::aggregate_fn::AggregateFnVTable;
use crate::aggregate_fn::DynAccumulator;
use crate::aggregate_fn::EmptyOptions;
use crate::dtype::DType;
use crate::dtype::Nullability::NonNullable;
use crate::dtype::PType;
use crate::expr::stats::Precision;
use crate::expr::stats::Stat;
use crate::expr::stats::StatsProvider;
use crate::scalar::Scalar;
use crate::scalar::ScalarValue;

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
    let count_usize = usize::try_from(count).vortex_expect("Cannot be more nulls than usize::MAX");

    array
        .statistics()
        .set(Stat::NullCount, Precision::Exact(ScalarValue::from(count)));

    Ok(count_usize)
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
        AggregateFnId::new("vortex.null_count")
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
        _options: &Self::Options,
        _input_dtype: &DType,
    ) -> VortexResult<Self::Partial> {
        Ok(0)
    }

    fn combine_partials(&self, partial: &mut Self::Partial, other: Scalar) -> VortexResult<()> {
        let count = other
            .as_primitive()
            .typed_value::<u64>()
            .vortex_expect("null_count partial should not be null");
        *partial += count;
        Ok(())
    }

    fn to_scalar(&self, partial: &Self::Partial) -> VortexResult<Scalar> {
        Ok(Scalar::primitive(*partial, NonNullable))
    }

    fn reset(&self, partial: &mut Self::Partial) {
        *partial = 0;
    }

    #[inline]
    fn is_saturated(&self, _partial: &Self::Partial) -> bool {
        false
    }

    fn try_accumulate(
        &self,
        state: &mut Self::Partial,
        batch: &ArrayRef,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<bool> {
        *state += batch.invalid_count(ctx)? as u64;
        Ok(true)
    }

    fn accumulate(
        &self,
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

    fn finalize(&self, partials: ArrayRef) -> VortexResult<ArrayRef> {
        Ok(partials)
    }

    fn finalize_scalar(&self, partial: &Self::Partial) -> VortexResult<Scalar> {
        self.to_scalar(partial)
    }
}

#[cfg(test)]
mod tests {
    use vortex_error::VortexResult;

    use crate::IntoArray;
    use crate::VortexSessionExecute;
    use crate::aggregate_fn::Accumulator;
    use crate::aggregate_fn::DynAccumulator;
    use crate::aggregate_fn::EmptyOptions;
    use crate::aggregate_fn::fns::null_count::NullCount;
    use crate::aggregate_fn::fns::null_count::null_count;
    use crate::array_session;
    use crate::arrays::PrimitiveArray;
    use crate::dtype::DType;
    use crate::dtype::Nullability;
    use crate::dtype::PType;
    use crate::expr::stats::Precision;
    use crate::expr::stats::Stat;
    use crate::expr::stats::StatsProviderExt;

    #[test]
    fn null_count_with_nulls() -> VortexResult<()> {
        let array =
            PrimitiveArray::from_option_iter([Some(1i32), None, Some(3), None]).into_array();
        let mut ctx = array_session().create_execution_ctx();

        assert_eq!(null_count(&array, &mut ctx)?, 2);
        assert_eq!(
            array.statistics().get_as::<u64>(Stat::NullCount),
            Precision::exact(2u64)
        );
        Ok(())
    }

    #[test]
    fn null_count_multi_batch() -> VortexResult<()> {
        let mut ctx = array_session().create_execution_ctx();
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
