// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_error::VortexExpect;
use vortex_error::VortexResult;

use crate::ArrayRef;
use crate::Columnar;
use crate::ExecutionCtx;
use crate::aggregate_fn::AggregateFnId;
use crate::aggregate_fn::AggregateFnVTable;
use crate::aggregate_fn::EmptyOptions;
use crate::dtype::DType;
use crate::dtype::Nullability;
use crate::dtype::PType;
use crate::scalar::Scalar;

/// Aggregate that returns the input length, including nulls.
///
/// Applies to all input dtypes, returns a non-nullable `u64`, and has zero as
/// its identity value.
///
/// Unlike [`Count`][crate::aggregate_fn::fns::count::Count], `RowCount`
/// includes null elements. It is primarily used in pruning predicates that need
/// to compare a statistic with the current evaluation scope's row count.
#[derive(Clone, Debug)]
pub struct RowCount;

impl AggregateFnVTable for RowCount {
    type Options = EmptyOptions;
    type Partial = u64;

    fn id(&self) -> AggregateFnId {
        AggregateFnId::new("vortex.row_count")
    }

    fn serialize(&self, _options: &Self::Options) -> VortexResult<Option<Vec<u8>>> {
        unimplemented!("RowCount is not yet serializable");
    }

    fn return_dtype(&self, _options: &Self::Options, _input_dtype: &DType) -> Option<DType> {
        Some(DType::Primitive(PType::U64, Nullability::NonNullable))
    }

    fn partial_dtype(&self, options: &Self::Options, input_dtype: &DType) -> Option<DType> {
        self.return_dtype(options, input_dtype)
    }

    fn empty_partial(
        &self,
        _options: &Self::Options,
        _input_dtype: &DType,
    ) -> VortexResult<Self::Partial> {
        Ok(0u64)
    }

    fn combine_partials(&self, partial: &mut Self::Partial, other: Scalar) -> VortexResult<()> {
        let val = other
            .as_primitive()
            .typed_value::<u64>()
            .vortex_expect("row_count partial should not be null");
        *partial += val;
        Ok(())
    }

    fn to_scalar(&self, partial: &Self::Partial) -> VortexResult<Scalar> {
        Ok(Scalar::primitive(*partial, Nullability::NonNullable))
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
        _ctx: &mut ExecutionCtx,
    ) -> VortexResult<bool> {
        *state += batch.len() as u64;
        Ok(true)
    }

    fn accumulate(
        &self,
        _partial: &mut Self::Partial,
        _batch: &Columnar,
        _ctx: &mut ExecutionCtx,
    ) -> VortexResult<()> {
        unreachable!("RowCount::try_accumulate handles all arrays")
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
    use std::sync::LazyLock;

    use vortex_buffer::buffer;
    use vortex_error::VortexResult;
    use vortex_session::VortexSession;

    use crate::IntoArray;
    use crate::VortexSessionExecute;
    use crate::aggregate_fn::Accumulator;
    use crate::aggregate_fn::DynAccumulator;
    use crate::aggregate_fn::EmptyOptions;
    use crate::aggregate_fn::fns::row_count::RowCount;
    use crate::arrays::PrimitiveArray;
    use crate::session::ArraySession;

    static SESSION: LazyLock<VortexSession> =
        LazyLock::new(|| VortexSession::empty().with::<ArraySession>());

    #[test]
    fn row_count_all_valid() -> VortexResult<()> {
        let array = buffer![1i32, 2, 3, 4, 5].into_array();
        let mut acc = Accumulator::try_new(RowCount, EmptyOptions, array.dtype().clone())?;
        acc.accumulate(&array, &mut SESSION.create_execution_ctx())?;
        let result = acc.finish()?;
        assert_eq!(result.as_primitive().typed_value::<u64>(), Some(5));
        Ok(())
    }

    #[test]
    fn row_count_includes_nulls() -> VortexResult<()> {
        let array = PrimitiveArray::from_option_iter([Some(1i32), None, Some(3), None, Some(5)])
            .into_array();
        let mut acc = Accumulator::try_new(RowCount, EmptyOptions, array.dtype().clone())?;
        acc.accumulate(&array, &mut SESSION.create_execution_ctx())?;
        let result = acc.finish()?;
        assert_eq!(result.as_primitive().typed_value::<u64>(), Some(5));
        Ok(())
    }
}
