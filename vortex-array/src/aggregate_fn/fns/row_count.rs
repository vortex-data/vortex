// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_error::VortexResult;
use vortex_error::vortex_err;
use vortex_session::VortexSession;

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

/// An aggregate function that returns the total row count of the input.
///
/// This counts all rows, including those with null values. The result is always
/// a non-nullable `u64`.
#[derive(Debug, Clone)]
pub struct RowCount;

impl AggregateFnVTable for RowCount {
    type Options = EmptyOptions;
    type Partial = u64;

    fn id(&self) -> AggregateFnId {
        AggregateFnId::new_ref("vortex.row_count")
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
        Some(DType::Primitive(PType::U64, Nullability::NonNullable))
    }

    fn partial_dtype(&self, _options: &Self::Options, _input_dtype: &DType) -> Option<DType> {
        Some(DType::Primitive(PType::U64, Nullability::NonNullable))
    }

    fn empty_partial(
        &self,
        _options: &Self::Options,
        _input_dtype: &DType,
    ) -> VortexResult<Self::Partial> {
        Ok(0)
    }

    fn combine_partials(&self, partial: &mut Self::Partial, other: Scalar) -> VortexResult<()> {
        let other = other
            .as_primitive()
            .as_::<u64>()
            .ok_or_else(|| vortex_err!("Cannot cast partial to u64: {other}"))?;
        *partial += other;

        Ok(())
    }

    fn to_scalar(&self, partial: &Self::Partial) -> VortexResult<Scalar> {
        Ok(Scalar::primitive(*partial, Nullability::NonNullable))
    }

    fn reset(&self, partial: &mut Self::Partial) {
        *partial = 0;
    }

    fn is_saturated(&self, _state: &Self::Partial) -> bool {
        false
    }

    fn accumulate(
        &self,
        state: &mut Self::Partial,
        batch: &Columnar,
        _ctx: &mut ExecutionCtx,
    ) -> VortexResult<()> {
        *state += batch.len() as u64;
        Ok(())
    }

    fn finalize(&self, states: ArrayRef) -> VortexResult<ArrayRef> {
        Ok(states)
    }
}

#[cfg(test)]
mod tests {
    use vortex_buffer::buffer;
    use vortex_error::VortexResult;

    use crate::IntoArray;
    use crate::LEGACY_SESSION;
    use crate::VortexSessionExecute;
    use crate::aggregate_fn::Accumulator;
    use crate::aggregate_fn::AggregateFnVTable;
    use crate::aggregate_fn::DynAccumulator;
    use crate::aggregate_fn::EmptyOptions;
    use crate::aggregate_fn::fns::row_count::RowCount;
    use crate::arrays::PrimitiveArray;
    use crate::dtype::DType;
    use crate::dtype::Nullability;
    use crate::dtype::PType;
    use crate::scalar::Scalar;
    use crate::validity::Validity;

    #[test]
    fn row_count_multi_batch() -> VortexResult<()> {
        let mut ctx = LEGACY_SESSION.create_execution_ctx();
        let dtype = DType::Primitive(PType::I32, Nullability::NonNullable);
        let mut acc = Accumulator::try_new(RowCount, EmptyOptions, dtype)?;

        let batch1 = PrimitiveArray::new(buffer![1i32, 2, 3], Validity::NonNullable).into_array();
        acc.accumulate(&batch1, &mut ctx)?;

        let batch2 = PrimitiveArray::new(buffer![4i32, 5], Validity::NonNullable).into_array();
        acc.accumulate(&batch2, &mut ctx)?;

        let result = acc.finish()?;
        assert_eq!(result.as_primitive().typed_value::<u64>(), Some(5));
        Ok(())
    }

    #[test]
    fn row_count_finish_resets_state() -> VortexResult<()> {
        let mut ctx = LEGACY_SESSION.create_execution_ctx();
        let dtype = DType::Primitive(PType::I32, Nullability::NonNullable);
        let mut acc = Accumulator::try_new(RowCount, EmptyOptions, dtype)?;

        let batch1 = PrimitiveArray::new(buffer![1i32, 2, 3], Validity::NonNullable).into_array();
        acc.accumulate(&batch1, &mut ctx)?;
        let result1 = acc.finish()?;
        assert_eq!(result1.as_primitive().typed_value::<u64>(), Some(3));

        let batch2 = PrimitiveArray::new(buffer![4i32, 5], Validity::NonNullable).into_array();
        acc.accumulate(&batch2, &mut ctx)?;
        let result2 = acc.finish()?;
        assert_eq!(result2.as_primitive().typed_value::<u64>(), Some(2));
        Ok(())
    }

    #[test]
    fn row_count_state_merge() -> VortexResult<()> {
        let dtype = DType::Primitive(PType::I32, Nullability::NonNullable);
        let mut state = RowCount.empty_partial(&EmptyOptions, &dtype)?;

        let scalar1 = Scalar::primitive(10u64, Nullability::NonNullable);
        RowCount.combine_partials(&mut state, scalar1)?;

        let scalar2 = Scalar::primitive(7u64, Nullability::NonNullable);
        RowCount.combine_partials(&mut state, scalar2)?;

        let result = RowCount.to_scalar(&state)?;
        assert_eq!(result.as_primitive().typed_value::<u64>(), Some(17));
        Ok(())
    }

    #[test]
    fn row_count_empty() -> VortexResult<()> {
        let dtype = DType::Primitive(PType::I32, Nullability::NonNullable);
        let mut acc = Accumulator::try_new(RowCount, EmptyOptions, dtype)?;
        let result = acc.finish()?;
        assert_eq!(result.as_primitive().typed_value::<u64>(), Some(0));
        Ok(())
    }

    #[test]
    fn row_count_with_nulls() -> VortexResult<()> {
        let mut ctx = LEGACY_SESSION.create_execution_ctx();
        let dtype = DType::Primitive(PType::I32, Nullability::Nullable);
        let mut acc = Accumulator::try_new(RowCount, EmptyOptions, dtype)?;

        // Row count should count all rows, including nulls
        let batch =
            PrimitiveArray::from_option_iter([Some(1i32), None, Some(3), None]).into_array();
        acc.accumulate(&batch, &mut ctx)?;

        let result = acc.finish()?;
        assert_eq!(result.as_primitive().typed_value::<u64>(), Some(4));
        Ok(())
    }
}
