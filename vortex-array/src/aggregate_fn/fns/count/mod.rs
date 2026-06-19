// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

mod grouped;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;

use crate::ArrayRef;
use crate::Columnar;
use crate::ExecutionCtx;
use crate::IntoArray;
use crate::aggregate_fn::AggregateFnId;
use crate::aggregate_fn::AggregateFnVTable;
use crate::aggregate_fn::EmptyOptions;
use crate::arrays::PrimitiveArray;
use crate::dtype::DType;
use crate::dtype::Nullability;
use crate::dtype::PType;
use crate::scalar::Scalar;

/// Count the number of non-null elements in an array.
///
/// Applies to all types. Returns a `u64` count.
/// The identity value is zero.
#[derive(Clone, Debug)]
pub struct Count;

impl AggregateFnVTable for Count {
    type Options = EmptyOptions;
    type Partial = u64;

    fn id(&self) -> AggregateFnId {
        AggregateFnId::new("vortex.count")
    }

    fn serialize(&self, _options: &Self::Options) -> VortexResult<Option<Vec<u8>>> {
        unimplemented!("Count is not yet serializable");
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
            .vortex_expect("count partial should not be null");
        *partial += val;
        Ok(())
    }

    fn to_scalar(&self, partial: &Self::Partial) -> VortexResult<Scalar> {
        Ok(Scalar::primitive(*partial, Nullability::NonNullable))
    }

    fn partials_to_array(
        &self,
        partials: &[Self::Partial],
        _partial_dtype: &DType,
    ) -> VortexResult<Option<ArrayRef>> {
        Ok(Some(
            PrimitiveArray::from_iter(partials.iter().copied()).into_array(),
        ))
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
        *state += batch.valid_count(ctx)? as u64;
        Ok(true)
    }

    fn try_accumulate_grouped(
        &self,
        states: &mut [Self::Partial],
        batch: &ArrayRef,
        group_ids: &[u32],
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<bool> {
        grouped::try_accumulate_grouped(states, batch, group_ids, ctx)
    }

    fn accumulate(
        &self,
        _partial: &mut Self::Partial,
        _batch: &Columnar,
        _ctx: &mut ExecutionCtx,
    ) -> VortexResult<()> {
        unreachable!("Count::try_accumulate handles all arrays")
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
    use vortex_buffer::buffer;
    use vortex_error::VortexExpect;
    use vortex_error::VortexResult;

    use crate::ArrayRef;
    use crate::ExecutionCtx;
    use crate::IntoArray;
    use crate::LEGACY_SESSION;
    use crate::VortexSessionExecute;
    use crate::aggregate_fn::Accumulator;
    use crate::aggregate_fn::AggregateFnVTable;
    use crate::aggregate_fn::DynAccumulator;
    use crate::aggregate_fn::DynGroupedAccumulator;
    use crate::aggregate_fn::EmptyOptions;
    use crate::aggregate_fn::GroupedAccumulator;
    use crate::aggregate_fn::fns::count::Count;
    use crate::arrays::ChunkedArray;
    use crate::arrays::ConstantArray;
    use crate::arrays::PrimitiveArray;
    use crate::arrays::VarBinViewArray;
    use crate::assert_arrays_eq;
    use crate::dtype::DType;
    use crate::dtype::Nullability;
    use crate::dtype::PType;
    use crate::scalar::Scalar;
    use crate::validity::Validity;

    pub fn count(array: &ArrayRef, ctx: &mut ExecutionCtx) -> VortexResult<usize> {
        let mut acc = Accumulator::try_new(Count, EmptyOptions, array.dtype().clone())?;
        acc.accumulate(array, ctx)?;
        let result = acc.finish()?;

        Ok(usize::try_from(
            result
                .as_primitive()
                .typed_value::<u64>()
                .vortex_expect("count result should not be null"),
        )?)
    }

    #[test]
    fn count_all_valid() -> VortexResult<()> {
        let array =
            PrimitiveArray::new(buffer![1i32, 2, 3, 4, 5], Validity::NonNullable).into_array();
        let mut ctx = LEGACY_SESSION.create_execution_ctx();
        assert_eq!(count(&array, &mut ctx)?, 5);
        Ok(())
    }

    #[test]
    fn count_with_nulls() -> VortexResult<()> {
        let array = PrimitiveArray::from_option_iter([Some(1i32), None, Some(3), None, Some(5)])
            .into_array();
        let mut ctx = LEGACY_SESSION.create_execution_ctx();
        assert_eq!(count(&array, &mut ctx)?, 3);
        Ok(())
    }

    #[test]
    fn count_all_null() -> VortexResult<()> {
        let array = PrimitiveArray::from_option_iter::<i32, _>([None, None, None]).into_array();
        let mut ctx = LEGACY_SESSION.create_execution_ctx();
        assert_eq!(count(&array, &mut ctx)?, 0);
        Ok(())
    }

    #[test]
    fn count_empty() -> VortexResult<()> {
        let dtype = DType::Primitive(PType::I32, Nullability::NonNullable);
        let mut acc = Accumulator::try_new(Count, EmptyOptions, dtype)?;
        let result = acc.finish()?;
        assert_eq!(result.as_primitive().typed_value::<u64>(), Some(0));
        Ok(())
    }

    #[test]
    fn count_multi_batch() -> VortexResult<()> {
        let mut ctx = LEGACY_SESSION.create_execution_ctx();
        let dtype = DType::Primitive(PType::I32, Nullability::Nullable);
        let mut acc = Accumulator::try_new(Count, EmptyOptions, dtype)?;

        let batch1 = PrimitiveArray::from_option_iter([Some(1i32), None, Some(3)]).into_array();
        acc.accumulate(&batch1, &mut ctx)?;

        let batch2 = PrimitiveArray::from_option_iter([None, Some(5i32)]).into_array();
        acc.accumulate(&batch2, &mut ctx)?;

        let result = acc.finish()?;
        assert_eq!(result.as_primitive().typed_value::<u64>(), Some(3));
        Ok(())
    }

    #[test]
    fn count_finish_resets_state() -> VortexResult<()> {
        let mut ctx = LEGACY_SESSION.create_execution_ctx();
        let dtype = DType::Primitive(PType::I32, Nullability::Nullable);
        let mut acc = Accumulator::try_new(Count, EmptyOptions, dtype)?;

        let batch1 = PrimitiveArray::from_option_iter([Some(1i32), None]).into_array();
        acc.accumulate(&batch1, &mut ctx)?;
        let result1 = acc.finish()?;
        assert_eq!(result1.as_primitive().typed_value::<u64>(), Some(1));

        let batch2 = PrimitiveArray::from_option_iter([Some(2i32), Some(3), None]).into_array();
        acc.accumulate(&batch2, &mut ctx)?;
        let result2 = acc.finish()?;
        assert_eq!(result2.as_primitive().typed_value::<u64>(), Some(2));
        Ok(())
    }

    #[test]
    fn count_state_merge() -> VortexResult<()> {
        let dtype = DType::Primitive(PType::I32, Nullability::NonNullable);
        let mut state = Count.empty_partial(&EmptyOptions, &dtype)?;

        let scalar1 = Scalar::primitive(5u64, Nullability::NonNullable);
        Count.combine_partials(&mut state, scalar1)?;

        let scalar2 = Scalar::primitive(3u64, Nullability::NonNullable);
        Count.combine_partials(&mut state, scalar2)?;

        let result = Count.to_scalar(&state)?;
        Count.reset(&mut state);
        assert_eq!(result.as_primitive().typed_value::<u64>(), Some(8));
        Ok(())
    }

    fn run_grouped_count(
        values: &ArrayRef,
        group_ids: &[u32],
        num_groups: usize,
    ) -> VortexResult<ArrayRef> {
        let mut acc = GroupedAccumulator::try_new(Count, EmptyOptions, values.dtype().clone())?;
        acc.accumulate(
            values,
            group_ids,
            num_groups,
            &mut LEGACY_SESSION.create_execution_ctx(),
        )?;
        acc.finish(num_groups)
    }

    #[test]
    fn grouped_count_dense_ids() -> VortexResult<()> {
        let values =
            PrimitiveArray::from_option_iter([Some(1i32), None, Some(3), Some(4), None, Some(6)])
                .into_array();
        let actual = run_grouped_count(&values, &[0, 0, 1, 1, 2, 2], 3)?;

        let expected = PrimitiveArray::from_iter([1u64, 2, 1]).into_array();
        assert_arrays_eq!(&actual, &expected);
        Ok(())
    }

    #[test]
    fn grouped_count_omitted_group() -> VortexResult<()> {
        let values =
            PrimitiveArray::new(buffer![1i32, 2, 3, 4, 5, 6], Validity::NonNullable).into_array();
        let actual = run_grouped_count(&values, &[0, 0, 1, 2, 2, 2], 4)?;

        let expected = PrimitiveArray::from_iter([2u64, 1, 3, 0]).into_array();
        assert_arrays_eq!(&actual, &expected);
        Ok(())
    }

    #[test]
    fn grouped_count_varbinview_with_nulls() -> VortexResult<()> {
        let values = VarBinViewArray::from_iter_nullable_str([
            Some("a"),
            None,
            Some("bbb"),
            None,
            Some("cc"),
        ])
        .into_array();
        let actual = run_grouped_count(&values, &[0, 0, 1, 1, 2], 3)?;

        let expected = PrimitiveArray::from_iter([1u64, 1, 1]).into_array();
        assert_arrays_eq!(&actual, &expected);
        Ok(())
    }

    #[test]
    fn grouped_count_rejects_out_of_range_group_id() -> VortexResult<()> {
        let values = PrimitiveArray::new(buffer![1i32, 2], Validity::NonNullable).into_array();
        let mut acc = GroupedAccumulator::try_new(Count, EmptyOptions, values.dtype().clone())?;
        let mut ctx = LEGACY_SESSION.create_execution_ctx();

        assert!(acc.accumulate(&values, &[0, 2], 2, &mut ctx).is_err());
        Ok(())
    }

    #[test]
    fn grouped_count_accumulate_partials_and_merge_group() -> VortexResult<()> {
        let dtype = DType::Primitive(PType::I32, Nullability::Nullable);
        let partials = PrimitiveArray::from_iter([2u64, 3, 5]).into_array();
        let mut ctx = LEGACY_SESSION.create_execution_ctx();

        let mut left = GroupedAccumulator::try_new(Count, EmptyOptions, dtype.clone())?;
        left.accumulate_partials(&partials, &[0, 1, 1], 2, &mut ctx)?;

        let mut right = GroupedAccumulator::try_new(Count, EmptyOptions, dtype)?;
        right.merge_group(0, &left, 1)?;

        let actual = right.finish(1)?;
        let expected = PrimitiveArray::from_iter([8u64]).into_array();
        assert_arrays_eq!(&actual, &expected);
        Ok(())
    }

    #[test]
    fn count_constant_non_null() -> VortexResult<()> {
        let array = ConstantArray::new(42i32, 10);
        let mut ctx = LEGACY_SESSION.create_execution_ctx();
        assert_eq!(count(&array.into_array(), &mut ctx)?, 10);
        Ok(())
    }

    #[test]
    fn count_constant_null() -> VortexResult<()> {
        let array = ConstantArray::new(
            Scalar::null(DType::Primitive(PType::I32, Nullability::Nullable)),
            10,
        );
        let mut ctx = LEGACY_SESSION.create_execution_ctx();
        assert_eq!(count(&array.into_array(), &mut ctx)?, 0);
        Ok(())
    }

    #[test]
    fn count_chunked() -> VortexResult<()> {
        let chunk1 = PrimitiveArray::from_option_iter([Some(1i32), None, Some(3)]);
        let chunk2 = PrimitiveArray::from_option_iter([None, Some(5i32), None]);
        let dtype = chunk1.dtype().clone();
        let chunked = ChunkedArray::try_new(vec![chunk1.into_array(), chunk2.into_array()], dtype)?;
        let mut ctx = LEGACY_SESSION.create_execution_ctx();
        assert_eq!(count(&chunked.into_array(), &mut ctx)?, 3);
        Ok(())
    }
}
