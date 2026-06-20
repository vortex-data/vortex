// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

mod grouped;
pub(crate) use grouped::CountGroupedKernel;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;

use crate::ArrayRef;
use crate::Columnar;
use crate::ExecutionCtx;
use crate::aggregate_fn::AggregateFnId;
use crate::aggregate_fn::AggregateFnVTable;
use crate::aggregate_fn::NumericalAggregateOpts;
use crate::aggregate_fn::fns::nan_count::nan_count;
use crate::dtype::DType;
use crate::dtype::Nullability;
use crate::dtype::PType;
use crate::scalar::Scalar;

/// Count the number of non-null elements in an array.
///
/// Applies to all types. Returns a `u64` count.
/// The identity value is zero.
///
/// For float inputs, NaN handling is controlled by [`NumericalAggregateOpts`]: with `skip_nans` (the
/// default) NaN values are treated as missing and excluded from the count, otherwise they are
/// counted like any other non-null value.
#[derive(Clone, Debug)]
pub struct Count;

/// Partial accumulator state for the count aggregate.
pub struct CountPartial {
    count: u64,
    /// Whether NaN values must be excluded from the count (float input with `skip_nans`).
    exclude_nans: bool,
}

impl AggregateFnVTable for Count {
    type Options = NumericalAggregateOpts;
    type Partial = CountPartial;

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
        options: &Self::Options,
        input_dtype: &DType,
    ) -> VortexResult<Self::Partial> {
        Ok(CountPartial {
            count: 0,
            exclude_nans: options.skip_nans && input_dtype.is_float(),
        })
    }

    fn combine_partials(&self, partial: &mut Self::Partial, other: Scalar) -> VortexResult<()> {
        let val = other
            .as_primitive()
            .typed_value::<u64>()
            .vortex_expect("count partial should not be null");
        partial.count += val;
        Ok(())
    }

    fn to_scalar(&self, partial: &Self::Partial) -> VortexResult<Scalar> {
        Ok(Scalar::primitive(partial.count, Nullability::NonNullable))
    }

    fn reset(&self, partial: &mut Self::Partial) {
        partial.count = 0;
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
        let mut count = batch.valid_count(ctx)? as u64;
        if state.exclude_nans {
            // `nan_count` shortcircuits on an exact `Stat::NaNCount` before scanning the batch.
            count = count.saturating_sub(nan_count(batch, ctx)? as u64);
        }
        state.count += count;
        Ok(true)
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
    use std::sync::LazyLock;

    use vortex_buffer::buffer;
    use vortex_error::VortexExpect;
    use vortex_error::VortexResult;
    use vortex_session::VortexSession;

    use crate::ArrayRef;
    use crate::ExecutionCtx;
    use crate::IntoArray;
    use crate::VortexSessionExecute;
    use crate::aggregate_fn::Accumulator;
    use crate::aggregate_fn::AggregateFnVTable;
    use crate::aggregate_fn::DynAccumulator;
    use crate::aggregate_fn::NumericalAggregateOpts;
    use crate::aggregate_fn::fns::count::Count;
    use crate::arrays::ChunkedArray;
    use crate::arrays::ConstantArray;
    use crate::arrays::PrimitiveArray;
    use crate::dtype::DType;
    use crate::dtype::Nullability;
    use crate::dtype::PType;
    use crate::expr::stats::Precision;
    use crate::expr::stats::Stat;
    use crate::scalar::Scalar;
    use crate::scalar::ScalarValue;
    use crate::validity::Validity;

    static SESSION: LazyLock<VortexSession> = LazyLock::new(vortex_array::array_session);

    pub fn count(array: &ArrayRef, ctx: &mut ExecutionCtx) -> VortexResult<usize> {
        let mut acc = Accumulator::try_new(
            Count,
            NumericalAggregateOpts::default(),
            array.dtype().clone(),
        )?;
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
        let mut ctx = SESSION.create_execution_ctx();
        assert_eq!(count(&array, &mut ctx)?, 5);
        Ok(())
    }

    #[test]
    fn count_with_nulls() -> VortexResult<()> {
        let array = PrimitiveArray::from_option_iter([Some(1i32), None, Some(3), None, Some(5)])
            .into_array();
        let mut ctx = SESSION.create_execution_ctx();
        assert_eq!(count(&array, &mut ctx)?, 3);
        Ok(())
    }

    #[test]
    fn count_all_null() -> VortexResult<()> {
        let array = PrimitiveArray::from_option_iter::<i32, _>([None, None, None]).into_array();
        let mut ctx = SESSION.create_execution_ctx();
        assert_eq!(count(&array, &mut ctx)?, 0);
        Ok(())
    }

    #[test]
    fn count_empty() -> VortexResult<()> {
        let dtype = DType::Primitive(PType::I32, Nullability::NonNullable);
        let mut acc = Accumulator::try_new(Count, NumericalAggregateOpts::default(), dtype)?;
        let result = acc.finish()?;
        assert_eq!(result.as_primitive().typed_value::<u64>(), Some(0));
        Ok(())
    }

    #[test]
    fn count_multi_batch() -> VortexResult<()> {
        let mut ctx = SESSION.create_execution_ctx();
        let dtype = DType::Primitive(PType::I32, Nullability::Nullable);
        let mut acc = Accumulator::try_new(Count, NumericalAggregateOpts::default(), dtype)?;

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
        let mut ctx = SESSION.create_execution_ctx();
        let dtype = DType::Primitive(PType::I32, Nullability::Nullable);
        let mut acc = Accumulator::try_new(Count, NumericalAggregateOpts::default(), dtype)?;

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
        let mut state = Count.empty_partial(&NumericalAggregateOpts::default(), &dtype)?;

        let scalar1 = Scalar::primitive(5u64, Nullability::NonNullable);
        Count.combine_partials(&mut state, scalar1)?;

        let scalar2 = Scalar::primitive(3u64, Nullability::NonNullable);
        Count.combine_partials(&mut state, scalar2)?;

        let result = Count.to_scalar(&state)?;
        Count.reset(&mut state);
        assert_eq!(result.as_primitive().typed_value::<u64>(), Some(8));
        Ok(())
    }

    fn count_with_options(
        array: &ArrayRef,
        ctx: &mut ExecutionCtx,
        options: NumericalAggregateOpts,
    ) -> VortexResult<u64> {
        let mut acc = Accumulator::try_new(Count, options, array.dtype().clone())?;
        acc.accumulate(array, ctx)?;
        Ok(acc
            .finish()?
            .as_primitive()
            .typed_value::<u64>()
            .vortex_expect("count result should not be null"))
    }

    #[test]
    fn count_float_excludes_nans_by_default() -> VortexResult<()> {
        let array =
            PrimitiveArray::from_option_iter([Some(1.0f64), Some(f64::NAN), None, Some(3.0)])
                .into_array();
        let mut ctx = SESSION.create_execution_ctx();
        assert_eq!(count(&array, &mut ctx)?, 2);
        Ok(())
    }

    #[test]
    fn count_float_includes_nans_when_not_skipping() -> VortexResult<()> {
        let array =
            PrimitiveArray::from_option_iter([Some(1.0f64), Some(f64::NAN), None, Some(3.0)])
                .into_array();
        let mut ctx = SESSION.create_execution_ctx();
        assert_eq!(
            count_with_options(&array, &mut ctx, NumericalAggregateOpts::include_nans())?,
            3
        );
        Ok(())
    }

    #[test]
    fn count_float_shortcircuits_on_exact_nan_count_stat() -> VortexResult<()> {
        // The array has no NaNs; a planted exact NaNCount stat proves the count is derived from
        // the stat rather than a scan.
        let array =
            PrimitiveArray::new(buffer![1.0f64, 2.0, 3.0, 4.0], Validity::NonNullable).into_array();
        array
            .statistics()
            .set(Stat::NaNCount, Precision::Exact(ScalarValue::from(3u64)));
        let mut ctx = SESSION.create_execution_ctx();
        assert_eq!(count(&array, &mut ctx)?, 1);
        Ok(())
    }

    #[test]
    fn count_constant_nan() -> VortexResult<()> {
        let array = ConstantArray::new(f64::NAN, 5).into_array();
        let mut ctx = SESSION.create_execution_ctx();
        assert_eq!(count(&array, &mut ctx)?, 0);
        assert_eq!(
            count_with_options(&array, &mut ctx, NumericalAggregateOpts::include_nans())?,
            5
        );
        Ok(())
    }

    #[test]
    fn count_constant_non_null() -> VortexResult<()> {
        let array = ConstantArray::new(42i32, 10);
        let mut ctx = SESSION.create_execution_ctx();
        assert_eq!(count(&array.into_array(), &mut ctx)?, 10);
        Ok(())
    }

    #[test]
    fn count_constant_null() -> VortexResult<()> {
        let array = ConstantArray::new(
            Scalar::null(DType::Primitive(PType::I32, Nullability::Nullable)),
            10,
        );
        let mut ctx = SESSION.create_execution_ctx();
        assert_eq!(count(&array.into_array(), &mut ctx)?, 0);
        Ok(())
    }

    #[test]
    fn count_chunked() -> VortexResult<()> {
        let chunk1 = PrimitiveArray::from_option_iter([Some(1i32), None, Some(3)]);
        let chunk2 = PrimitiveArray::from_option_iter([None, Some(5i32), None]);
        let dtype = chunk1.dtype().clone();
        let chunked = ChunkedArray::try_new(vec![chunk1.into_array(), chunk2.into_array()], dtype)?;
        let mut ctx = SESSION.create_execution_ctx();
        assert_eq!(count(&chunked.into_array(), &mut ctx)?, 3);
        Ok(())
    }
}
