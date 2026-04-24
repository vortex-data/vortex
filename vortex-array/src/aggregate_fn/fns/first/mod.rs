// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_error::VortexResult;

use crate::ArrayRef;
use crate::Columnar;
use crate::ExecutionCtx;
use crate::aggregate_fn::Accumulator;
use crate::aggregate_fn::AggregateFnId;
use crate::aggregate_fn::AggregateFnVTable;
use crate::aggregate_fn::DynAccumulator;
use crate::aggregate_fn::EmptyOptions;
use crate::dtype::DType;
use crate::scalar::Scalar;

/// Return the first non-null value of an array.
///
/// See [`First`] for details.
pub fn first(array: &ArrayRef, ctx: &mut ExecutionCtx) -> VortexResult<Scalar> {
    let mut acc = Accumulator::try_new(First, EmptyOptions, array.dtype().clone())?;
    acc.accumulate(array, ctx)?;
    acc.finish()
}

/// Return the first non-null value seen across all batches.
#[derive(Clone, Debug)]
pub struct First;

/// Partial accumulator state for the [`First`] aggregate.
pub struct FirstPartial {
    /// The nullable version of the input dtype, used for the result and for empty/all-null inputs.
    return_dtype: DType,
    /// The first non-null value seen so far, or `None` if no non-null value has been observed.
    value: Option<Scalar>,
}

impl AggregateFnVTable for First {
    type Options = EmptyOptions;
    type Partial = FirstPartial;

    fn id(&self) -> AggregateFnId {
        AggregateFnId::new("vortex.first")
    }

    fn serialize(&self, _options: &Self::Options) -> VortexResult<Option<Vec<u8>>> {
        unimplemented!("First is not yet serializable");
    }

    fn return_dtype(&self, _options: &Self::Options, input_dtype: &DType) -> Option<DType> {
        Some(input_dtype.as_nullable())
    }

    fn partial_dtype(&self, options: &Self::Options, input_dtype: &DType) -> Option<DType> {
        self.return_dtype(options, input_dtype)
    }

    fn empty_partial(
        &self,
        _options: &Self::Options,
        input_dtype: &DType,
    ) -> VortexResult<Self::Partial> {
        Ok(FirstPartial {
            return_dtype: input_dtype.as_nullable(),
            value: None,
        })
    }

    fn combine_partials(&self, partial: &mut Self::Partial, other: Scalar) -> VortexResult<()> {
        // Only the first non-null partial wins; later ones are ignored.
        if partial.value.is_none() && !other.is_null() {
            partial.value = Some(other);
        }
        Ok(())
    }

    fn to_scalar(&self, partial: &Self::Partial) -> VortexResult<Scalar> {
        Ok(match &partial.value {
            Some(v) => v.clone(),
            None => Scalar::null(partial.return_dtype.clone()),
        })
    }

    fn reset(&self, partial: &mut Self::Partial) {
        partial.value = None;
    }

    #[inline]
    fn is_saturated(&self, partial: &Self::Partial) -> bool {
        partial.value.is_some()
    }

    fn try_accumulate(
        &self,
        partial: &mut Self::Partial,
        batch: &ArrayRef,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<bool> {
        if partial.value.is_some() {
            return Ok(true);
        }
        if let Some(idx) = batch.validity()?.execute_mask(batch.len(), ctx)?.first() {
            let scalar = batch.execute_scalar(idx, ctx)?;
            partial.value = Some(scalar.into_nullable());
        }
        Ok(true)
    }

    fn accumulate(
        &self,
        _partial: &mut Self::Partial,
        _batch: &Columnar,
        _ctx: &mut ExecutionCtx,
    ) -> VortexResult<()> {
        unreachable!("First::try_accumulate handles all arrays")
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
    use vortex_error::VortexResult;

    use crate::IntoArray;
    use crate::LEGACY_SESSION;
    use crate::VortexSessionExecute;
    use crate::aggregate_fn::Accumulator;
    use crate::aggregate_fn::AggregateFnVTable;
    use crate::aggregate_fn::DynAccumulator;
    use crate::aggregate_fn::EmptyOptions;
    use crate::aggregate_fn::fns::first::First;
    use crate::aggregate_fn::fns::first::first;
    use crate::arrays::ChunkedArray;
    use crate::arrays::ConstantArray;
    use crate::arrays::PrimitiveArray;
    use crate::arrays::VarBinArray;
    use crate::dtype::DType;
    use crate::dtype::Nullability;
    use crate::dtype::Nullability::Nullable;
    use crate::dtype::PType;
    use crate::scalar::Scalar;
    use crate::validity::Validity;

    #[test]
    fn first_non_null() -> VortexResult<()> {
        let array = PrimitiveArray::new(buffer![10i32, 20, 30], Validity::NonNullable).into_array();
        let mut ctx = LEGACY_SESSION.create_execution_ctx();
        assert_eq!(first(&array, &mut ctx)?, Scalar::primitive(10i32, Nullable));
        Ok(())
    }

    #[test]
    fn first_skips_leading_nulls() -> VortexResult<()> {
        let array =
            PrimitiveArray::from_option_iter([None, None, Some(7i32), Some(8)]).into_array();
        let mut ctx = LEGACY_SESSION.create_execution_ctx();
        assert_eq!(first(&array, &mut ctx)?, Scalar::primitive(7i32, Nullable));
        Ok(())
    }

    #[test]
    fn first_all_null() -> VortexResult<()> {
        let array = PrimitiveArray::from_option_iter::<i32, _>([None, None, None]).into_array();
        let mut ctx = LEGACY_SESSION.create_execution_ctx();
        let dtype = DType::Primitive(PType::I32, Nullable);
        assert_eq!(first(&array, &mut ctx)?, Scalar::null(dtype));
        Ok(())
    }

    #[test]
    fn first_empty() -> VortexResult<()> {
        let dtype = DType::Primitive(PType::I32, Nullability::NonNullable);
        let mut acc = Accumulator::try_new(First, EmptyOptions, dtype)?;
        let result = acc.finish()?;
        assert_eq!(result, Scalar::null(DType::Primitive(PType::I32, Nullable)));
        Ok(())
    }

    #[test]
    fn first_constant() -> VortexResult<()> {
        let array = ConstantArray::new(42i32, 10).into_array();
        let mut ctx = LEGACY_SESSION.create_execution_ctx();
        assert_eq!(first(&array, &mut ctx)?, Scalar::primitive(42i32, Nullable));
        Ok(())
    }

    #[test]
    fn first_constant_null() -> VortexResult<()> {
        let dtype = DType::Primitive(PType::I32, Nullable);
        let array = ConstantArray::new(Scalar::null(dtype.clone()), 10).into_array();
        let mut ctx = LEGACY_SESSION.create_execution_ctx();
        assert_eq!(first(&array, &mut ctx)?, Scalar::null(dtype));
        Ok(())
    }

    #[test]
    fn first_varbin() -> VortexResult<()> {
        let array = VarBinArray::from_iter(
            vec![None, Some("hello"), Some("world")],
            DType::Utf8(Nullable),
        )
        .into_array();
        let mut ctx = LEGACY_SESSION.create_execution_ctx();
        assert_eq!(first(&array, &mut ctx)?, Scalar::utf8("hello", Nullable));
        Ok(())
    }

    #[test]
    fn first_multi_batch_picks_earliest_non_null() -> VortexResult<()> {
        let mut ctx = LEGACY_SESSION.create_execution_ctx();
        let dtype = DType::Primitive(PType::I32, Nullable);
        let mut acc = Accumulator::try_new(First, EmptyOptions, dtype)?;

        // First batch is all null - should not saturate.
        let batch1 = PrimitiveArray::from_option_iter::<i32, _>([None, None]).into_array();
        acc.accumulate(&batch1, &mut ctx)?;
        assert!(!acc.is_saturated());

        // Second batch contains the first non-null value.
        let batch2 = PrimitiveArray::from_option_iter([None, Some(99i32), Some(100)]).into_array();
        acc.accumulate(&batch2, &mut ctx)?;
        assert!(acc.is_saturated());

        // Third batch must be ignored - First is already saturated.
        let batch3 = PrimitiveArray::from_option_iter([Some(1i32)]).into_array();
        acc.accumulate(&batch3, &mut ctx)?;

        let result = acc.finish()?;
        assert_eq!(result, Scalar::primitive(99i32, Nullable));
        Ok(())
    }

    #[test]
    fn first_finish_resets_state() -> VortexResult<()> {
        let mut ctx = LEGACY_SESSION.create_execution_ctx();
        let dtype = DType::Primitive(PType::I32, Nullability::NonNullable);
        let mut acc = Accumulator::try_new(First, EmptyOptions, dtype)?;

        let batch1 = PrimitiveArray::new(buffer![10i32, 20], Validity::NonNullable).into_array();
        acc.accumulate(&batch1, &mut ctx)?;
        assert_eq!(acc.finish()?, Scalar::primitive(10i32, Nullable));

        let batch2 = PrimitiveArray::new(buffer![3i32, 6, 9], Validity::NonNullable).into_array();
        acc.accumulate(&batch2, &mut ctx)?;
        assert_eq!(acc.finish()?, Scalar::primitive(3i32, Nullable));
        Ok(())
    }

    #[test]
    fn first_state_merge() -> VortexResult<()> {
        let dtype = DType::Primitive(PType::I32, Nullability::NonNullable);
        let mut state = First.empty_partial(&EmptyOptions, &dtype)?;

        // A null partial means the sub-accumulator saw nothing valid - should be ignored.
        First.combine_partials(&mut state, Scalar::null(dtype.as_nullable()))?;
        assert!(!First.is_saturated(&state));

        First.combine_partials(&mut state, Scalar::primitive(5i32, Nullable))?;
        assert!(First.is_saturated(&state));

        // Subsequent valid partials are dropped.
        First.combine_partials(&mut state, Scalar::primitive(7i32, Nullable))?;
        assert_eq!(First.to_scalar(&state)?, Scalar::primitive(5i32, Nullable));
        Ok(())
    }

    #[test]
    fn first_chunked() -> VortexResult<()> {
        let chunk1 = PrimitiveArray::from_option_iter::<i32, _>([None, None]);
        let chunk2 = PrimitiveArray::from_option_iter([None, Some(42i32), Some(100)]);
        let dtype = chunk1.dtype().clone();
        let chunked = ChunkedArray::try_new(vec![chunk1.into_array(), chunk2.into_array()], dtype)?;
        let mut ctx = LEGACY_SESSION.create_execution_ctx();
        assert_eq!(
            first(&chunked.into_array(), &mut ctx)?,
            Scalar::primitive(42i32, Nullable)
        );
        Ok(())
    }
}
