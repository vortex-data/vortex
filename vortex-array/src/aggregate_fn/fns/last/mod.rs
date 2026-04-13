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

/// Return the last non-null value of an array.
///
/// See [`Last`] for details.
pub fn last(array: &ArrayRef, ctx: &mut ExecutionCtx) -> VortexResult<Scalar> {
    let mut acc = Accumulator::try_new(Last, EmptyOptions, array.dtype().clone())?;
    acc.accumulate(array, ctx)?;
    acc.finish()
}

/// Return the last non-null value seen across all batches.
#[derive(Clone, Debug)]
pub struct Last;

/// Partial accumulator state for the [`Last`] aggregate.
pub struct LastPartial {
    /// The nullable version of the input dtype, used for the result and for empty/all-null inputs.
    return_dtype: DType,
    /// The last non-null value seen so far, or `None` if no non-null value has been observed.
    value: Option<Scalar>,
}

impl AggregateFnVTable for Last {
    type Options = EmptyOptions;
    type Partial = LastPartial;

    fn id(&self) -> AggregateFnId {
        AggregateFnId::new("vortex.last")
    }

    fn serialize(&self, _options: &Self::Options) -> VortexResult<Option<Vec<u8>>> {
        unimplemented!("Last is not yet serializable");
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
        Ok(LastPartial {
            return_dtype: input_dtype.as_nullable(),
            value: None,
        })
    }

    fn combine_partials(&self, partial: &mut Self::Partial, other: Scalar) -> VortexResult<()> {
        // Each new non-null partial replaces the previous one; nulls are ignored.
        if !other.is_null() {
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
    fn is_saturated(&self, _partial: &Self::Partial) -> bool {
        // Last can never short-circuit: a later batch can always supersede the current value.
        false
    }

    fn try_accumulate(
        &self,
        partial: &mut Self::Partial,
        batch: &ArrayRef,
        _ctx: &mut ExecutionCtx,
    ) -> VortexResult<bool> {
        if let Some(idx) = batch.validity_mask()?.last() {
            let scalar = batch.scalar_at(idx)?;
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
        unreachable!("Last::try_accumulate handles all arrays")
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
    use crate::aggregate_fn::fns::last::Last;
    use crate::aggregate_fn::fns::last::last;
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
    fn last_non_null() -> VortexResult<()> {
        let array = PrimitiveArray::new(buffer![10i32, 20, 30], Validity::NonNullable).into_array();
        let mut ctx = LEGACY_SESSION.create_execution_ctx();
        assert_eq!(last(&array, &mut ctx)?, Scalar::primitive(30i32, Nullable));
        Ok(())
    }

    #[test]
    fn last_skips_trailing_nulls() -> VortexResult<()> {
        let array =
            PrimitiveArray::from_option_iter([Some(7i32), Some(8), None, None]).into_array();
        let mut ctx = LEGACY_SESSION.create_execution_ctx();
        assert_eq!(last(&array, &mut ctx)?, Scalar::primitive(8i32, Nullable));
        Ok(())
    }

    #[test]
    fn last_all_null() -> VortexResult<()> {
        let array = PrimitiveArray::from_option_iter::<i32, _>([None, None, None]).into_array();
        let mut ctx = LEGACY_SESSION.create_execution_ctx();
        let dtype = DType::Primitive(PType::I32, Nullable);
        assert_eq!(last(&array, &mut ctx)?, Scalar::null(dtype));
        Ok(())
    }

    #[test]
    fn last_empty() -> VortexResult<()> {
        let dtype = DType::Primitive(PType::I32, Nullability::NonNullable);
        let mut acc = Accumulator::try_new(Last, EmptyOptions, dtype)?;
        let result = acc.finish()?;
        assert_eq!(result, Scalar::null(DType::Primitive(PType::I32, Nullable)));
        Ok(())
    }

    #[test]
    fn last_constant() -> VortexResult<()> {
        let array = ConstantArray::new(42i32, 10).into_array();
        let mut ctx = LEGACY_SESSION.create_execution_ctx();
        assert_eq!(last(&array, &mut ctx)?, Scalar::primitive(42i32, Nullable));
        Ok(())
    }

    #[test]
    fn last_constant_null() -> VortexResult<()> {
        let dtype = DType::Primitive(PType::I32, Nullable);
        let array = ConstantArray::new(Scalar::null(dtype.clone()), 10).into_array();
        let mut ctx = LEGACY_SESSION.create_execution_ctx();
        assert_eq!(last(&array, &mut ctx)?, Scalar::null(dtype));
        Ok(())
    }

    #[test]
    fn last_varbin() -> VortexResult<()> {
        let array = VarBinArray::from_iter(
            vec![Some("hello"), Some("world"), None],
            DType::Utf8(Nullable),
        )
        .into_array();
        let mut ctx = LEGACY_SESSION.create_execution_ctx();
        assert_eq!(last(&array, &mut ctx)?, Scalar::utf8("world", Nullable));
        Ok(())
    }

    #[test]
    fn last_multi_batch_picks_latest_non_null() -> VortexResult<()> {
        let mut ctx = LEGACY_SESSION.create_execution_ctx();
        let dtype = DType::Primitive(PType::I32, Nullable);
        let mut acc = Accumulator::try_new(Last, EmptyOptions, dtype)?;

        let batch1 = PrimitiveArray::from_option_iter([Some(1i32), Some(2)]).into_array();
        acc.accumulate(&batch1, &mut ctx)?;

        // All-null batch must not clobber the previously-stored value.
        let batch2 = PrimitiveArray::from_option_iter::<i32, _>([None, None]).into_array();
        acc.accumulate(&batch2, &mut ctx)?;

        let batch3 = PrimitiveArray::from_option_iter([Some(99i32), None]).into_array();
        acc.accumulate(&batch3, &mut ctx)?;

        // Last is never saturated; later batches keep updating it.
        assert!(!acc.is_saturated());

        let result = acc.finish()?;
        assert_eq!(result, Scalar::primitive(99i32, Nullable));
        Ok(())
    }

    #[test]
    fn last_finish_resets_state() -> VortexResult<()> {
        let mut ctx = LEGACY_SESSION.create_execution_ctx();
        let dtype = DType::Primitive(PType::I32, Nullability::NonNullable);
        let mut acc = Accumulator::try_new(Last, EmptyOptions, dtype)?;

        let batch1 = PrimitiveArray::new(buffer![10i32, 20], Validity::NonNullable).into_array();
        acc.accumulate(&batch1, &mut ctx)?;
        assert_eq!(acc.finish()?, Scalar::primitive(20i32, Nullable));

        let batch2 = PrimitiveArray::new(buffer![3i32, 6, 9], Validity::NonNullable).into_array();
        acc.accumulate(&batch2, &mut ctx)?;
        assert_eq!(acc.finish()?, Scalar::primitive(9i32, Nullable));
        Ok(())
    }

    #[test]
    fn last_state_merge() -> VortexResult<()> {
        let dtype = DType::Primitive(PType::I32, Nullability::NonNullable);
        let mut state = Last.empty_partial(&EmptyOptions, &dtype)?;

        Last.combine_partials(&mut state, Scalar::primitive(5i32, Nullable))?;
        assert_eq!(Last.to_scalar(&state)?, Scalar::primitive(5i32, Nullable));

        // A later non-null partial replaces the prior value.
        Last.combine_partials(&mut state, Scalar::primitive(7i32, Nullable))?;
        assert_eq!(Last.to_scalar(&state)?, Scalar::primitive(7i32, Nullable));

        // A null partial must not clobber the stored value.
        Last.combine_partials(&mut state, Scalar::null(dtype.as_nullable()))?;
        assert_eq!(Last.to_scalar(&state)?, Scalar::primitive(7i32, Nullable));
        Ok(())
    }

    #[test]
    fn last_chunked() -> VortexResult<()> {
        let chunk1 = PrimitiveArray::from_option_iter([Some(42i32), Some(100)]);
        let chunk2 = PrimitiveArray::from_option_iter::<i32, _>([None, None]);
        let dtype = chunk1.dtype().clone();
        let chunked = ChunkedArray::try_new(vec![chunk1.into_array(), chunk2.into_array()], dtype)?;
        let mut ctx = LEGACY_SESSION.create_execution_ctx();
        assert_eq!(
            last(&chunked.into_array(), &mut ctx)?,
            Scalar::primitive(100i32, Nullable)
        );
        Ok(())
    }
}
