// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_error::VortexResult;
use vortex_session::registry::CachedId;

use crate::ArrayRef;
use crate::Columnar;
use crate::ExecutionCtx;
use crate::aggregate_fn::Accumulator;
use crate::aggregate_fn::AggregateDTypes;
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
    /// The last non-null value seen so far, or `None` if no non-null value has been observed.
    value: Option<Scalar>,
}

impl AggregateFnVTable for Last {
    type Options = EmptyOptions;
    type Partial = LastPartial;

    fn id(&self) -> AggregateFnId {
        static ID: CachedId = CachedId::new("vortex.last");
        *ID
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
        _dtypes: AggregateDTypes<'_>,
    ) -> VortexResult<Self::Partial> {
        Ok(LastPartial { value: None })
    }

    fn partial_from_scalar(
        &self,
        _options: &Self::Options,
        _dtypes: AggregateDTypes<'_>,
        scalar: Scalar,
    ) -> VortexResult<Self::Partial> {
        // A null partial means the producing accumulator saw nothing valid.
        Ok(LastPartial {
            value: (!scalar.is_null()).then_some(scalar),
        })
    }

    fn merge_partials(
        &self,
        _options: &Self::Options,
        _dtypes: AggregateDTypes<'_>,
        first: Self::Partial,
        second: Self::Partial,
    ) -> VortexResult<Self::Partial> {
        // The later non-empty partial wins; an empty later partial changes nothing.
        Ok(LastPartial {
            value: second.value.or(first.value),
        })
    }

    fn to_scalar(
        &self,
        _options: &Self::Options,
        dtypes: AggregateDTypes<'_>,
        partial: &Self::Partial,
    ) -> VortexResult<Scalar> {
        Ok(match &partial.value {
            Some(v) => v.clone(),
            None => Scalar::null(dtypes.result.clone()),
        })
    }

    #[inline]
    fn is_saturated(
        &self,
        _options: &Self::Options,
        _dtypes: AggregateDTypes<'_>,
        _partial: &Self::Partial,
    ) -> bool {
        // Last can never short-circuit: a later batch can always supersede the current value.
        false
    }

    fn try_accumulate(
        &self,
        _options: &Self::Options,
        _dtypes: AggregateDTypes<'_>,
        partial: &mut Self::Partial,
        batch: &ArrayRef,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<bool> {
        if let Some(idx) = batch.validity()?.execute_mask(batch.len(), ctx)?.last() {
            let scalar = batch.execute_scalar(idx, ctx)?;
            partial.value = Some(scalar.into_nullable());
        }
        Ok(true)
    }

    fn accumulate(
        &self,
        _options: &Self::Options,
        _dtypes: AggregateDTypes<'_>,
        _partial: &mut Self::Partial,
        _batch: &Columnar,
        _ctx: &mut ExecutionCtx,
    ) -> VortexResult<()> {
        unreachable!("Last::try_accumulate handles all arrays")
    }

    fn finalize(
        &self,
        _options: &Self::Options,
        _dtypes: AggregateDTypes<'_>,
        partials: ArrayRef,
    ) -> VortexResult<ArrayRef> {
        Ok(partials)
    }

    fn finalize_scalar(
        &self,
        options: &Self::Options,
        dtypes: AggregateDTypes<'_>,
        partial: &Self::Partial,
    ) -> VortexResult<Scalar> {
        self.to_scalar(options, dtypes, partial)
    }
}

#[cfg(test)]
mod tests {
    use vortex_buffer::buffer;
    use vortex_error::VortexResult;

    use crate::IntoArray;
    use crate::VortexSessionExecute;
    use crate::aggregate_fn::Accumulator;
    use crate::aggregate_fn::AggregateFnVTable;
    use crate::aggregate_fn::DynAccumulator;
    use crate::aggregate_fn::EmptyOptions;
    use crate::aggregate_fn::OwnedAggregateDTypes;
    use crate::aggregate_fn::fns::last::Last;
    use crate::aggregate_fn::fns::last::LastPartial;
    use crate::aggregate_fn::fns::last::last;
    use crate::array_session;
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
        let mut ctx = array_session().create_execution_ctx();
        assert_eq!(last(&array, &mut ctx)?, Scalar::primitive(30i32, Nullable));
        Ok(())
    }

    #[test]
    fn last_skips_trailing_nulls() -> VortexResult<()> {
        let array =
            PrimitiveArray::from_option_iter([Some(7i32), Some(8), None, None]).into_array();
        let mut ctx = array_session().create_execution_ctx();
        assert_eq!(last(&array, &mut ctx)?, Scalar::primitive(8i32, Nullable));
        Ok(())
    }

    #[test]
    fn last_all_null() -> VortexResult<()> {
        let array = PrimitiveArray::from_option_iter::<i32, _>([None, None, None]).into_array();
        let mut ctx = array_session().create_execution_ctx();
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
        let mut ctx = array_session().create_execution_ctx();
        assert_eq!(last(&array, &mut ctx)?, Scalar::primitive(42i32, Nullable));
        Ok(())
    }

    #[test]
    fn last_constant_null() -> VortexResult<()> {
        let dtype = DType::Primitive(PType::I32, Nullable);
        let array = ConstantArray::new(Scalar::null(dtype.clone()), 10).into_array();
        let mut ctx = array_session().create_execution_ctx();
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
        let mut ctx = array_session().create_execution_ctx();
        assert_eq!(last(&array, &mut ctx)?, Scalar::utf8("world", Nullable));
        Ok(())
    }

    #[test]
    fn last_multi_batch_picks_latest_non_null() -> VortexResult<()> {
        let mut ctx = array_session().create_execution_ctx();
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
        let mut ctx = array_session().create_execution_ctx();
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
        let owned = OwnedAggregateDTypes::try_new(&Last, &EmptyOptions, dtype)?;
        let dtypes = owned.borrow();
        let partial_of = |value: Option<Scalar>| LastPartial { value };

        let five = partial_of(Some(Scalar::primitive(5i32, Nullable)));
        let seven = partial_of(Some(Scalar::primitive(7i32, Nullable)));
        // An empty partial must not clobber a prior value.
        let empty = partial_of(None);

        // The last non-empty partial in order replaces the prior values.
        let state = Last.reduce_partials(&EmptyOptions, dtypes, [five, seven, empty])?;
        assert_eq!(
            Last.to_scalar(&EmptyOptions, dtypes, &state)?,
            Scalar::primitive(7i32, Nullable)
        );
        Ok(())
    }

    #[test]
    fn last_chunked() -> VortexResult<()> {
        let chunk1 = PrimitiveArray::from_option_iter([Some(42i32), Some(100)]);
        let chunk2 = PrimitiveArray::from_option_iter::<i32, _>([None, None]);
        let dtype = chunk1.dtype().clone();
        let chunked = ChunkedArray::try_new(vec![chunk1.into_array(), chunk2.into_array()], dtype)?;
        let mut ctx = array_session().create_execution_ctx();
        assert_eq!(
            last(&chunked.into_array(), &mut ctx)?,
            Scalar::primitive(100i32, Nullable)
        );
        Ok(())
    }
}
