// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_error::VortexResult;

use crate::ArrayRef;
use crate::Columnar;
use crate::ExecutionCtx;
use crate::IntoArray;
use crate::aggregate_fn::AggregateFnId;
use crate::aggregate_fn::AggregateFnVTable;
use crate::aggregate_fn::EmptyOptions;
use crate::aggregate_fn::fns::nan_count::nan_count;
use crate::dtype::DType;
use crate::dtype::Nullability;
use crate::scalar::Scalar;

/// Compute whether every value in an array is NaN.
///
/// This is a pruning aggregate, not just a convenience wrapper around
/// [`NanCount`][crate::aggregate_fn::fns::nan_count::NanCount]. Pruning aggregates must prove a
/// row-wise fact for every value in the scope, so their partials remain valid when a stats column is
/// sliced or concatenated alongside the data. [`NanCount`][crate::aggregate_fn::fns::nan_count::NanCount]
/// carries cross-row count information instead, so it is useful as a legacy storage format but not
/// as the pruning expression itself.
#[derive(Clone, Debug)]
pub struct AllNan;

impl AggregateFnVTable for AllNan {
    type Options = EmptyOptions;
    type Partial = bool;

    fn id(&self) -> AggregateFnId {
        AggregateFnId::new("vortex.all_nan")
    }

    fn serialize(&self, _options: &Self::Options) -> VortexResult<Option<Vec<u8>>> {
        Ok(None)
    }

    fn return_dtype(&self, _options: &Self::Options, _input_dtype: &DType) -> Option<DType> {
        Some(DType::Bool(Nullability::NonNullable))
    }

    fn partial_dtype(&self, options: &Self::Options, input_dtype: &DType) -> Option<DType> {
        self.return_dtype(options, input_dtype)
    }

    fn empty_partial(
        &self,
        _options: &Self::Options,
        input_dtype: &DType,
    ) -> VortexResult<Self::Partial> {
        Ok(has_nans(input_dtype))
    }

    fn combine_partials(&self, partial: &mut Self::Partial, other: Scalar) -> VortexResult<()> {
        *partial &= bool::try_from(&other)?;
        Ok(())
    }

    fn to_scalar(&self, partial: &Self::Partial) -> VortexResult<Scalar> {
        Ok(Scalar::bool(*partial, Nullability::NonNullable))
    }

    fn reset(&self, partial: &mut Self::Partial) {
        *partial = true;
    }

    fn is_saturated(&self, partial: &Self::Partial) -> bool {
        !*partial
    }

    fn try_accumulate(
        &self,
        state: &mut Self::Partial,
        batch: &ArrayRef,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<bool> {
        if !has_nans(batch.dtype()) {
            *state = false;
            return Ok(true);
        }

        *state &= nan_count(batch, ctx)? == batch.len();
        Ok(true)
    }

    fn accumulate(
        &self,
        partial: &mut Self::Partial,
        batch: &Columnar,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<()> {
        let array = match batch {
            Columnar::Constant(c) => c.clone().into_array(),
            Columnar::Canonical(c) => c.clone().into_array(),
        };
        if !has_nans(array.dtype()) {
            *partial = false;
            return Ok(());
        }

        *partial &= nan_count(&array, ctx)? == array.len();
        Ok(())
    }

    fn finalize(&self, partials: ArrayRef) -> VortexResult<ArrayRef> {
        Ok(partials)
    }

    fn finalize_scalar(&self, partial: &Self::Partial) -> VortexResult<Scalar> {
        self.to_scalar(partial)
    }
}

fn has_nans(dtype: &DType) -> bool {
    matches!(dtype, DType::Primitive(ptype, _) if ptype.is_float())
}

#[cfg(test)]
mod tests {
    use vortex_error::VortexResult;

    use crate::IntoArray;
    use crate::LEGACY_SESSION;
    use crate::VortexSessionExecute;
    use crate::aggregate_fn::Accumulator;
    use crate::aggregate_fn::DynAccumulator;
    use crate::aggregate_fn::EmptyOptions;
    use crate::aggregate_fn::fns::all_nan::AllNan;
    use crate::arrays::PrimitiveArray;
    use crate::dtype::DType;
    use crate::dtype::Nullability;
    use crate::dtype::PType;

    #[test]
    fn all_nan_aggregate_fn() -> VortexResult<()> {
        let mut ctx = LEGACY_SESSION.create_execution_ctx();
        let dtype = DType::Primitive(PType::F32, Nullability::Nullable);
        let mut acc = Accumulator::try_new(AllNan, EmptyOptions, dtype)?;

        let batch = PrimitiveArray::from_option_iter([Some(f32::NAN), Some(f32::NAN)]).into_array();
        acc.accumulate(&batch, &mut ctx)?;

        assert!(bool::try_from(&acc.finish()?)?);
        Ok(())
    }

    #[test]
    fn all_nan_false_with_non_nan() -> VortexResult<()> {
        let mut ctx = LEGACY_SESSION.create_execution_ctx();
        let dtype = DType::Primitive(PType::F32, Nullability::Nullable);
        let mut acc = Accumulator::try_new(AllNan, EmptyOptions, dtype)?;

        let batch = PrimitiveArray::from_option_iter([Some(f32::NAN), Some(1.0f32)]).into_array();
        acc.accumulate(&batch, &mut ctx)?;

        assert!(!bool::try_from(&acc.finish()?)?);
        Ok(())
    }

    #[test]
    fn all_nan_false_for_non_float_values() -> VortexResult<()> {
        let mut ctx = LEGACY_SESSION.create_execution_ctx();
        let dtype = DType::Primitive(PType::I32, Nullability::Nullable);
        let mut acc = Accumulator::try_new(AllNan, EmptyOptions, dtype)?;

        let batch = PrimitiveArray::from_option_iter([Some(1i32), None]).into_array();
        acc.accumulate(&batch, &mut ctx)?;

        assert!(!bool::try_from(&acc.finish()?)?);
        Ok(())
    }

    #[test]
    fn all_nan_false_for_empty_non_float_values() -> VortexResult<()> {
        let dtype = DType::Primitive(PType::I32, Nullability::Nullable);
        let mut acc = Accumulator::try_new(AllNan, EmptyOptions, dtype)?;

        assert!(!bool::try_from(&acc.finish()?)?);
        Ok(())
    }
}
