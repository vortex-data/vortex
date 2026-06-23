// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_error::VortexResult;
use vortex_session::VortexSession;

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

/// Compute whether every value in an array is not NaN.
///
/// Like other `all` aggregates, this is vacuously true for empty input.
///
/// This is a pruning aggregate, not just a convenience wrapper around
/// [`NanCount`][crate::aggregate_fn::fns::nan_count::NanCount]. Pruning aggregates must prove a
/// row-wise fact for every value in the scope, so their partials remain valid when a stats column is
/// sliced or concatenated alongside the data. [`NanCount`][crate::aggregate_fn::fns::nan_count::NanCount]
/// carries cross-row count information instead, so it is useful as a legacy storage format but not
/// as the pruning expression itself.
#[derive(Clone, Debug)]
pub struct AllNonNan;

impl AggregateFnVTable for AllNonNan {
    type Options = EmptyOptions;
    type Partial = bool;

    fn id(&self) -> AggregateFnId {
        AggregateFnId::new("vortex.all_non_nan")
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

    fn return_dtype(&self, _options: &Self::Options, input_dtype: &DType) -> Option<DType> {
        matches!(input_dtype, DType::Primitive(ptype, _) if ptype.is_float())
            .then_some(DType::Bool(Nullability::Nullable))
    }

    fn partial_dtype(&self, options: &Self::Options, input_dtype: &DType) -> Option<DType> {
        self.return_dtype(options, input_dtype)
    }

    fn empty_partial(
        &self,
        _options: &Self::Options,
        _input_dtype: &DType,
    ) -> VortexResult<Self::Partial> {
        Ok(true)
    }

    fn combine_partials(&self, partial: &mut Self::Partial, other: Scalar) -> VortexResult<()> {
        *partial &= bool::try_from(&other)?;
        Ok(())
    }

    fn to_scalar(&self, partial: &Self::Partial) -> VortexResult<Scalar> {
        Ok(Scalar::bool(*partial, Nullability::Nullable))
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
        *state &= nan_count(batch, ctx)? == 0;
        Ok(true)
    }

    fn accumulate(
        &self,
        partial: &mut Self::Partial,
        batch: &Columnar,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<()> {
        // Normal array dispatch is handled by `try_accumulate`, which always short-circuits.
        // Keep this fallback in sync for direct Columnar accumulation paths.
        let array = match batch {
            Columnar::Constant(c) => c.clone().into_array(),
            Columnar::Canonical(c) => c.clone().into_array(),
        };
        *partial &= nan_count(&array, ctx)? == 0;
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
    use crate::aggregate_fn::fns::all_non_nan::AllNonNan;
    use crate::array_session;
    use crate::arrays::PrimitiveArray;
    use crate::dtype::DType;
    use crate::dtype::Nullability;
    use crate::dtype::PType;

    #[test]
    fn all_non_nan_aggregate_fn() -> VortexResult<()> {
        let mut ctx = array_session().create_execution_ctx();
        let dtype = DType::Primitive(PType::F32, Nullability::Nullable);
        let mut acc = Accumulator::try_new(AllNonNan, EmptyOptions, dtype)?;

        let batch = PrimitiveArray::from_option_iter([Some(1.0f32), None, Some(3.0)]).into_array();
        acc.accumulate(&batch, &mut ctx)?;

        assert!(bool::try_from(&acc.finish()?)?);
        Ok(())
    }

    #[test]
    fn all_non_nan_false_with_nan() -> VortexResult<()> {
        let mut ctx = array_session().create_execution_ctx();
        let dtype = DType::Primitive(PType::F32, Nullability::Nullable);
        let mut acc = Accumulator::try_new(AllNonNan, EmptyOptions, dtype)?;

        let batch = PrimitiveArray::from_option_iter([Some(1.0f32), Some(f32::NAN)]).into_array();
        acc.accumulate(&batch, &mut ctx)?;

        assert!(!bool::try_from(&acc.finish()?)?);
        Ok(())
    }

    #[test]
    fn all_non_nan_unsupported_for_non_float() -> VortexResult<()> {
        let dtype = DType::Primitive(PType::I32, Nullability::Nullable);
        assert!(Accumulator::try_new(AllNonNan, EmptyOptions, dtype).is_err());
        Ok(())
    }

    #[test]
    fn all_non_nan_true_for_empty_float() -> VortexResult<()> {
        let dtype = DType::Primitive(PType::F32, Nullability::Nullable);
        let mut acc = Accumulator::try_new(AllNonNan, EmptyOptions, dtype)?;

        assert!(bool::try_from(&acc.finish()?)?);
        Ok(())
    }

    #[test]
    fn all_non_nan_true_with_nulls() -> VortexResult<()> {
        let mut ctx = array_session().create_execution_ctx();
        let dtype = DType::Primitive(PType::F32, Nullability::Nullable);
        let mut acc = Accumulator::try_new(AllNonNan, EmptyOptions, dtype)?;

        let batch = PrimitiveArray::from_option_iter([Some(1.0f32), None]).into_array();
        acc.accumulate(&batch, &mut ctx)?;

        assert!(bool::try_from(&acc.finish()?)?);
        Ok(())
    }
}
