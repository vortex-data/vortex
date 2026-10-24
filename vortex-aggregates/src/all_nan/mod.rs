// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_array::ArrayRef;
use vortex_array::Columnar;
use vortex_array::ExecutionCtx;
use vortex_array::aggregate_fn::AggregateArgs;
use vortex_array::aggregate_fn::AggregateFnId;
use vortex_array::aggregate_fn::AggregateFnVTable;
use vortex_array::aggregate_fn::EmptyOptions;
use vortex_array::aggregate_fn::stat_aggregates;
use vortex_array::dtype::DType;
use vortex_array::dtype::Nullability;
use vortex_array::scalar::Scalar;
use vortex_error::VortexResult;
use vortex_session::VortexSession;

use crate::nan_count::columnar_nan_count;
use crate::nan_count::nan_count;

/// Compute whether every value in an array is NaN.
///
/// Like other `all` aggregates, this is vacuously true for empty input.
///
/// This is a pruning aggregate, not just a convenience wrapper around
/// [`NanCount`][crate::nan_count::NanCount]. Pruning aggregates must prove a
/// row-wise fact for every value in the scope, so their partials remain valid when a stats column is
/// sliced or concatenated alongside the data. [`NanCount`][crate::nan_count::NanCount]
/// carries cross-row count information instead, so it is useful as a legacy storage format but not
/// as the pruning expression itself.
#[derive(Clone, Debug)]
pub struct AllNan;

impl AggregateFnVTable for AllNan {
    type Options = EmptyOptions;
    type Partial = bool;

    fn id(&self) -> AggregateFnId {
        *stat_aggregates::ALL_NAN_ID
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
        _args: AggregateArgs<'_, Self::Options>,
    ) -> VortexResult<Self::Partial> {
        Ok(true)
    }

    fn partial_from_scalar(
        &self,
        _args: AggregateArgs<'_, Self::Options>,
        scalar: &Scalar,
    ) -> VortexResult<Self::Partial> {
        bool::try_from(scalar)
    }

    fn merge_partials(
        &self,
        _args: AggregateArgs<'_, Self::Options>,
        first: Self::Partial,
        second: Self::Partial,
    ) -> VortexResult<Self::Partial> {
        Ok(first && second)
    }

    fn to_scalar(
        &self,
        _args: AggregateArgs<'_, Self::Options>,
        partial: &Self::Partial,
    ) -> VortexResult<Scalar> {
        Ok(Scalar::bool(*partial, Nullability::Nullable))
    }

    fn is_saturated(
        &self,
        _args: AggregateArgs<'_, Self::Options>,
        partial: &Self::Partial,
    ) -> bool {
        !*partial
    }

    fn try_accumulate(
        &self,
        _args: AggregateArgs<'_, Self::Options>,
        state: &mut Self::Partial,
        batch: &ArrayRef,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<bool> {
        if !matches!(batch.dtype(), DType::Primitive(ptype, _) if ptype.is_float()) {
            *state = false;
            return Ok(true);
        }

        *state &= nan_count(batch, ctx)? == batch.len();
        Ok(true)
    }

    fn accumulate(
        &self,
        _args: AggregateArgs<'_, Self::Options>,
        partial: &mut Self::Partial,
        batch: &Columnar,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<()> {
        // Normal array dispatch is handled by `try_accumulate`, which always short-circuits.
        // Keep this fallback in sync for direct Columnar accumulation paths.
        if !matches!(batch.dtype(), DType::Primitive(ptype, _) if ptype.is_float()) {
            *partial = false;
            return Ok(());
        }

        *partial &= columnar_nan_count(batch, ctx)? == batch.len() as u64;
        Ok(())
    }

    fn finalize(
        &self,
        _args: AggregateArgs<'_, Self::Options>,
        partials: ArrayRef,
    ) -> VortexResult<ArrayRef> {
        Ok(partials)
    }

    fn finalize_scalar(
        &self,
        args: AggregateArgs<'_, Self::Options>,
        partial: &Self::Partial,
    ) -> VortexResult<Scalar> {
        self.to_scalar(args, partial)
    }
}

#[cfg(test)]
mod tests {

    use std::sync::LazyLock;

    use vortex_array::IntoArray;
    use vortex_array::VortexSessionExecute;
    use vortex_array::aggregate_fn::Accumulator;
    use vortex_array::aggregate_fn::DynAccumulator;
    use vortex_array::aggregate_fn::EmptyOptions;
    use vortex_array::arrays::PrimitiveArray;
    use vortex_array::dtype::DType;
    use vortex_array::dtype::Nullability;
    use vortex_array::dtype::PType;
    use vortex_error::VortexResult;
    use vortex_session::VortexSession;

    use crate::all_nan::AllNan;

    static SESSION: LazyLock<VortexSession> = LazyLock::new(|| {
        let session = VortexSession::empty();
        crate::initialize(&session);
        session
    });

    #[test]
    fn all_nan_aggregate_fn() -> VortexResult<()> {
        let mut ctx = SESSION.create_execution_ctx();
        let dtype = DType::Primitive(PType::F32, Nullability::Nullable);
        let mut acc = Accumulator::try_new(AllNan, EmptyOptions, dtype)?;

        let batch = PrimitiveArray::from_option_iter([Some(f32::NAN), Some(f32::NAN)]).into_array();
        acc.accumulate(&batch, &mut ctx)?;

        assert!(bool::try_from(&acc.finish()?)?);
        Ok(())
    }

    #[test]
    fn all_nan_false_with_non_nan() -> VortexResult<()> {
        let mut ctx = SESSION.create_execution_ctx();
        let dtype = DType::Primitive(PType::F32, Nullability::Nullable);
        let mut acc = Accumulator::try_new(AllNan, EmptyOptions, dtype)?;

        let batch = PrimitiveArray::from_option_iter([Some(f32::NAN), Some(1.0f32)]).into_array();
        acc.accumulate(&batch, &mut ctx)?;

        assert!(!bool::try_from(&acc.finish()?)?);
        Ok(())
    }

    #[test]
    fn all_nan_unsupported_for_non_float_values() -> VortexResult<()> {
        let dtype = DType::Primitive(PType::I32, Nullability::Nullable);
        assert!(Accumulator::try_new(AllNan, EmptyOptions, dtype).is_err());
        Ok(())
    }

    #[test]
    fn all_nan_false_with_null() -> VortexResult<()> {
        let mut ctx = SESSION.create_execution_ctx();
        let dtype = DType::Primitive(PType::F32, Nullability::Nullable);
        let mut acc = Accumulator::try_new(AllNan, EmptyOptions, dtype)?;

        let batch = PrimitiveArray::from_option_iter([Some(f32::NAN), None]).into_array();
        acc.accumulate(&batch, &mut ctx)?;

        assert!(!bool::try_from(&acc.finish()?)?);
        Ok(())
    }

    #[test]
    fn all_nan_true_for_empty_float_values() -> VortexResult<()> {
        let dtype = DType::Primitive(PType::F32, Nullability::Nullable);
        let mut acc = Accumulator::try_new(AllNan, EmptyOptions, dtype)?;

        assert!(bool::try_from(&acc.finish()?)?);
        Ok(())
    }
}
