// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

mod primitive;

use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_err;

use self::primitive::accumulate_primitive;
use crate::ArrayRef;
use crate::Canonical;
use crate::Columnar;
use crate::ExecutionCtx;
use crate::aggregate_fn::Accumulator;
use crate::aggregate_fn::AggregateFnId;
use crate::aggregate_fn::AggregateFnVTable;
use crate::aggregate_fn::DynAccumulator;
use crate::aggregate_fn::EmptyOptions;
use crate::dtype::DType;
use crate::dtype::Nullability::NonNullable;
use crate::dtype::PType;
use crate::expr::stats::Precision;
use crate::expr::stats::Stat;
use crate::expr::stats::StatsProvider;
use crate::scalar::Scalar;
use crate::scalar::ScalarValue;

/// Return the number of NaN values in an array.
///
/// See [`NanCount`] for details.
pub fn nan_count(array: &ArrayRef, ctx: &mut ExecutionCtx) -> VortexResult<usize> {
    // Short-circuit using cached array statistics.
    if let Some(Precision::Exact(nan_count_scalar)) = array.statistics().get(Stat::NaNCount) {
        return usize::try_from(&nan_count_scalar)
            .map_err(|e| vortex_err!("Failed to convert NaN count stat to usize: {e}"));
    }

    // Short-circuit for non-float types.
    if NanCount
        .return_dtype(&EmptyOptions, array.dtype())
        .is_none()
    {
        return Ok(0);
    }

    // Short-circuit for empty arrays or all-null arrays.
    if array.is_empty() || array.valid_count(ctx)? == 0 {
        return Ok(0);
    }

    // Compute using Accumulator<NanCount>.
    let mut acc = Accumulator::try_new(NanCount, EmptyOptions, array.dtype().clone())?;
    acc.accumulate(array, ctx)?;
    let result = acc.finish()?;

    let count = result
        .as_primitive()
        .typed_value::<u64>()
        .vortex_expect("nan_count result should not be null");
    let count_usize = usize::try_from(count).vortex_expect("Cannot be more nans than usize::MAX");

    // Cache the computed NaN count as a statistic.
    array
        .statistics()
        .set(Stat::NaNCount, Precision::Exact(ScalarValue::from(count)));

    Ok(count_usize)
}

/// Count the number of NaN values in an array.
///
/// Only applies to floating-point primitive types. Returns a `u64` count.
/// If the array is all-invalid, the NaN count is zero.
#[derive(Clone, Debug)]
pub struct NanCount;

impl AggregateFnVTable for NanCount {
    type Options = EmptyOptions;
    type Partial = u64;

    fn id(&self) -> AggregateFnId {
        AggregateFnId::new("vortex.nan_count")
    }

    fn serialize(&self, _options: &Self::Options) -> VortexResult<Option<Vec<u8>>> {
        unimplemented!("NanCount is not yet serializable");
    }

    fn return_dtype(&self, _options: &Self::Options, input_dtype: &DType) -> Option<DType> {
        if let DType::Primitive(ptype, ..) = input_dtype
            && ptype.is_float()
        {
            Some(DType::Primitive(PType::U64, NonNullable))
        } else {
            None
        }
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
            .vortex_expect("nan_count partial should not be null");
        *partial += val;
        Ok(())
    }

    fn to_scalar(&self, partial: &Self::Partial) -> VortexResult<Scalar> {
        Ok(Scalar::primitive(*partial, NonNullable))
    }

    fn reset(&self, partial: &mut Self::Partial) {
        *partial = 0;
    }

    #[inline]
    fn is_saturated(&self, _partial: &Self::Partial) -> bool {
        false
    }

    fn accumulate(
        &self,
        partial: &mut Self::Partial,
        batch: &Columnar,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<()> {
        match batch {
            Columnar::Constant(c) => {
                if c.scalar().is_null() {
                    // Null values are not NaN.
                    return Ok(());
                }
                if c.scalar().as_primitive().is_nan() {
                    *partial += c.len() as u64;
                }
                Ok(())
            }
            Columnar::Canonical(c) => match c {
                Canonical::Primitive(p) => accumulate_primitive(partial, p, ctx),
                _ => vortex_bail!(
                    "Unsupported canonical type for nan_count: {}",
                    batch.dtype()
                ),
            },
        }
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
    use crate::aggregate_fn::fns::nan_count::NanCount;
    use crate::aggregate_fn::fns::nan_count::nan_count;
    use crate::arrays::ChunkedArray;
    use crate::arrays::ConstantArray;
    use crate::arrays::PrimitiveArray;
    use crate::dtype::DType;
    use crate::dtype::Nullability;
    use crate::dtype::PType;
    use crate::scalar::Scalar;
    use crate::validity::Validity;

    #[test]
    fn nan_count_multi_batch() -> VortexResult<()> {
        let mut ctx = LEGACY_SESSION.create_execution_ctx();
        let dtype = DType::Primitive(PType::F64, Nullability::NonNullable);
        let mut acc = Accumulator::try_new(NanCount, EmptyOptions, dtype)?;

        let batch1 =
            PrimitiveArray::new(buffer![f64::NAN, 1.0f64, f64::NAN], Validity::NonNullable)
                .into_array();
        acc.accumulate(&batch1, &mut ctx)?;

        let batch2 =
            PrimitiveArray::new(buffer![2.0f64, f64::NAN], Validity::NonNullable).into_array();
        acc.accumulate(&batch2, &mut ctx)?;

        let result = acc.finish()?;
        assert_eq!(result.as_primitive().typed_value::<u64>(), Some(3));
        Ok(())
    }

    #[test]
    fn nan_count_finish_resets_state() -> VortexResult<()> {
        let mut ctx = LEGACY_SESSION.create_execution_ctx();
        let dtype = DType::Primitive(PType::F64, Nullability::NonNullable);
        let mut acc = Accumulator::try_new(NanCount, EmptyOptions, dtype)?;

        let batch1 =
            PrimitiveArray::new(buffer![f64::NAN, 1.0f64], Validity::NonNullable).into_array();
        acc.accumulate(&batch1, &mut ctx)?;
        let result1 = acc.finish()?;
        assert_eq!(result1.as_primitive().typed_value::<u64>(), Some(1));

        let batch2 = PrimitiveArray::new(buffer![f64::NAN, f64::NAN, 2.0], Validity::NonNullable)
            .into_array();
        acc.accumulate(&batch2, &mut ctx)?;
        let result2 = acc.finish()?;
        assert_eq!(result2.as_primitive().typed_value::<u64>(), Some(2));
        Ok(())
    }

    #[test]
    fn nan_count_state_merge() -> VortexResult<()> {
        let dtype = DType::Primitive(PType::F64, Nullability::NonNullable);
        let mut state = NanCount.empty_partial(&EmptyOptions, &dtype)?;

        let scalar1 = Scalar::primitive(5u64, Nullability::NonNullable);
        NanCount.combine_partials(&mut state, scalar1)?;

        let scalar2 = Scalar::primitive(3u64, Nullability::NonNullable);
        NanCount.combine_partials(&mut state, scalar2)?;

        let result = NanCount.to_scalar(&state)?;
        NanCount.reset(&mut state);
        assert_eq!(result.as_primitive().typed_value::<u64>(), Some(8));
        Ok(())
    }

    #[test]
    fn nan_count_constant_nan() -> VortexResult<()> {
        let array = ConstantArray::new(f64::NAN, 10);
        let mut ctx = LEGACY_SESSION.create_execution_ctx();
        assert_eq!(nan_count(&array.into_array(), &mut ctx)?, 10);
        Ok(())
    }

    #[test]
    fn nan_count_constant_non_nan() -> VortexResult<()> {
        let array = ConstantArray::new(1.0f64, 10);
        let mut ctx = LEGACY_SESSION.create_execution_ctx();
        assert_eq!(nan_count(&array.into_array(), &mut ctx)?, 0);
        Ok(())
    }

    #[test]
    fn nan_count_empty() -> VortexResult<()> {
        let dtype = DType::Primitive(PType::F64, Nullability::NonNullable);
        let mut acc = Accumulator::try_new(NanCount, EmptyOptions, dtype)?;
        let result = acc.finish()?;
        assert_eq!(result.as_primitive().typed_value::<u64>(), Some(0));
        Ok(())
    }

    #[test]
    fn nan_count_chunked() -> VortexResult<()> {
        let chunk1 = PrimitiveArray::from_option_iter([Some(f64::NAN), None, Some(1.0)]);
        let chunk2 = PrimitiveArray::from_option_iter([Some(f64::NAN), Some(f64::NAN), None]);
        let dtype = chunk1.dtype().clone();
        let chunked = ChunkedArray::try_new(vec![chunk1.into_array(), chunk2.into_array()], dtype)?;
        let mut ctx = LEGACY_SESSION.create_execution_ctx();
        assert_eq!(nan_count(&chunked.into_array(), &mut ctx)?, 3);
        Ok(())
    }

    #[test]
    fn nan_count_all_null() -> VortexResult<()> {
        let p = PrimitiveArray::from_option_iter::<f64, _>([None, None, None]);
        let mut ctx = LEGACY_SESSION.create_execution_ctx();
        assert_eq!(nan_count(&p.into_array(), &mut ctx)?, 0);
        Ok(())
    }
}
