// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_error::VortexResult;

use crate::ArrayRef;
use crate::ExecutionCtx;
use crate::aggregate_fn::combined::{BinaryCombined, Combined, PairOptions};
use crate::aggregate_fn::fns::count::Count;
use crate::aggregate_fn::fns::sum::Sum;
use crate::aggregate_fn::{
    Accumulator, AggregateFnId, AggregateFnVTable, DynAccumulator, EmptyOptions,
};
use crate::builtins::ArrayBuiltins;
use crate::dtype::{DType, DecimalDType, MAX_PRECISION, MAX_SCALE, Nullability, PType};
use crate::scalar::Scalar;
use crate::scalar_fn::fns::operators::Operator;

/// Compute the arithmetic mean of an array.
///
/// See [`Mean`] for details.
pub fn mean(array: &ArrayRef, ctx: &mut ExecutionCtx) -> VortexResult<Scalar> {
    let options = PairOptions(EmptyOptions, EmptyOptions);
    let vtable = Mean::combined();

    let coerced_dtype = vtable.coerce_args(&options, array.dtype())?;
    let coerced = array.cast(coerced_dtype.clone())?;

    let mut acc = Accumulator::try_new(vtable, options, coerced_dtype)?;
    acc.accumulate(&coerced, ctx)?;
    acc.finish()
}

/// Compute the arithmetic mean of an array.
///
/// Implemented as `Sum / Count` via [`BinaryCombined`].
///
/// Coercion / return type:
/// - Booleans and primitive numeric types are coerced to `f64` and the result
///   is a nullable `f64`.
/// - Decimals are kept as decimals with widened precision and scale
///   (`+4` each, capped at [`MAX_PRECISION`] / [`MAX_SCALE`]), matching
///   DataFusion's `coerce_avg_type`.
#[derive(Clone, Debug)]
pub struct Mean;

impl Mean {
    pub fn combined() -> Combined<Self> {
        Combined(Mean)
    }
}

impl BinaryCombined for Mean {
    type Left = Sum;
    type Right = Count;

    fn id(&self) -> AggregateFnId {
        AggregateFnId::new_ref("vortex.mean")
    }

    fn left(&self) -> Sum {
        Sum
    }

    fn right(&self) -> Count {
        Count
    }

    fn left_name(&self) -> &'static str {
        "sum"
    }

    fn right_name(&self) -> &'static str {
        "count"
    }

    fn return_dtype(&self, input_dtype: &DType) -> Option<DType> {
        let coerced = coerced_dtype(input_dtype)?;
        // Mean is always nullable: an empty / all-null group returns null.
        Some(coerced.with_nullability(Nullability::Nullable))
    }

    fn finalize(&self, sum: ArrayRef, count: ArrayRef) -> VortexResult<ArrayRef> {
        let count_cast = count.cast(sum.dtype().clone())?;
        sum.binary(count_cast, Operator::Div)
    }

    fn coerce_args(
        &self,
        _options: &PairOptions<<Sum as AggregateFnVTable>::Options, <Count as AggregateFnVTable>::Options>,
        input_dtype: &DType,
    ) -> VortexResult<DType> {
        Ok(coerced_dtype(input_dtype).unwrap_or_else(|| input_dtype.clone()))
    }
}

/// Decide what to coerce the input dtype to before feeding it to `Sum` and `Count`.
///
/// Returns `None` for unsupported input types so callers can fall through.
fn coerced_dtype(input_dtype: &DType) -> Option<DType> {
    match input_dtype {
        DType::Bool(n) | DType::Primitive(_, n) => {
            Some(DType::Primitive(PType::F64, *n))
        }
        DType::Decimal(d, n) => {
            // Mirrors DataFusion's `coerce_avg_type`: precision and scale each
            // grow by 4, capped at the maximum allowed.
            let new_precision = u8::min(MAX_PRECISION, d.precision().saturating_add(4));
            let new_scale = i8::min(MAX_SCALE, d.scale().saturating_add(4));
            Some(DType::Decimal(
                DecimalDType::new(new_precision, new_scale),
                *n,
            ))
        }
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use vortex_buffer::buffer;
    use vortex_error::VortexResult;

    use super::*;
    use crate::IntoArray;
    use crate::LEGACY_SESSION;
    use crate::VortexSessionExecute;
    use crate::arrays::{BoolArray, ChunkedArray, ConstantArray, PrimitiveArray};
    use crate::validity::Validity;

    #[test]
    fn mean_all_valid() -> VortexResult<()> {
        let array =
            PrimitiveArray::new(buffer![1.0f64, 2.0, 3.0, 4.0, 5.0], Validity::NonNullable)
                .into_array();
        let mut ctx = LEGACY_SESSION.create_execution_ctx();
        let result = mean(&array, &mut ctx)?;
        assert_eq!(result.as_primitive().as_::<f64>(), Some(3.0));
        Ok(())
    }

    #[test]
    fn mean_with_nulls() -> VortexResult<()> {
        let array = PrimitiveArray::from_option_iter([Some(2.0f64), None, Some(4.0)]).into_array();
        let mut ctx = LEGACY_SESSION.create_execution_ctx();
        let result = mean(&array, &mut ctx)?;
        assert_eq!(result.as_primitive().as_::<f64>(), Some(3.0));
        Ok(())
    }

    #[test]
    fn mean_integers() -> VortexResult<()> {
        let array = PrimitiveArray::new(buffer![10i32, 20, 30], Validity::NonNullable).into_array();
        let mut ctx = LEGACY_SESSION.create_execution_ctx();
        let result = mean(&array, &mut ctx)?;
        assert_eq!(result.as_primitive().as_::<f64>(), Some(20.0));
        Ok(())
    }

    #[test]
    fn mean_bool() -> VortexResult<()> {
        let array: BoolArray = [true, false, true, true].into_iter().collect();
        let mut ctx = LEGACY_SESSION.create_execution_ctx();
        let result = mean(&array.into_array(), &mut ctx)?;
        assert_eq!(result.as_primitive().as_::<f64>(), Some(0.75));
        Ok(())
    }

    #[test]
    fn mean_constant_non_null() -> VortexResult<()> {
        let array = ConstantArray::new(5.0f64, 4);
        let mut ctx = LEGACY_SESSION.create_execution_ctx();
        let result = mean(&array.into_array(), &mut ctx)?;
        assert_eq!(result.as_primitive().as_::<f64>(), Some(5.0));
        Ok(())
    }

    #[test]
    fn mean_chunked() -> VortexResult<()> {
        let chunk1 = PrimitiveArray::from_option_iter([Some(1.0f64), None, Some(3.0)]);
        let chunk2 = PrimitiveArray::from_option_iter([Some(5.0f64), None]);
        let dtype = chunk1.dtype().clone();
        let chunked = ChunkedArray::try_new(vec![chunk1.into_array(), chunk2.into_array()], dtype)?;
        let mut ctx = LEGACY_SESSION.create_execution_ctx();
        let result = mean(&chunked.into_array(), &mut ctx)?;
        assert_eq!(result.as_primitive().as_::<f64>(), Some(3.0));
        Ok(())
    }

    #[test]
    fn mean_multi_batch() -> VortexResult<()> {
        let mut ctx = LEGACY_SESSION.create_execution_ctx();
        let dtype = DType::Primitive(PType::F64, Nullability::NonNullable);
        let mut acc = Accumulator::try_new(
            Mean::combined(),
            PairOptions(EmptyOptions, EmptyOptions),
            dtype,
        )?;

        let batch1 =
            PrimitiveArray::new(buffer![1.0f64, 2.0, 3.0], Validity::NonNullable).into_array();
        acc.accumulate(&batch1, &mut ctx)?;

        let batch2 = PrimitiveArray::new(buffer![4.0f64, 5.0], Validity::NonNullable).into_array();
        acc.accumulate(&batch2, &mut ctx)?;

        let result = acc.finish()?;
        assert_eq!(result.as_primitive().as_::<f64>(), Some(3.0));
        Ok(())
    }
}
