// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_error::VortexResult;

use crate::ArrayRef;
use crate::ExecutionCtx;
use crate::aggregate_fn::Accumulator;
use crate::aggregate_fn::AggregateFnId;
use crate::aggregate_fn::AggregateFnVTable;
use crate::aggregate_fn::DynAccumulator;
use crate::aggregate_fn::EmptyOptions;
use crate::aggregate_fn::combined::{BinaryCombined, CombinedOptions};
use crate::aggregate_fn::combined::Combined;
use crate::aggregate_fn::combined::PairOptions;
use crate::aggregate_fn::fns::count::Count;
use crate::aggregate_fn::fns::sum::Sum;
use crate::builtins::ArrayBuiltins;
use crate::dtype::DType;
use crate::dtype::DecimalDType;
use crate::dtype::MAX_PRECISION;
use crate::dtype::MAX_SCALE;
use crate::dtype::Nullability;
use crate::dtype::PType;
use crate::scalar::Scalar;
use crate::scalar_fn::fns::operators::Operator;

/// Compute the arithmetic mean of an array.
///
/// See [`Mean`] for details.
pub fn mean(array: &ArrayRef, ctx: &mut ExecutionCtx) -> VortexResult<Scalar> {
    let mut acc = Accumulator::try_new(
        Mean::combined(),
        PairOptions(EmptyOptions, EmptyOptions),
        array.dtype().clone(),
    )?;
    acc.accumulate(array, ctx)?;
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
        Some(mean_output_dtype(input_dtype)?.with_nullability(Nullability::Nullable))
    }

    fn finalize(&self, sum: ArrayRef, count: ArrayRef) -> VortexResult<ArrayRef> {
        let target = match sum.dtype() {
            DType::Decimal(..) => sum.dtype().with_nullability(Nullability::Nullable),
            _ => DType::Primitive(PType::F64, Nullability::Nullable),
        };
        let sum_cast = sum.cast(target.clone())?;
        let count_cast = count.cast(target)?;
        sum_cast.binary(count_cast, Operator::Div)
    }

    fn serialize(&self, _options: &CombinedOptions<Self>) -> VortexResult<Option<Vec<u8>>> {
        unimplemented!("Mean is not yet serializable");
    }
    
    fn coerce_args(
        &self,
        _options: &PairOptions<
            <Sum as AggregateFnVTable>::Options,
            <Count as AggregateFnVTable>::Options,
        >,
        input_dtype: &DType,
    ) -> VortexResult<DType> {
        // Advisory hint for query planners: where possible, cast input to the
        // type we're going to compute the mean in.
        Ok(coerced_input_dtype(input_dtype).unwrap_or_else(|| input_dtype.clone()))
    }
}

/// Hint for callers: what to cast the input to before accumulation.
///
/// - Bool stays as bool — `Sum` has a native bool path and bool → f64 isn't
///   currently a direct cast in vortex.
/// - Primitive numerics → `f64` so the sum and finalize work without overflow.
/// - Decimals → decimal with widened precision and scale (`+4` each, capped),
///   matching DataFusion's `coerce_avg_type`.
fn coerced_input_dtype(input_dtype: &DType) -> Option<DType> {
    match input_dtype {
        DType::Bool(_) => Some(input_dtype.clone()),
        DType::Primitive(_, n) => Some(DType::Primitive(PType::F64, *n)),
        DType::Decimal(d, n) => {
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

fn mean_output_dtype(input_dtype: &DType) -> Option<DType> {
    match input_dtype {
        DType::Bool(_) | DType::Primitive(..) => {
            Some(DType::Primitive(PType::F64, Nullability::Nullable))
        }
        DType::Decimal(d, _) => {
            let new_precision = u8::min(MAX_PRECISION, d.precision().saturating_add(10));
            Some(DType::Decimal(
                DecimalDType::new(new_precision, d.scale()),
                Nullability::Nullable,
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
    use crate::arrays::BoolArray;
    use crate::arrays::ChunkedArray;
    use crate::arrays::ConstantArray;
    use crate::arrays::DecimalArray;
    use crate::arrays::PrimitiveArray;
    use crate::scalar::DecimalValue;
    use crate::validity::Validity;

    #[test]
    fn mean_all_valid() -> VortexResult<()> {
        let array = PrimitiveArray::new(buffer![1.0f64, 2.0, 3.0, 4.0, 5.0], Validity::NonNullable)
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

    // TODO: vortex's cast kernel doesn't currently support `u64 → decimal`,
    #[test]
    #[ignore = "u64 → decimal cast not yet supported"]
    fn mean_decimal() -> VortexResult<()> {
        // 1.00, 2.00, 3.00 in decimal(5, 2). Mean = 2.00.
        let values = buffer![100i128, 200i128, 300i128];
        let dt = DecimalDType::new(5, 2);
        let array = DecimalArray::new(values, dt, Validity::NonNullable).into_array();
        let mut ctx = LEGACY_SESSION.create_execution_ctx();
        let result = mean(&array, &mut ctx)?;
        // `Sum` widens precision by +10, so the result lives in decimal(15, 2).
        // 2.00 in scale=2 is the integer 200.
        assert_eq!(
            result.as_decimal().decimal_value(),
            Some(DecimalValue::I128(200))
        );
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
