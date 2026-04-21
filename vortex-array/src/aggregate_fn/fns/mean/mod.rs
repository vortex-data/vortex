// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_error::vortex_bail;
use vortex_error::VortexResult;

use crate::aggregate_fn::combined::BinaryCombined;
use crate::aggregate_fn::combined::Combined;
use crate::aggregate_fn::combined::CombinedOptions;
use crate::aggregate_fn::combined::PairOptions;
use crate::aggregate_fn::fns::count::Count;
use crate::aggregate_fn::fns::sum::Sum;
use crate::aggregate_fn::Accumulator;
use crate::aggregate_fn::AggregateFnId;
use crate::aggregate_fn::AggregateFnVTable;
use crate::aggregate_fn::DynAccumulator;
use crate::aggregate_fn::EmptyOptions;
use crate::builtins::ArrayBuiltins;
use crate::dtype::DType;
use crate::dtype::Nullability;
use crate::dtype::PType;
use crate::scalar::Scalar;
use crate::scalar_fn::fns::operators::Operator;
use crate::ArrayRef;
use crate::ExecutionCtx;

/// Compute the arithmetic avg of an array.
///
/// See [`Avg`] for details.
pub fn avg(array: &ArrayRef, ctx: &mut ExecutionCtx) -> VortexResult<Scalar> {
    let mut acc = Accumulator::try_new(
        Avg::combined(),
        PairOptions(EmptyOptions, EmptyOptions),
        array.dtype().clone(),
    )?;
    acc.accumulate(array, ctx)?;
    acc.finish()
}

/// Compute the arithmetic avg of an array.
///
/// Implemented as `Sum / Count` via [`BinaryCombined`].
///
/// Coercion / return type:
/// - Booleans and primitive numeric types are coerced to `f64` and the result
///   is a nullable `f64`.
/// - Decimals are kept as decimals but not implemented currently
#[derive(Clone, Debug)]
pub struct Avg;

impl Avg {
    pub fn combined() -> Combined<Self> {
        Combined(Avg)
    }
}

impl BinaryCombined for Avg {
    type Left = Sum;
    type Right = Count;

    fn id(&self) -> AggregateFnId {
        AggregateFnId::new("vortex.avg")
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
        Some(avg_output_dtype(input_dtype)?.with_nullability(Nullability::Nullable))
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

    fn finalize_scalar(&self, left_scalar: Scalar, right_scalar: Scalar) -> VortexResult<Scalar> {
        if let DType::Decimal(..) = left_scalar.dtype() {
            vortex_bail!("avg::finalize_scalar not yet implemented for decimal inputs");
        }

        let target = DType::Primitive(PType::F64, Nullability::Nullable);
        let sum_cast = left_scalar.cast(&target)?;
        let count_cast = right_scalar.cast(&target)?;

        let sum = sum_cast.as_primitive().typed_value::<f64>();
        let count = count_cast.as_primitive().typed_value::<f64>();
        let value = match (sum, count) {
            (None, _) | (_, None) | (_, Some(0.0)) => return Ok(Scalar::null(target)), // Sum overflowed
            (Some(s), Some(c)) => s / c,
        };
        Ok(Scalar::primitive(value, Nullability::Nullable))
    }

    fn serialize(&self, _options: &CombinedOptions<Self>) -> VortexResult<Option<Vec<u8>>> {
        unimplemented!("avg is not yet serializable");
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
        // type we're going to compute the avg in.
        Ok(coerced_input_dtype(input_dtype).unwrap_or_else(|| input_dtype.clone()))
    }
}

/// Hint for callers: what to cast the input to before accumulation.
///
/// - Bool stays as bool — `Sum` has a native bool path and bool → f64 isn't
///   currently a direct cast in vortex.
/// - Primitive numerics → `f64` so the sum and finalize work without overflow.
fn coerced_input_dtype(input_dtype: &DType) -> Option<DType> {
    match input_dtype {
        DType::Bool(_) => Some(input_dtype.clone()),
        DType::Primitive(_, n) => Some(DType::Primitive(PType::F64, *n)),
        DType::Decimal(_, _) => {
            unimplemented!("Avg is not implemented for decimals yet")
        }
        _ => None,
    }
}

fn avg_output_dtype(input_dtype: &DType) -> Option<DType> {
    match input_dtype {
        DType::Bool(_) | DType::Primitive(..) => {
            Some(DType::Primitive(PType::F64, Nullability::Nullable))
        }
        DType::Decimal(d, _) => {
            unimplemented!("Avg for decimals is not yet implemented");
        }
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use vortex_buffer::buffer;
    use vortex_error::VortexResult;

    use super::*;
    use crate::arrays::BoolArray;
    use crate::arrays::ChunkedArray;
    use crate::arrays::ConstantArray;
    use crate::arrays::PrimitiveArray;
    use crate::validity::Validity;
    use crate::IntoArray;
    use crate::VortexSessionExecute;
    use crate::LEGACY_SESSION;

    #[test]
    fn avg_all_valid() -> VortexResult<()> {
        let array = PrimitiveArray::new(buffer![1.0f64, 2.0, 3.0, 4.0, 5.0], Validity::NonNullable)
            .into_array();
        let mut ctx = LEGACY_SESSION.create_execution_ctx();
        let result = avg(&array, &mut ctx)?;
        assert_eq!(result.as_primitive().as_::<f64>(), Some(3.0));
        Ok(())
    }

    #[test]
    fn avg_with_nulls() -> VortexResult<()> {
        let array = PrimitiveArray::from_option_iter([Some(2.0f64), None, Some(4.0)]).into_array();
        let mut ctx = LEGACY_SESSION.create_execution_ctx();
        let result = avg(&array, &mut ctx)?;
        assert_eq!(result.as_primitive().as_::<f64>(), Some(3.0));
        Ok(())
    }

    #[test]
    fn avg_integers() -> VortexResult<()> {
        let array = PrimitiveArray::new(buffer![10i32, 20, 30], Validity::NonNullable).into_array();
        let mut ctx = LEGACY_SESSION.create_execution_ctx();
        let result = avg(&array, &mut ctx)?;
        assert_eq!(result.as_primitive().as_::<f64>(), Some(20.0));
        Ok(())
    }

    #[test]
    fn avg_bool() -> VortexResult<()> {
        let array: BoolArray = [true, false, true, true].into_iter().collect();
        let mut ctx = LEGACY_SESSION.create_execution_ctx();
        let result = avg(&array.into_array(), &mut ctx)?;
        assert_eq!(result.as_primitive().as_::<f64>(), Some(0.75));
        Ok(())
    }

    #[test]
    fn avg_constant_non_null() -> VortexResult<()> {
        let array = ConstantArray::new(5.0f64, 4);
        let mut ctx = LEGACY_SESSION.create_execution_ctx();
        let result = avg(&array.into_array(), &mut ctx)?;
        assert_eq!(result.as_primitive().as_::<f64>(), Some(5.0));
        Ok(())
    }

    #[test]
    fn avg_chunked() -> VortexResult<()> {
        let chunk1 = PrimitiveArray::from_option_iter([Some(1.0f64), None, Some(3.0)]);
        let chunk2 = PrimitiveArray::from_option_iter([Some(5.0f64), None]);
        let dtype = chunk1.dtype().clone();
        let chunked = ChunkedArray::try_new(vec![chunk1.into_array(), chunk2.into_array()], dtype)?;
        let mut ctx = LEGACY_SESSION.create_execution_ctx();
        let result = avg(&chunked.into_array(), &mut ctx)?;
        assert_eq!(result.as_primitive().as_::<f64>(), Some(3.0));
        Ok(())
    }


    #[test]
    fn avg_multi_batch() -> VortexResult<()> {
        let mut ctx = LEGACY_SESSION.create_execution_ctx();
        let dtype = DType::Primitive(PType::F64, Nullability::NonNullable);
        let mut acc = Accumulator::try_new(
            Avg::combined(),
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
