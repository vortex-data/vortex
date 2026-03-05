// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::fmt::Display;
use std::fmt::Formatter;

use vortex_error::VortexResult;
use vortex_error::vortex_err;
use vortex_error::vortex_panic;

use crate::aggregate_fn::AggregateFnId;
use crate::aggregate_fn::AggregateFnVTable;
use crate::dtype::DType;
use crate::dtype::PType;
use crate::expr::stats::Stat;
use crate::scalar::DecimalValue;

#[derive(Clone, Debug)]
pub struct Sum;

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct SumOptions {
    checked: bool,
}

impl Display for SumOptions {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        if self.checked {
            write!(f, "checked")
        } else {
            write!(f, "unchecked")
        }
    }
}

impl AggregateFnVTable for Sum {
    type Options = SumOptions;
    type GroupState = Option<SumState>;

    fn id(&self) -> AggregateFnId {
        AggregateFnId::new_ref("vortex.sum")
    }

    fn return_dtype(&self, _options: &Self::Options, input_dtype: &DType) -> VortexResult<DType> {
        Stat::Sum
            .dtype(input_dtype)
            .ok_or_else(|| vortex_err!("Cannot sum {}", input_dtype))
    }

    fn state_dtype(&self, options: &Self::Options, input_dtype: &DType) -> VortexResult<DType> {
        self.return_dtype(options, input_dtype)
    }

    fn state_new(
        &self,
        _options: &Self::Options,
        input_dtype: &DType,
    ) -> VortexResult<Self::GroupState> {
        Ok(Some(
            match Stat::Sum
                .dtype(input_dtype)
                .ok_or_else(|| vortex_err!("Cannot sum {}", input_dtype))?
            {
                DType::Primitive(ptype, _) => match ptype {
                    PType::U8 | PType::U16 | PType::U32 | PType::U64 => SumState::Unsigned(0),
                    PType::I8 | PType::I16 | PType::I32 | PType::I64 => SumState::Signed(0),
                    PType::F16 | PType::F32 | PType::F64 => SumState::Float(0.0),
                },
                DType::Decimal(decimal, _) => SumState::Decimal(DecimalValue::zero(&decimal)),
                _ => vortex_panic!("Unsupported sum types"),
            },
        ))
    }

    fn state_is_saturated(&self, state: &Self::GroupState) -> bool {
        // On overflow, the state is set to `None`.
        state.is_none()
    }
}

/// The state of a sum aggregate function, which may be used to optimize accumulation by
/// short-circuiting
pub enum SumState {
    Unsigned(u64),
    Signed(i64),
    Float(f64),
    Decimal(DecimalValue),
}

#[test]
mod tests {
    use vortex_error::VortexResult;

    use crate::ArrayRef;
    use crate::DynArray;
    use crate::aggregate_fn::Accumulator;
    use crate::aggregate_fn::fns::sum::Sum;
    use crate::aggregate_fn::fns::sum::SumOptions;

    fn run_sum(batch: &ArrayRef, options: &SumOptions) -> VortexResult<ArrayRef> {
        let mut acc = Accumulator::try_new(Sum, options.clone(), batch.dtype().clone())?;
        acc.accumulate(&batch.to_canonical()?)?;
        acc.flush()?;
        acc.finish()
    }

    fn get_i64_value(array: &ArrayRef, idx: usize) -> VortexResult<Option<i64>> {
        let scalar = array.scalar_at(idx)?;
        Ok(scalar.as_primitive().typed_value::<i64>())
    }

    fn get_u64_value(array: &ArrayRef, idx: usize) -> VortexResult<Option<u64>> {
        let scalar = array.scalar_at(idx)?;
        Ok(scalar.as_primitive().typed_value::<u64>())
    }

    fn get_f64_value(array: &ArrayRef, idx: usize) -> VortexResult<Option<f64>> {
        let scalar = array.scalar_at(idx)?;
        Ok(scalar.as_primitive().typed_value::<f64>())
    }

    #[test]
    fn sum_i32() -> VortexResult<()> {
        let arr = PrimitiveArray::new(buffer![1i32, 2, 3, 4], Validity::NonNullable).into_array();
        let result = run_sum(&arr, &checked_opts())?;
        assert_eq!(get_i64_value(&result, 0)?, Some(10));
        Ok(())
    }

    #[test]
    fn sum_u8() -> VortexResult<()> {
        let arr = PrimitiveArray::new(buffer![10u8, 20, 30], Validity::NonNullable).into_array();
        let result = run_sum(&arr, &checked_opts())?;
        assert_eq!(get_u64_value(&result, 0)?, Some(60));
        Ok(())
    }

    #[test]
    fn sum_f64() -> VortexResult<()> {
        let arr =
            PrimitiveArray::new(buffer![1.5f64, 2.5, 3.0], Validity::NonNullable).into_array();
        let result = run_sum(&arr, &checked_opts())?;
        assert_eq!(get_f64_value(&result, 0)?, Some(7.0));
        Ok(())
    }

    #[test]
    fn sum_with_nulls() -> VortexResult<()> {
        let arr = PrimitiveArray::from_option_iter([Some(2i32), None, Some(4)]).into_array();
        let result = run_sum(&arr, &checked_opts())?;
        assert_eq!(get_i64_value(&result, 0)?, Some(6));
        Ok(())
    }

    #[test]
    fn sum_all_null() -> VortexResult<()> {
        let arr = PrimitiveArray::from_option_iter([None::<i32>, None, None]).into_array();
        let result = run_sum(&arr, &checked_opts())?;
        assert_eq!(get_i64_value(&result, 0)?, None);
        Ok(())
    }

    #[test]
    fn sum_empty_flush_produces_zero() -> VortexResult<()> {
        let mut acc = Sum.accumulator(
            &checked_opts(),
            &DType::Primitive(PType::I32, Nullability::NonNullable),
        )?;
        acc.flush()?;
        let result = acc.finish()?;
        assert_eq!(get_i64_value(&result, 0)?, Some(0));
        Ok(())
    }

    #[test]
    fn sum_empty_flush_f64_produces_zero() -> VortexResult<()> {
        let mut acc = Sum.accumulator(
            &checked_opts(),
            &DType::Primitive(PType::F64, Nullability::NonNullable),
        )?;
        acc.flush()?;
        let result = acc.finish()?;
        assert_eq!(get_f64_value(&result, 0)?, Some(0.0));
        Ok(())
    }

    #[test]
    fn sum_multi_group() -> VortexResult<()> {
        let mut acc = Sum.accumulator(
            &checked_opts(),
            &DType::Primitive(PType::I32, Nullability::NonNullable),
        )?;

        let batch1 = PrimitiveArray::new(buffer![10i32, 20], Validity::NonNullable).into_array();
        acc.accumulate(&batch1)?;
        acc.flush()?;

        let batch2 = PrimitiveArray::new(buffer![3i32, 6, 9], Validity::NonNullable).into_array();
        acc.accumulate(&batch2)?;
        acc.flush()?;

        let result = acc.finish()?;
        assert_eq!(get_i64_value(&result, 0)?, Some(30));
        assert_eq!(get_i64_value(&result, 1)?, Some(18));
        Ok(())
    }

    #[test]
    fn sum_merge() -> VortexResult<()> {
        let mut acc = Sum.accumulator(
            &checked_opts(),
            &DType::Primitive(PType::I32, Nullability::NonNullable),
        )?;

        let state1 = Scalar::primitive(100i64, Nullability::Nullable);
        acc.merge(&state1)?;

        let state2 = Scalar::primitive(50i64, Nullability::Nullable);
        acc.merge(&state2)?;

        acc.flush()?;
        let result = acc.finish()?;
        assert_eq!(get_i64_value(&result, 0)?, Some(150));
        Ok(())
    }

    #[test]
    fn sum_checked_overflow() -> VortexResult<()> {
        let arr = PrimitiveArray::new(buffer![i64::MAX, 1i64], Validity::NonNullable).into_array();
        let result = run_sum(&arr, &checked_opts())?;
        assert_eq!(get_i64_value(&result, 0)?, None);
        Ok(())
    }

    #[test]
    fn sum_checked_overflow_is_saturated() -> VortexResult<()> {
        let mut acc = Sum.accumulator(
            &checked_opts(),
            &DType::Primitive(PType::I64, Nullability::NonNullable),
        )?;
        assert!(!acc.is_saturated());

        let batch =
            PrimitiveArray::new(buffer![i64::MAX, 1i64], Validity::NonNullable).into_array();
        acc.accumulate(&batch)?;
        assert!(acc.is_saturated());

        acc.flush()?;
        assert!(!acc.is_saturated());
        Ok(())
    }

    #[test]
    fn sum_unchecked_wrapping() -> VortexResult<()> {
        let arr = PrimitiveArray::new(buffer![i64::MAX, 1i64], Validity::NonNullable).into_array();
        let result = run_sum(&arr, &unchecked_opts())?;
        assert_eq!(get_i64_value(&result, 0)?, Some(i64::MAX.wrapping_add(1)));
        Ok(())
    }

    // Boolean sum tests

    #[test]
    fn sum_bool_all_true() -> VortexResult<()> {
        let arr: BoolArray = [true, true, true].into_iter().collect();
        let result = run_sum(&arr.into_array(), &checked_opts())?;
        assert_eq!(get_u64_value(&result, 0)?, Some(3));
        Ok(())
    }

    #[test]
    fn sum_bool_mixed() -> VortexResult<()> {
        let arr: BoolArray = [true, false, true, false, true].into_iter().collect();
        let result = run_sum(&arr.into_array(), &checked_opts())?;
        assert_eq!(get_u64_value(&result, 0)?, Some(3));
        Ok(())
    }

    #[test]
    fn sum_bool_all_false() -> VortexResult<()> {
        let arr: BoolArray = [false, false, false].into_iter().collect();
        let result = run_sum(&arr.into_array(), &checked_opts())?;
        assert_eq!(get_u64_value(&result, 0)?, Some(0));
        Ok(())
    }

    #[test]
    fn sum_bool_with_nulls() -> VortexResult<()> {
        let arr = BoolArray::from_iter([Some(true), None, Some(true), Some(false)]);
        let result = run_sum(&arr.into_array(), &checked_opts())?;
        assert_eq!(get_u64_value(&result, 0)?, Some(2));
        Ok(())
    }

    #[test]
    fn sum_bool_all_null() -> VortexResult<()> {
        let arr = BoolArray::from_iter([None::<bool>, None, None]);
        let result = run_sum(&arr.into_array(), &checked_opts())?;
        assert_eq!(get_u64_value(&result, 0)?, None);
        Ok(())
    }

    #[test]
    fn sum_bool_empty_flush_produces_zero() -> VortexResult<()> {
        let mut acc = Sum.accumulator(&checked_opts(), &DType::Bool(Nullability::NonNullable))?;
        acc.flush()?;
        let result = acc.finish()?;
        assert_eq!(get_u64_value(&result, 0)?, Some(0));
        Ok(())
    }

    #[test]
    fn sum_bool_multi_group() -> VortexResult<()> {
        let mut acc = Sum.accumulator(&checked_opts(), &DType::Bool(Nullability::NonNullable))?;

        let batch1: BoolArray = [true, true, false].into_iter().collect();
        acc.accumulate(&batch1.into_array())?;
        acc.flush()?;

        let batch2: BoolArray = [false, true].into_iter().collect();
        acc.accumulate(&batch2.into_array())?;
        acc.flush()?;

        let result = acc.finish()?;
        assert_eq!(get_u64_value(&result, 0)?, Some(2));
        assert_eq!(get_u64_value(&result, 1)?, Some(1));
        Ok(())
    }

    #[test]
    fn sum_bool_return_dtype() -> VortexResult<()> {
        let dtype = Sum.return_dtype(&checked_opts(), &DType::Bool(Nullability::NonNullable))?;
        assert_eq!(dtype, DType::Primitive(PType::U64, Nullability::Nullable));
        Ok(())
    }
}
