// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::fmt::Display;
use std::fmt::Formatter;

use vortex_error::VortexResult;
use vortex_error::vortex_err;
use vortex_error::vortex_panic;

use crate::Canonical;
use crate::ExecutionCtx;
use crate::aggregate_fn::AggregateFnId;
use crate::aggregate_fn::AggregateFnVTable;
use crate::dtype::DType;
use crate::dtype::PType;
use crate::expr::stats::Stat;
use crate::scalar::DecimalValue;
use crate::scalar::Scalar;

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

    fn state_reset(&self, state: &mut Self::GroupState) {
        todo!()
    }

    fn state_merge(&self, state: &mut Self::GroupState, other: Scalar) -> VortexResult<()> {
        todo!()
    }

    fn state_result(&self, state: &Self::GroupState) -> Scalar {
        todo!()
    }

    fn state_is_saturated(&self, state: &Self::GroupState) -> bool {
        // On overflow, the state is set to `None`.
        state.is_none()
    }

    fn state_accumulate(
        &self,
        state: &mut Self::GroupState,
        batch: &Canonical,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<()> {
        todo!()
    }
}

/// The state of a sum aggregate function, which may be used to optimize accumulation by
/// short-circuiting
///
// TODO(ngates): instead of an enum, we should use a Box<dyn State> to avoid dispatcher over the
//  input type every time? Perhaps?
pub enum SumState {
    Unsigned(u64),
    Signed(i64),
    Float(f64),
    Decimal(DecimalValue),
}

#[cfg(test)]
mod tests {
    use vortex_buffer::buffer;
    use vortex_error::VortexResult;
    use vortex_session::VortexSession;

    use crate::ArrayRef;
    use crate::IntoArray;
    use crate::aggregate_fn::Accumulator;
    use crate::aggregate_fn::AggregateFnVTable;
    use crate::aggregate_fn::DynAccumulator;
    use crate::aggregate_fn::fns::sum::Sum;
    use crate::aggregate_fn::fns::sum::SumOptions;
    use crate::arrays::BoolArray;
    use crate::arrays::PrimitiveArray;
    use crate::dtype::DType;
    use crate::dtype::Nullability;
    use crate::dtype::PType;
    use crate::scalar::Scalar;
    use crate::validity::Validity;

    fn session() -> VortexSession {
        VortexSession::empty()
    }

    fn checked_opts() -> SumOptions {
        SumOptions { checked: true }
    }

    fn unchecked_opts() -> SumOptions {
        SumOptions { checked: false }
    }

    fn run_sum(batch: &ArrayRef, options: &SumOptions) -> VortexResult<Scalar> {
        let mut acc = Accumulator::try_new(Sum, options.clone(), batch.dtype().clone(), session())?;
        acc.accumulate(batch)?;
        Ok(acc.finish())
    }

    // Primitive sum tests

    #[test]
    fn sum_i32() -> VortexResult<()> {
        let arr = PrimitiveArray::new(buffer![1i32, 2, 3, 4], Validity::NonNullable).into_array();
        let result = run_sum(&arr, &checked_opts())?;
        assert_eq!(result.as_primitive().typed_value::<i64>(), Some(10));
        Ok(())
    }

    #[test]
    fn sum_u8() -> VortexResult<()> {
        let arr = PrimitiveArray::new(buffer![10u8, 20, 30], Validity::NonNullable).into_array();
        let result = run_sum(&arr, &checked_opts())?;
        assert_eq!(result.as_primitive().typed_value::<u64>(), Some(60));
        Ok(())
    }

    #[test]
    fn sum_f64() -> VortexResult<()> {
        let arr =
            PrimitiveArray::new(buffer![1.5f64, 2.5, 3.0], Validity::NonNullable).into_array();
        let result = run_sum(&arr, &checked_opts())?;
        assert_eq!(result.as_primitive().typed_value::<f64>(), Some(7.0));
        Ok(())
    }

    #[test]
    fn sum_with_nulls() -> VortexResult<()> {
        let arr = PrimitiveArray::from_option_iter([Some(2i32), None, Some(4)]).into_array();
        let result = run_sum(&arr, &checked_opts())?;
        assert_eq!(result.as_primitive().typed_value::<i64>(), Some(6));
        Ok(())
    }

    #[test]
    fn sum_all_null() -> VortexResult<()> {
        let arr = PrimitiveArray::from_option_iter([None::<i32>, None, None]).into_array();
        let result = run_sum(&arr, &checked_opts())?;
        assert!(result.is_null());
        Ok(())
    }

    // Empty accumulator tests

    #[test]
    fn sum_empty_produces_zero() -> VortexResult<()> {
        let dtype = DType::Primitive(PType::I32, Nullability::NonNullable);
        let mut acc = Accumulator::try_new(Sum, checked_opts(), dtype, session())?;
        let result = acc.finish();
        assert_eq!(result.as_primitive().typed_value::<i64>(), Some(0));
        Ok(())
    }

    #[test]
    fn sum_empty_f64_produces_zero() -> VortexResult<()> {
        let dtype = DType::Primitive(PType::F64, Nullability::NonNullable);
        let mut acc = Accumulator::try_new(Sum, checked_opts(), dtype, session())?;
        let result = acc.finish();
        assert_eq!(result.as_primitive().typed_value::<f64>(), Some(0.0));
        Ok(())
    }

    // Multi-batch and reset tests

    #[test]
    fn sum_multi_batch() -> VortexResult<()> {
        let dtype = DType::Primitive(PType::I32, Nullability::NonNullable);
        let mut acc = Accumulator::try_new(Sum, checked_opts(), dtype, session())?;

        let batch1 = PrimitiveArray::new(buffer![10i32, 20], Validity::NonNullable).into_array();
        acc.accumulate(&batch1)?;

        let batch2 = PrimitiveArray::new(buffer![3i32, 6, 9], Validity::NonNullable).into_array();
        acc.accumulate(&batch2)?;

        let result = acc.finish();
        assert_eq!(result.as_primitive().typed_value::<i64>(), Some(48));
        Ok(())
    }

    #[test]
    fn sum_finish_resets_state() -> VortexResult<()> {
        let dtype = DType::Primitive(PType::I32, Nullability::NonNullable);
        let mut acc = Accumulator::try_new(Sum, checked_opts(), dtype, session())?;

        let batch1 = PrimitiveArray::new(buffer![10i32, 20], Validity::NonNullable).into_array();
        acc.accumulate(&batch1)?;
        let result1 = acc.finish();
        assert_eq!(result1.as_primitive().typed_value::<i64>(), Some(30));

        let batch2 = PrimitiveArray::new(buffer![3i32, 6, 9], Validity::NonNullable).into_array();
        acc.accumulate(&batch2)?;
        let result2 = acc.finish();
        assert_eq!(result2.as_primitive().typed_value::<i64>(), Some(18));
        Ok(())
    }

    // State merge tests (vtable-level)

    #[test]
    fn sum_state_merge() -> VortexResult<()> {
        let dtype = DType::Primitive(PType::I32, Nullability::NonNullable);
        let mut state = Sum.state_new(&checked_opts(), &dtype)?;

        let scalar1 = Scalar::primitive(100i64, Nullability::Nullable);
        Sum.state_merge(&mut state, scalar1)?;

        let scalar2 = Scalar::primitive(50i64, Nullability::Nullable);
        Sum.state_merge(&mut state, scalar2)?;

        let result = Sum.state_result(&state);
        assert_eq!(result.as_primitive().typed_value::<i64>(), Some(150));
        Ok(())
    }

    // Overflow tests

    #[test]
    fn sum_checked_overflow() -> VortexResult<()> {
        let arr = PrimitiveArray::new(buffer![i64::MAX, 1i64], Validity::NonNullable).into_array();
        let result = run_sum(&arr, &checked_opts())?;
        assert!(result.is_null());
        Ok(())
    }

    #[test]
    fn sum_checked_overflow_is_saturated() -> VortexResult<()> {
        let dtype = DType::Primitive(PType::I64, Nullability::NonNullable);
        let mut acc = Accumulator::try_new(Sum, checked_opts(), dtype, session())?;
        assert!(!acc.is_saturated());

        let batch =
            PrimitiveArray::new(buffer![i64::MAX, 1i64], Validity::NonNullable).into_array();
        acc.accumulate(&batch)?;
        assert!(acc.is_saturated());

        // finish resets state, clearing saturation
        drop(acc.finish());
        assert!(!acc.is_saturated());
        Ok(())
    }

    #[test]
    fn sum_unchecked_wrapping() -> VortexResult<()> {
        let arr = PrimitiveArray::new(buffer![i64::MAX, 1i64], Validity::NonNullable).into_array();
        let result = run_sum(&arr, &unchecked_opts())?;
        assert_eq!(
            result.as_primitive().typed_value::<i64>(),
            Some(i64::MAX.wrapping_add(1))
        );
        Ok(())
    }

    // Boolean sum tests

    #[test]
    fn sum_bool_all_true() -> VortexResult<()> {
        let arr: BoolArray = [true, true, true].into_iter().collect();
        let result = run_sum(&arr.into_array(), &checked_opts())?;
        assert_eq!(result.as_primitive().typed_value::<u64>(), Some(3));
        Ok(())
    }

    #[test]
    fn sum_bool_mixed() -> VortexResult<()> {
        let arr: BoolArray = [true, false, true, false, true].into_iter().collect();
        let result = run_sum(&arr.into_array(), &checked_opts())?;
        assert_eq!(result.as_primitive().typed_value::<u64>(), Some(3));
        Ok(())
    }

    #[test]
    fn sum_bool_all_false() -> VortexResult<()> {
        let arr: BoolArray = [false, false, false].into_iter().collect();
        let result = run_sum(&arr.into_array(), &checked_opts())?;
        assert_eq!(result.as_primitive().typed_value::<u64>(), Some(0));
        Ok(())
    }

    #[test]
    fn sum_bool_with_nulls() -> VortexResult<()> {
        let arr = BoolArray::from_iter([Some(true), None, Some(true), Some(false)]);
        let result = run_sum(&arr.into_array(), &checked_opts())?;
        assert_eq!(result.as_primitive().typed_value::<u64>(), Some(2));
        Ok(())
    }

    #[test]
    fn sum_bool_all_null() -> VortexResult<()> {
        let arr = BoolArray::from_iter([None::<bool>, None, None]);
        let result = run_sum(&arr.into_array(), &checked_opts())?;
        assert!(result.is_null());
        Ok(())
    }

    #[test]
    fn sum_bool_empty_produces_zero() -> VortexResult<()> {
        let dtype = DType::Bool(Nullability::NonNullable);
        let mut acc = Accumulator::try_new(Sum, checked_opts(), dtype, session())?;
        let result = acc.finish();
        assert_eq!(result.as_primitive().typed_value::<u64>(), Some(0));
        Ok(())
    }

    #[test]
    fn sum_bool_finish_resets_state() -> VortexResult<()> {
        let dtype = DType::Bool(Nullability::NonNullable);
        let mut acc = Accumulator::try_new(Sum, checked_opts(), dtype, session())?;

        let batch1: BoolArray = [true, true, false].into_iter().collect();
        acc.accumulate(&batch1.into_array())?;
        let result1 = acc.finish();
        assert_eq!(result1.as_primitive().typed_value::<u64>(), Some(2));

        let batch2: BoolArray = [false, true].into_iter().collect();
        acc.accumulate(&batch2.into_array())?;
        let result2 = acc.finish();
        assert_eq!(result2.as_primitive().typed_value::<u64>(), Some(1));
        Ok(())
    }

    #[test]
    fn sum_bool_return_dtype() -> VortexResult<()> {
        let dtype = Sum.return_dtype(&checked_opts(), &DType::Bool(Nullability::NonNullable))?;
        assert_eq!(dtype, DType::Primitive(PType::U64, Nullability::Nullable));
        Ok(())
    }
}
