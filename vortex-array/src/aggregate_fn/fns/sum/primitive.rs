// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use itertools::Itertools;
use num_traits::ToPrimitive;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_error::vortex_panic;
use vortex_mask::AllOr;

use super::SumState;
use super::checked_add_i64;
use super::checked_add_u64;
use crate::ExecutionCtx;
use crate::arrays::PrimitiveArray;
use crate::match_each_native_ptype;

pub(super) fn accumulate_primitive(
    inner: &mut SumState,
    p: &PrimitiveArray,
    ctx: &mut ExecutionCtx,
) -> VortexResult<bool> {
    let mask = p.as_ref().validity()?.execute_mask(p.as_ref().len(), ctx)?;
    match mask.bit_buffer() {
        AllOr::None => Ok(false),
        AllOr::All => accumulate_primitive_all(inner, p),
        AllOr::Some(validity) => accumulate_primitive_valid(inner, p, validity),
    }
}

fn accumulate_primitive_all(inner: &mut SumState, p: &PrimitiveArray) -> VortexResult<bool> {
    match inner {
        SumState::Unsigned(acc) => match_each_native_ptype!(p.ptype(),
            unsigned: |T| {
                for &v in p.as_slice::<T>() {
                    if checked_add_u64(acc, v.to_u64().vortex_expect("unsigned to u64")) {
                        return Ok(true);
                    }
                }
                Ok(false)
            },
            signed: |_T| { vortex_panic!("unsigned sum state with signed input") },
            floating: |_T| { vortex_panic!("unsigned sum state with float input") }
        ),
        SumState::Signed(acc) => match_each_native_ptype!(p.ptype(),
            unsigned: |_T| { vortex_panic!("signed sum state with unsigned input") },
            signed: |T| {
                for &v in p.as_slice::<T>() {
                    if checked_add_i64(acc, v.to_i64().vortex_expect("signed to i64")) {
                        return Ok(true);
                    }
                }
                Ok(false)
            },
            floating: |_T| { vortex_panic!("signed sum state with float input") }
        ),
        SumState::Float(acc) => match_each_native_ptype!(p.ptype(),
            unsigned: |_T| { vortex_panic!("float sum state with unsigned input") },
            signed: |_T| { vortex_panic!("float sum state with signed input") },
            floating: |T| {
                for &v in p.as_slice::<T>() {
                    if !v.is_nan() {
                        *acc += ToPrimitive::to_f64(&v).vortex_expect("float to f64");
                    }
                }
                Ok(false)
            }
        ),
        SumState::Decimal { .. } => vortex_panic!("decimal sum state with primitive input"),
    }
}

fn accumulate_primitive_valid(
    inner: &mut SumState,
    p: &PrimitiveArray,
    validity: &vortex_buffer::BitBuffer,
) -> VortexResult<bool> {
    match inner {
        SumState::Unsigned(acc) => match_each_native_ptype!(p.ptype(),
            unsigned: |T| {
                for (&v, valid) in p.as_slice::<T>().iter().zip_eq(validity.iter()) {
                    if valid && checked_add_u64(acc, v.to_u64().vortex_expect("unsigned to u64")) {
                        return Ok(true);
                    }
                }
                Ok(false)
            },
            signed: |_T| { vortex_panic!("unsigned sum state with signed input") },
            floating: |_T| { vortex_panic!("unsigned sum state with float input") }
        ),
        SumState::Signed(acc) => match_each_native_ptype!(p.ptype(),
            unsigned: |_T| { vortex_panic!("signed sum state with unsigned input") },
            signed: |T| {
                for (&v, valid) in p.as_slice::<T>().iter().zip_eq(validity.iter()) {
                    if valid && checked_add_i64(acc, v.to_i64().vortex_expect("signed to i64")) {
                        return Ok(true);
                    }
                }
                Ok(false)
            },
            floating: |_T| { vortex_panic!("signed sum state with float input") }
        ),
        SumState::Float(acc) => match_each_native_ptype!(p.ptype(),
            unsigned: |_T| { vortex_panic!("float sum state with unsigned input") },
            signed: |_T| { vortex_panic!("float sum state with signed input") },
            floating: |T| {
                for (&v, valid) in p.as_slice::<T>().iter().zip_eq(validity.iter()) {
                    if valid && !v.is_nan() {
                        *acc += ToPrimitive::to_f64(&v).vortex_expect("float to f64");
                    }
                }
                Ok(false)
            }
        ),
        SumState::Decimal { .. } => vortex_panic!("decimal sum state with primitive input"),
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
    use crate::aggregate_fn::DynAccumulator;
    use crate::aggregate_fn::EmptyOptions;
    use crate::aggregate_fn::fns::sum::Sum;
    use crate::aggregate_fn::fns::sum::sum;
    use crate::arrays::PrimitiveArray;
    use crate::dtype::DType;
    use crate::dtype::Nullability;
    use crate::dtype::Nullability::Nullable;
    use crate::dtype::PType;
    use crate::scalar::Scalar;
    use crate::validity::Validity;

    #[test]
    fn sum_i32() -> VortexResult<()> {
        let arr = PrimitiveArray::new(buffer![1i32, 2, 3, 4], Validity::NonNullable).into_array();
        let result = sum(&arr, &mut LEGACY_SESSION.create_execution_ctx())?;
        assert_eq!(result.as_primitive().typed_value::<i64>(), Some(10));
        Ok(())
    }

    #[test]
    fn sum_u8() -> VortexResult<()> {
        let arr = PrimitiveArray::new(buffer![10u8, 20, 30], Validity::NonNullable).into_array();
        let result = sum(&arr, &mut LEGACY_SESSION.create_execution_ctx())?;
        assert_eq!(result.as_primitive().typed_value::<u64>(), Some(60));
        Ok(())
    }

    #[test]
    fn sum_f64() -> VortexResult<()> {
        let arr =
            PrimitiveArray::new(buffer![1.5f64, 2.5, 3.0], Validity::NonNullable).into_array();
        let result = sum(&arr, &mut LEGACY_SESSION.create_execution_ctx())?;
        assert_eq!(result.as_primitive().typed_value::<f64>(), Some(7.0));
        Ok(())
    }

    #[test]
    fn sum_with_nulls() -> VortexResult<()> {
        let arr = PrimitiveArray::from_option_iter([Some(2i32), None, Some(4)]).into_array();
        let result = sum(&arr, &mut LEGACY_SESSION.create_execution_ctx())?;
        assert_eq!(result.as_primitive().typed_value::<i64>(), Some(6));
        Ok(())
    }

    #[test]
    fn sum_all_null() -> VortexResult<()> {
        let arr = PrimitiveArray::from_option_iter([None::<i32>, None, None]).into_array();
        let result = sum(&arr, &mut LEGACY_SESSION.create_execution_ctx())?;
        assert_eq!(result.as_primitive().typed_value::<i64>(), Some(0));
        Ok(())
    }

    #[test]
    fn sum_all_invalid_float() -> VortexResult<()> {
        let arr = PrimitiveArray::from_option_iter::<f32, _>([None, None, None]).into_array();
        let result = sum(&arr, &mut LEGACY_SESSION.create_execution_ctx())?;
        assert_eq!(result, Scalar::primitive(0f64, Nullable));
        Ok(())
    }

    #[test]
    fn sum_buffer_i32() -> VortexResult<()> {
        let arr = buffer![1, 1, 1, 1].into_array();
        let result = sum(&arr, &mut LEGACY_SESSION.create_execution_ctx())?;
        assert_eq!(result.as_primitive().as_::<i32>(), Some(4));
        Ok(())
    }

    #[test]
    fn sum_buffer_f64() -> VortexResult<()> {
        let arr = buffer![1., 1., 1., 1.].into_array();
        let result = sum(&arr, &mut LEGACY_SESSION.create_execution_ctx())?;
        assert_eq!(result.as_primitive().as_::<f32>(), Some(4.));
        Ok(())
    }

    #[test]
    fn sum_empty_produces_zero() -> VortexResult<()> {
        let dtype = DType::Primitive(PType::I32, Nullability::NonNullable);
        let mut acc = Accumulator::try_new(Sum, EmptyOptions, dtype)?;
        let result = acc.finish()?;
        assert_eq!(result.as_primitive().typed_value::<i64>(), Some(0));
        Ok(())
    }

    #[test]
    fn sum_empty_f64_produces_zero() -> VortexResult<()> {
        let dtype = DType::Primitive(PType::F64, Nullability::NonNullable);
        let mut acc = Accumulator::try_new(Sum, EmptyOptions, dtype)?;
        let result = acc.finish()?;
        assert_eq!(result.as_primitive().typed_value::<f64>(), Some(0.0));
        Ok(())
    }

    #[test]
    fn sum_f64_with_nan() -> VortexResult<()> {
        let arr = PrimitiveArray::new(
            buffer![1.0f64, f64::NAN, 2.0, f64::NAN, 3.0],
            Validity::NonNullable,
        )
        .into_array();
        let result = sum(&arr, &mut LEGACY_SESSION.create_execution_ctx())?;
        assert_eq!(result.as_primitive().typed_value::<f64>(), Some(6.0));
        Ok(())
    }

    #[test]
    fn sum_f32_with_nan() -> VortexResult<()> {
        let arr =
            PrimitiveArray::new(buffer![1.0f32, f32::NAN, 4.0], Validity::NonNullable).into_array();
        let result = sum(&arr, &mut LEGACY_SESSION.create_execution_ctx())?;
        assert_eq!(result.as_primitive().typed_value::<f64>(), Some(5.0));
        Ok(())
    }

    #[test]
    fn sum_f64_with_nan_and_nulls() -> VortexResult<()> {
        let arr = PrimitiveArray::from_option_iter([Some(1.0f64), None, Some(f64::NAN), Some(3.0)])
            .into_array();
        let result = sum(&arr, &mut LEGACY_SESSION.create_execution_ctx())?;
        assert_eq!(result.as_primitive().typed_value::<f64>(), Some(4.0));
        Ok(())
    }

    #[test]
    fn sum_all_nan() -> VortexResult<()> {
        let arr =
            PrimitiveArray::new(buffer![f64::NAN, f64::NAN], Validity::NonNullable).into_array();
        let result = sum(&arr, &mut LEGACY_SESSION.create_execution_ctx())?;
        assert_eq!(result.as_primitive().typed_value::<f64>(), Some(0.0));
        Ok(())
    }

    #[test]
    fn sum_f64_with_infinity() -> VortexResult<()> {
        let batch = PrimitiveArray::new(
            buffer![1.0f64, f64::INFINITY, f64::NEG_INFINITY, 2.0],
            Validity::NonNullable,
        )
        .into_array();
        let acc = sum(&batch, &mut LEGACY_SESSION.create_execution_ctx())?;
        // INFINITY + NEG_INFINITY = NaN, which is treated as saturated
        assert!(acc.as_primitive().typed_value::<f64>().unwrap().is_nan());

        let mut acc = Accumulator::try_new(
            Sum,
            EmptyOptions,
            DType::Primitive(PType::F64, Nullability::NonNullable),
        )?;
        acc.accumulate(&batch, &mut LEGACY_SESSION.create_execution_ctx())?;
        assert!(acc.is_saturated());
        Ok(())
    }

    #[test]
    fn sum_checked_overflow() -> VortexResult<()> {
        let arr = PrimitiveArray::new(buffer![i64::MAX, 1i64], Validity::NonNullable).into_array();
        let result = sum(&arr, &mut LEGACY_SESSION.create_execution_ctx())?;
        assert!(result.is_null());
        Ok(())
    }

    #[test]
    fn sum_checked_overflow_is_saturated() -> VortexResult<()> {
        let dtype = DType::Primitive(PType::I64, Nullability::NonNullable);
        let mut acc = Accumulator::try_new(Sum, EmptyOptions, dtype)?;
        assert!(!acc.is_saturated());

        let batch =
            PrimitiveArray::new(buffer![i64::MAX, 1i64], Validity::NonNullable).into_array();
        acc.accumulate(&batch, &mut LEGACY_SESSION.create_execution_ctx())?;
        assert!(acc.is_saturated());

        // finish resets state, clearing saturation
        drop(acc.finish()?);
        assert!(!acc.is_saturated());
        Ok(())
    }
}
