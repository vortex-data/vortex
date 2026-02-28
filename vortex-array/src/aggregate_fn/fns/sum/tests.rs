// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_buffer::buffer;
use vortex_error::VortexResult;

use crate::ArrayRef;
use crate::IntoArray;
use crate::aggregate_fn::AggregateFnVTable;
use crate::aggregate_fn::fns::sum::Sum;
use crate::aggregate_fn::fns::sum::SumOptions;
use crate::arrays::BoolArray;
use crate::arrays::PrimitiveArray;
use crate::dtype::DType;
use crate::dtype::Nullability;
use crate::dtype::PType;
use crate::scalar::Scalar;
use crate::validity::Validity;

fn checked_opts() -> SumOptions {
    SumOptions { checked: true }
}

fn unchecked_opts() -> SumOptions {
    SumOptions { checked: false }
}

fn run_sum(batch: &ArrayRef, options: &SumOptions) -> VortexResult<ArrayRef> {
    let mut acc = Sum.accumulator(options, batch.dtype())?;
    acc.accumulate(batch)?;
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
    let arr = PrimitiveArray::new(buffer![1.5f64, 2.5, 3.0], Validity::NonNullable).into_array();
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

    let batch = PrimitiveArray::new(buffer![i64::MAX, 1i64], Validity::NonNullable).into_array();
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
