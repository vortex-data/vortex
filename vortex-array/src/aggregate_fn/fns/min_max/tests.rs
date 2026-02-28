// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_buffer::buffer;
use vortex_error::VortexResult;

use crate::ArrayRef;
use crate::IntoArray;
use crate::aggregate_fn::AggregateFnVTable;
use crate::aggregate_fn::fns::min_max::Max;
use crate::aggregate_fn::fns::min_max::Min;
use crate::arrays::BoolArray;
use crate::arrays::PrimitiveArray;
use crate::dtype::DType;
use crate::dtype::Nullability;
use crate::dtype::PType;
use crate::scalar::Scalar;
use crate::scalar_fn::EmptyOptions;
use crate::validity::Validity;

fn run_min(batch: &ArrayRef) -> VortexResult<ArrayRef> {
    let mut acc = Min.accumulator(&EmptyOptions, batch.dtype())?;
    acc.accumulate(batch)?;
    acc.flush()?;
    acc.finish()
}

fn run_max(batch: &ArrayRef) -> VortexResult<ArrayRef> {
    let mut acc = Max.accumulator(&EmptyOptions, batch.dtype())?;
    acc.accumulate(batch)?;
    acc.flush()?;
    acc.finish()
}

fn get_i32_value(array: &ArrayRef, idx: usize) -> VortexResult<Option<i32>> {
    let scalar = array.scalar_at(idx)?;
    Ok(scalar.as_primitive().typed_value::<i32>())
}

fn get_f64_value(array: &ArrayRef, idx: usize) -> VortexResult<Option<f64>> {
    let scalar = array.scalar_at(idx)?;
    Ok(scalar.as_primitive().typed_value::<f64>())
}

fn get_bool_value(array: &ArrayRef, idx: usize) -> VortexResult<Option<bool>> {
    let scalar = array.scalar_at(idx)?;
    Ok(scalar.as_bool().value())
}

// -- Min tests --

#[test]
fn min_i32() -> VortexResult<()> {
    let arr = PrimitiveArray::new(buffer![3i32, 1, 4, 1, 5], Validity::NonNullable).into_array();
    let result = run_min(&arr)?;
    assert_eq!(get_i32_value(&result, 0)?, Some(1));
    Ok(())
}

#[test]
fn max_i32() -> VortexResult<()> {
    let arr = PrimitiveArray::new(buffer![3i32, 1, 4, 1, 5], Validity::NonNullable).into_array();
    let result = run_max(&arr)?;
    assert_eq!(get_i32_value(&result, 0)?, Some(5));
    Ok(())
}

#[test]
fn min_f64() -> VortexResult<()> {
    let arr = PrimitiveArray::new(buffer![3.0f64, 1.5, 2.0], Validity::NonNullable).into_array();
    let result = run_min(&arr)?;
    assert_eq!(get_f64_value(&result, 0)?, Some(1.5));
    Ok(())
}

#[test]
fn max_f64() -> VortexResult<()> {
    let arr = PrimitiveArray::new(buffer![3.0f64, 1.5, 2.0], Validity::NonNullable).into_array();
    let result = run_max(&arr)?;
    assert_eq!(get_f64_value(&result, 0)?, Some(3.0));
    Ok(())
}

#[test]
fn min_with_nulls() -> VortexResult<()> {
    let arr = PrimitiveArray::from_option_iter([Some(5i32), None, Some(2)]).into_array();
    let result = run_min(&arr)?;
    assert_eq!(get_i32_value(&result, 0)?, Some(2));
    Ok(())
}

#[test]
fn max_with_nulls() -> VortexResult<()> {
    let arr = PrimitiveArray::from_option_iter([Some(5i32), None, Some(2)]).into_array();
    let result = run_max(&arr)?;
    assert_eq!(get_i32_value(&result, 0)?, Some(5));
    Ok(())
}

#[test]
fn min_all_null() -> VortexResult<()> {
    let arr = PrimitiveArray::from_option_iter([None::<i32>, None, None]).into_array();
    let result = run_min(&arr)?;
    assert_eq!(get_i32_value(&result, 0)?, None);
    Ok(())
}

#[test]
fn max_all_null() -> VortexResult<()> {
    let arr = PrimitiveArray::from_option_iter([None::<i32>, None, None]).into_array();
    let result = run_max(&arr)?;
    assert_eq!(get_i32_value(&result, 0)?, None);
    Ok(())
}

#[test]
fn min_empty_flush_produces_null() -> VortexResult<()> {
    let mut acc = Min.accumulator(
        &EmptyOptions,
        &DType::Primitive(PType::I32, Nullability::NonNullable),
    )?;
    acc.flush()?;
    let result = acc.finish()?;
    assert_eq!(get_i32_value(&result, 0)?, None);
    Ok(())
}

#[test]
fn max_empty_flush_produces_null() -> VortexResult<()> {
    let mut acc = Max.accumulator(
        &EmptyOptions,
        &DType::Primitive(PType::I32, Nullability::NonNullable),
    )?;
    acc.flush()?;
    let result = acc.finish()?;
    assert_eq!(get_i32_value(&result, 0)?, None);
    Ok(())
}

#[test]
fn min_multi_group() -> VortexResult<()> {
    let mut acc = Min.accumulator(
        &EmptyOptions,
        &DType::Primitive(PType::I32, Nullability::NonNullable),
    )?;

    let batch1 = PrimitiveArray::new(buffer![10i32, 20], Validity::NonNullable).into_array();
    acc.accumulate(&batch1)?;
    acc.flush()?;

    let batch2 = PrimitiveArray::new(buffer![3i32, 6, 1], Validity::NonNullable).into_array();
    acc.accumulate(&batch2)?;
    acc.flush()?;

    let result = acc.finish()?;
    assert_eq!(get_i32_value(&result, 0)?, Some(10));
    assert_eq!(get_i32_value(&result, 1)?, Some(1));
    Ok(())
}

#[test]
fn min_merge() -> VortexResult<()> {
    let mut acc = Min.accumulator(
        &EmptyOptions,
        &DType::Primitive(PType::I32, Nullability::NonNullable),
    )?;

    let state1 = Scalar::primitive(100i32, Nullability::Nullable);
    acc.merge(&state1)?;

    let state2 = Scalar::primitive(50i32, Nullability::Nullable);
    acc.merge(&state2)?;

    acc.flush()?;
    let result = acc.finish()?;
    assert_eq!(get_i32_value(&result, 0)?, Some(50));
    Ok(())
}

#[test]
fn min_is_saturated() -> VortexResult<()> {
    let mut acc = Min.accumulator(
        &EmptyOptions,
        &DType::Primitive(PType::I32, Nullability::NonNullable),
    )?;
    assert!(!acc.is_saturated());

    let batch = PrimitiveArray::new(buffer![i32::MIN, 5i32], Validity::NonNullable).into_array();
    acc.accumulate(&batch)?;
    assert!(acc.is_saturated());
    Ok(())
}

#[test]
fn max_is_saturated() -> VortexResult<()> {
    let mut acc = Max.accumulator(
        &EmptyOptions,
        &DType::Primitive(PType::I32, Nullability::NonNullable),
    )?;
    assert!(!acc.is_saturated());

    let batch = PrimitiveArray::new(buffer![i32::MAX, 5i32], Validity::NonNullable).into_array();
    acc.accumulate(&batch)?;
    assert!(acc.is_saturated());
    Ok(())
}

// NaN handling

#[test]
fn min_skips_nan() -> VortexResult<()> {
    let arr =
        PrimitiveArray::new(buffer![f64::NAN, 3.0f64, 1.0], Validity::NonNullable).into_array();
    let result = run_min(&arr)?;
    assert_eq!(get_f64_value(&result, 0)?, Some(1.0));
    Ok(())
}

#[test]
fn max_skips_nan() -> VortexResult<()> {
    let arr =
        PrimitiveArray::new(buffer![f64::NAN, 3.0f64, 1.0], Validity::NonNullable).into_array();
    let result = run_max(&arr)?;
    assert_eq!(get_f64_value(&result, 0)?, Some(3.0));
    Ok(())
}

// Boolean min/max

#[test]
fn min_bool_mixed() -> VortexResult<()> {
    let arr: BoolArray = [true, false, true].into_iter().collect();
    let result = run_min(&arr.into_array())?;
    assert_eq!(get_bool_value(&result, 0)?, Some(false));
    Ok(())
}

#[test]
fn max_bool_mixed() -> VortexResult<()> {
    let arr: BoolArray = [false, true, false].into_iter().collect();
    let result = run_max(&arr.into_array())?;
    assert_eq!(get_bool_value(&result, 0)?, Some(true));
    Ok(())
}

#[test]
fn min_bool_all_true() -> VortexResult<()> {
    let arr: BoolArray = [true, true, true].into_iter().collect();
    let result = run_min(&arr.into_array())?;
    assert_eq!(get_bool_value(&result, 0)?, Some(true));
    Ok(())
}

#[test]
fn max_bool_all_false() -> VortexResult<()> {
    let arr: BoolArray = [false, false, false].into_iter().collect();
    let result = run_max(&arr.into_array())?;
    assert_eq!(get_bool_value(&result, 0)?, Some(false));
    Ok(())
}

#[test]
fn min_bool_with_nulls() -> VortexResult<()> {
    let arr = BoolArray::from_iter([Some(true), None, Some(false)]);
    let result = run_min(&arr.into_array())?;
    assert_eq!(get_bool_value(&result, 0)?, Some(false));
    Ok(())
}

#[test]
fn max_bool_all_null() -> VortexResult<()> {
    let arr = BoolArray::from_iter([None::<bool>, None]);
    let result = run_max(&arr.into_array())?;
    assert_eq!(get_bool_value(&result, 0)?, None);
    Ok(())
}

#[test]
fn min_bool_empty_flush_produces_null() -> VortexResult<()> {
    let mut acc = Min.accumulator(&EmptyOptions, &DType::Bool(Nullability::NonNullable))?;
    acc.flush()?;
    let result = acc.finish()?;
    assert_eq!(get_bool_value(&result, 0)?, None);
    Ok(())
}

#[test]
fn max_bool_empty_flush_produces_null() -> VortexResult<()> {
    let mut acc = Max.accumulator(&EmptyOptions, &DType::Bool(Nullability::NonNullable))?;
    acc.flush()?;
    let result = acc.finish()?;
    assert_eq!(get_bool_value(&result, 0)?, None);
    Ok(())
}

#[test]
fn min_bool_saturated() -> VortexResult<()> {
    let mut acc = Min.accumulator(&EmptyOptions, &DType::Bool(Nullability::NonNullable))?;
    assert!(!acc.is_saturated());

    let batch: BoolArray = [true, false].into_iter().collect();
    acc.accumulate(&batch.into_array())?;
    assert!(acc.is_saturated());
    Ok(())
}

#[test]
fn max_bool_saturated() -> VortexResult<()> {
    let mut acc = Max.accumulator(&EmptyOptions, &DType::Bool(Nullability::NonNullable))?;
    assert!(!acc.is_saturated());

    let batch: BoolArray = [false, true].into_iter().collect();
    acc.accumulate(&batch.into_array())?;
    assert!(acc.is_saturated());
    Ok(())
}

// Return dtype

#[test]
fn min_return_dtype() -> VortexResult<()> {
    let dtype = Min.return_dtype(
        &EmptyOptions,
        &DType::Primitive(PType::I32, Nullability::NonNullable),
    )?;
    assert_eq!(dtype, DType::Primitive(PType::I32, Nullability::Nullable));
    Ok(())
}

#[test]
fn max_return_dtype_bool() -> VortexResult<()> {
    let dtype = Max.return_dtype(&EmptyOptions, &DType::Bool(Nullability::NonNullable))?;
    assert_eq!(dtype, DType::Bool(Nullability::Nullable));
    Ok(())
}
