// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::cmp::Ordering;

use num_traits::CheckedSub;
use rstest::rstest;
use vortex_dtype::DType;
use vortex_dtype::FromPrimitiveOrF16;
use vortex_dtype::Nullability;
use vortex_dtype::PType;
use vortex_dtype::ToBytes;
use vortex_dtype::half::f16;
use vortex_error::VortexExpect;
use vortex_utils::aliases::hash_set::HashSet;

use super::pvalue::CoercePValue;
use super::*;
use crate::scalar::PValue;
use crate::scalar::PrimitiveScalar;
use crate::scalar::ScalarValue;

#[test]
fn test_integer_subtract() {
    let dtype = DType::Primitive(PType::I32, Nullability::NonNullable);
    let value1 = ScalarValue::Primitive(PValue::I32(5));
    let value2 = ScalarValue::Primitive(PValue::I32(4));
    let p_scalar1 = PrimitiveScalar::try_new(&dtype, Some(&value1)).unwrap();
    let p_scalar2 = PrimitiveScalar::try_new(&dtype, Some(&value2)).unwrap();
    let pscalar_or_overflow = p_scalar1.checked_sub(&p_scalar2);
    let value_or_null_or_type_error = pscalar_or_overflow.unwrap().as_::<i32>();
    assert_eq!(value_or_null_or_type_error.unwrap(), 1);

    assert_eq!((p_scalar1 - p_scalar2).as_::<i32>().unwrap(), 1);
}

#[test]
#[should_panic(expected = "PrimitiveScalar subtract: overflow or underflow")]
fn test_integer_subtract_overflow() {
    let dtype = DType::Primitive(PType::I32, Nullability::NonNullable);
    let value1 = ScalarValue::Primitive(PValue::I32(i32::MIN));
    let value2 = ScalarValue::Primitive(PValue::I32(i32::MAX));
    let p_scalar1 = PrimitiveScalar::try_new(&dtype, Some(&value1)).unwrap();
    let p_scalar2 = PrimitiveScalar::try_new(&dtype, Some(&value2)).unwrap();
    let _ = p_scalar1 - p_scalar2;
}

#[test]
fn test_float_subtract() {
    let dtype = DType::Primitive(PType::F32, Nullability::NonNullable);
    let value1 = ScalarValue::Primitive(PValue::F32(1.99f32));
    let value2 = ScalarValue::Primitive(PValue::F32(1.0f32));
    let p_scalar1 = PrimitiveScalar::try_new(&dtype, Some(&value1)).unwrap();
    let p_scalar2 = PrimitiveScalar::try_new(&dtype, Some(&value2)).unwrap();
    let pscalar_or_overflow = p_scalar1.checked_sub(&p_scalar2).unwrap();
    let value_or_null_or_type_error = pscalar_or_overflow.as_::<f32>();
    assert_eq!(value_or_null_or_type_error.unwrap(), 0.99f32);

    assert_eq!((p_scalar1 - p_scalar2).as_::<f32>().unwrap(), 0.99f32);
}

#[test]
fn test_primitive_scalar_equality() {
    let dtype = DType::Primitive(PType::I32, Nullability::NonNullable);
    let value1 = ScalarValue::Primitive(PValue::I32(42));
    let value2 = ScalarValue::Primitive(PValue::I32(42));
    let value3 = ScalarValue::Primitive(PValue::I32(43));
    let scalar1 = PrimitiveScalar::try_new(&dtype, Some(&value1)).unwrap();
    let scalar2 = PrimitiveScalar::try_new(&dtype, Some(&value2)).unwrap();
    let scalar3 = PrimitiveScalar::try_new(&dtype, Some(&value3)).unwrap();

    assert_eq!(scalar1, scalar2);
    assert_ne!(scalar1, scalar3);
}

#[test]
fn test_primitive_scalar_partial_ord() {
    let dtype = DType::Primitive(PType::I32, Nullability::NonNullable);
    let value1 = ScalarValue::Primitive(PValue::I32(10));
    let value2 = ScalarValue::Primitive(PValue::I32(20));
    let scalar1 = PrimitiveScalar::try_new(&dtype, Some(&value1)).unwrap();
    let scalar2 = PrimitiveScalar::try_new(&dtype, Some(&value2)).unwrap();

    assert!(scalar1 < scalar2);
    assert!(scalar2 > scalar1);
    assert_eq!(scalar1.partial_cmp(&scalar1), Some(Ordering::Equal));
}

#[test]
fn test_primitive_scalar_null_handling() {
    let dtype = DType::Primitive(PType::I32, Nullability::Nullable);
    let null_scalar = PrimitiveScalar::try_new(&dtype, None).unwrap();

    assert_eq!(null_scalar.pvalue(), None);
    assert_eq!(null_scalar.typed_value::<i32>(), None);
}

#[test]
fn test_typed_value_correct_type() {
    let dtype = DType::Primitive(PType::F64, Nullability::NonNullable);
    let value = ScalarValue::Primitive(PValue::F64(3.5));
    let scalar = PrimitiveScalar::try_new(&dtype, Some(&value)).unwrap();

    assert_eq!(scalar.typed_value::<f64>(), Some(3.5));
}

#[test]
#[should_panic(expected = "Attempting to read")]
fn test_typed_value_wrong_type() {
    let dtype = DType::Primitive(PType::F64, Nullability::NonNullable);
    let value = ScalarValue::Primitive(PValue::F64(3.5));
    let scalar = PrimitiveScalar::try_new(&dtype, Some(&value)).unwrap();

    let _ = scalar.typed_value::<i32>();
}

#[rstest]
#[case(PType::I8, 127i32, PType::I16, true)]
#[case(PType::I8, 127i32, PType::I32, true)]
#[case(PType::I8, 127i32, PType::I64, true)]
#[case(PType::U8, 255i32, PType::U16, true)]
#[case(PType::U8, 255i32, PType::U32, true)]
#[case(PType::I32, 42i32, PType::F32, true)]
#[case(PType::I32, 42i32, PType::F64, true)]
// Overflow cases
#[case(PType::I32, 300i32, PType::U8, false)]
#[case(PType::I32, -1i32, PType::U32, false)]
#[case(PType::I32, 256i32, PType::I8, false)]
#[case(PType::U16, 65535i32, PType::I8, false)]
fn test_primitive_cast(
    #[case] source_type: PType,
    #[case] source_value: i32,
    #[case] target_type: PType,
    #[case] should_succeed: bool,
) {
    let source_pvalue = match source_type {
        PType::I8 => PValue::I8(i8::try_from(source_value).vortex_expect("cannot cast")),
        PType::U8 => PValue::U8(u8::try_from(source_value).vortex_expect("cannot cast")),
        PType::U16 => PValue::U16(u16::try_from(source_value).vortex_expect("cannot cast")),
        PType::I32 => PValue::I32(source_value),
        _ => unreachable!("Test case uses unexpected source type"),
    };

    let dtype = DType::Primitive(source_type, Nullability::NonNullable);
    let value = ScalarValue::Primitive(source_pvalue);
    let scalar = PrimitiveScalar::try_new(&dtype, Some(&value)).unwrap();

    let target_dtype = DType::Primitive(target_type, Nullability::NonNullable);
    let result = scalar.cast(&target_dtype);

    if should_succeed {
        assert!(
            result.is_ok(),
            "Cast from {:?} to {:?} should succeed",
            source_type,
            target_type
        );
    } else {
        assert!(
            result.is_err(),
            "Cast from {:?} to {:?} should fail due to overflow",
            source_type,
            target_type
        );
    }
}

#[test]
fn test_as_conversion_success() {
    let dtype = DType::Primitive(PType::I32, Nullability::NonNullable);
    let value = ScalarValue::Primitive(PValue::I32(42));
    let scalar = PrimitiveScalar::try_new(&dtype, Some(&value)).unwrap();

    assert_eq!(scalar.as_::<i64>(), Some(42i64));
    assert_eq!(scalar.as_::<f64>(), Some(42.0));
}

#[test]
fn test_as_conversion_overflow() {
    let dtype = DType::Primitive(PType::I32, Nullability::NonNullable);
    let value = ScalarValue::Primitive(PValue::I32(-1));
    let scalar = PrimitiveScalar::try_new(&dtype, Some(&value)).unwrap();

    // Converting -1 to u32 should fail
    let result = scalar.as_opt::<u32>();
    assert!(result.is_none());
}

#[test]
fn test_as_conversion_null() {
    let dtype = DType::Primitive(PType::I32, Nullability::Nullable);
    let scalar = PrimitiveScalar::try_new(&dtype, None).unwrap();

    assert_eq!(scalar.as_::<i32>(), None);
    assert_eq!(scalar.as_::<f64>(), None);
}

#[test]
fn test_numeric_operator_swap() {
    assert_eq!(NumericOperator::Add.swap(), NumericOperator::Add);
    assert_eq!(NumericOperator::Sub.swap(), NumericOperator::RSub);
    assert_eq!(NumericOperator::RSub.swap(), NumericOperator::Sub);
    assert_eq!(NumericOperator::Mul.swap(), NumericOperator::Mul);
    assert_eq!(NumericOperator::Div.swap(), NumericOperator::RDiv);
    assert_eq!(NumericOperator::RDiv.swap(), NumericOperator::Div);
}

#[test]
fn test_checked_binary_numeric_add() {
    let dtype = DType::Primitive(PType::I32, Nullability::NonNullable);
    let value1 = ScalarValue::Primitive(PValue::I32(10));
    let value2 = ScalarValue::Primitive(PValue::I32(20));
    let scalar1 = PrimitiveScalar::try_new(&dtype, Some(&value1)).unwrap();
    let scalar2 = PrimitiveScalar::try_new(&dtype, Some(&value2)).unwrap();

    let result = scalar1
        .checked_binary_numeric(&scalar2, NumericOperator::Add)
        .unwrap();
    assert_eq!(result.typed_value::<i32>(), Some(30));
}

#[test]
fn test_checked_binary_numeric_overflow() {
    let dtype = DType::Primitive(PType::I32, Nullability::NonNullable);
    let value1 = ScalarValue::Primitive(PValue::I32(i32::MAX));
    let value2 = ScalarValue::Primitive(PValue::I32(1));
    let scalar1 = PrimitiveScalar::try_new(&dtype, Some(&value1)).unwrap();
    let scalar2 = PrimitiveScalar::try_new(&dtype, Some(&value2)).unwrap();

    // Add should overflow and return None
    let result = scalar1.checked_binary_numeric(&scalar2, NumericOperator::Add);
    assert!(result.is_none());
}

#[test]
fn test_checked_binary_numeric_with_null() {
    let dtype = DType::Primitive(PType::I32, Nullability::Nullable);
    let value = ScalarValue::Primitive(PValue::I32(10));
    let scalar1 = PrimitiveScalar::try_new(&dtype, Some(&value)).unwrap();
    let null_scalar = PrimitiveScalar::try_new(&dtype, None).unwrap();

    // Operation with null should return null
    let result = scalar1
        .checked_binary_numeric(&null_scalar, NumericOperator::Add)
        .unwrap();
    assert_eq!(result.pvalue(), None);
}

#[test]
fn test_checked_binary_numeric_mul() {
    let dtype = DType::Primitive(PType::I32, Nullability::NonNullable);
    let value1 = ScalarValue::Primitive(PValue::I32(5));
    let value2 = ScalarValue::Primitive(PValue::I32(6));
    let scalar1 = PrimitiveScalar::try_new(&dtype, Some(&value1)).unwrap();
    let scalar2 = PrimitiveScalar::try_new(&dtype, Some(&value2)).unwrap();

    let result = scalar1
        .checked_binary_numeric(&scalar2, NumericOperator::Mul)
        .unwrap();
    assert_eq!(result.typed_value::<i32>(), Some(30));
}

#[test]
fn test_checked_binary_numeric_div() {
    let dtype = DType::Primitive(PType::I32, Nullability::NonNullable);
    let value1 = ScalarValue::Primitive(PValue::I32(20));
    let value2 = ScalarValue::Primitive(PValue::I32(4));
    let scalar1 = PrimitiveScalar::try_new(&dtype, Some(&value1)).unwrap();
    let scalar2 = PrimitiveScalar::try_new(&dtype, Some(&value2)).unwrap();

    let result = scalar1
        .checked_binary_numeric(&scalar2, NumericOperator::Div)
        .unwrap();
    assert_eq!(result.typed_value::<i32>(), Some(5));
}

#[test]
fn test_checked_binary_numeric_rdiv() {
    let dtype = DType::Primitive(PType::I32, Nullability::NonNullable);
    let value1 = ScalarValue::Primitive(PValue::I32(4));
    let value2 = ScalarValue::Primitive(PValue::I32(20));
    let scalar1 = PrimitiveScalar::try_new(&dtype, Some(&value1)).unwrap();
    let scalar2 = PrimitiveScalar::try_new(&dtype, Some(&value2)).unwrap();

    // RDiv means right / left, so 20 / 4 = 5
    let result = scalar1
        .checked_binary_numeric(&scalar2, NumericOperator::RDiv)
        .unwrap();
    assert_eq!(result.typed_value::<i32>(), Some(5));
}

#[test]
fn test_checked_binary_numeric_div_by_zero() {
    let dtype = DType::Primitive(PType::I32, Nullability::NonNullable);
    let value1 = ScalarValue::Primitive(PValue::I32(10));
    let value2 = ScalarValue::Primitive(PValue::I32(0));
    let scalar1 = PrimitiveScalar::try_new(&dtype, Some(&value1)).unwrap();
    let scalar2 = PrimitiveScalar::try_new(&dtype, Some(&value2)).unwrap();

    // Division by zero should return None for integers
    let result = scalar1.checked_binary_numeric(&scalar2, NumericOperator::Div);
    assert!(result.is_none());
}

#[test]
fn test_checked_binary_numeric_float_ops() {
    let dtype = DType::Primitive(PType::F32, Nullability::NonNullable);
    let value1 = ScalarValue::Primitive(PValue::F32(10.0));
    let value2 = ScalarValue::Primitive(PValue::F32(2.5));
    let scalar1 = PrimitiveScalar::try_new(&dtype, Some(&value1)).unwrap();
    let scalar2 = PrimitiveScalar::try_new(&dtype, Some(&value2)).unwrap();

    // Test all operations with floats
    let add_result = scalar1
        .checked_binary_numeric(&scalar2, NumericOperator::Add)
        .unwrap();
    assert_eq!(add_result.typed_value::<f32>(), Some(12.5));

    let sub_result = scalar1
        .checked_binary_numeric(&scalar2, NumericOperator::Sub)
        .unwrap();
    assert_eq!(sub_result.typed_value::<f32>(), Some(7.5));

    let mul_result = scalar1
        .checked_binary_numeric(&scalar2, NumericOperator::Mul)
        .unwrap();
    assert_eq!(mul_result.typed_value::<f32>(), Some(25.0));

    let div_result = scalar1
        .checked_binary_numeric(&scalar2, NumericOperator::Div)
        .unwrap();
    assert_eq!(div_result.typed_value::<f32>(), Some(4.0));
}

#[test]
fn test_from_primitive_or_f16() {
    // Test f16 to f32 conversion
    let f16_val = f16::from_f32(3.5);
    assert!(f32::from_f16(f16_val).is_some());

    // Test f16 to f64 conversion
    assert!(f64::from_f16(f16_val).is_some());

    // Test PValue::F16(f16) to integer conversion (should fail)
    assert!(i32::try_from(PValue::from(f16_val)).is_err());
    assert!(u32::try_from(PValue::from(f16_val)).is_err());
}

#[test]
fn test_partial_ord_different_types() {
    let dtype1 = DType::Primitive(PType::I32, Nullability::NonNullable);
    let dtype2 = DType::Primitive(PType::F32, Nullability::NonNullable);

    let value1 = ScalarValue::Primitive(PValue::I32(10));
    let value2 = ScalarValue::Primitive(PValue::F32(10.0));
    let scalar1 = PrimitiveScalar::try_new(&dtype1, Some(&value1)).unwrap();
    let scalar2 = PrimitiveScalar::try_new(&dtype2, Some(&value2)).unwrap();

    // Different types should not be comparable
    assert_eq!(scalar1.partial_cmp(&scalar2), None);
}

#[test]
fn test_scalar_value_from_usize() {
    let value: ScalarValue = 42usize.into();
    assert!(matches!(value, ScalarValue::Primitive(PValue::U64(42))));
}

#[test]
fn test_getters() {
    let dtype = DType::Primitive(PType::I32, Nullability::NonNullable);
    let value = ScalarValue::Primitive(PValue::I32(42));
    let scalar = PrimitiveScalar::try_new(&dtype, Some(&value)).unwrap();

    assert_eq!(scalar.dtype(), &dtype);
    assert_eq!(scalar.ptype(), PType::I32);
    assert_eq!(scalar.pvalue(), Some(PValue::I32(42)));
}

#[test]
pub fn test_is_instance_of() {
    assert!(PValue::U8(10).is_instance_of(&PType::U8));
    assert!(!PValue::U8(10).is_instance_of(&PType::U16));
    assert!(!PValue::U8(10).is_instance_of(&PType::I8));
    assert!(!PValue::U8(10).is_instance_of(&PType::F16));

    assert!(PValue::I8(10).is_instance_of(&PType::I8));
    assert!(!PValue::I8(10).is_instance_of(&PType::I16));
    assert!(!PValue::I8(10).is_instance_of(&PType::U8));
    assert!(!PValue::I8(10).is_instance_of(&PType::F16));

    assert!(PValue::F16(f16::from_f32(10.0)).is_instance_of(&PType::F16));
    assert!(!PValue::F16(f16::from_f32(10.0)).is_instance_of(&PType::F32));
    assert!(!PValue::F16(f16::from_f32(10.0)).is_instance_of(&PType::U16));
    assert!(!PValue::F16(f16::from_f32(10.0)).is_instance_of(&PType::I16));
}

#[test]
fn test_compare_different_types() {
    assert_eq!(
        PValue::I8(4).partial_cmp(&PValue::I8(5)),
        Some(Ordering::Less)
    );
    assert_eq!(
        PValue::I8(4).partial_cmp(&PValue::I64(5)),
        Some(Ordering::Less)
    );
}

#[test]
fn test_hash() {
    let set = HashSet::from([
        PValue::U8(1),
        PValue::U16(1),
        PValue::U32(1),
        PValue::U64(1),
        PValue::I8(1),
        PValue::I16(1),
        PValue::I32(1),
        PValue::I64(1),
        PValue::I8(-1),
        PValue::I16(-1),
        PValue::I32(-1),
        PValue::I64(-1),
    ]);
    assert_eq!(set.len(), 2);
}

#[test]
fn test_zero_values() {
    assert_eq!(PValue::zero(&PType::U8), PValue::U8(0));
    assert_eq!(PValue::zero(&PType::U16), PValue::U16(0));
    assert_eq!(PValue::zero(&PType::U32), PValue::U32(0));
    assert_eq!(PValue::zero(&PType::U64), PValue::U64(0));
    assert_eq!(PValue::zero(&PType::I8), PValue::I8(0));
    assert_eq!(PValue::zero(&PType::I16), PValue::I16(0));
    assert_eq!(PValue::zero(&PType::I32), PValue::I32(0));
    assert_eq!(PValue::zero(&PType::I64), PValue::I64(0));
    assert_eq!(PValue::zero(&PType::F16), PValue::F16(f16::from_f32(0.0)));
    assert_eq!(PValue::zero(&PType::F32), PValue::F32(0.0));
    assert_eq!(PValue::zero(&PType::F64), PValue::F64(0.0));
}

#[test]
fn test_ptype() {
    assert_eq!(PValue::U8(10).ptype(), PType::U8);
    assert_eq!(PValue::U16(10).ptype(), PType::U16);
    assert_eq!(PValue::U32(10).ptype(), PType::U32);
    assert_eq!(PValue::U64(10).ptype(), PType::U64);
    assert_eq!(PValue::I8(10).ptype(), PType::I8);
    assert_eq!(PValue::I16(10).ptype(), PType::I16);
    assert_eq!(PValue::I32(10).ptype(), PType::I32);
    assert_eq!(PValue::I64(10).ptype(), PType::I64);
    assert_eq!(PValue::F16(f16::from_f32(10.0)).ptype(), PType::F16);
    assert_eq!(PValue::F32(10.0).ptype(), PType::F32);
    assert_eq!(PValue::F64(10.0).ptype(), PType::F64);
}

#[test]
fn test_reinterpret_cast_same_type() {
    let value = PValue::U32(42);
    assert_eq!(value.reinterpret_cast(PType::U32), value);
}

#[test]
fn test_reinterpret_cast_u8_i8() {
    let value = PValue::U8(255);
    let casted = value.reinterpret_cast(PType::I8);
    assert_eq!(casted, PValue::I8(-1));
}

#[test]
fn test_reinterpret_cast_u16_types() {
    let value = PValue::U16(12345);

    // U16 -> I16
    let as_i16 = value.reinterpret_cast(PType::I16);
    assert_eq!(as_i16, PValue::I16(12345));

    // U16 -> F16
    let as_f16 = value.reinterpret_cast(PType::F16);
    assert_eq!(as_f16, PValue::F16(f16::from_bits(12345)));
}

#[test]
fn test_reinterpret_cast_u32_types() {
    let value = PValue::U32(0x3f800000); // 1.0 in float bits

    // U32 -> F32
    let as_f32 = value.reinterpret_cast(PType::F32);
    assert_eq!(as_f32, PValue::F32(1.0));

    // U32 -> I32
    let value2 = PValue::U32(0x80000000);
    let as_i32 = value2.reinterpret_cast(PType::I32);
    assert_eq!(as_i32, PValue::I32(i32::MIN));
}

#[test]
fn test_reinterpret_cast_f32_to_u32() {
    let value = PValue::F32(1.0);
    let as_u32 = value.reinterpret_cast(PType::U32);
    assert_eq!(as_u32, PValue::U32(0x3f800000));
}

#[test]
fn test_reinterpret_cast_f64_to_i64() {
    let value = PValue::F64(1.0);
    let as_i64 = value.reinterpret_cast(PType::I64);
    assert_eq!(as_i64, PValue::I64(0x3ff0000000000000_i64));
}

#[test]
#[should_panic(expected = "Cannot reinterpret cast between types of different widths")]
fn test_reinterpret_cast_different_widths() {
    let value = PValue::U8(42);
    let _ = value.reinterpret_cast(PType::U16);
}

#[test]
fn test_as_primitive_conversions() {
    // Test as_u8
    assert_eq!(PValue::U8(42).as_u8(), Some(42));
    assert_eq!(PValue::I8(42).as_u8(), Some(42));
    assert_eq!(PValue::U16(255).as_u8(), Some(255));
    assert_eq!(PValue::U16(256).as_u8(), None); // Overflow

    // Test as_i32
    assert_eq!(PValue::I32(42).as_i32(), Some(42));
    assert_eq!(PValue::U32(42).as_i32(), Some(42));
    assert_eq!(PValue::I64(42).as_i32(), Some(42));
    assert_eq!(PValue::U64(u64::MAX).as_i32(), None); // Overflow

    // Test as_f64
    assert_eq!(PValue::F64(42.5).as_f64(), Some(42.5));
    assert_eq!(PValue::F32(42.5).as_f64(), Some(42.5f64));
    assert_eq!(PValue::I32(42).as_f64(), Some(42.0));
}

#[test]
fn test_try_from_pvalue_integers() {
    // Test u8 conversion
    assert_eq!(u8::try_from(PValue::U8(42)).unwrap(), 42);
    assert_eq!(u8::try_from(PValue::I8(42)).unwrap(), 42);
    assert!(u8::try_from(PValue::I8(-1)).is_err());
    assert!(u8::try_from(PValue::U16(256)).is_err());

    // Test i32 conversion
    assert_eq!(i32::try_from(PValue::I32(42)).unwrap(), 42);
    assert_eq!(i32::try_from(PValue::I16(-100)).unwrap(), -100);
    assert!(i32::try_from(PValue::U64(u64::MAX)).is_err());

    // Float to int should fail
    assert!(i32::try_from(PValue::F32(42.5)).is_err());
}

#[test]
fn test_try_from_pvalue_floats() {
    // Test f32 conversion
    assert_eq!(f32::try_from(PValue::F32(42.5)).unwrap(), 42.5);
    assert_eq!(f32::try_from(PValue::I32(42)).unwrap(), 42.0);
    assert_eq!(f32::try_from(PValue::U8(255)).unwrap(), 255.0);

    // Test f64 conversion
    assert_eq!(f64::try_from(PValue::F64(42.5)).unwrap(), 42.5);
    assert_eq!(f64::try_from(PValue::F32(42.5)).unwrap(), 42.5f64);
    assert_eq!(f64::try_from(PValue::I64(-100)).unwrap(), -100.0);
}

#[test]
fn test_from_usize() {
    let value: PValue = 42usize.into();
    assert_eq!(value, PValue::U64(42));

    let max_value: PValue = usize::MAX.into();
    assert_eq!(max_value, PValue::U64(usize::MAX as u64));
}

#[test]
fn test_equality_cross_types() {
    // Same numeric value, different types
    assert_eq!(PValue::U8(42), PValue::U16(42));
    assert_eq!(PValue::U8(42), PValue::U32(42));
    assert_eq!(PValue::U8(42), PValue::U64(42));
    assert_eq!(PValue::I8(42), PValue::I16(42));
    assert_eq!(PValue::I8(42), PValue::I32(42));
    assert_eq!(PValue::I8(42), PValue::I64(42));

    // Unsigned vs signed with same value (they compare equal even though different categories)
    assert_eq!(PValue::U8(42), PValue::I8(42));
    assert_eq!(PValue::U32(42), PValue::I32(42));

    // Float equality
    assert_eq!(PValue::F32(42.0), PValue::F32(42.0));
    assert_eq!(PValue::F64(42.0), PValue::F64(42.0));
    assert_ne!(PValue::F32(42.0), PValue::F64(42.0)); // Different types

    // Float vs int should not be equal
    assert_ne!(PValue::F32(42.0), PValue::I32(42));
}

#[test]
fn test_partial_ord_cross_types() {
    // Unsigned comparisons
    assert_eq!(
        PValue::U8(10).partial_cmp(&PValue::U16(20)),
        Some(Ordering::Less)
    );
    assert_eq!(
        PValue::U32(30).partial_cmp(&PValue::U8(20)),
        Some(Ordering::Greater)
    );

    // Signed comparisons
    assert_eq!(
        PValue::I8(-10).partial_cmp(&PValue::I64(0)),
        Some(Ordering::Less)
    );
    assert_eq!(
        PValue::I32(10).partial_cmp(&PValue::I16(10)),
        Some(Ordering::Equal)
    );

    // Float comparisons (same type only)
    assert_eq!(
        PValue::F32(1.0).partial_cmp(&PValue::F32(2.0)),
        Some(Ordering::Less)
    );
    assert_eq!(
        PValue::F64(2.0).partial_cmp(&PValue::F64(1.0)),
        Some(Ordering::Greater)
    );

    // Cross-category comparisons - unsigned vs signed work, float vs int don't
    assert_eq!(
        PValue::U32(42).partial_cmp(&PValue::I32(42)),
        Some(Ordering::Equal)
    ); // Actually works
    assert_eq!(PValue::F32(42.0).partial_cmp(&PValue::I32(42)), None);
    assert_eq!(PValue::F32(42.0).partial_cmp(&PValue::F64(42.0)), None);
}

#[test]
fn test_to_le_bytes() {
    assert_eq!(PValue::U8(0x12).to_le_bytes(), &[0x12]);
    assert_eq!(PValue::U16(0x1234).to_le_bytes(), &[0x34, 0x12]);
    assert_eq!(
        PValue::U32(0x12345678).to_le_bytes(),
        &[0x78, 0x56, 0x34, 0x12]
    );

    assert_eq!(PValue::I8(-1).to_le_bytes(), &[0xFF]);
    assert_eq!(PValue::I16(-1).to_le_bytes(), &[0xFF, 0xFF]);

    let f32_bytes = PValue::F32(1.0).to_le_bytes();
    assert_eq!(f32_bytes.len(), 4);

    let f64_bytes = PValue::F64(1.0).to_le_bytes();
    assert_eq!(f64_bytes.len(), 8);
}

#[test]
fn test_f16_special_values() {
    // Test F16 NaN handling
    let nan = f16::NAN;
    let nan_value = PValue::F16(nan);
    assert!(nan_value.as_f16().unwrap().is_nan());

    // Test F16 infinity
    let inf = f16::INFINITY;
    let inf_value = PValue::F16(inf);
    assert!(inf_value.as_f16().unwrap().is_infinite());

    // Test F16 comparison with NaN
    assert_eq!(
        PValue::F16(nan).partial_cmp(&PValue::F16(nan)),
        Some(Ordering::Equal)
    );
}

#[test]
fn test_coerce_pvalue() {
    // Test integer coercion
    assert_eq!(u32::coerce(PValue::U16(42)).unwrap(), 42u32);
    assert_eq!(i64::coerce(PValue::I32(-42)).unwrap(), -42i64);

    // Test float coercion from bits
    assert_eq!(f32::coerce(PValue::U32(0x3f800000)).unwrap(), 1.0f32);
    assert_eq!(
        f64::coerce(PValue::U64(0x3ff0000000000000)).unwrap(),
        1.0f64
    );
}

#[test]
fn test_coerce_f16_beyond_u16_max() {
    // Test U32 to f16 coercion within valid range
    assert!(f16::coerce(PValue::U32(u16::MAX as u32)).is_ok());
    assert_eq!(
        f16::coerce(PValue::U32(0x3C00)).unwrap(),
        f16::from_bits(0x3C00) // 1.0 in f16
    );

    // Test U32 to f16 coercion beyond u16::MAX - should fail
    assert!(f16::coerce(PValue::U32((u16::MAX as u32) + 1)).is_err());
    assert!(f16::coerce(PValue::U32(u32::MAX)).is_err());

    // Test U64 to f16 coercion within valid range
    assert!(f16::coerce(PValue::U64(u16::MAX as u64)).is_ok());
    assert_eq!(
        f16::coerce(PValue::U64(0x3C00)).unwrap(),
        f16::from_bits(0x3C00) // 1.0 in f16
    );

    // Test U64 to f16 coercion beyond u16::MAX - should fail
    assert!(f16::coerce(PValue::U64((u16::MAX as u64) + 1)).is_err());
    assert!(f16::coerce(PValue::U64(u32::MAX as u64)).is_err());
    assert!(f16::coerce(PValue::U64(u64::MAX)).is_err());
}

#[test]
fn test_coerce_f32_beyond_u32_max() {
    // Test U64 to f32 coercion within valid range
    assert!(f32::coerce(PValue::U64(u32::MAX as u64)).is_ok());
    assert_eq!(
        f32::coerce(PValue::U64(0x3f800000)).unwrap(),
        1.0f32 // 0x3f800000 is 1.0 in f32
    );

    // Test U64 to f32 coercion beyond u32::MAX - should fail
    assert!(f32::coerce(PValue::U64((u32::MAX as u64) + 1)).is_err());
    assert!(f32::coerce(PValue::U64(u64::MAX)).is_err());

    // Test smaller types still work
    assert!(f32::coerce(PValue::U8(255)).is_ok());
    assert!(f32::coerce(PValue::U16(u16::MAX)).is_ok());
    assert!(f32::coerce(PValue::U32(u32::MAX)).is_ok());
}

#[test]
fn test_coerce_f64_all_unsigned() {
    // Test f64 can accept all unsigned integer values as bit patterns
    assert!(f64::coerce(PValue::U8(u8::MAX)).is_ok());
    assert!(f64::coerce(PValue::U16(u16::MAX)).is_ok());
    assert!(f64::coerce(PValue::U32(u32::MAX)).is_ok());
    assert!(f64::coerce(PValue::U64(u64::MAX)).is_ok());

    // Verify specific bit patterns
    assert_eq!(
        f64::coerce(PValue::U64(0x3ff0000000000000)).unwrap(),
        1.0f64 // 0x3ff0000000000000 is 1.0 in f64
    );
}

#[test]
fn test_f16_nans_equal() {
    let nan1 = f16::from_le_bytes([154, 253]);
    assert!(nan1.is_nan());
    let nan3 = f16::from_f16(nan1).unwrap();
    assert_eq!(nan1.to_bits(), nan3.to_bits(),);
}
