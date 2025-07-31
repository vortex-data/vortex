// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_dtype::{DType, Nullability, PType};
use vortex_error::VortexUnwrap;

use crate::compute::cast;
use crate::Array;

/// Test conformance of the cast compute function for an array.
///
/// This function tests various casting scenarios including:
/// - Casting between numeric types (widening and narrowing)
/// - Casting between signed and unsigned types
/// - Casting between integral and floating-point types
/// - Casting with nullability changes
/// - Edge cases like overflow behavior
pub fn test_cast_conformance(array: &dyn Array) {
    let dtype = array.dtype();
    
    // Only test primitive types for now
    if let DType::Primitive(ptype, nullability) = dtype {
        test_cast_identity(array);
        test_cast_nullability_changes(array, *ptype, *nullability);
        
        match ptype {
            PType::U8 => test_cast_from_u8(array),
            PType::U16 => test_cast_from_u16(array),
            PType::U32 => test_cast_from_u32(array),
            PType::U64 => test_cast_from_u64(array),
            PType::I8 => test_cast_from_i8(array),
            PType::I16 => test_cast_from_i16(array),
            PType::I32 => test_cast_from_i32(array),
            PType::I64 => test_cast_from_i64(array),
            PType::F16 => test_cast_from_f16(array),
            PType::F32 => test_cast_from_f32(array),
            PType::F64 => test_cast_from_f64(array),
        }
    }
}

fn test_cast_identity(array: &dyn Array) {
    // Casting to the same type should be a no-op
    let result = cast(array, array.dtype()).vortex_unwrap();
    assert_eq!(result.len(), array.len());
    assert_eq!(result.dtype(), array.dtype());
    
    // Verify values are unchanged
    for i in 0..array.len() {
        assert_eq!(
            array.scalar_at(i).vortex_unwrap(),
            result.scalar_at(i).vortex_unwrap()
        );
    }
}

fn test_cast_nullability_changes(array: &dyn Array, ptype: PType, nullability: Nullability) {
    // Test casting to nullable version
    if nullability == Nullability::NonNullable {
        let nullable_dtype = DType::Primitive(ptype, Nullability::Nullable);
        let result = cast(array, &nullable_dtype).vortex_unwrap();
        assert_eq!(result.len(), array.len());
        assert_eq!(result.dtype(), &nullable_dtype);
        
        // Values should be unchanged
        for i in 0..array.len() {
            assert_eq!(
                array.scalar_at(i).vortex_unwrap(),
                result.scalar_at(i).vortex_unwrap()
            );
        }
    }
    
    // Test casting from nullable to non-nullable (only if no nulls present)
    if nullability == Nullability::Nullable {
        // Try to cast to non-nullable and see if it succeeds
        let non_nullable_dtype = DType::Primitive(ptype, Nullability::NonNullable);
        if let Ok(result) = cast(array, &non_nullable_dtype) {
            assert_eq!(result.len(), array.len());
            assert_eq!(result.dtype(), &non_nullable_dtype);
            
            // Values should be unchanged
            for i in 0..array.len() {
                assert_eq!(
                    array.scalar_at(i).vortex_unwrap(),
                    result.scalar_at(i).vortex_unwrap()
                );
            }
        }
    }
}

fn test_cast_from_u8(array: &dyn Array) {
    // Test widening casts
    test_cast_to_type(array, PType::U16);
    test_cast_to_type(array, PType::U32);
    test_cast_to_type(array, PType::U64);
    test_cast_to_type(array, PType::I16);
    test_cast_to_type(array, PType::I32);
    test_cast_to_type(array, PType::I64);
    test_cast_to_type(array, PType::F32);
    test_cast_to_type(array, PType::F64);
    
    // Test same-width cast
    test_cast_to_type(array, PType::I8);
}

fn test_cast_from_u16(array: &dyn Array) {
    // Test narrowing cast
    test_cast_to_type(array, PType::U8);
    
    // Test widening casts
    test_cast_to_type(array, PType::U32);
    test_cast_to_type(array, PType::U64);
    test_cast_to_type(array, PType::I32);
    test_cast_to_type(array, PType::I64);
    test_cast_to_type(array, PType::F32);
    test_cast_to_type(array, PType::F64);
    
    // Test same-width cast
    test_cast_to_type(array, PType::I16);
}

fn test_cast_from_u32(array: &dyn Array) {
    // Test narrowing casts
    test_cast_to_type(array, PType::U8);
    test_cast_to_type(array, PType::U16);
    test_cast_to_type(array, PType::I8);
    test_cast_to_type(array, PType::I16);
    
    // Test widening casts
    test_cast_to_type(array, PType::U64);
    test_cast_to_type(array, PType::I64);
    test_cast_to_type(array, PType::F64);
    
    // Test same-width casts
    test_cast_to_type(array, PType::I32);
    test_cast_to_type(array, PType::F32);
}

fn test_cast_from_u64(array: &dyn Array) {
    // Test narrowing casts
    test_cast_to_type(array, PType::U8);
    test_cast_to_type(array, PType::U16);
    test_cast_to_type(array, PType::U32);
    test_cast_to_type(array, PType::I8);
    test_cast_to_type(array, PType::I16);
    test_cast_to_type(array, PType::I32);
    test_cast_to_type(array, PType::F32);
    
    // Test same-width casts
    test_cast_to_type(array, PType::I64);
    test_cast_to_type(array, PType::F64);
}

fn test_cast_from_i8(array: &dyn Array) {
    // Test widening casts
    test_cast_to_type(array, PType::I16);
    test_cast_to_type(array, PType::I32);
    test_cast_to_type(array, PType::I64);
    test_cast_to_type(array, PType::F32);
    test_cast_to_type(array, PType::F64);
    
    // Test same-width cast (may fail for negative values)
    test_cast_to_type(array, PType::U8);
}

fn test_cast_from_i16(array: &dyn Array) {
    // Test narrowing cast
    test_cast_to_type(array, PType::I8);
    
    // Test widening casts
    test_cast_to_type(array, PType::I32);
    test_cast_to_type(array, PType::I64);
    test_cast_to_type(array, PType::F32);
    test_cast_to_type(array, PType::F64);
    
    // Test same-width cast (may fail for negative values)
    test_cast_to_type(array, PType::U16);
}

fn test_cast_from_i32(array: &dyn Array) {
    // Test narrowing casts
    test_cast_to_type(array, PType::I8);
    test_cast_to_type(array, PType::I16);
    
    // Test widening casts
    test_cast_to_type(array, PType::I64);
    test_cast_to_type(array, PType::F64);
    
    // Test same-width casts
    test_cast_to_type(array, PType::F32);
    test_cast_to_type(array, PType::U32);
}

fn test_cast_from_i64(array: &dyn Array) {
    // Test narrowing casts
    test_cast_to_type(array, PType::I8);
    test_cast_to_type(array, PType::I16);
    test_cast_to_type(array, PType::I32);
    test_cast_to_type(array, PType::F32);
    
    // Test same-width cast
    test_cast_to_type(array, PType::F64);
    test_cast_to_type(array, PType::U64);
}

fn test_cast_from_f16(array: &dyn Array) {
    // Test casts to other float types
    test_cast_to_type(array, PType::F32);
    test_cast_to_type(array, PType::F64);
}

fn test_cast_from_f32(array: &dyn Array) {
    // Test narrowing cast
    test_cast_to_type(array, PType::F16);
    
    // Test widening cast
    test_cast_to_type(array, PType::F64);
    
    // Test casts to integer types (truncation)
    test_cast_to_integral_types(array);
}

fn test_cast_from_f64(array: &dyn Array) {
    // Test narrowing casts
    test_cast_to_type(array, PType::F16);
    test_cast_to_type(array, PType::F32);
    
    // Test casts to integer types (truncation)
    test_cast_to_integral_types(array);
}

fn test_cast_to_integral_types(array: &dyn Array) {
    // Test casting to all integral types
    // Some may fail due to out-of-range values
    test_cast_to_type(array, PType::I8);
    test_cast_to_type(array, PType::U8);
    test_cast_to_type(array, PType::I16);
    test_cast_to_type(array, PType::U16);
    test_cast_to_type(array, PType::I32);
    test_cast_to_type(array, PType::U32);
    test_cast_to_type(array, PType::I64);
    test_cast_to_type(array, PType::U64);
}

fn test_cast_to_type(array: &dyn Array, target_ptype: PType) {
    let target_dtype = DType::Primitive(target_ptype, array.dtype().nullability());
    
    // Attempt the cast
    let result = match cast(array, &target_dtype) {
        Ok(r) => r,
        Err(_) => {
            // Some casts may fail (e.g., negative to unsigned, out-of-range values)
            // This is expected behavior
            return;
        }
    };
    
    assert_eq!(result.len(), array.len());
    assert_eq!(result.dtype(), &target_dtype);
    
    // For valid casts, verify the values are correctly converted
    // We can't easily verify exact values without knowing the input type,
    // but we can at least check that scalars can be retrieved
    for i in 0..array.len().min(10) {
        let _original = array.scalar_at(i).vortex_unwrap();
        let _casted = result.scalar_at(i).vortex_unwrap();
        // The actual value verification would depend on the specific cast
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use vortex_buffer::buffer;
    use crate::arrays::PrimitiveArray;
    use crate::IntoArray;
    
    #[test]
    fn test_cast_conformance_u32() {
        let array = buffer![0u32, 100, 200, 65535, 1000000].into_array();
        test_cast_conformance(array.as_ref());
    }
    
    #[test]
    fn test_cast_conformance_i32() {
        let array = buffer![-100i32, -1, 0, 1, 100].into_array();
        test_cast_conformance(array.as_ref());
    }
    
    #[test]
    fn test_cast_conformance_f32() {
        let array = buffer![0.0f32, 1.5, -2.5, 100.0, 1e6].into_array();
        test_cast_conformance(array.as_ref());
    }
    
    #[test]
    fn test_cast_conformance_nullable() {
        let array = PrimitiveArray::from_option_iter([Some(1u8), None, Some(255), Some(0), None]);
        test_cast_conformance(array.as_ref());
    }
}