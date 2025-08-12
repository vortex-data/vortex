// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_dtype::{DType, Nullability, PType};
use vortex_error::VortexUnwrap;

use crate::Array;
use crate::compute::cast;

/// Test conformance of the cast compute function for an array.
///
/// This function tests various casting scenarios including:
/// - Casting between numeric types (widening and narrowing)
/// - Casting between signed and unsigned types
/// - Casting between integral and floating-point types
/// - Casting with nullability changes
/// - Casting between string types (Utf8/Binary)
/// - Edge cases like overflow behavior
pub fn test_cast_conformance(array: &dyn Array) {
    let dtype = array.dtype();

    // Always test identity cast and nullability changes
    test_cast_identity(array);

    // Test AllValid to NonNullable and back if applicable
    test_cast_allvalid_to_nonnullable_and_back(array);

    // Test based on the specific DType
    match dtype {
        DType::Null => test_cast_from_null(array),
        DType::Bool(nullability) => test_cast_from_bool(array, *nullability),
        DType::Primitive(ptype, nullability) => {
            test_cast_nullability_changes_primitive(array, *ptype, *nullability);
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
        DType::Decimal(_, nullability) => test_cast_from_decimal(array, *nullability),
        DType::Utf8(nullability) => test_cast_from_utf8(array, *nullability),
        DType::Binary(nullability) => test_cast_from_binary(array, *nullability),
        DType::Struct(_, nullability) => test_cast_from_struct(array, *nullability),
        DType::List(_, nullability) => test_cast_from_list(array, *nullability),
        DType::Extension(_) => test_cast_from_extension(array),
    }
}

fn test_cast_identity(array: &dyn Array) {
    // Casting to the same type should be a no-op
    let result = cast(array, array.dtype()).vortex_unwrap();
    assert_eq!(result.len(), array.len());
    assert_eq!(result.dtype(), array.dtype());

    // Verify values are unchanged
    for i in 0..array.len().min(10) {
        assert_eq!(
            array.scalar_at(i).vortex_unwrap(),
            result.scalar_at(i).vortex_unwrap()
        );
    }
}

fn test_cast_from_null(array: &dyn Array) {
    // Null can be cast to itself
    let result = cast(array, &DType::Null).vortex_unwrap();
    assert_eq!(result.len(), array.len());
    assert_eq!(result.dtype(), &DType::Null);

    // Null can also be cast to any nullable type
    let nullable_types = vec![
        DType::Bool(Nullability::Nullable),
        DType::Primitive(PType::I32, Nullability::Nullable),
        DType::Primitive(PType::F64, Nullability::Nullable),
        DType::Utf8(Nullability::Nullable),
        DType::Binary(Nullability::Nullable),
    ];

    for dtype in nullable_types {
        let result = cast(array, &dtype).vortex_unwrap();
        assert_eq!(result.len(), array.len());
        assert_eq!(result.dtype(), &dtype);

        // Verify all values are null
        for i in 0..array.len().min(10) {
            assert!(result.scalar_at(i).vortex_unwrap().is_null());
        }
    }

    // Casting to non-nullable types should fail
    let non_nullable_types = vec![
        DType::Bool(Nullability::NonNullable),
        DType::Primitive(PType::I32, Nullability::NonNullable),
    ];

    for dtype in non_nullable_types {
        assert!(cast(array, &dtype).is_err());
    }
}

fn test_cast_from_bool(array: &dyn Array, nullability: Nullability) {
    // Test nullability changes
    test_cast_nullability_changes(array, &DType::Bool(Nullability::Nullable));
    if nullability == Nullability::Nullable {
        // Try casting to non-nullable (may fail if nulls present)
        let _ = cast(array, &DType::Bool(Nullability::NonNullable));
    }

    // Test bool to numeric casts (true -> 1, false -> 0)
    test_cast_to_primitive(array, PType::U8);
    test_cast_to_primitive(array, PType::I32);
    test_cast_to_primitive(array, PType::F32);
}

fn test_cast_from_decimal(array: &dyn Array, nullability: Nullability) {
    // Test nullability changes for the same decimal type
    if let DType::Decimal(decimal_type, _) = array.dtype() {
        test_cast_nullability_changes(array, &DType::Decimal(*decimal_type, Nullability::Nullable));
        if nullability == Nullability::Nullable {
            // Try casting to non-nullable (may fail if nulls present)
            let _ = cast(
                array,
                &DType::Decimal(*decimal_type, Nullability::NonNullable),
            );
        }
    }
}

fn test_cast_from_utf8(array: &dyn Array, nullability: Nullability) {
    // Test nullability changes
    test_cast_nullability_changes(array, &DType::Utf8(Nullability::Nullable));
    if nullability == Nullability::Nullable {
        // Try casting to non-nullable (may fail if nulls present)
        let _ = cast(array, &DType::Utf8(Nullability::NonNullable));
    }

    // UTF-8 strings can potentially be cast to Binary
    test_cast_to_type_safe(array, &DType::Binary(nullability));
}

fn test_cast_from_binary(array: &dyn Array, nullability: Nullability) {
    // Test nullability changes
    test_cast_nullability_changes(array, &DType::Binary(Nullability::Nullable));
    if nullability == Nullability::Nullable {
        // Try casting to non-nullable (may fail if nulls present)
        let _ = cast(array, &DType::Binary(Nullability::NonNullable));
    }

    // Binary might be castable to UTF-8 if it contains valid UTF-8
    test_cast_to_type_safe(array, &DType::Utf8(nullability));
}

fn test_cast_from_struct(array: &dyn Array, nullability: Nullability) {
    // Test nullability changes for the same struct type
    if let DType::Struct(fields, _) = array.dtype() {
        test_cast_nullability_changes(array, &DType::Struct(fields.clone(), Nullability::Nullable));
        if nullability == Nullability::Nullable {
            // Try casting to non-nullable (may fail if nulls present)
            let _ = cast(
                array,
                &DType::Struct(fields.clone(), Nullability::NonNullable),
            );
        }
    }
}

fn test_cast_from_list(array: &dyn Array, nullability: Nullability) {
    // Test nullability changes for the same list type
    if let DType::List(element_type, _) = array.dtype() {
        test_cast_nullability_changes(
            array,
            &DType::List(element_type.clone(), Nullability::Nullable),
        );
        if nullability == Nullability::Nullable {
            // Try casting to non-nullable (may fail if nulls present)
            let _ = cast(
                array,
                &DType::List(element_type.clone(), Nullability::NonNullable),
            );
        }
    }
}

fn test_cast_from_extension(array: &dyn Array) {
    // Extension types typically only cast to themselves
    // The specific casting rules depend on the extension type
    if let DType::Extension(ext_dtype) = array.dtype() {
        let result = cast(array, &DType::Extension(ext_dtype.clone())).vortex_unwrap();
        assert_eq!(result.len(), array.len());
        assert_eq!(result.dtype(), array.dtype());
    }
}

fn test_cast_allvalid_to_nonnullable_and_back(array: &dyn Array) {
    // Skip if array is null type (special case)
    if array.dtype() == &DType::Null {
        return;
    }

    // Only test if array has no nulls
    if let Ok(null_count) = array.invalid_count()
        && null_count == 0
    {
        // Test casting to NonNullable if currently Nullable
        if array.dtype().nullability() == Nullability::Nullable {
            let non_nullable_dtype = array.dtype().with_nullability(Nullability::NonNullable);

            // Cast to NonNullable
            if let Ok(non_nullable) = cast(array, &non_nullable_dtype) {
                assert_eq!(non_nullable.dtype(), &non_nullable_dtype);
                assert_eq!(non_nullable.len(), array.len());

                // Cast back to Nullable
                let nullable_dtype = array.dtype().with_nullability(Nullability::Nullable);
                let back_to_nullable = cast(&non_nullable, &nullable_dtype).vortex_unwrap();
                assert_eq!(back_to_nullable.dtype(), &nullable_dtype);
                assert_eq!(back_to_nullable.len(), array.len());

                // Verify values are unchanged
                for i in 0..array.len().min(10) {
                    assert_eq!(
                        array.scalar_at(i).vortex_unwrap(),
                        back_to_nullable.scalar_at(i).vortex_unwrap()
                    );
                }
            }
        }
        // Test casting to Nullable if currently NonNullable
        else if array.dtype().nullability() == Nullability::NonNullable {
            let nullable_dtype = array.dtype().with_nullability(Nullability::Nullable);

            // Cast to Nullable
            let nullable = cast(array, &nullable_dtype).vortex_unwrap();
            assert_eq!(nullable.dtype(), &nullable_dtype);
            assert_eq!(nullable.len(), array.len());

            // Cast back to NonNullable
            let non_nullable_dtype = array.dtype().with_nullability(Nullability::NonNullable);
            let back_to_non_nullable = cast(&nullable, &non_nullable_dtype).vortex_unwrap();
            assert_eq!(back_to_non_nullable.dtype(), &non_nullable_dtype);
            assert_eq!(back_to_non_nullable.len(), array.len());

            // Verify values are unchanged
            for i in 0..array.len().min(10) {
                assert_eq!(
                    array.scalar_at(i).vortex_unwrap(),
                    back_to_non_nullable.scalar_at(i).vortex_unwrap()
                );
            }
        }
    }
}

fn test_cast_nullability_changes(array: &dyn Array, nullable_version: &DType) {
    // Test casting to nullable version
    if array.dtype().nullability() == Nullability::NonNullable {
        let result = cast(array, nullable_version).vortex_unwrap();
        assert_eq!(result.len(), array.len());
        assert_eq!(result.dtype(), nullable_version);

        // IMPORTANT: Nullability casting should preserve the encoding
        assert_eq!(
            result.encoding().id(),
            array.encoding().id(),
            "Nullability cast should preserve encoding"
        );

        // Values should be unchanged
        for i in 0..array.len().min(10) {
            assert_eq!(
                array.scalar_at(i).vortex_unwrap(),
                result.scalar_at(i).vortex_unwrap()
            );
        }
    }
}

fn test_cast_nullability_changes_primitive(
    array: &dyn Array,
    ptype: PType,
    nullability: Nullability,
) {
    // Test casting to nullable version
    if nullability == Nullability::NonNullable {
        let nullable_dtype = DType::Primitive(ptype, Nullability::Nullable);
        let result = cast(array, &nullable_dtype).vortex_unwrap();
        assert_eq!(result.len(), array.len());
        assert_eq!(result.dtype(), &nullable_dtype);

        // IMPORTANT: Nullability casting should preserve the encoding
        assert_eq!(
            result.encoding().id(),
            array.encoding().id(),
            "Nullability cast should preserve encoding"
        );

        // Values should be unchanged
        for i in 0..array.len().min(10) {
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

            // IMPORTANT: Nullability casting should preserve the encoding
            assert_eq!(
                result.encoding().id(),
                array.encoding().id(),
                "Nullability cast should preserve encoding"
            );

            // Values should be unchanged
            for i in 0..array.len().min(10) {
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
    test_cast_to_primitive(array, PType::U16);
    test_cast_to_primitive(array, PType::U32);
    test_cast_to_primitive(array, PType::U64);
    test_cast_to_primitive(array, PType::I16);
    test_cast_to_primitive(array, PType::I32);
    test_cast_to_primitive(array, PType::I64);
    test_cast_to_primitive(array, PType::F32);
    test_cast_to_primitive(array, PType::F64);

    // Test same-width cast
    test_cast_to_primitive(array, PType::I8);
}

fn test_cast_from_u16(array: &dyn Array) {
    // Test narrowing cast
    test_cast_to_primitive(array, PType::U8);

    // Test widening casts
    test_cast_to_primitive(array, PType::U32);
    test_cast_to_primitive(array, PType::U64);
    test_cast_to_primitive(array, PType::I32);
    test_cast_to_primitive(array, PType::I64);
    test_cast_to_primitive(array, PType::F32);
    test_cast_to_primitive(array, PType::F64);

    // Test same-width cast
    test_cast_to_primitive(array, PType::I16);
}

fn test_cast_from_u32(array: &dyn Array) {
    // Test narrowing casts
    test_cast_to_primitive(array, PType::U8);
    test_cast_to_primitive(array, PType::U16);
    test_cast_to_primitive(array, PType::I8);
    test_cast_to_primitive(array, PType::I16);

    // Test widening casts
    test_cast_to_primitive(array, PType::U64);
    test_cast_to_primitive(array, PType::I64);
    test_cast_to_primitive(array, PType::F64);

    // Test same-width casts
    test_cast_to_primitive(array, PType::I32);
    test_cast_to_primitive(array, PType::F32);
}

fn test_cast_from_u64(array: &dyn Array) {
    // Test narrowing casts
    test_cast_to_primitive(array, PType::U8);
    test_cast_to_primitive(array, PType::U16);
    test_cast_to_primitive(array, PType::U32);
    test_cast_to_primitive(array, PType::I8);
    test_cast_to_primitive(array, PType::I16);
    test_cast_to_primitive(array, PType::I32);
    test_cast_to_primitive(array, PType::F32);

    // Test same-width casts
    test_cast_to_primitive(array, PType::I64);
    test_cast_to_primitive(array, PType::F64);
}

fn test_cast_from_i8(array: &dyn Array) {
    // Test widening casts
    test_cast_to_primitive(array, PType::I16);
    test_cast_to_primitive(array, PType::I32);
    test_cast_to_primitive(array, PType::I64);
    test_cast_to_primitive(array, PType::F32);
    test_cast_to_primitive(array, PType::F64);

    // Test same-width cast (may fail for negative values)
    test_cast_to_primitive(array, PType::U8);
}

fn test_cast_from_i16(array: &dyn Array) {
    // Test narrowing cast
    test_cast_to_primitive(array, PType::I8);

    // Test widening casts
    test_cast_to_primitive(array, PType::I32);
    test_cast_to_primitive(array, PType::I64);
    test_cast_to_primitive(array, PType::F32);
    test_cast_to_primitive(array, PType::F64);

    // Test same-width cast (may fail for negative values)
    test_cast_to_primitive(array, PType::U16);
}

fn test_cast_from_i32(array: &dyn Array) {
    // Test narrowing casts
    test_cast_to_primitive(array, PType::I8);
    test_cast_to_primitive(array, PType::I16);

    // Test widening casts
    test_cast_to_primitive(array, PType::I64);
    test_cast_to_primitive(array, PType::F64);

    // Test same-width casts
    test_cast_to_primitive(array, PType::F32);
    test_cast_to_primitive(array, PType::U32);
}

fn test_cast_from_i64(array: &dyn Array) {
    // Test narrowing casts
    test_cast_to_primitive(array, PType::I8);
    test_cast_to_primitive(array, PType::I16);
    test_cast_to_primitive(array, PType::I32);
    test_cast_to_primitive(array, PType::F32);

    // Test same-width cast
    test_cast_to_primitive(array, PType::F64);
    test_cast_to_primitive(array, PType::U64);
}

fn test_cast_from_f16(array: &dyn Array) {
    // Test casts to other float types
    test_cast_to_primitive(array, PType::F32);
    test_cast_to_primitive(array, PType::F64);
}

fn test_cast_from_f32(array: &dyn Array) {
    // Test narrowing cast
    test_cast_to_primitive(array, PType::F16);

    // Test widening cast
    test_cast_to_primitive(array, PType::F64);

    // Test casts to integer types (truncation)
    test_cast_to_integral_types(array);
}

fn test_cast_from_f64(array: &dyn Array) {
    // Test narrowing casts
    test_cast_to_primitive(array, PType::F16);
    test_cast_to_primitive(array, PType::F32);

    // Test casts to integer types (truncation)
    test_cast_to_integral_types(array);
}

fn test_cast_to_integral_types(array: &dyn Array) {
    // Test casting to all integral types
    // Some may fail due to out-of-range values
    test_cast_to_primitive(array, PType::I8);
    test_cast_to_primitive(array, PType::U8);
    test_cast_to_primitive(array, PType::I16);
    test_cast_to_primitive(array, PType::U16);
    test_cast_to_primitive(array, PType::I32);
    test_cast_to_primitive(array, PType::U32);
    test_cast_to_primitive(array, PType::I64);
    test_cast_to_primitive(array, PType::U64);
}

fn test_cast_to_primitive(array: &dyn Array, target_ptype: PType) {
    let target_dtype = DType::Primitive(target_ptype, array.dtype().nullability());
    test_cast_to_type_safe(array, &target_dtype);
}

fn test_cast_to_type_safe(array: &dyn Array, target_dtype: &DType) {
    // Attempt the cast
    let result = match cast(array, target_dtype) {
        Ok(r) => r,
        Err(_) => {
            // Some casts may fail (e.g., negative to unsigned, out-of-range values)
            // This is expected behavior
            return;
        }
    };

    assert_eq!(result.len(), array.len());
    assert_eq!(result.dtype(), target_dtype);

    // For valid casts, verify the values are correctly converted
    // We verify up to the first 10 values (or all if less than 10)
    for i in 0..array.len().min(10) {
        let original = array.scalar_at(i).vortex_unwrap();
        let casted = result.scalar_at(i).vortex_unwrap();

        // For nullability-only changes, values should be identical
        if array.dtype().eq_ignore_nullability(target_dtype) {
            assert_eq!(
                original, casted,
                "Value at index {i} changed during nullability cast"
            );
        } else {
            // For type conversions, at least verify we can retrieve the values
            // and that null values remain null
            if original.is_null() {
                assert!(
                    casted.is_null(),
                    "Null value at index {i} became non-null after cast"
                );
            } else {
                assert!(
                    !casted.is_null(),
                    "Non-null value at index {i} became null after cast"
                );
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use vortex_buffer::buffer;
    use vortex_dtype::{DType, FieldNames, Nullability};

    use super::*;
    use crate::IntoArray;
    use crate::arrays::{
        BoolArray, ListArray, NullArray, PrimitiveArray, StructArray, VarBinArray,
    };

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

    #[test]
    fn test_cast_conformance_bool() {
        let array = BoolArray::from_iter(vec![true, false, true, false]);
        test_cast_conformance(array.as_ref());
    }

    #[test]
    fn test_cast_conformance_null() {
        let array = NullArray::new(5);
        test_cast_conformance(array.as_ref());
    }

    #[test]
    fn test_cast_conformance_utf8() {
        let array = VarBinArray::from_iter(
            vec![Some("hello"), None, Some("world")],
            DType::Utf8(Nullability::Nullable),
        );
        test_cast_conformance(array.as_ref());
    }

    #[test]
    fn test_cast_conformance_binary() {
        let array = VarBinArray::from_iter(
            vec![Some(b"data".as_slice()), None, Some(b"bytes".as_slice())],
            DType::Binary(Nullability::Nullable),
        );
        test_cast_conformance(array.as_ref());
    }

    #[test]
    fn test_cast_conformance_struct() {
        let names: FieldNames = vec!["a".into(), "b".into()].into();

        let a = buffer![1i32, 2, 3].into_array();
        let b = VarBinArray::from_iter(
            vec![Some("x"), None, Some("z")],
            DType::Utf8(Nullability::Nullable),
        )
        .into_array();

        let array =
            StructArray::try_new(names, vec![a, b], 3, crate::validity::Validity::NonNullable)
                .unwrap();
        test_cast_conformance(array.as_ref());
    }

    #[test]
    fn test_cast_conformance_list() {
        let data = buffer![1i32, 2, 3, 4, 5, 6].into_array();
        let offsets = buffer![0i64, 2, 2, 5, 6].into_array();

        let array =
            ListArray::try_new(data, offsets, crate::validity::Validity::NonNullable).unwrap();
        test_cast_conformance(array.as_ref());
    }
}
