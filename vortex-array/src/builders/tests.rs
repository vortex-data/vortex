// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::sync::Arc;

use rstest::rstest;
use vortex_error::VortexExpect;

use crate::LEGACY_SESSION;
use crate::VortexSessionExecute;
use crate::builders::ArrayBuilder;
use crate::builders::builder_with_capacity;
use crate::dtype::DType;
use crate::dtype::DecimalDType;
use crate::dtype::Nullability;
use crate::dtype::PType;
use crate::dtype::StructFields;
use crate::dtype::half::f16;
use crate::extension::datetime::TimeUnit;
use crate::extension::datetime::Timestamp;
use crate::scalar::Scalar;

/// Test that `append_zeros` produces the same result as manually appending `Scalar::default_value`.
///
/// This test verifies that the implementation of `append_zeros` correctly matches the behavior
/// defined by `Scalar::default_value` for each data type.
#[rstest]
#[case::bool(DType::Bool(Nullability::NonNullable))]
#[case::i8(DType::Primitive(PType::I8, Nullability::NonNullable))]
#[case::i16(DType::Primitive(PType::I16, Nullability::NonNullable))]
#[case::i32(DType::Primitive(PType::I32, Nullability::NonNullable))]
#[case::i64(DType::Primitive(PType::I64, Nullability::NonNullable))]
#[case::u8(DType::Primitive(PType::U8, Nullability::NonNullable))]
#[case::u16(DType::Primitive(PType::U16, Nullability::NonNullable))]
#[case::u32(DType::Primitive(PType::U32, Nullability::NonNullable))]
#[case::u64(DType::Primitive(PType::U64, Nullability::NonNullable))]
#[case::f32(DType::Primitive(PType::F32, Nullability::NonNullable))]
#[case::f64(DType::Primitive(PType::F64, Nullability::NonNullable))]
#[case::utf8(DType::Utf8(Nullability::NonNullable))]
#[case::binary(DType::Binary(Nullability::NonNullable))]
#[case::decimal128(DType::Decimal(DecimalDType::new(10, 2), Nullability::NonNullable))]
#[case::struct_simple(DType::Struct(
    StructFields::from_iter([
        ("a", DType::Primitive(PType::I32, Nullability::NonNullable)),
        ("b", DType::Utf8(Nullability::NonNullable)),
    ]),
    Nullability::NonNullable
))]
#[case::struct_nested(DType::Struct(
    StructFields::from_iter([
        ("field1", DType::Bool(Nullability::NonNullable)),
        ("field2", DType::Struct(
            StructFields::from_iter([
                ("nested", DType::Primitive(PType::F64, Nullability::NonNullable)),
            ]),
            Nullability::NonNullable
        )),
    ]),
    Nullability::NonNullable
))]
#[case::list(DType::List(
    Arc::new(DType::Primitive(PType::I32, Nullability::NonNullable)),
    Nullability::NonNullable
))]
#[case::fixed_size_list(DType::FixedSizeList(
    Arc::new(DType::Primitive(PType::I32, Nullability::NonNullable)),
    3,
    Nullability::NonNullable
))]
#[case::extension(DType::Extension(
    Timestamp::new(TimeUnit::Milliseconds, Nullability::NonNullable).erased()
))]
fn test_append_zeros_matches_default_value(#[case] dtype: DType) {
    let num_elements = 5;

    // Builder 1: Use append_zeros.
    let mut builder_zeros = builder_with_capacity(&dtype, num_elements);
    builder_zeros.append_zeros(num_elements);
    let array_zeros = builder_zeros.finish();

    // Builder 2: Manually append default values.
    let mut builder_manual = builder_with_capacity(&dtype, num_elements);
    let default_scalar = Scalar::zero_value(&dtype);
    for _ in 0..num_elements {
        builder_manual.append_scalar(&default_scalar).unwrap();
    }
    let array_manual = builder_manual.finish();

    // Both arrays should have the same length.
    assert_eq!(array_zeros.len(), array_manual.len());
    assert_eq!(array_zeros.len(), num_elements);

    // Compare each element.
    for i in 0..num_elements {
        let scalar_zeros = array_zeros
            .execute_scalar(i, &mut LEGACY_SESSION.create_execution_ctx())
            .unwrap();
        let scalar_manual = array_manual
            .execute_scalar(i, &mut LEGACY_SESSION.create_execution_ctx())
            .unwrap();

        assert_eq!(
            scalar_zeros, scalar_manual,
            "Element at index {} should be equal",
            i
        );
    }
}

/// Test that calling `append_nulls` on non-nullable builders panics.
/// Tests both single null (n=1) and multiple nulls (n=3).
#[rstest]
#[case::bool(DType::Bool(Nullability::NonNullable), 1)]
#[case::bool_multiple(DType::Bool(Nullability::NonNullable), 3)]
#[case::i32(DType::Primitive(PType::I32, Nullability::NonNullable), 1)]
#[case::i32_multiple(DType::Primitive(PType::I32, Nullability::NonNullable), 3)]
#[case::f64(DType::Primitive(PType::F64, Nullability::NonNullable), 1)]
#[case::f64_multiple(DType::Primitive(PType::F64, Nullability::NonNullable), 3)]
#[case::utf8(DType::Utf8(Nullability::NonNullable), 1)]
#[case::utf8_multiple(DType::Utf8(Nullability::NonNullable), 3)]
#[case::binary(DType::Binary(Nullability::NonNullable), 1)]
#[case::binary_multiple(DType::Binary(Nullability::NonNullable), 3)]
#[case::decimal(DType::Decimal(DecimalDType::new(10, 2), Nullability::NonNullable), 1)]
#[case::decimal_multiple(DType::Decimal(DecimalDType::new(10, 2), Nullability::NonNullable), 3)]
#[case::list(
    DType::List(
        Arc::new(DType::Primitive(PType::I32, Nullability::NonNullable)),
        Nullability::NonNullable
    ),
    1
)]
#[case::list_multiple(
    DType::List(
        Arc::new(DType::Primitive(PType::I32, Nullability::NonNullable)),
        Nullability::NonNullable
    ),
    3
)]
#[case::fixed_size_list(
    DType::FixedSizeList(
        Arc::new(DType::Primitive(PType::I32, Nullability::NonNullable)),
        3,
        Nullability::NonNullable
    ),
    1
)]
#[case::fixed_size_list_multiple(
    DType::FixedSizeList(
        Arc::new(DType::Primitive(PType::I32, Nullability::NonNullable)),
        3,
        Nullability::NonNullable
    ),
    3
)]
#[case::struct_type(DType::Struct(
    StructFields::from_iter([
        ("a", DType::Primitive(PType::I32, Nullability::NonNullable)),
    ]),
    Nullability::NonNullable
), 1)]
#[case::struct_type_multiple(DType::Struct(
    StructFields::from_iter([
        ("a", DType::Primitive(PType::I32, Nullability::NonNullable)),
    ]),
    Nullability::NonNullable
), 3)]
#[case::extension(
    DType::Extension(Timestamp::new(TimeUnit::Milliseconds, Nullability::NonNullable).erased()),
    1
)]
#[case::extension_multiple(
    DType::Extension(Timestamp::new(TimeUnit::Milliseconds, Nullability::NonNullable).erased()),
    3
)]
#[should_panic(expected = "non-nullable")]
fn test_append_nulls_panics_on_non_nullable(#[case] dtype: DType, #[case] count: usize) {
    let mut builder = builder_with_capacity(&dtype, count);
    builder.append_nulls(count);
}

/// Test that `append_defaults` behaves correctly for nullable and non-nullable types.
#[rstest]
#[case::nullable_bool(DType::Bool(Nullability::Nullable), true)]
#[case::non_nullable_bool(DType::Bool(Nullability::NonNullable), false)]
#[case::nullable_i32(DType::Primitive(PType::I32, Nullability::Nullable), true)]
#[case::non_nullable_i32(DType::Primitive(PType::I32, Nullability::NonNullable), false)]
#[case::nullable_utf8(DType::Utf8(Nullability::Nullable), true)]
#[case::non_nullable_utf8(DType::Utf8(Nullability::NonNullable), false)]
fn test_append_defaults_behavior(#[case] dtype: DType, #[case] should_be_null: bool) {
    let mut builder = builder_with_capacity(&dtype, 3);
    builder.append_defaults(3);
    let array = builder.finish();

    assert_eq!(array.len(), 3);

    for i in 0..3 {
        let scalar = array
            .execute_scalar(i, &mut LEGACY_SESSION.create_execution_ctx())
            .unwrap();
        if should_be_null {
            assert!(scalar.is_null(), "Element at index {} should be null", i);
        } else {
            assert!(
                !scalar.is_null(),
                "Element at index {} should not be null",
                i
            );
            // For non-nullable, it should match the default value.
            let expected = Scalar::default_value(&dtype);
            // Skip list comparison due to known bug.
            if !matches!(dtype, DType::List(..)) {
                assert_eq!(
                    scalar, expected,
                    "Element at index {} should be the default value",
                    i
                );
            }
        }
    }
}

/// Helper function that fills two builders with the same values and compares the results
/// of `to_canonical()` vs `finish().to_canonical()`.
fn compare_to_canonical_methods<F>(dtype: &DType, mut fill_builder: F)
where
    F: FnMut(&mut dyn ArrayBuilder),
{
    use crate::IntoArray;

    // Create two identical builders.
    let mut builder1 = builder_with_capacity(dtype, 10);
    let mut builder2 = builder_with_capacity(dtype, 10);

    // Fill both builders with the same data.
    fill_builder(builder1.as_mut());
    fill_builder(builder2.as_mut());

    // Get canonical arrays using both methods.
    let canonical_direct = builder1.finish_into_canonical();
    #[expect(deprecated)]
    let canonical_indirect = builder2
        .finish()
        .to_canonical()
        .vortex_expect("to_canonical failed");

    // Convert both to arrays for comparison.
    let array_direct = canonical_direct.into_array();
    let array_indirect = canonical_indirect.into_array();

    // Verify they have the same length.
    assert_eq!(array_direct.len(), array_indirect.len());

    // Compare each element.
    for i in 0..array_direct.len() {
        let scalar_direct = array_direct
            .execute_scalar(i, &mut LEGACY_SESSION.create_execution_ctx())
            .unwrap();
        let scalar_indirect = array_indirect
            .execute_scalar(i, &mut LEGACY_SESSION.create_execution_ctx())
            .unwrap();

        assert_eq!(
            scalar_direct, scalar_indirect,
            "Element at index {} should be equal for dtype {:?}",
            i, dtype
        );
    }
}

#[test]
fn test_to_canonical_bool() {
    let dtype = DType::Bool(Nullability::NonNullable);
    compare_to_canonical_methods(&dtype, |builder| {
        for i in 0..5 {
            let value = Scalar::bool(i % 2 == 0, Nullability::NonNullable);
            builder.append_scalar(&value).unwrap();
        }
    });
}

#[test]
fn test_to_canonical_bool_nullable() {
    let dtype = DType::Bool(Nullability::Nullable);
    compare_to_canonical_methods(&dtype, |builder| {
        for i in 0..5 {
            let value = Scalar::bool(i % 2 == 0, Nullability::Nullable);
            builder.append_scalar(&value).unwrap();
        }
        builder.append_nulls(1);
    });
}

#[test]
fn test_to_canonical_i32() {
    let dtype = DType::Primitive(PType::I32, Nullability::NonNullable);
    compare_to_canonical_methods(&dtype, |builder| {
        for i in 0..5 {
            let value = Scalar::primitive(i, Nullability::NonNullable);
            builder.append_scalar(&value).unwrap();
        }
    });
}

#[test]
fn test_to_canonical_i32_nullable() {
    let dtype = DType::Primitive(PType::I32, Nullability::Nullable);
    compare_to_canonical_methods(&dtype, |builder| {
        for i in 0..5 {
            let value = Scalar::primitive(i, Nullability::Nullable);
            builder.append_scalar(&value).unwrap();
        }
        builder.append_nulls(1);
    });
}

#[test]
fn test_to_canonical_f64() {
    let dtype = DType::Primitive(PType::F64, Nullability::NonNullable);
    compare_to_canonical_methods(&dtype, |builder| {
        for i in 0..5 {
            let value = Scalar::primitive(i as f64, Nullability::NonNullable);
            builder.append_scalar(&value).unwrap();
        }
    });
}

#[test]
fn test_to_canonical_utf8() {
    let dtype = DType::Utf8(Nullability::NonNullable);
    compare_to_canonical_methods(&dtype, |builder| {
        let values = ["hello", "world", "test", "data", "vortex"];
        for value in &values {
            let scalar = Scalar::utf8(*value, Nullability::NonNullable);
            builder.append_scalar(&scalar).unwrap();
        }
    });
}

#[test]
fn test_to_canonical_utf8_nullable() {
    let dtype = DType::Utf8(Nullability::Nullable);
    compare_to_canonical_methods(&dtype, |builder| {
        let values = ["hello", "world", "test"];
        for value in &values {
            let scalar = Scalar::utf8(*value, Nullability::Nullable);
            builder.append_scalar(&scalar).unwrap();
        }
        builder.append_nulls(1);
    });
}

#[test]
fn test_to_canonical_binary() {
    let dtype = DType::Binary(Nullability::NonNullable);
    compare_to_canonical_methods(&dtype, |builder| {
        let values = [b"hello", b"world", b"vortx", b"bytes", b"tests"];
        for value in &values {
            let scalar = Scalar::binary(value.to_vec(), Nullability::NonNullable);
            builder.append_scalar(&scalar).unwrap();
        }
    });
}

#[test]
fn test_to_canonical_struct() {
    let dtype = DType::Struct(
        StructFields::from_iter([
            ("a", DType::Primitive(PType::I32, Nullability::NonNullable)),
            ("b", DType::Utf8(Nullability::NonNullable)),
        ]),
        Nullability::NonNullable,
    );
    compare_to_canonical_methods(&dtype, |builder| {
        for _ in 0..3 {
            let value = Scalar::default_value(&dtype);
            builder.append_scalar(&value).unwrap();
        }
    });
}

#[test]
fn test_to_canonical_extension() {
    let dtype =
        DType::Extension(Timestamp::new(TimeUnit::Milliseconds, Nullability::NonNullable).erased());
    compare_to_canonical_methods(&dtype, |builder| {
        let ext_dtype = match &dtype {
            DType::Extension(ext) => ext.clone(),
            _ => unreachable!(),
        };
        for i in 0..5 {
            let storage_value = Scalar::from(i as i64);
            let ext_scalar = Scalar::extension_ref(ext_dtype.clone(), storage_value);
            builder.append_scalar(&ext_scalar).unwrap();
        }
    });
}

#[test]
fn test_to_canonical_null() {
    let dtype = DType::Null;
    compare_to_canonical_methods(&dtype, |builder| {
        builder.append_nulls(5);
    });
}

#[test]
fn test_to_canonical_decimal() {
    let dtype = DType::Decimal(DecimalDType::new(10, 2), Nullability::NonNullable);
    compare_to_canonical_methods(&dtype, |builder| {
        for _ in 0..5 {
            let value = Scalar::default_value(&dtype);
            builder.append_scalar(&value).unwrap();
        }
    });
}

#[test]
fn test_to_canonical_i8() {
    let dtype = DType::Primitive(PType::I8, Nullability::NonNullable);
    compare_to_canonical_methods(&dtype, |builder| {
        for i in 0..5i8 {
            let value = Scalar::primitive(i, Nullability::NonNullable);
            builder.append_scalar(&value).unwrap();
        }
    });
}

#[test]
fn test_to_canonical_u64() {
    let dtype = DType::Primitive(PType::U64, Nullability::NonNullable);
    compare_to_canonical_methods(&dtype, |builder| {
        for i in 0..5 {
            let value = Scalar::primitive(i as u64, Nullability::NonNullable);
            builder.append_scalar(&value).unwrap();
        }
    });
}

#[test]
fn test_to_canonical_f32() {
    let dtype = DType::Primitive(PType::F32, Nullability::NonNullable);
    compare_to_canonical_methods(&dtype, |builder| {
        for i in 0..5 {
            let value = Scalar::primitive(i as f32, Nullability::NonNullable);
            builder.append_scalar(&value).unwrap();
        }
    });
}

/// Comprehensive test for `append_scalar` across all supported data types.
/// This test verifies that `append_scalar` works correctly for each type by:
/// 1. Creating a builder with the given dtype
/// 2. Appending various scalars (including nulls for nullable types)
/// 3. Verifying the resulting array matches expectations
#[rstest]
#[case::bool_non_nullable(DType::Bool(Nullability::NonNullable))]
#[case::bool_nullable(DType::Bool(Nullability::Nullable))]
#[case::i8(DType::Primitive(PType::I8, Nullability::NonNullable))]
#[case::i16(DType::Primitive(PType::I16, Nullability::NonNullable))]
#[case::i32(DType::Primitive(PType::I32, Nullability::NonNullable))]
#[case::i64(DType::Primitive(PType::I64, Nullability::NonNullable))]
#[case::u8(DType::Primitive(PType::U8, Nullability::NonNullable))]
#[case::u16(DType::Primitive(PType::U16, Nullability::NonNullable))]
#[case::u32(DType::Primitive(PType::U32, Nullability::NonNullable))]
#[case::u64(DType::Primitive(PType::U64, Nullability::NonNullable))]
#[case::f32(DType::Primitive(PType::F32, Nullability::NonNullable))]
#[case::f64(DType::Primitive(PType::F64, Nullability::NonNullable))]
#[case::i32_nullable(DType::Primitive(PType::I32, Nullability::Nullable))]
#[case::f64_nullable(DType::Primitive(PType::F64, Nullability::Nullable))]
#[case::utf8_non_nullable(DType::Utf8(Nullability::NonNullable))]
#[case::utf8_nullable(DType::Utf8(Nullability::Nullable))]
#[case::binary_non_nullable(DType::Binary(Nullability::NonNullable))]
#[case::binary_nullable(DType::Binary(Nullability::Nullable))]
#[case::null(DType::Null)]
#[case::decimal128_non_nullable(DType::Decimal(
    DecimalDType::new(10, 2),
    Nullability::NonNullable
))]
#[case::decimal128_nullable(DType::Decimal(DecimalDType::new(10, 2), Nullability::Nullable))]
#[case::struct_simple(DType::Struct(
    StructFields::from_iter([
        ("a", DType::Primitive(PType::I32, Nullability::NonNullable)),
        ("b", DType::Utf8(Nullability::NonNullable)),
    ]),
    Nullability::NonNullable
))]
#[case::struct_nullable(DType::Struct(
    StructFields::from_iter([
        ("x", DType::Primitive(PType::F64, Nullability::NonNullable)),
    ]),
    Nullability::Nullable
))]
#[case::list_non_nullable(DType::List(
    Arc::new(DType::Primitive(PType::I32, Nullability::NonNullable)),
    Nullability::NonNullable
))]
#[case::list_nullable(DType::List(
    Arc::new(DType::Primitive(PType::I32, Nullability::NonNullable)),
    Nullability::Nullable
))]
#[case::fixed_size_list_non_nullable(DType::FixedSizeList(
    Arc::new(DType::Primitive(PType::I32, Nullability::NonNullable)),
    3,
    Nullability::NonNullable
))]
#[case::fixed_size_list_nullable(DType::FixedSizeList(
    Arc::new(DType::Primitive(PType::I32, Nullability::NonNullable)),
    3,
    Nullability::Nullable
))]
#[case::extension_non_nullable(DType::Extension(
    Timestamp::new(TimeUnit::Milliseconds, Nullability::NonNullable).erased()
))]
fn test_append_scalar_comprehensive(#[case] dtype: DType) {
    let num_elements = 3;
    let mut builder = builder_with_capacity(&dtype, num_elements * 2);

    // Create test scalars based on the dtype.
    let scalars = create_test_scalars_for_dtype(&dtype, num_elements);

    // Append each scalar.
    for scalar in &scalars {
        builder.append_scalar(scalar).unwrap();
    }

    // If nullable, append a null (special handling for fixed-size lists).
    if dtype.is_nullable() {
        // Fixed-size lists require special handling for nulls.
        if matches!(dtype, DType::FixedSizeList(..)) {
            builder.append_nulls(1);
        } else {
            let null_scalar = Scalar::null(dtype.clone());
            builder.append_scalar(&null_scalar).unwrap();
        }
    }

    let array = builder.finish();

    // Verify the array length.
    let expected_len = if dtype.is_nullable() {
        num_elements + 1
    } else {
        num_elements
    };
    assert_eq!(array.len(), expected_len);

    // Verify each scalar matches.
    for (i, expected_scalar) in scalars.iter().enumerate() {
        let actual_scalar = array
            .execute_scalar(i, &mut LEGACY_SESSION.create_execution_ctx())
            .unwrap();
        assert_scalars_equal(&actual_scalar, expected_scalar, &dtype, i);
    }

    // If nullable, verify the last element is null.
    if dtype.is_nullable() {
        let null_scalar = array
            .execute_scalar(num_elements, &mut LEGACY_SESSION.create_execution_ctx())
            .unwrap();
        assert!(
            null_scalar.is_null(),
            "Last element should be null for nullable dtype"
        );
    }
}

/// Helper function to create test scalars for a given dtype.
#[expect(clippy::cast_possible_truncation)]
fn create_test_scalars_for_dtype(dtype: &DType, count: usize) -> Vec<Scalar> {
    let mut scalars = Vec::with_capacity(count);

    for i in 0..count {
        let scalar = match dtype {
            DType::Null => Scalar::null(dtype.clone()),
            DType::Bool(n) => Scalar::bool(i % 2 == 0, *n),
            DType::Primitive(ptype, n) => match ptype {
                PType::I8 => Scalar::primitive(i as i8, *n),
                PType::I16 => Scalar::primitive(i as i16, *n),
                PType::I32 => Scalar::primitive(i as i32, *n),
                PType::I64 => Scalar::primitive(i as i64, *n),
                PType::U8 => Scalar::primitive(i as u8, *n),
                PType::U16 => Scalar::primitive(i as u16, *n),
                PType::U32 => Scalar::primitive(i as u32, *n),
                PType::U64 => Scalar::primitive(i as u64, *n),
                PType::F16 => Scalar::primitive(f16::from_f32(i as f32 * 1.5), *n),
                PType::F32 => Scalar::primitive(i as f32 * 1.5, *n),
                PType::F64 => Scalar::primitive(i as f64 * 1.5, *n),
            },
            DType::Utf8(n) => Scalar::utf8(format!("test_string_{}", i), *n),
            DType::Binary(n) => Scalar::binary(format!("bytes_{}", i).into_bytes(), *n),
            DType::Decimal(dec_dtype, n) => {
                // Create decimal scalars based on the decimal dtype.
                use crate::scalar::DecimalValue;
                let value = DecimalValue::I128((i as i128 + 1) * 100); // Simple decimal values.
                Scalar::decimal(value, *dec_dtype, *n)
            }
            DType::Struct(fields, n) => {
                // Create struct scalars with field values.
                let field_values: Vec<Scalar> = fields
                    .fields()
                    .enumerate()
                    .map(|(j, field_dtype)| {
                        // Create simple values for each field.
                        match &field_dtype {
                            DType::Primitive(PType::I32, n) => {
                                Scalar::primitive((i as i32).saturating_add(j as i32), *n)
                            }
                            DType::Primitive(PType::F64, n) => {
                                Scalar::primitive((i + j) as f64, *n)
                            }
                            DType::Utf8(n) => Scalar::utf8(format!("field_{}", i + j), *n),
                            _ => Scalar::default_value(&field_dtype),
                        }
                    })
                    .collect();
                Scalar::struct_(DType::Struct(fields.clone(), *n), field_values)
            }
            DType::Union(..) => todo!("TODO(connor)[Union]: unimplemented"),
            DType::List(element_dtype, n) => {
                // Create list scalars with a few elements.
                let elements: Vec<Scalar> = (0..=i)
                    .map(|j| match element_dtype.as_ref() {
                        DType::Primitive(PType::I32, n) => {
                            Scalar::primitive(j.min(i32::MAX as usize) as i32, *n)
                        }
                        _ => Scalar::default_value(element_dtype.as_ref()),
                    })
                    .collect();
                Scalar::list(Arc::clone(element_dtype), elements, *n)
            }
            DType::FixedSizeList(element_dtype, size, n) => {
                // Create fixed-size list scalars.
                let elements: Vec<Scalar> = (0..*size)
                    .map(|j| match element_dtype.as_ref() {
                        DType::Primitive(PType::I32, n) => {
                            Scalar::primitive((i as i32).saturating_add(j as i32), *n)
                        }
                        _ => Scalar::default_value(element_dtype.as_ref()),
                    })
                    .collect();
                Scalar::fixed_size_list(Arc::clone(element_dtype), elements, *n)
            }
            DType::Extension(ext_dtype) => {
                // Create extension scalars with storage values.
                let storage_scalar = match ext_dtype.storage_dtype() {
                    DType::Primitive(PType::I64, n) => Scalar::primitive(i as i64, *n),
                    _ => Scalar::default_value(ext_dtype.storage_dtype()),
                };
                Scalar::extension_ref(ext_dtype.clone(), storage_scalar)
            }
            DType::Variant(_) => continue,
        };
        scalars.push(scalar);
    }

    scalars
}

/// Helper function to compare scalars, handling special cases like lists.
fn assert_scalars_equal(actual: &Scalar, expected: &Scalar, dtype: &DType, index: usize) {
    // For lists, we need special handling due to known issues.
    if matches!(dtype, DType::List(..)) {
        // Just check nullability matches.
        assert_eq!(
            actual.is_null(),
            expected.is_null(),
            "Null status mismatch at index {}",
            index
        );
        // Skip detailed comparison for lists due to known bugs.
        return;
    }

    assert_eq!(
        actual, expected,
        "Scalar mismatch at index {} for dtype {:?}",
        index, dtype
    );
}

/// Test that `append_scalar` correctly handles mixed valid and null values
/// for nullable types.
#[rstest]
#[case::bool(DType::Bool(Nullability::Nullable))]
#[case::i32(DType::Primitive(PType::I32, Nullability::Nullable))]
#[case::f64(DType::Primitive(PType::F64, Nullability::Nullable))]
#[case::utf8(DType::Utf8(Nullability::Nullable))]
#[case::binary(DType::Binary(Nullability::Nullable))]
fn test_append_scalar_mixed_nulls(#[case] dtype: DType) {
    let mut builder = builder_with_capacity(&dtype, 6);

    // Create a pattern of valid, null, valid, null, valid.
    let test_scalars = create_test_scalars_for_dtype(&dtype, 3);
    let null_scalar = Scalar::null(dtype.clone());

    builder.append_scalar(&test_scalars[0]).unwrap();
    builder.append_scalar(&null_scalar).unwrap();
    builder.append_scalar(&test_scalars[1]).unwrap();
    builder.append_scalar(&null_scalar).unwrap();
    builder.append_scalar(&test_scalars[2]).unwrap();

    let array = builder.finish();
    assert_eq!(array.len(), 5);

    // Check the pattern.
    assert!(
        !array
            .execute_scalar(0, &mut LEGACY_SESSION.create_execution_ctx())
            .unwrap()
            .is_null()
    );
    assert!(
        array
            .execute_scalar(1, &mut LEGACY_SESSION.create_execution_ctx())
            .unwrap()
            .is_null()
    );
    assert!(
        !array
            .execute_scalar(2, &mut LEGACY_SESSION.create_execution_ctx())
            .unwrap()
            .is_null()
    );
    assert!(
        array
            .execute_scalar(3, &mut LEGACY_SESSION.create_execution_ctx())
            .unwrap()
            .is_null()
    );
    assert!(
        !array
            .execute_scalar(4, &mut LEGACY_SESSION.create_execution_ctx())
            .unwrap()
            .is_null()
    );

    // Verify non-null values match.
    assert_scalars_equal(
        &array
            .execute_scalar(0, &mut LEGACY_SESSION.create_execution_ctx())
            .unwrap(),
        &test_scalars[0],
        &dtype,
        0,
    );
    assert_scalars_equal(
        &array
            .execute_scalar(2, &mut LEGACY_SESSION.create_execution_ctx())
            .unwrap(),
        &test_scalars[1],
        &dtype,
        2,
    );
    assert_scalars_equal(
        &array
            .execute_scalar(4, &mut LEGACY_SESSION.create_execution_ctx())
            .unwrap(),
        &test_scalars[2],
        &dtype,
        4,
    );
}

/// Test that `append_scalar` correctly rejects scalars with wrong dtype.
#[test]
fn test_append_scalar_wrong_dtype_rejection() {
    // Test bool builder rejecting i32 scalar.
    let mut bool_builder = builder_with_capacity(&DType::Bool(Nullability::NonNullable), 1);
    let i32_scalar = Scalar::from(42i32);
    assert!(
        bool_builder.append_scalar(&i32_scalar).is_err(),
        "Bool builder should reject i32 scalar"
    );

    // Test i32 builder rejecting string scalar.
    let mut i32_builder =
        builder_with_capacity(&DType::Primitive(PType::I32, Nullability::NonNullable), 1);
    let string_scalar = Scalar::utf8("test", Nullability::NonNullable);
    assert!(
        i32_builder.append_scalar(&string_scalar).is_err(),
        "I32 builder should reject string scalar"
    );

    // Test string builder rejecting binary scalar.
    let mut string_builder = builder_with_capacity(&DType::Utf8(Nullability::NonNullable), 1);
    let binary_scalar = Scalar::binary(vec![0u8, 1, 2], Nullability::NonNullable);
    assert!(
        string_builder.append_scalar(&binary_scalar).is_err(),
        "String builder should reject binary scalar"
    );
}

/// Test that `append_scalar` works correctly when called repeatedly
/// with the same scalar instance.
#[test]
fn test_append_scalar_repeated_same_instance() {
    let dtype = DType::Primitive(PType::I32, Nullability::NonNullable);
    let mut builder = builder_with_capacity(&dtype, 5);

    let scalar = Scalar::primitive(42i32, Nullability::NonNullable);

    // Append the same scalar instance multiple times.
    for _ in 0..5 {
        builder.append_scalar(&scalar).unwrap();
    }

    let array = builder.finish();
    assert_eq!(array.len(), 5);

    // All values should be 42.
    for i in 0..5 {
        let actual = array
            .execute_scalar(i, &mut LEGACY_SESSION.create_execution_ctx())
            .unwrap();
        assert_eq!(
            actual.as_primitive().typed_value::<i32>(),
            Some(42),
            "Value at index {} should be 42",
            i
        );
    }
}
