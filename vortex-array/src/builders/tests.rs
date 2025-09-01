// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::sync::Arc;

use rstest::rstest;
use vortex_dtype::{DType, DecimalDType, ExtDType, ExtID, Nullability, PType, StructFields};
use vortex_scalar::Scalar;

use crate::builders::builder_with_capacity;

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
// TODO(connor): This test case is expected to fail due to a known bug where append_zeros creates
// lists of size 1 instead of empty lists.
// #[case::list(DType::List(
//     Arc::new(DType::Primitive(PType::I32, Nullability::NonNullable)),
//     Nullability::NonNullable
// ))]
#[case::fixed_size_list(DType::FixedSizeList(
    Arc::new(DType::Primitive(PType::I32, Nullability::NonNullable)),
    3,
    Nullability::NonNullable
))]
#[case::extension_simple(DType::Extension(Arc::new(ExtDType::new(
    ExtID::from("test.extension"),
    Arc::new(DType::Primitive(PType::I64, Nullability::NonNullable)),
    None
))))]
#[case::extension_with_metadata(DType::Extension(Arc::new(ExtDType::new(
    ExtID::from("test.temperature"),
    Arc::new(DType::Primitive(PType::F64, Nullability::NonNullable)),
    Some([0u8].as_slice().into())
))))]
fn test_append_zeros_matches_default_value(#[case] dtype: DType) {
    let num_elements = 5;

    // Builder 1: Use append_zeros.
    let mut builder_zeros = builder_with_capacity(&dtype, num_elements);
    builder_zeros.append_zeros(num_elements);
    let array_zeros = builder_zeros.finish();

    // Builder 2: Manually append default values.
    let mut builder_manual = builder_with_capacity(&dtype, num_elements);
    let default_scalar = Scalar::default_value(dtype.clone());
    for _ in 0..num_elements {
        builder_manual.append_scalar(&default_scalar).unwrap();
    }
    let array_manual = builder_manual.finish();

    // Both arrays should have the same length.
    assert_eq!(array_zeros.len(), array_manual.len());
    assert_eq!(array_zeros.len(), num_elements);

    // Compare each element.
    for i in 0..num_elements {
        let scalar_zeros = array_zeros.scalar_at(i);
        let scalar_manual = array_manual.scalar_at(i);

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
    DType::Extension(Arc::new(ExtDType::new(
        ExtID::from("test.ext"),
        Arc::new(DType::Primitive(PType::I64, Nullability::NonNullable)),
        None
    ))),
    1
)]
#[case::extension_multiple(
    DType::Extension(Arc::new(ExtDType::new(
        ExtID::from("test.ext"),
        Arc::new(DType::Primitive(PType::I64, Nullability::NonNullable)),
        None
    ))),
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
        let scalar = array.scalar_at(i);
        if should_be_null {
            assert!(scalar.is_null(), "Element at index {} should be null", i);
        } else {
            assert!(
                !scalar.is_null(),
                "Element at index {} should not be null",
                i
            );
            // For non-nullable, it should match the default value.
            let expected = Scalar::default_value(dtype.clone());
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
