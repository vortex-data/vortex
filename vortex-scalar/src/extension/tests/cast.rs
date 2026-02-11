// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_dtype::{extension::EmptyMetadata, DType, Nullability, PType};

use crate::Scalar;

use super::*;

#[test]
fn test_ext_scalar_cast_to_storage() {
    let scalar = Scalar::extension::<TrivialExtType>(
        EmptyMetadata,
        Scalar::primitive(42i32, Nullability::NonNullable),
    );

    let ext = scalar.as_extension();

    // Cast to storage type
    let casted = ext
        .cast(&DType::Primitive(PType::I32, Nullability::NonNullable))
        .unwrap();
    assert_eq!(
        casted.dtype(),
        &DType::Primitive(PType::I32, Nullability::NonNullable)
    );
    assert_eq!(casted.as_primitive().typed_value::<i32>(), Some(42));

    // Cast to nullable storage type
    let casted_nullable = ext
        .cast(&DType::Primitive(PType::I32, Nullability::Nullable))
        .unwrap();
    assert_eq!(
        casted_nullable.dtype(),
        &DType::Primitive(PType::I32, Nullability::Nullable)
    );
    assert_eq!(
        casted_nullable.as_primitive().typed_value::<i32>(),
        Some(42)
    );
}

#[test]
fn test_ext_scalar_cast_to_self() {
    let ext_dtype = TrivialExtType::new_non_nullable();

    let scalar = Scalar::extension::<TrivialExtType>(
        EmptyMetadata,
        Scalar::primitive(42i32, Nullability::NonNullable),
    );

    let ext = scalar.as_extension();
    let ext_dtype = ext_dtype.erased();

    // Cast to same extension type
    let casted = ext.cast(&DType::Extension(ext_dtype.clone())).unwrap();
    assert_eq!(casted.dtype(), &DType::Extension(ext_dtype.clone()));

    // Cast to nullable version of same extension type
    let nullable_ext = DType::Extension(ext_dtype).as_nullable();
    let casted_nullable = ext.cast(&nullable_ext).unwrap();
    assert_eq!(casted_nullable.dtype(), &nullable_ext);
}

#[test]
fn test_ext_scalar_cast_incompatible() {
    let scalar = Scalar::extension::<TrivialExtType>(
        EmptyMetadata,
        Scalar::primitive(42i32, Nullability::NonNullable),
    );

    let ext = scalar.as_extension();

    // Cast to incompatible type should fail
    let result = ext.cast(&DType::Utf8(Nullability::NonNullable));
    assert!(result.is_err());
}

#[test]
fn test_ext_scalar_cast_null_to_non_nullable() {
    let scalar = Scalar::extension::<TrivialExtType>(
        EmptyMetadata,
        Scalar::null(DType::Primitive(PType::I32, Nullability::Nullable)),
    );

    let ext = scalar.as_extension();

    // Cast null to non-nullable should fail
    let result = ext.cast(&DType::Primitive(PType::I32, Nullability::NonNullable));
    assert!(result.is_err());
}
