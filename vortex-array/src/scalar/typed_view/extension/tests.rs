// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_error::VortexResult;

use crate::dtype::DType;
use crate::dtype::ExtDType;
use crate::dtype::ExtID;
use crate::dtype::Nullability;
use crate::dtype::PType;
use crate::dtype::extension::EmptyMetadata;
use crate::dtype::extension::ExtDTypeVTable;
use crate::scalar::ExtScalar;
use crate::scalar::PValue;
use crate::scalar::Scalar;
use crate::scalar::ScalarValue;

#[derive(Clone, Debug, Default, PartialEq, Eq, Hash)]
struct TestI32Ext;
impl ExtDTypeVTable for TestI32Ext {
    type Metadata = EmptyMetadata;

    fn id(&self) -> ExtID {
        ExtID::new_ref("test_ext")
    }

    fn validate_dtype(
        &self,
        _options: &Self::Metadata,
        _storage_dtype: &DType,
    ) -> VortexResult<()> {
        Ok(())
    }
}

impl TestI32Ext {
    fn new_non_nullable() -> ExtDType<TestI32Ext> {
        ExtDType::try_new(
            EmptyMetadata,
            DType::Primitive(PType::I32, Nullability::NonNullable),
        )
        .unwrap()
    }
}

#[test]
fn test_ext_scalar_equality() {
    let scalar1 = Scalar::extension::<TestI32Ext>(
        EmptyMetadata,
        Scalar::primitive(42i32, Nullability::NonNullable),
    );
    let scalar2 = Scalar::extension::<TestI32Ext>(
        EmptyMetadata,
        Scalar::primitive(42i32, Nullability::NonNullable),
    );
    let scalar3 = Scalar::extension::<TestI32Ext>(
        EmptyMetadata,
        Scalar::primitive(43i32, Nullability::NonNullable),
    );

    let ext1 = scalar1.as_extension();
    let ext2 = scalar2.as_extension();
    let ext3 = scalar3.as_extension();

    assert_eq!(ext1, ext2);
    assert_ne!(ext1, ext3);
}

#[test]
fn test_ext_scalar_partial_ord() {
    let scalar1 = Scalar::extension::<TestI32Ext>(
        EmptyMetadata,
        Scalar::primitive(10i32, Nullability::NonNullable),
    );
    let scalar2 = Scalar::extension::<TestI32Ext>(
        EmptyMetadata,
        Scalar::primitive(20i32, Nullability::NonNullable),
    );

    let ext1 = scalar1.as_extension();
    let ext2 = scalar2.as_extension();

    assert!(ext1 < ext2);
    assert!(ext2 > ext1);
}

#[test]
fn test_ext_scalar_partial_ord_different_types() {
    #[derive(Clone, Debug, Default, PartialEq, Eq, Hash)]
    struct TestExt2;
    impl ExtDTypeVTable for TestExt2 {
        type Metadata = EmptyMetadata;

        fn id(&self) -> ExtID {
            ExtID::new_ref("test_ext_2")
        }

        fn validate_dtype(
            &self,
            _options: &Self::Metadata,
            _storage_dtype: &DType,
        ) -> VortexResult<()> {
            Ok(())
        }
    }

    let scalar1 = Scalar::extension::<TestI32Ext>(
        EmptyMetadata,
        Scalar::primitive(10i32, Nullability::NonNullable),
    );
    let scalar2 = Scalar::extension::<TestExt2>(
        EmptyMetadata,
        Scalar::primitive(20i32, Nullability::NonNullable),
    );

    let ext1 = scalar1.as_extension();
    let ext2 = scalar2.as_extension();

    // Different extension types should not be comparable
    assert_eq!(ext1.partial_cmp(&ext2), None);
}

#[test]
fn test_ext_scalar_hash() {
    use vortex_utils::aliases::hash_set::HashSet;

    let scalar1 = Scalar::extension::<TestI32Ext>(
        EmptyMetadata,
        Scalar::primitive(42i32, Nullability::NonNullable),
    );
    let scalar2 = Scalar::extension::<TestI32Ext>(
        EmptyMetadata,
        Scalar::primitive(42i32, Nullability::NonNullable),
    );

    let mut set = HashSet::new();
    set.insert(scalar2);
    set.insert(scalar1);

    // Same value should hash the same
    assert_eq!(set.len(), 1);

    // Different value should hash differently
    let scalar3 = Scalar::extension::<TestI32Ext>(
        EmptyMetadata,
        Scalar::primitive(43i32, Nullability::NonNullable),
    );
    set.insert(scalar3);
    assert_eq!(set.len(), 2);
}

#[test]
fn test_ext_scalar_storage() {
    let storage_scalar = Scalar::primitive(42i32, Nullability::NonNullable);
    let ext_scalar = Scalar::extension::<TestI32Ext>(EmptyMetadata, storage_scalar.clone());

    let ext = ext_scalar.as_extension();
    assert_eq!(ext.to_storage_scalar(), storage_scalar);
}

#[test]
fn test_ext_scalar_ext_dtype() {
    let ext_dtype = TestI32Ext::new_non_nullable();
    let scalar = Scalar::extension::<TestI32Ext>(
        EmptyMetadata.clone(),
        Scalar::primitive(42i32, Nullability::NonNullable),
    );

    let ext = scalar.as_extension();
    assert_eq!(ext.ext_dtype().id(), ext_dtype.id());
    assert_eq!(ext.ext_dtype(), &ext_dtype.erased());
}

#[test]
fn test_ext_scalar_cast_to_storage() {
    let scalar = Scalar::extension::<TestI32Ext>(
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
    let ext_dtype = TestI32Ext::new_non_nullable();

    let scalar = Scalar::extension::<TestI32Ext>(
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
    let scalar = Scalar::extension::<TestI32Ext>(
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
    let scalar = Scalar::extension::<TestI32Ext>(
        EmptyMetadata,
        Scalar::null(DType::Primitive(PType::I32, Nullability::Nullable)),
    );

    let ext = scalar.as_extension();

    // Cast null to non-nullable should fail
    let result = ext.cast(&DType::Primitive(PType::I32, Nullability::NonNullable));
    assert!(result.is_err());
}

#[test]
fn test_ext_scalar_try_new_non_extension() {
    let dtype = DType::Primitive(PType::I32, Nullability::NonNullable);
    let value = ScalarValue::Primitive(PValue::I32(42));

    let result = ExtScalar::try_new(&dtype, Some(&value));
    assert!(result.is_err());
}

#[test]
fn test_ext_scalar_with_metadata() {
    #[derive(Clone, Debug, Default, PartialEq, Eq, Hash)]
    struct TestExtMetadata;
    impl ExtDTypeVTable for TestExtMetadata {
        type Metadata = usize;

        fn id(&self) -> ExtID {
            ExtID::new_ref("test_ext_metadata")
        }

        fn validate_dtype(
            &self,
            _options: &Self::Metadata,
            _storage_dtype: &DType,
        ) -> VortexResult<()> {
            Ok(())
        }
    }

    let scalar = Scalar::extension::<TestExtMetadata>(
        1234,
        Scalar::primitive(42i32, Nullability::NonNullable),
    );

    let ext = scalar.as_extension();
    assert_eq!(ext.ext_dtype().metadata::<TestExtMetadata>(), &1234);
}
