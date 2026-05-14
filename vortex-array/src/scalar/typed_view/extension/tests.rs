// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_error::VortexResult;
use vortex_error::vortex_bail;

use crate::dtype::DType;
use crate::dtype::Nullability;
use crate::dtype::PType;
use crate::dtype::extension::ExtDType;
use crate::dtype::extension::ExtId;
use crate::dtype::extension::ExtVTable;
use crate::extension::EmptyMetadata;
use crate::scalar::Scalar;
use crate::scalar::ScalarValue;

#[derive(Clone, Debug, Default, PartialEq, Eq, Hash)]
struct TestI32Ext;
impl ExtVTable for TestI32Ext {
    type Metadata = EmptyMetadata;
    type NativeValue<'a> = &'a str;

    fn id(&self) -> ExtId {
        ExtId::new("test_ext")
    }

    fn serialize_metadata(&self, _metadata: &Self::Metadata) -> VortexResult<Vec<u8>> {
        Ok(vec![])
    }

    fn deserialize_metadata(&self, _data: &[u8]) -> VortexResult<Self::Metadata> {
        Ok(EmptyMetadata)
    }

    fn validate_dtype(_ext_dtype: &ExtDType<Self>) -> VortexResult<()> {
        Ok(())
    }

    fn unpack_native<'a>(
        _ext_dtype: &'a ExtDType<Self>,
        _storage_value: &'a ScalarValue,
    ) -> VortexResult<Self::NativeValue<'a>> {
        Ok("")
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
    impl ExtVTable for TestExt2 {
        type Metadata = EmptyMetadata;
        type NativeValue<'a> = &'a str;

        fn id(&self) -> ExtId {
            ExtId::new("test_ext_2")
        }

        fn serialize_metadata(&self, _metadata: &Self::Metadata) -> VortexResult<Vec<u8>> {
            Ok(vec![])
        }

        fn deserialize_metadata(&self, _data: &[u8]) -> VortexResult<Self::Metadata> {
            Ok(EmptyMetadata)
        }

        fn validate_dtype(_ext_dtype: &ExtDType<Self>) -> VortexResult<()> {
            Ok(())
        }

        fn unpack_native<'a>(
            _ext_dtype: &'a ExtDType<Self>,
            _storage_value: &'a ScalarValue,
        ) -> VortexResult<Self::NativeValue<'a>> {
            Ok("")
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
fn test_ext_scalar_with_metadata() {
    #[derive(Clone, Debug, Default, PartialEq, Eq, Hash)]
    struct TestExtMetadata;
    impl ExtVTable for TestExtMetadata {
        type Metadata = usize;
        type NativeValue<'a> = &'a str;

        fn id(&self) -> ExtId {
            ExtId::new("test_ext_metadata")
        }

        fn serialize_metadata(&self, _metadata: &Self::Metadata) -> VortexResult<Vec<u8>> {
            vortex_bail!("not implemented")
        }

        fn deserialize_metadata(&self, _data: &[u8]) -> VortexResult<Self::Metadata> {
            vortex_bail!("not implemented")
        }

        fn validate_dtype(_ext_dtype: &ExtDType<Self>) -> VortexResult<()> {
            Ok(())
        }

        fn unpack_native<'a>(
            _ext_dtype: &'a ExtDType<Self>,
            _storage_value: &'a ScalarValue,
        ) -> VortexResult<Self::NativeValue<'a>> {
            Ok("")
        }
    }

    let scalar = Scalar::extension::<TestExtMetadata>(
        1234,
        Scalar::primitive(42i32, Nullability::NonNullable),
    );

    let ext = scalar.as_extension();
    assert_eq!(ext.ext_dtype().metadata::<TestExtMetadata>(), &1234);
}
