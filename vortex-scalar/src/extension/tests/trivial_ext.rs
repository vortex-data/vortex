// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_dtype::DType;
use vortex_dtype::ExtDType;
use vortex_dtype::ExtID;
use vortex_dtype::Nullability;
use vortex_dtype::PType;
use vortex_dtype::extension::EmptyMetadata;
use vortex_dtype::extension::ExtDTypeVTable;
use vortex_error::VortexResult;
use vortex_error::vortex_ensure;

use crate::Scalar;
use crate::ScalarValue;
use crate::extension::ExtScalarVTable;

#[derive(Clone, Debug, Default, PartialEq, Eq, Hash)]
pub struct TrivialI32ExtVTable;

impl ExtDTypeVTable for TrivialI32ExtVTable {
    type Metadata = EmptyMetadata;

    fn id(&self) -> ExtID {
        ExtID::new_ref("trivial_ext")
    }

    fn validate_dtype(
        &self,
        _options: &Self::Metadata,
        _storage_dtype: &DType,
    ) -> VortexResult<()> {
        Ok(())
    }
}

impl ExtScalarVTable for TrivialI32ExtVTable {
    type Value<'a> = i32;

    fn unpack<'a>(
        &self,
        _metadata: &'a <Self as ExtDTypeVTable>::Metadata,
        storage_dtype: &'a DType,
        storage_value: &'a ScalarValue,
    ) -> Self::Value<'a> {
        assert!(storage_dtype.is_primitive());
        storage_value.as_primitive().as_i32().unwrap()
    }

    fn validate_scalar_value(
        &self,
        _metadata: &<Self as ExtDTypeVTable>::Metadata,
        storage_dtype: &DType,
        storage_value: &ScalarValue,
    ) -> VortexResult<()> {
        vortex_ensure!(storage_dtype.is_primitive());
        vortex_ensure!(storage_value.as_primitive().as_i32().is_some());

        Ok(())
    }
}

#[test]
fn test_trivial_ext_scalar_equality() {
    let scalar1 = Scalar::extension::<TrivialI32ExtVTable>(
        EmptyMetadata,
        Scalar::primitive(42i32, Nullability::NonNullable),
    );
    let scalar2 = Scalar::extension::<TrivialI32ExtVTable>(
        EmptyMetadata,
        Scalar::primitive(42i32, Nullability::NonNullable),
    );
    let scalar3 = Scalar::extension::<TrivialI32ExtVTable>(
        EmptyMetadata,
        Scalar::primitive(43i32, Nullability::NonNullable),
    );

    let ext1 = scalar1.as_extension().as_value::<TrivialI32ExtVTable>();
    let ext2 = scalar2.as_extension().as_value::<TrivialI32ExtVTable>();
    let ext3 = scalar3.as_extension().as_value::<TrivialI32ExtVTable>();

    assert_eq!(ext1, ext2);
    assert_ne!(ext1, ext3);
}

#[test]
fn test_trivial_ext_scalar_partial_ord() {
    let scalar1 = Scalar::extension::<TrivialI32ExtVTable>(
        EmptyMetadata,
        Scalar::primitive(10i32, Nullability::NonNullable),
    );
    let scalar2 = Scalar::extension::<TrivialI32ExtVTable>(
        EmptyMetadata,
        Scalar::primitive(20i32, Nullability::NonNullable),
    );

    let ext1 = scalar1.as_extension().as_value::<TrivialI32ExtVTable>();
    let ext2 = scalar2.as_extension().as_value::<TrivialI32ExtVTable>();

    assert!(ext1 < ext2);
    assert!(ext2 > ext1);
}

#[test]
fn test_trivial_ext_scalar_hash() {
    use vortex_utils::aliases::hash_set::HashSet;

    let scalar1 = Scalar::extension::<TrivialI32ExtVTable>(
        EmptyMetadata,
        Scalar::primitive(42i32, Nullability::NonNullable),
    );
    let scalar2 = Scalar::extension::<TrivialI32ExtVTable>(
        EmptyMetadata,
        Scalar::primitive(42i32, Nullability::NonNullable),
    );

    let mut set = HashSet::new();
    set.insert(scalar2);
    set.insert(scalar1);

    // Same value should hash the same
    assert_eq!(set.len(), 1);

    // Different value should hash differently
    let scalar3 = Scalar::extension::<TrivialI32ExtVTable>(
        EmptyMetadata,
        Scalar::primitive(43i32, Nullability::NonNullable),
    );
    set.insert(scalar3);
    assert_eq!(set.len(), 2);
}

#[test]
fn test_trivial_ext_scalar_storage() {
    let storage_scalar = Scalar::primitive(42i32, Nullability::NonNullable);
    let ext_scalar =
        Scalar::extension::<TrivialI32ExtVTable>(EmptyMetadata, storage_scalar.clone());

    let ext = ext_scalar.as_extension();
    assert_eq!(ext.to_storage_scalar(), storage_scalar);
}

#[test]
fn test_trivial_ext_scalar_ext_dtype() {
    let ext_dtype: ExtDType<TrivialI32ExtVTable> = ExtDType::try_new(
        EmptyMetadata,
        DType::Primitive(PType::I32, Nullability::NonNullable),
    )
    .unwrap();

    let scalar = Scalar::extension::<TrivialI32ExtVTable>(
        EmptyMetadata.clone(),
        Scalar::primitive(42i32, Nullability::NonNullable),
    );

    let ext = scalar.as_extension();
    assert_eq!(ext.ext_dtype().id(), ext_dtype.id());
    assert_eq!(ext.ext_dtype(), &ext_dtype.erased());
}
