// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::fmt::Display;
use std::fmt::Formatter;
use std::hash::Hash;

use vortex_dtype::DType;
use vortex_dtype::ExtDType;
use vortex_dtype::datetime::AnyTemporal;
use vortex_dtype::extension::ExtDTypeRef;
use vortex_dtype::extension::ExtDTypeVTable;
use vortex_error::VortexError;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;

use crate::Scalar;
use crate::ScalarValue;

/// A scalar value representing an extension type.
///
/// Extension types allow wrapping a storage type with custom semantics.
#[derive(Debug, Clone)]
pub struct ExtScalar<'a> {
    ext_dtype: &'a ExtDTypeRef,
    value: &'a ScalarValue,
}

impl Display for ExtScalar<'_> {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        // Specialized handling for date/time/timestamp builtin extension types.
        if let Some(temporal) = self.ext_dtype.metadata_opt::<AnyTemporal>() {
            let maybe_timestamp = self
                .storage()
                .as_primitive()
                .as_::<i64>()
                .map(|maybe_timestamp| temporal.to_jiff(maybe_timestamp))
                .transpose()
                .map_err(|_| std::fmt::Error)?;

            match maybe_timestamp {
                None => write!(f, "null"),
                Some(v) => write!(f, "{v}"),
            }
        } else {
            write!(f, "{}({})", self.ext_dtype().id(), self.storage())
        }
    }
}

impl PartialEq for ExtScalar<'_> {
    fn eq(&self, other: &Self) -> bool {
        self.ext_dtype.eq_ignore_nullability(other.ext_dtype) && self.storage() == other.storage()
    }
}

impl Eq for ExtScalar<'_> {}

// Ord is not implemented since it's undefined for different Extension DTypes
impl PartialOrd for ExtScalar<'_> {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        if !self.ext_dtype.eq_ignore_nullability(other.ext_dtype) {
            return None;
        }
        self.storage().partial_cmp(&other.storage())
    }
}

impl Hash for ExtScalar<'_> {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.ext_dtype.hash(state);
        self.storage().hash(state);
    }
}

impl<'a> ExtScalar<'a> {
    /// Creates a new extension scalar from a data type and scalar value.
    ///
    /// # Errors
    ///
    /// Returns an error if the data type is not an extension type.
    pub fn try_new(dtype: &'a DType, value: &'a ScalarValue) -> VortexResult<Self> {
        let DType::Extension(ext_dtype) = dtype else {
            vortex_bail!("Expected extension scalar, found {}", dtype)
        };

        Ok(Self { ext_dtype, value })
    }

    /// Returns the storage scalar of the extension scalar.
    pub fn storage(&self) -> Scalar {
        Scalar::new(self.ext_dtype.storage_dtype().clone(), self.value.clone())
    }

    /// Returns the extension data type.
    pub fn ext_dtype(&self) -> &'a ExtDTypeRef {
        self.ext_dtype
    }

    pub(crate) fn cast(&self, dtype: &DType) -> VortexResult<Scalar> {
        if self.value.is_null() && !dtype.is_nullable() {
            vortex_bail!(
                "cannot cast extension dtype with id {} and storage type {} to {}",
                self.ext_dtype.id(),
                self.ext_dtype.storage_dtype(),
                dtype
            );
        }

        if self.ext_dtype.storage_dtype().eq_ignore_nullability(dtype) {
            // Casting from an extension type to the underlying storage type is OK.
            return Ok(Scalar::new(dtype.clone(), self.value.clone()));
        }

        if let DType::Extension(ext_dtype) = dtype
            && self.ext_dtype.eq_ignore_nullability(ext_dtype)
        {
            return Ok(Scalar::new(dtype.clone(), self.value.clone()));
        }

        vortex_bail!(
            "cannot cast extension dtype with id {} and storage type {} to {}",
            self.ext_dtype.id(),
            self.ext_dtype.storage_dtype(),
            dtype
        );
    }
}

impl<'a> TryFrom<&'a Scalar> for ExtScalar<'a> {
    type Error = VortexError;

    fn try_from(scalar: &'a Scalar) -> Result<Self, Self::Error> {
        ExtScalar::try_new(scalar.dtype(), scalar.value())
    }
}

impl Scalar {
    /// Creates a new extension scalar wrapping the given storage value.
    pub fn extension<V: ExtDTypeVTable + Default>(options: V::Metadata, value: Scalar) -> Self {
        let ext_dtype = ExtDType::<V>::try_new(options, value.dtype().clone())
            .vortex_expect("Failed to create extension dtype");
        Self::new(DType::Extension(ext_dtype.erased()), value.value().clone())
    }

    /// Creates a new extension scalar wrapping the given storage value.
    pub fn extension_ref(ext_dtype: ExtDTypeRef, value: Scalar) -> Self {
        assert_eq!(ext_dtype.storage_dtype(), value.dtype());
        Self::new(DType::Extension(ext_dtype), value.value().clone())
    }
}

#[cfg(test)]
mod tests {
    use vortex_dtype::DType;
    use vortex_dtype::ExtDType;
    use vortex_dtype::ExtID;
    use vortex_dtype::Nullability;
    use vortex_dtype::PType;
    use vortex_dtype::extension::EmptyMetadata;
    use vortex_dtype::extension::ExtDTypeVTable;
    use vortex_error::VortexResult;

    use crate::ExtScalar;
    use crate::InnerScalarValue;
    use crate::Scalar;
    use crate::ScalarValue;

    #[derive(Clone, Debug, Default, PartialEq, Eq, Hash)]
    struct TestExt;
    impl ExtDTypeVTable for TestExt {
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

    impl TestExt {
        fn new_non_nullable() -> ExtDType<TestExt> {
            ExtDType::try_new(
                EmptyMetadata,
                DType::Primitive(PType::I32, Nullability::NonNullable),
            )
            .unwrap()
        }
    }

    #[test]
    fn test_ext_scalar_equality() {
        let scalar1 = Scalar::extension::<TestExt>(
            EmptyMetadata,
            Scalar::primitive(42i32, Nullability::NonNullable),
        );
        let scalar2 = Scalar::extension::<TestExt>(
            EmptyMetadata,
            Scalar::primitive(42i32, Nullability::NonNullable),
        );
        let scalar3 = Scalar::extension::<TestExt>(
            EmptyMetadata,
            Scalar::primitive(43i32, Nullability::NonNullable),
        );

        let ext1 = ExtScalar::try_from(&scalar1).unwrap();
        let ext2 = ExtScalar::try_from(&scalar2).unwrap();
        let ext3 = ExtScalar::try_from(&scalar3).unwrap();

        assert_eq!(ext1, ext2);
        assert_ne!(ext1, ext3);
    }

    #[test]
    fn test_ext_scalar_partial_ord() {
        let scalar1 = Scalar::extension::<TestExt>(
            EmptyMetadata,
            Scalar::primitive(10i32, Nullability::NonNullable),
        );
        let scalar2 = Scalar::extension::<TestExt>(
            EmptyMetadata,
            Scalar::primitive(20i32, Nullability::NonNullable),
        );

        let ext1 = ExtScalar::try_from(&scalar1).unwrap();
        let ext2 = ExtScalar::try_from(&scalar2).unwrap();

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

        let scalar1 = Scalar::extension::<TestExt>(
            EmptyMetadata,
            Scalar::primitive(10i32, Nullability::NonNullable),
        );
        let scalar2 = Scalar::extension::<TestExt2>(
            EmptyMetadata,
            Scalar::primitive(20i32, Nullability::NonNullable),
        );

        let ext1 = ExtScalar::try_from(&scalar1).unwrap();
        let ext2 = ExtScalar::try_from(&scalar2).unwrap();

        // Different extension types should not be comparable
        assert_eq!(ext1.partial_cmp(&ext2), None);
    }

    #[test]
    fn test_ext_scalar_hash() {
        use vortex_utils::aliases::hash_set::HashSet;

        let scalar1 = Scalar::extension::<TestExt>(
            EmptyMetadata,
            Scalar::primitive(42i32, Nullability::NonNullable),
        );
        let scalar2 = Scalar::extension::<TestExt>(
            EmptyMetadata,
            Scalar::primitive(42i32, Nullability::NonNullable),
        );

        let mut set = HashSet::new();
        set.insert(scalar2);
        set.insert(scalar1);

        // Same value should hash the same
        assert_eq!(set.len(), 1);

        // Different value should hash differently
        let scalar3 = Scalar::extension::<TestExt>(
            EmptyMetadata,
            Scalar::primitive(43i32, Nullability::NonNullable),
        );
        set.insert(scalar3);
        assert_eq!(set.len(), 2);
    }

    #[test]
    fn test_ext_scalar_storage() {
        let storage_scalar = Scalar::primitive(42i32, Nullability::NonNullable);
        let ext_scalar = Scalar::extension::<TestExt>(EmptyMetadata, storage_scalar.clone());

        let ext = ExtScalar::try_from(&ext_scalar).unwrap();
        assert_eq!(ext.storage(), storage_scalar);
    }

    #[test]
    fn test_ext_scalar_ext_dtype() {
        let ext_dtype = TestExt::new_non_nullable();
        let scalar = Scalar::extension::<TestExt>(
            EmptyMetadata.clone(),
            Scalar::primitive(42i32, Nullability::NonNullable),
        );

        let ext = ExtScalar::try_from(&scalar).unwrap();
        assert_eq!(ext.ext_dtype().id(), ext_dtype.id());
        assert_eq!(ext.ext_dtype(), &ext_dtype.erased());
    }

    #[test]
    fn test_ext_scalar_cast_to_storage() {
        let scalar = Scalar::extension::<TestExt>(
            EmptyMetadata,
            Scalar::primitive(42i32, Nullability::NonNullable),
        );

        let ext = ExtScalar::try_from(&scalar).unwrap();

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
        let ext_dtype = TestExt::new_non_nullable();

        let scalar = Scalar::extension::<TestExt>(
            EmptyMetadata,
            Scalar::primitive(42i32, Nullability::NonNullable),
        );

        let ext = ExtScalar::try_from(&scalar).unwrap();
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
        let scalar = Scalar::extension::<TestExt>(
            EmptyMetadata,
            Scalar::primitive(42i32, Nullability::NonNullable),
        );

        let ext = ExtScalar::try_from(&scalar).unwrap();

        // Cast to incompatible type should fail
        let result = ext.cast(&DType::Utf8(Nullability::NonNullable));
        assert!(result.is_err());
    }

    #[test]
    fn test_ext_scalar_cast_null_to_non_nullable() {
        let scalar = Scalar::extension::<TestExt>(
            EmptyMetadata,
            Scalar::null(DType::Primitive(PType::I32, Nullability::Nullable)),
        );

        let ext = ExtScalar::try_from(&scalar).unwrap();

        // Cast null to non-nullable should fail
        let result = ext.cast(&DType::Primitive(PType::I32, Nullability::NonNullable));
        assert!(result.is_err());
    }

    #[test]
    fn test_ext_scalar_try_new_non_extension() {
        let dtype = DType::Primitive(PType::I32, Nullability::NonNullable);
        let value = ScalarValue(InnerScalarValue::Primitive(crate::PValue::I32(42)));

        let result = ExtScalar::try_new(&dtype, &value);
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

        let ext = ExtScalar::try_from(&scalar).unwrap();
        assert_eq!(ext.ext_dtype().metadata::<TestExtMetadata>(), &1234);
    }

    #[test]
    fn test_ext_scalar_equality_ignores_nullability() {
        let scalar1 = Scalar::extension::<TestExt>(
            EmptyMetadata,
            Scalar::primitive(42i32, Nullability::NonNullable),
        );
        let scalar2 = Scalar::extension::<TestExt>(
            EmptyMetadata,
            Scalar::primitive(42i32, Nullability::Nullable),
        );

        let ext1 = ExtScalar::try_from(&scalar1).unwrap();
        let ext2 = ExtScalar::try_from(&scalar2).unwrap();

        // Equality should ignore nullability differences
        assert_eq!(ext1, ext2);
    }
}
