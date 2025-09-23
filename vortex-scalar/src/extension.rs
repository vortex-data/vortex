// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::fmt::{Display, Formatter};
use std::hash::Hash;
use std::sync::Arc;

use vortex_dtype::datetime::{TemporalMetadata, is_temporal_ext_type};
use vortex_dtype::{DType, ExtDType};
use vortex_error::{VortexError, VortexResult, vortex_bail};

use crate::{Scalar, ScalarValue};

/// A scalar value representing an extension type.
///
/// Extension types allow wrapping a storage type with custom semantics.
#[derive(Debug)]
pub struct ExtScalar<'a> {
    ext_dtype: &'a ExtDType,
    value: &'a ScalarValue,
}

impl Display for ExtScalar<'_> {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        // Specialized handling for date/time/timestamp builtin extension types.
        if is_temporal_ext_type(self.ext_dtype().id()) {
            let metadata =
                TemporalMetadata::try_from(self.ext_dtype()).map_err(|_| std::fmt::Error)?;

            let maybe_timestamp = self
                .storage()
                .as_primitive()
                .as_::<i64>()
                .map(|maybe_timestamp| metadata.to_jiff(maybe_timestamp))
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
    pub fn ext_dtype(&self) -> &'a ExtDType {
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
    pub fn extension(ext_dtype: Arc<ExtDType>, value: Scalar) -> Self {
        // TODO(joe): enable once we use rust duckdb
        // assert_eq!(ext_dtype.storage_dtype(), value.dtype());
        Self::new(DType::Extension(ext_dtype), value.value().clone())
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use vortex_dtype::{DType, ExtDType, ExtID, ExtMetadata, Nullability, PType};

    use crate::{ExtScalar, InnerScalarValue, Scalar, ScalarValue};

    #[test]
    fn test_ext_scalar_equality() {
        let ext_dtype = Arc::new(ExtDType::new(
            ExtID::new("test_ext".into()),
            Arc::new(DType::Primitive(PType::I32, Nullability::NonNullable)),
            None,
        ));

        let scalar1 = Scalar::extension(
            ext_dtype.clone(),
            Scalar::primitive(42i32, Nullability::NonNullable),
        );
        let scalar2 = Scalar::extension(
            ext_dtype.clone(),
            Scalar::primitive(42i32, Nullability::NonNullable),
        );
        let scalar3 = Scalar::extension(
            ext_dtype,
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
        let ext_dtype = Arc::new(ExtDType::new(
            ExtID::new("test_ext".into()),
            Arc::new(DType::Primitive(PType::I32, Nullability::NonNullable)),
            None,
        ));

        let scalar1 = Scalar::extension(
            ext_dtype.clone(),
            Scalar::primitive(10i32, Nullability::NonNullable),
        );
        let scalar2 = Scalar::extension(
            ext_dtype,
            Scalar::primitive(20i32, Nullability::NonNullable),
        );

        let ext1 = ExtScalar::try_from(&scalar1).unwrap();
        let ext2 = ExtScalar::try_from(&scalar2).unwrap();

        assert!(ext1 < ext2);
        assert!(ext2 > ext1);
    }

    #[test]
    fn test_ext_scalar_partial_ord_different_types() {
        let ext_dtype1 = Arc::new(ExtDType::new(
            ExtID::new("type1".into()),
            Arc::new(DType::Primitive(PType::I32, Nullability::NonNullable)),
            None,
        ));
        let ext_dtype2 = Arc::new(ExtDType::new(
            ExtID::new("type2".into()),
            Arc::new(DType::Primitive(PType::I32, Nullability::NonNullable)),
            None,
        ));

        let scalar1 = Scalar::extension(
            ext_dtype1,
            Scalar::primitive(10i32, Nullability::NonNullable),
        );
        let scalar2 = Scalar::extension(
            ext_dtype2,
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

        let ext_dtype = Arc::new(ExtDType::new(
            ExtID::new("test_ext".into()),
            Arc::new(DType::Primitive(PType::I32, Nullability::NonNullable)),
            None,
        ));

        let scalar1 = Scalar::extension(
            ext_dtype.clone(),
            Scalar::primitive(42i32, Nullability::NonNullable),
        );
        let scalar2 = Scalar::extension(
            ext_dtype,
            Scalar::primitive(42i32, Nullability::NonNullable),
        );

        let mut set = HashSet::new();
        set.insert(scalar2);
        set.insert(scalar1);

        // Same value should hash the same
        assert_eq!(set.len(), 1);

        // Different value should hash differently
        let ext_dtype2 = Arc::new(ExtDType::new(
            ExtID::new("test_ext".into()),
            Arc::new(DType::Primitive(PType::I32, Nullability::NonNullable)),
            None,
        ));
        let scalar3 = Scalar::extension(
            ext_dtype2,
            Scalar::primitive(43i32, Nullability::NonNullable),
        );
        set.insert(scalar3);
        assert_eq!(set.len(), 2);
    }

    #[test]
    fn test_ext_scalar_storage() {
        let ext_dtype = Arc::new(ExtDType::new(
            ExtID::new("test_ext".into()),
            Arc::new(DType::Primitive(PType::I32, Nullability::NonNullable)),
            None,
        ));

        let storage_scalar = Scalar::primitive(42i32, Nullability::NonNullable);
        let ext_scalar = Scalar::extension(ext_dtype, storage_scalar.clone());

        let ext = ExtScalar::try_from(&ext_scalar).unwrap();
        assert_eq!(ext.storage(), storage_scalar);
    }

    #[test]
    fn test_ext_scalar_ext_dtype() {
        let ext_id = ExtID::new("test_ext".into());
        let ext_dtype = Arc::new(ExtDType::new(
            ext_id.clone(),
            Arc::new(DType::Primitive(PType::I32, Nullability::NonNullable)),
            None,
        ));

        let scalar = Scalar::extension(
            ext_dtype.clone(),
            Scalar::primitive(42i32, Nullability::NonNullable),
        );

        let ext = ExtScalar::try_from(&scalar).unwrap();
        assert_eq!(ext.ext_dtype().id(), &ext_id);
        assert_eq!(ext.ext_dtype(), ext_dtype.as_ref());
    }

    #[test]
    fn test_ext_scalar_cast_to_storage() {
        let ext_dtype = Arc::new(ExtDType::new(
            ExtID::new("test_ext".into()),
            Arc::new(DType::Primitive(PType::I32, Nullability::NonNullable)),
            None,
        ));

        let scalar = Scalar::extension(
            ext_dtype,
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
        let ext_dtype = Arc::new(ExtDType::new(
            ExtID::new("test_ext".into()),
            Arc::new(DType::Primitive(PType::I32, Nullability::NonNullable)),
            None,
        ));

        let scalar = Scalar::extension(
            ext_dtype.clone(),
            Scalar::primitive(42i32, Nullability::NonNullable),
        );

        let ext = ExtScalar::try_from(&scalar).unwrap();

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
        let ext_dtype = Arc::new(ExtDType::new(
            ExtID::new("test_ext".into()),
            Arc::new(DType::Primitive(PType::I32, Nullability::NonNullable)),
            None,
        ));

        let scalar = Scalar::extension(
            ext_dtype,
            Scalar::primitive(42i32, Nullability::NonNullable),
        );

        let ext = ExtScalar::try_from(&scalar).unwrap();

        // Cast to incompatible type should fail
        let result = ext.cast(&DType::Utf8(Nullability::NonNullable));
        assert!(result.is_err());
    }

    #[test]
    fn test_ext_scalar_cast_null_to_non_nullable() {
        let ext_dtype = Arc::new(ExtDType::new(
            ExtID::new("test_ext".into()),
            Arc::new(DType::Primitive(PType::I32, Nullability::Nullable)),
            None,
        ));

        let scalar = Scalar::extension(
            ext_dtype,
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
        let metadata = ExtMetadata::new(vec![1u8, 2, 3].into());
        let ext_dtype = Arc::new(ExtDType::new(
            ExtID::new("test_ext_with_meta".into()),
            Arc::new(DType::Primitive(PType::I32, Nullability::NonNullable)),
            Some(metadata),
        ));

        let scalar = Scalar::extension(
            ext_dtype.clone(),
            Scalar::primitive(42i32, Nullability::NonNullable),
        );

        let ext = ExtScalar::try_from(&scalar).unwrap();
        assert_eq!(ext.ext_dtype(), ext_dtype.as_ref());
        assert!(ext.ext_dtype().metadata().is_some());
    }

    #[test]
    fn test_ext_scalar_equality_ignores_nullability() {
        let ext_dtype_non_null = Arc::new(ExtDType::new(
            ExtID::new("test_ext".into()),
            Arc::new(DType::Primitive(PType::I32, Nullability::NonNullable)),
            None,
        ));
        let ext_dtype_nullable = Arc::new(ExtDType::new(
            ExtID::new("test_ext".into()),
            Arc::new(DType::Primitive(PType::I32, Nullability::Nullable)),
            None,
        ));

        let scalar1 = Scalar::extension(
            ext_dtype_non_null,
            Scalar::primitive(42i32, Nullability::NonNullable),
        );
        let scalar2 = Scalar::extension(
            ext_dtype_nullable,
            Scalar::primitive(42i32, Nullability::Nullable),
        );

        let ext1 = ExtScalar::try_from(&scalar1).unwrap();
        let ext2 = ExtScalar::try_from(&scalar2).unwrap();

        // Equality should ignore nullability differences
        assert_eq!(ext1, ext2);
    }
}
