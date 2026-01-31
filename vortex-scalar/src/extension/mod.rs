// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

mod matcher;
mod vtable;

use std::any::Any;
use std::any::type_name;
use std::fmt::Debug;
use std::fmt::Display;
use std::fmt::Formatter;
use std::hash::Hash;
use std::hash::Hasher;
use std::mem::discriminant;
use std::sync::Arc;

pub use matcher::*;
use vortex_dtype::DType;
use vortex_dtype::ExtDType;
use vortex_dtype::ExtDTypeRef;
use vortex_dtype::ExtID;
use vortex_dtype::Nullability;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_err;
use vortex_session::VortexSession;
pub use vtable::*;

use crate::Scalar;
use crate::ScalarValue;
use crate::session::ScalarSessionExt;

pub struct ExtensionScalar<'a> {
    pub(super) ext_dtype: &'a ExtDTypeRef,
    pub(super) ext_value: Option<&'a ExtScalarRef>,
}

/// A typed extension scalar.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct ExtScalar<V: ExtScalarVTable>(Arc<ExtScalarAdapter<V>>);

impl<V: ExtScalarVTable + Default> ExtScalar<V> {
    /// Creates a new extension scalar from a scalar value.
    pub fn try_new(
        metadata: V::Metadata,
        value: Option<V::Value>,
        nullability: Nullability,
    ) -> VortexResult<Self> {
        let vtable = V::default();
        let storage_scalar = vtable.pack(&metadata, value.as_ref(), nullability)?;
        let ext_dtype = ExtDType::try_new(metadata, storage_scalar.dtype().clone())?;
        Self::try_with_vtable(V::default(), ext_dtype, value)
    }

    /// Creates a new extension scalar from a type-erased dtype and scalar value.
    pub fn try_from_scalar(dtype: ExtDTypeRef, value: &ScalarValue) -> VortexResult<Self> {
        let vtable = V::default();
        let dtype = dtype
            .try_downcast::<V>()
            .map_err(|_| vortex_err!("Failed to downcast ExtDTypeRef to {}", type_name::<V>()))?;

        let value = if value.is_null() {
            None
        } else {
            Some(dtype.vtable().unpack(&dtype, value)?)
        };

        Ok(Self(Arc::new(ExtScalarAdapter::<V> {
            vtable,
            dtype,
            value,
        })))
    }
}

impl<V: ExtScalarVTable> ExtScalar<V> {
    /// Creates a new extension scalar from a vtable, metadata, and scalar value.
    pub fn try_with_vtable(
        vtable: V,
        dtype: ExtDType<V>,
        value: Option<V::Value>,
    ) -> VortexResult<Self> {
        // Ensure the value is permitted by the dtype's metadata and nullability
        let _storage_scalar = vtable.pack(
            dtype.metadata(),
            value.as_ref(),
            dtype.storage_dtype().nullability(),
        )?;
        Ok(Self(Arc::new(ExtScalarAdapter::<V> {
            vtable,
            dtype,
            value,
        })))
    }

    /// Returns the identifier of the extension scalar.
    pub fn id(&self) -> ExtID {
        self.0.dtype.id()
    }

    /// Returns the vtable of this extension scalar.
    pub fn vtable(&self) -> &V {
        self.0.dtype.vtable()
    }

    /// Returns the value of this extension scalar.
    pub fn value(&self) -> Option<&V::Value> {
        self.0.value.as_ref()
    }

    /// Erase the concrete type information, returning a type-erased extension scalar.
    pub fn erased(self) -> ExtScalarRef {
        ExtScalarRef(self.0)
    }

    // pub(crate) fn cast(&self, dtype: &DType) -> VortexResult<Scalar> {
    //     if self.value.is_none() && !dtype.is_nullable() {
    //         vortex_bail!(
    //             "cannot cast extension dtype with id {} and storage type {} to {}",
    //             self.ext_dtype.id(),
    //             self.ext_dtype.storage_dtype(),
    //             dtype
    //         );
    //     }
    //
    //     if self.ext_dtype.storage_dtype().eq_ignore_nullability(dtype) {
    //         // Casting from an extension type to the underlying storage type is OK.
    //         return Ok(Scalar::new(dtype.clone(), self.value.clone()));
    //     }
    //
    //     if let DType::Extension(ext_dtype) = dtype
    //         && self.ext_dtype.eq_ignore_nullability(ext_dtype)
    //     {
    //         return Ok(Scalar::new(dtype.clone(), self.value.clone()));
    //     }
    //
    //     vortex_bail!(
    //         "cannot cast extension dtype with id {} and storage type {} to {}",
    //         self.ext_dtype.id(),
    //         self.ext_dtype.storage_dtype(),
    //         dtype
    //     );
    // }
}

/// A type-erased extension scalar.
#[derive(Clone)]
pub struct ExtScalarRef(Arc<dyn ExtScalarImpl>);

impl ExtScalarRef {
    pub fn try_from_scalar(
        dtype: ExtDTypeRef,
        value: &ScalarValue,
        session: &VortexSession,
    ) -> VortexResult<Self> {
        let vtable = session
            .scalars()
            .registry()
            .find(&dtype.id())
            .ok_or_else(|| {
                vortex_err!(
                    "No registered vtable for extension scalar with id {}",
                    dtype.id()
                )
            })?;
        vtable.unpack(&dtype, value)
    }
}

impl Display for ExtScalarRef {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        self.0.value_display(f)
    }
}

impl Debug for ExtScalarRef {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ExtScalar")
            .field("dtype", &self.dtype())
            .field("value", &self.value_erased())
            .finish()
    }
}

impl PartialEq for ExtScalarRef {
    fn eq(&self, other: &Self) -> bool {
        self.dtype() == other.dtype() && self.0.value_eq(other.0.value_any())
    }
}
impl Eq for ExtScalarRef {}

impl Hash for ExtScalarRef {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.dtype().hash(state);
        self.value_erased().hash(state);
    }
}

impl ExtScalarRef {
    /// Returns the identifier of the extension scalar.
    pub fn id(&self) -> ExtID {
        self.0.id()
    }

    /// Returns the type-erased dtype of this extension scalar.
    pub fn dtype(&self) -> ExtDTypeRef {
        self.0.dtype()
    }

    /// Returns the type-erased value of this extension scalar.
    pub fn value_erased(&self) -> ExtScalarValue<'_> {
        ExtScalarValue { scalar: self }
    }
}

/// Methods for downcasting type-erased extension scalars.
impl ExtScalarRef {
    /// Check if the extension scalar is of the concrete type.
    pub fn is<M: Matcher>(&self) -> bool {
        M::matches(self)
    }

    /// Extract the value of the ExtScalar per the given [`Matcher`].
    pub fn value_opt<M: Matcher>(&self) -> Option<M::Match<'_>> {
        M::try_match(self)
    }

    /// Extract the value of the ExtScalar per the given [`Matcher`].
    ///
    /// # Panics
    ///
    /// Panics if the match fails.
    pub fn value<M: Matcher>(&self) -> M::Match<'_> {
        self.value_opt::<M>()
            .vortex_expect("Failed to downcast ExtScalar")
    }

    /// Downcast to the concrete [`ExtScalar`].
    ///
    /// Returns `Err(self)` if the downcast fails.
    pub fn try_downcast<V: ExtScalarVTable>(self) -> Result<ExtScalar<V>, ExtScalarRef> {
        // Check if the concrete type matches
        if self.0.as_any().is::<ExtScalarAdapter<V>>() {
            // SAFETY: type matches and ExtScalarAdapter<V> is the only implementor
            let ptr = Arc::into_raw(self.0) as *const ExtScalarAdapter<V>;
            let inner = unsafe { Arc::from_raw(ptr) };
            Ok(ExtScalar(inner))
        } else {
            Err(self)
        }
    }

    /// Downcast to the concrete [`ExtScalar`].
    ///
    /// # Panics
    ///
    /// Panics if the downcast fails.
    pub fn downcast<V: ExtScalarVTable>(self) -> ExtScalar<V> {
        self.try_downcast::<V>()
            .map_err(|this| {
                vortex_err!(
                    "Failed to downcast ExtScalar {} to {}",
                    this.0.id(),
                    type_name::<V>(),
                )
            })
            .vortex_expect("Failed to downcast ExtScalar")
    }
}

/// A type-erased reference to an extension scalar value.
pub struct ExtScalarValue<'a> {
    scalar: &'a ExtScalarRef,
}

impl Display for ExtScalarValue<'_> {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        self.scalar.0.value_display(f)
    }
}

impl Debug for ExtScalarValue<'_> {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        self.scalar.0.value_debug(f)
    }
}

impl PartialEq for ExtScalarValue<'_> {
    fn eq(&self, other: &Self) -> bool {
        self.scalar.0.value_eq(other.scalar.0.value_any())
    }
}
impl Eq for ExtScalarValue<'_> {}

impl Hash for ExtScalarValue<'_> {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.scalar.0.value_hash(state);
    }
}

trait ExtScalarImpl: 'static + Send + Sync {
    fn as_any(&self) -> &dyn Any;
    fn id(&self) -> ExtID;
    fn dtype(&self) -> ExtDTypeRef;
    fn value_any(&self) -> Option<&dyn Any>;
    fn value_debug(&self, f: &mut Formatter<'_>) -> std::fmt::Result;
    fn value_display(&self, f: &mut Formatter<'_>) -> std::fmt::Result;
    fn value_eq(&self, other: Option<&dyn Any>) -> bool;
    fn value_hash(&self, state: &mut dyn Hasher);
}

#[derive(Debug, Hash, PartialEq, Eq)]
struct ExtScalarAdapter<V: ExtScalarVTable> {
    vtable: V,
    dtype: ExtDType<V>,
    value: Option<V::Value>,
}

impl<V: ExtScalarVTable> ExtScalarImpl for ExtScalarAdapter<V> {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn id(&self) -> ExtID {
        self.dtype.id()
    }

    fn dtype(&self) -> ExtDTypeRef {
        self.dtype.clone().erased()
    }

    fn value_any(&self) -> Option<&dyn Any> {
        self.value.as_ref().map(|v| v as &dyn Any)
    }

    fn value_debug(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match &self.value {
            None => return write!(f, "null"),
            Some(value) => <V::Value as Debug>::fmt(value, f),
        }
    }

    fn value_display(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match &self.value {
            None => write!(f, "null"),
            Some(value) => <V::Value as Display>::fmt(value, f),
        }
    }

    fn value_eq(&self, other: Option<&dyn Any>) -> bool {
        match (&self.value, other) {
            (None, None) => true,
            (Some(_), None) | (None, Some(_)) => false,
            (Some(value), Some(other)) => {
                let Some(other) = other.downcast_ref::<V::Value>() else {
                    return false;
                };
                <V::Value as PartialEq>::eq(value, other)
            }
        }
    }

    fn value_hash(&self, mut state: &mut dyn Hasher) {
        self.dtype.hash(&mut state);
        discriminant(&self.value).hash(&mut state);
        if let Some(value) = self.value.as_ref() {
            <V::Value as Hash>::hash(value, &mut state);
        }
    }
}

impl Scalar {
    /// Creates a new extension scalar wrapping the given storage value.
    pub fn extension<V: ExtScalarVTable + Default>(
        metadata: V::Metadata,
        value: Option<V::Value>,
        nullability: Nullability,
    ) -> VortexResult<Self> {
        if value.is_none() && nullability == Nullability::NonNullable {
            vortex_bail!(
                "Cannot create non-nullable extension scalar of type {} with null value",
                type_name::<V>(),
            );
        }

        let vtable = V::default();
        let storage_scalar = vtable.pack(&metadata, value.as_ref(), nullability)?;

        let ext_dtype = ExtDType::<V>::try_new(metadata, storage_scalar.dtype().clone())
            .vortex_expect("Failed to create extension dtype");

        Ok(Self::new(
            DType::Extension(ext_dtype.erased()),
            storage_scalar.into_value(),
        ))
    }

    /// Creates a new extension scalar wrapping the given storage value.
    pub fn extension_ref(ext_dtype: ExtDTypeRef, value: Scalar) -> Self {
        assert_eq!(ext_dtype.storage_dtype(), value.dtype());
        Self::new(DType::Extension(ext_dtype), value.value().clone())
    }
}

// #[cfg(test)]
// mod tests {
//     use vortex_dtype::DType;
//     use vortex_dtype::ExtDType;
//     use vortex_dtype::ExtID;
//     use vortex_dtype::Nullability;
//     use vortex_dtype::PType;
//     use vortex_dtype::extension::EmptyMetadata;
//     use vortex_dtype::extension::ExtDTypeVTable;
//     use vortex_error::VortexResult;
//
//     use crate::ExtScalar;
//
//     use crate::Scalar;
//     use crate::ScalarValue;
//
//     #[derive(Debug, Clone, Default)]
//     struct TestExt;
//     impl ExtDTypeVTable for TestExt {
//         type Metadata = EmptyMetadata;
//
//         fn id(&self) -> ExtID {
//             ExtID::new_ref("test_ext")
//         }
//
//         fn validate(&self, _options: &Self::Metadata, _storage_dtype: &DType) -> VortexResult<()> {
//             Ok(())
//         }
//     }
//
//     impl TestExt {
//         fn new_non_nullable() -> ExtDType<TestExt> {
//             ExtDType::try_new(
//                 EmptyMetadata,
//                 DType::Primitive(PType::I32, Nullability::NonNullable),
//             )
//             .unwrap()
//         }
//     }
//
//     #[test]
//     fn test_ext_scalar_equality() {
//         let scalar1 = Scalar::extension::<TestExt>(
//             EmptyMetadata,
//             Scalar::primitive(42i32, Nullability::NonNullable),
//         );
//         let scalar2 = Scalar::extension::<TestExt>(
//             EmptyMetadata,
//             Scalar::primitive(42i32, Nullability::NonNullable),
//         );
//         let scalar3 = Scalar::extension::<TestExt>(
//             EmptyMetadata,
//             Scalar::primitive(43i32, Nullability::NonNullable),
//         );
//
//         let ext1 = ExtScalar::try_from(&scalar1).unwrap();
//         let ext2 = ExtScalar::try_from(&scalar2).unwrap();
//         let ext3 = ExtScalar::try_from(&scalar3).unwrap();
//
//         assert_eq!(ext1, ext2);
//         assert_ne!(ext1, ext3);
//     }
//
//     #[test]
//     fn test_ext_scalar_partial_ord() {
//         let scalar1 = Scalar::extension::<TestExt>(
//             EmptyMetadata,
//             Scalar::primitive(10i32, Nullability::NonNullable),
//         );
//         let scalar2 = Scalar::extension::<TestExt>(
//             EmptyMetadata,
//             Scalar::primitive(20i32, Nullability::NonNullable),
//         );
//
//         let ext1 = ExtScalar::try_from(&scalar1).unwrap();
//         let ext2 = ExtScalar::try_from(&scalar2).unwrap();
//
//         assert!(ext1 < ext2);
//         assert!(ext2 > ext1);
//     }
//
//     #[test]
//     fn test_ext_scalar_partial_ord_different_types() {
//         #[derive(Clone, Debug, Default)]
//         struct TestExt2;
//         impl ExtDTypeVTable for TestExt2 {
//             type Metadata = EmptyMetadata;
//
//             fn id(&self) -> ExtID {
//                 ExtID::new_ref("test_ext_2")
//             }
//
//             fn validate(
//                 &self,
//                 _options: &Self::Metadata,
//                 _storage_dtype: &DType,
//             ) -> VortexResult<()> {
//                 Ok(())
//             }
//         }
//
//         let scalar1 = Scalar::extension::<TestExt>(
//             EmptyMetadata,
//             Scalar::primitive(10i32, Nullability::NonNullable),
//         );
//         let scalar2 = Scalar::extension::<TestExt2>(
//             EmptyMetadata,
//             Scalar::primitive(20i32, Nullability::NonNullable),
//         );
//
//         let ext1 = ExtScalar::try_from(&scalar1).unwrap();
//         let ext2 = ExtScalar::try_from(&scalar2).unwrap();
//
//         // Different extension types should not be comparable
//         assert_eq!(ext1.partial_cmp(&ext2), None);
//     }
//
//     #[test]
//     fn test_ext_scalar_hash() {
//         use vortex_utils::aliases::hash_set::HashSet;
//
//         let scalar1 = Scalar::extension::<TestExt>(
//             EmptyMetadata,
//             Scalar::primitive(42i32, Nullability::NonNullable),
//         );
//         let scalar2 = Scalar::extension::<TestExt>(
//             EmptyMetadata,
//             Scalar::primitive(42i32, Nullability::NonNullable),
//         );
//
//         let mut set = HashSet::new();
//         set.insert(scalar2);
//         set.insert(scalar1);
//
//         // Same value should hash the same
//         assert_eq!(set.len(), 1);
//
//         // Different value should hash differently
//         let scalar3 = Scalar::extension::<TestExt>(
//             EmptyMetadata,
//             Scalar::primitive(43i32, Nullability::NonNullable),
//         );
//         set.insert(scalar3);
//         assert_eq!(set.len(), 2);
//     }
//
//     #[test]
//     fn test_ext_scalar_storage() {
//         let storage_scalar = Scalar::primitive(42i32, Nullability::NonNullable);
//         let ext_scalar = Scalar::extension::<TestExt>(EmptyMetadata, storage_scalar.clone());
//
//         let ext = ExtScalar::try_from(&ext_scalar).unwrap();
//         assert_eq!(ext.storage(), storage_scalar);
//     }
//
//     #[test]
//     fn test_ext_scalar_ext_dtype() {
//         let ext_dtype = TestExt::new_non_nullable();
//         let scalar = Scalar::extension::<TestExt>(
//             EmptyMetadata.clone(),
//             Scalar::primitive(42i32, Nullability::NonNullable),
//         );
//
//         let ext = ExtScalar::try_from(&scalar).unwrap();
//         assert_eq!(ext.ext_dtype().id(), ext_dtype.id());
//         assert_eq!(ext.ext_dtype(), &ext_dtype.erased());
//     }
//
//     #[test]
//     fn test_ext_scalar_cast_to_storage() {
//         let scalar = Scalar::extension::<TestExt>(
//             EmptyMetadata,
//             Scalar::primitive(42i32, Nullability::NonNullable),
//         );
//
//         let ext = ExtScalar::try_from(&scalar).unwrap();
//
//         // Cast to storage type
//         let casted = ext
//             .cast(&DType::Primitive(PType::I32, Nullability::NonNullable))
//             .unwrap();
//         assert_eq!(
//             casted.dtype(),
//             &DType::Primitive(PType::I32, Nullability::NonNullable)
//         );
//         assert_eq!(casted.as_primitive().typed_value::<i32>(), Some(42));
//
//         // Cast to nullable storage type
//         let casted_nullable = ext
//             .cast(&DType::Primitive(PType::I32, Nullability::Nullable))
//             .unwrap();
//         assert_eq!(
//             casted_nullable.dtype(),
//             &DType::Primitive(PType::I32, Nullability::Nullable)
//         );
//         assert_eq!(
//             casted_nullable.as_primitive().typed_value::<i32>(),
//             Some(42)
//         );
//     }
//
//     #[test]
//     fn test_ext_scalar_cast_to_self() {
//         let ext_dtype = TestExt::new_non_nullable();
//
//         let scalar = Scalar::extension::<TestExt>(
//             EmptyMetadata,
//             Scalar::primitive(42i32, Nullability::NonNullable),
//         );
//
//         let ext = ExtScalar::try_from(&scalar).unwrap();
//         let ext_dtype = ext_dtype.erased();
//
//         // Cast to same extension type
//         let casted = ext.cast(&DType::Extension(ext_dtype.clone())).unwrap();
//         assert_eq!(casted.dtype(), &DType::Extension(ext_dtype.clone()));
//
//         // Cast to nullable version of same extension type
//         let nullable_ext = DType::Extension(ext_dtype).as_nullable();
//         let casted_nullable = ext.cast(&nullable_ext).unwrap();
//         assert_eq!(casted_nullable.dtype(), &nullable_ext);
//     }
//
//     #[test]
//     fn test_ext_scalar_cast_incompatible() {
//         let scalar = Scalar::extension::<TestExt>(
//             EmptyMetadata,
//             Scalar::primitive(42i32, Nullability::NonNullable),
//         );
//
//         let ext = ExtScalar::try_from(&scalar).unwrap();
//
//         // Cast to incompatible type should fail
//         let result = ext.cast(&DType::Utf8(Nullability::NonNullable));
//         assert!(result.is_err());
//     }
//
//     #[test]
//     fn test_ext_scalar_cast_null_to_non_nullable() {
//         let scalar = Scalar::extension::<TestExt>(
//             EmptyMetadata,
//             Scalar::null(DType::Primitive(PType::I32, Nullability::Nullable)),
//         );
//
//         let ext = ExtScalar::try_from(&scalar).unwrap();
//
//         // Cast null to non-nullable should fail
//         let result = ext.cast(&DType::Primitive(PType::I32, Nullability::NonNullable));
//         assert!(result.is_err());
//     }
//
//     #[test]
//     fn test_ext_scalar_try_new_non_extension() {
//         let dtype = DType::Primitive(PType::I32, Nullability::NonNullable);
//         let value = ScalarValue(InnerScalarValue::Primitive(crate::PValue::I32(42)));
//
//         let result = ExtScalar::try_new(&dtype, &value);
//         assert!(result.is_err());
//     }
//
//     #[test]
//     fn test_ext_scalar_with_metadata() {
//         #[derive(Clone, Debug, Default)]
//         struct TestExtMetadata;
//         impl ExtDTypeVTable for TestExtMetadata {
//             type Metadata = usize;
//
//             fn id(&self) -> ExtID {
//                 ExtID::new_ref("test_ext_metadata")
//             }
//
//             fn validate(
//                 &self,
//                 _options: &Self::Metadata,
//                 _storage_dtype: &DType,
//             ) -> VortexResult<()> {
//                 Ok(())
//             }
//         }
//
//         let scalar = Scalar::extension::<TestExtMetadata>(
//             1234,
//             Scalar::primitive(42i32, Nullability::NonNullable),
//         );
//
//         let ext = ExtScalar::try_from(&scalar).unwrap();
//         assert_eq!(ext.ext_dtype().metadata::<TestExtMetadata>(), &1234);
//     }
//
//     #[test]
//     fn test_ext_scalar_equality_ignores_nullability() {
//         let scalar1 = Scalar::extension::<TestExt>(
//             EmptyMetadata,
//             Scalar::primitive(42i32, Nullability::NonNullable),
//         );
//         let scalar2 = Scalar::extension::<TestExt>(
//             EmptyMetadata,
//             Scalar::primitive(42i32, Nullability::Nullable),
//         );
//
//         let ext1 = ExtScalar::try_from(&scalar1).unwrap();
//         let ext2 = ExtScalar::try_from(&scalar2).unwrap();
//
//         // Equality should ignore nullability differences
//         assert_eq!(ext1, ext2);
//     }
// }
