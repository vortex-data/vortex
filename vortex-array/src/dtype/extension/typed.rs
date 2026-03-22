// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Typed and inner representations of extension dtypes.
//!
//! - [`ExtDType<V>`]: The public typed wrapper, parameterized by a concrete [`ExtVTable`].
//! - [`ExtDTypeInner<V>`]: The private inner struct that holds the vtable + data.
//! - [`DynExtDType`]: The private sealed trait for type-erased dispatch.

use std::any::Any;
use std::fmt;
use std::hash::Hash;
use std::hash::Hasher;
use std::sync::Arc;

use vortex_error::VortexExpect;
use vortex_error::VortexResult;

use crate::dtype::DType;
use crate::dtype::Nullability;
use crate::dtype::extension::ExtDTypeRef;
use crate::dtype::extension::ExtId;
use crate::dtype::extension::ExtVTable;
use crate::scalar::ScalarValue;

/// A typed extension data type, parameterized by a concrete [`ExtVTable`].
///
/// You can construct one of these via [`try_new()`] (for zero-sized vtables) or
/// [`try_with_vtable()`], and erase the type with [`erased()`] to obtain an [`ExtDTypeRef`].
///
/// [`try_new()`]: ExtDType::try_new
/// [`try_with_vtable()`]: ExtDType::try_with_vtable
/// [`erased()`]: ExtDType::erased
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct ExtDType<V: ExtVTable> {
    /// The extension dtype vtable.
    vtable: V,
    /// The extension dtype metadata.
    metadata: V::Metadata,
    /// The underlying storage dtype.
    storage_dtype: DType,
}

/// Convenience implementation for zero-sized VTables (or VTables that implement `Default`).
impl<V: ExtVTable + Default> ExtDType<V> {
    /// Creates a new extension dtype with the given metadata and storage dtype.
    pub fn try_new(metadata: V::Metadata, storage_dtype: DType) -> VortexResult<Self> {
        Self::try_with_vtable(V::default(), metadata, storage_dtype)
    }
}

impl<V: ExtVTable> ExtDType<V> {
    /// Creates a new extension dtype with the given metadata and storage dtype.
    pub fn try_with_vtable(
        vtable: V,
        metadata: V::Metadata,
        storage_dtype: DType,
    ) -> VortexResult<Self> {
        let this = Self {
            vtable,
            metadata,
            storage_dtype,
        };

        V::validate_dtype(&this)?;

        Ok(this)
    }

    /// Returns the identifier of the extension type.
    pub fn id(&self) -> ExtId {
        self.vtable.id()
    }

    /// Returns the vtable of the extension type.
    pub fn vtable(&self) -> &V {
        &self.vtable
    }

    /// Returns the metadata of the extension type.
    pub fn metadata(&self) -> &V::Metadata {
        &self.metadata
    }

    /// Returns the storage dtype of the extension type.
    pub fn storage_dtype(&self) -> &DType {
        &self.storage_dtype
    }

    /// Erase the concrete type information, returning a type-erased extension dtype.
    pub fn erased(self) -> ExtDTypeRef {
        ExtDTypeRef(Arc::new(self))
    }
}

/// An object-safe, sealed trait encapsulating the behavior for extension dtypes.
///
/// This provides type-erased access to the extension dtype's identity, storage dtype, and
/// metadata. The only implementor is [`ExtDTypeInner`].
pub(super) trait DynExtDType: 'static + Send + Sync + super::sealed::Sealed {
    /// Returns `self` as a trait object for downcasting.
    fn as_any(&self) -> &dyn Any;
    /// Returns the [`ExtId`] identifying this extension type.
    fn ext_id(&self) -> ExtId;
    /// Returns a reference to the storage [`DType`].
    fn ext_storage_dtype(&self) -> &DType;
    /// Returns the metadata as a trait object for downcasting.
    fn metadata_any(&self) -> &dyn Any;
    /// Formats the metadata using [`Debug`].
    fn metadata_debug(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result;
    /// Formats the metadata using [`Display`].
    fn metadata_display(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result;
    /// Checks equality of the metadata against a type-erased value.
    fn metadata_eq(&self, other: &dyn Any) -> bool;
    /// Hashes the metadata into the given [`Hasher`].
    fn metadata_hash(&self, state: &mut dyn Hasher);
    /// Serializes the metadata into a byte vector.
    fn metadata_serialize(&self) -> VortexResult<Vec<u8>>;
    /// Returns a new [`ExtDTypeRef`] with the given nullability.
    fn with_nullability(&self, nullability: Nullability) -> ExtDTypeRef;
    /// Validates that the given storage scalar value is valid for this dtype.
    fn value_validate(&self, storage_value: &ScalarValue) -> VortexResult<()>;
    /// Formats an extension scalar value using the current dtype for metadata context.
    fn value_display(&self, f: &mut fmt::Formatter<'_>, storage_value: &ScalarValue)
    -> fmt::Result;
    /// Can a value of `other` be implicitly coerced into this extension type?
    fn coercion_can_coerce_from(&self, other: &DType) -> bool;
    /// Can this extension type be implicitly coerced into `other`?
    fn coercion_can_coerce_to(&self, other: &DType) -> bool;
    /// Compute the least supertype of this extension type and another type.
    fn coercion_least_supertype(&self, other: &DType) -> Option<DType>;
}

impl<V: ExtVTable> DynExtDType for ExtDType<V> {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn ext_id(&self) -> ExtId {
        self.vtable.id()
    }

    fn ext_storage_dtype(&self) -> &DType {
        &self.storage_dtype
    }

    fn metadata_any(&self) -> &dyn Any {
        &self.metadata
    }

    fn metadata_debug(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        <V::Metadata as fmt::Debug>::fmt(&self.metadata, f)
    }

    fn metadata_display(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        <V::Metadata as fmt::Display>::fmt(&self.metadata, f)
    }

    fn metadata_eq(&self, other: &dyn Any) -> bool {
        let Some(other) = other.downcast_ref::<V::Metadata>() else {
            return false;
        };
        <V::Metadata as PartialEq>::eq(&self.metadata, other)
    }

    fn metadata_hash(&self, mut state: &mut dyn Hasher) {
        <V::Metadata as Hash>::hash(&self.metadata, &mut state);
    }

    fn metadata_serialize(&self) -> VortexResult<Vec<u8>> {
        V::serialize_metadata(&self.vtable, &self.metadata)
    }

    fn with_nullability(&self, nullability: Nullability) -> ExtDTypeRef {
        let storage_dtype = self.storage_dtype.with_nullability(nullability);
        ExtDType::<V>::try_with_vtable(self.vtable.clone(), self.metadata.clone(), storage_dtype)
            .vortex_expect(
                "Extension DType should not fail validation with the same storage type \
                 but different nullability",
            )
            .erased()
    }

    fn value_validate(&self, storage_value: &ScalarValue) -> VortexResult<()> {
        V::validate_scalar_value(self, storage_value)
    }

    fn value_display(
        &self,
        f: &mut fmt::Formatter<'_>,
        storage_value: &ScalarValue,
    ) -> fmt::Result {
        match V::unpack_native(self, storage_value) {
            Ok(native) => fmt::Display::fmt(&native, f),
            Err(_) => write!(
                f,
                "<error unpacking native storage value {} for extension type {}>",
                storage_value,
                self.id()
            ),
        }
    }

    fn coercion_can_coerce_from(&self, other: &DType) -> bool {
        V::can_coerce_from(self, other)
    }

    fn coercion_can_coerce_to(&self, other: &DType) -> bool {
        V::can_coerce_to(self, other)
    }

    fn coercion_least_supertype(&self, other: &DType) -> Option<DType> {
        V::least_supertype(self, other)
    }
}
