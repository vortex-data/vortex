// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Typed and inner representations of extension dtypes.
//!
//! - [`ExtDType<V>`]: The public typed wrapper, parameterized by a concrete [`ExtVTable`].
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

#[expect(clippy::same_name_method)]
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

    /// Returns a new [`ExtDTypeRef`] with the given nullability.
    pub fn with_nullability(&self, nullability: Nullability) -> ExtDTypeRef {
        let storage_dtype = self.storage_dtype.with_nullability(nullability);
        ExtDType::<V>::try_with_vtable(self.vtable.clone(), self.metadata.clone(), storage_dtype)
            .vortex_expect(
                "Extension DType should not fail validation with the same storage type \
                 but different nullability",
            )
            .erased()
    }

    /// Serializes the metadata into a byte vector.
    pub fn serialize_metadata(&self) -> VortexResult<Vec<u8>> {
        V::serialize_metadata(&self.vtable, &self.metadata)
    }

    /// Validates that the given storage scalar value is valid for this dtype.
    pub fn validate_scalar_value(&self, storage_value: &ScalarValue) -> VortexResult<()> {
        V::validate_scalar_value(self, storage_value)
    }

    /// Erase the concrete type information, returning a type-erased extension dtype.
    pub fn erased(self) -> ExtDTypeRef {
        ExtDTypeRef(Arc::new(self))
    }
}

/// An object-safe, sealed trait for type-erased extension dtype dispatch.
///
/// Methods that have a corresponding inherent method on [`ExtDType<V>`] are thin forwarders
/// (e.g. `id`, `storage_dtype`). Methods that exist only for erased dispatch have no
/// inherent counterpart (e.g. `as_any`, `metadata_any`, `metadata_eq`).
pub(super) trait DynExtDType: 'static + Send + Sync + super::sealed::Sealed {
    fn as_any(&self) -> &dyn Any;
    fn id(&self) -> ExtId;
    fn storage_dtype(&self) -> &DType;
    fn metadata_any(&self) -> &dyn Any;
    fn metadata_debug(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result;
    fn metadata_display(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result;
    fn metadata_eq(&self, other: &dyn Any) -> bool;
    fn metadata_hash(&self, state: &mut dyn Hasher);
    fn serialize_metadata(&self) -> VortexResult<Vec<u8>>;
    fn with_nullability(&self, nullability: Nullability) -> ExtDTypeRef;
    fn validate_scalar_value(&self, storage_value: &ScalarValue) -> VortexResult<()>;
    fn value_display(&self, f: &mut fmt::Formatter<'_>, storage_value: &ScalarValue)
    -> fmt::Result;
}

/// Blanket impl: thin forwarder to `ExtDType<V>` inherent methods.
///
/// Rust's method resolution picks inherent methods over trait methods, so `self.id()` etc.
/// call the inherent impl, not this trait impl (no infinite recursion).
impl<V: ExtVTable> DynExtDType for ExtDType<V> {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn id(&self) -> ExtId {
        self.id()
    }

    fn storage_dtype(&self) -> &DType {
        self.storage_dtype()
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

    fn serialize_metadata(&self) -> VortexResult<Vec<u8>> {
        self.serialize_metadata()
    }

    fn with_nullability(&self, nullability: Nullability) -> ExtDTypeRef {
        self.with_nullability(nullability)
    }

    fn validate_scalar_value(&self, storage_value: &ScalarValue) -> VortexResult<()> {
        self.validate_scalar_value(storage_value)
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
}
