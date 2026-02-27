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
pub struct ExtDType<V: ExtVTable>(pub(super) Arc<ExtDTypeInner<V>>);

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
        vtable.validate_dtype(&metadata, &storage_dtype)?;

        Ok(Self(Arc::new(ExtDTypeInner::<V> {
            vtable,
            metadata,
            storage_dtype,
        })))
    }

    /// Returns the identifier of the extension type.
    pub fn id(&self) -> ExtId {
        self.0.vtable.id()
    }

    /// Returns the vtable of the extension type.
    pub fn vtable(&self) -> &V {
        &self.0.vtable
    }

    /// Returns the metadata of the extension type.
    pub fn metadata(&self) -> &V::Metadata {
        &self.0.metadata
    }

    /// Returns the storage dtype of the extension type.
    pub fn storage_dtype(&self) -> &DType {
        &self.0.storage_dtype
    }

    /// Erase the concrete type information, returning a type-erased extension dtype.
    pub fn erased(self) -> ExtDTypeRef {
        ExtDTypeRef(self.0)
    }
}

// ---------------------------------------------------------------------------
// Private inner struct + sealed trait
// ---------------------------------------------------------------------------

/// The private inner representation of an extension dtype, pairing a vtable with its metadata
/// and storage dtype.
///
/// This is the sole implementor of [`DynExtDType`], enabling [`ExtDTypeRef`] to safely downcast
/// back to the concrete vtable type via [`Any`].
#[derive(Debug, PartialEq, Eq, Hash)]
pub(super) struct ExtDTypeInner<V: ExtVTable> {
    /// The extension dtype vtable.
    pub(super) vtable: V,
    /// The extension dtype metadata.
    pub(super) metadata: V::Metadata,
    /// The underlying storage dtype.
    pub(super) storage_dtype: DType,
}

/// An object-safe, sealed trait encapsulating the behavior for extension dtypes.
///
/// This provides type-erased access to the extension dtype's identity, storage dtype, and
/// metadata. The only implementor is [`ExtDTypeInner`].
pub(super) trait DynExtDType: 'static + Send + Sync + super::sealed::Sealed {
    /// Returns `self` as a trait object for downcasting.
    fn as_any(&self) -> &dyn Any;
    /// Returns the [`ExtId`] identifying this extension type.
    fn id(&self) -> ExtId;
    /// Returns a reference to the storage [`DType`].
    fn storage_dtype(&self) -> &DType;
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
}

impl<V: ExtVTable> DynExtDType for ExtDTypeInner<V> {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn id(&self) -> ExtId {
        self.vtable.id()
    }

    fn storage_dtype(&self) -> &DType {
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
        self.vtable
            .validate_scalar_value(&self.metadata, &self.storage_dtype, storage_value)
    }

    fn value_display(
        &self,
        f: &mut fmt::Formatter<'_>,
        storage_value: &ScalarValue,
    ) -> fmt::Result {
        match self
            .vtable
            .unpack_native(&self.metadata, &self.storage_dtype, storage_value)
        {
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
