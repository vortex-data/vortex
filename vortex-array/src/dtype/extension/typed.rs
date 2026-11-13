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

use vortex_error::VortexResult;
use vortex_error::vortex_bail;

use crate::dtype::DType;
use crate::dtype::Nullability;
use crate::dtype::extension::ExtDTypeRef;
use crate::dtype::extension::ExtId;
use crate::dtype::extension::ExtVTable;
use crate::scalar::ScalarValue;

/// A typed extension data type, parameterized by a concrete [`ExtVTable`].
///
/// [`try_new()`] and [`try_with_vtable()`] construct a dtype from complete metadata.
/// [`try_from_storage`](Self::try_from_storage) reconstructs metadata from its stored metadata and storage.
/// [`erased()`] returns an [`ExtDTypeRef`] for type-erased dispatch.
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
    /// The declared storage dtype with this dtype's top-level nullability.
    ///
    /// Cached so that [`storage_dtype`](Self::storage_dtype) can return a reference.
    storage_dtype: DType,
}

/// Convenience implementation for zero-sized VTables (or VTables that implement `Default`).
impl<V: ExtVTable + Default> ExtDType<V> {
    /// Reconstructs a dtype from stored metadata and its complete storage dtype.
    ///
    /// See [`try_from_storage_with_vtable`](Self::try_from_storage_with_vtable) for the checks.
    ///
    /// ```
    /// use vortex_array::dtype::{DType, Nullability, PType};
    /// use vortex_array::dtype::extension::ExtDType;
    /// use vortex_array::extension::datetime::{Time, TimeUnit};
    ///
    /// let storage = DType::Primitive(PType::I32, Nullability::Nullable);
    /// let dtype = ExtDType::<Time>::try_from_storage(TimeUnit::Seconds, storage.clone())?;
    /// assert_eq!(dtype.storage_dtype(), &storage);
    /// # Ok::<(), vortex_error::VortexError>(())
    /// ```
    pub fn try_from_storage(
        stored_metadata: V::StoredMetadata,
        storage_dtype: DType,
    ) -> VortexResult<Self> {
        Self::try_from_storage_with_vtable(V::default(), stored_metadata, storage_dtype)
    }

    /// Creates a new extension dtype with the given metadata and nullability, as
    /// [`try_with_vtable`](Self::try_with_vtable) does.
    pub fn try_new(metadata: V::Metadata, nullability: Nullability) -> VortexResult<Self> {
        Self::try_with_vtable(V::default(), metadata, nullability)
    }
}

#[expect(clippy::same_name_method)]
impl<V: ExtVTable> ExtDType<V> {
    /// Reconstructs a dtype whose declaration exactly matches the supplied storage.
    ///
    /// The storage's top-level nullability belongs to the resulting dtype. Reconstruction sees
    /// a nullable outer level, so that level cannot affect the metadata.
    ///
    /// # Errors
    ///
    /// Returns an error if reconstruction or metadata validation fails, or the declared storage
    /// differs from the supplied dtype, including nested nullability.
    pub fn try_from_storage_with_vtable(
        vtable: V,
        stored_metadata: V::StoredMetadata,
        storage_dtype: DType,
    ) -> VortexResult<Self> {
        let normalized = storage_dtype.as_nullable();
        let metadata = vtable.metadata_from_storage(stored_metadata, &normalized)?;
        V::validate_metadata(&metadata)?;

        let declared = V::storage_dtype(&metadata).as_nullable();

        // Array reads decode storage children with the declaration. A different stored form,
        // including different nested nullability, would be misread instead of rejected.
        if declared != normalized {
            // Null storage has no non-nullable form. Report a mismatch without requesting one.
            let expected = if has_non_nullable_form(&declared) {
                declared.with_nullability(storage_dtype.nullability())
            } else {
                declared
            };
            vortex_bail!(
                "{} storage must be {expected}, got {storage_dtype}",
                vtable.id()
            );
        }

        Ok(Self {
            vtable,
            metadata,
            storage_dtype: declared.with_nullability(storage_dtype.nullability()),
        })
    }

    /// Creates a new extension dtype with the given metadata and nullability.
    ///
    /// The storage dtype is the one that [`ExtVTable::storage_dtype`] declares for `metadata`, with
    /// `nullability` at the top level.
    ///
    /// # Errors
    ///
    /// Returns an error if [`ExtVTable::validate_metadata`] rejects `metadata`, or if a
    /// non-nullable result is requested for storage without a non-nullable form, such as
    /// [`DType::Null`] or an extension dtype over it.
    pub fn try_with_vtable(
        vtable: V,
        metadata: V::Metadata,
        nullability: Nullability,
    ) -> VortexResult<Self> {
        V::validate_metadata(&metadata)?;
        let storage_dtype = V::storage_dtype(&metadata);

        if nullability == Nullability::NonNullable && !has_non_nullable_form(&storage_dtype) {
            vortex_bail!(
                "{} storage must have a non-nullable form, got {storage_dtype}",
                vtable.id()
            );
        }

        let storage_dtype = storage_dtype.with_nullability(nullability);

        Ok(Self {
            vtable,
            metadata,
            storage_dtype,
        })
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
    ///
    /// Rebuilds storage from the declaration and applies `nullability` at the top level. Metadata
    /// and nested nullability stay the same, so metadata validation is not repeated.
    ///
    /// # Panics
    ///
    /// Panics if a non-nullable result is requested for storage without a non-nullable form, such
    /// as [`DType::Null`] or an extension dtype over it.
    pub fn with_nullability(&self, nullability: Nullability) -> ExtDTypeRef {
        Self {
            vtable: self.vtable.clone(),
            metadata: self.metadata.clone(),
            storage_dtype: V::storage_dtype(&self.metadata).with_nullability(nullability),
        }
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

/// Returns whether `dtype` has a non-nullable form, without which [`DType::with_nullability`]
/// panics. [`DType::Null`] has none, and neither has an extension dtype whose storage has none.
fn has_non_nullable_form(dtype: &DType) -> bool {
    match dtype {
        DType::Null => false,
        DType::Extension(ext) => has_non_nullable_form(ext.storage_dtype()),
        _ => true,
    }
}
