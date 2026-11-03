// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::fmt::Debug;
use std::fmt::Display;
use std::hash::Hash;

use vortex_error::VortexResult;

use crate::dtype::DType;
use crate::dtype::extension::ExtDType;
use crate::dtype::extension::ExtId;
use crate::scalar::ScalarValue;

/// The public API for defining new extension types.
///
/// This is the non-object-safe trait that plugin authors implement to define a new extension type.
/// It specifies the type's identity, metadata, serialization, storage dtype, and scalar
/// compatibility.
///
/// An extension dtype is not a new physical array layout. It is a logical wrapper around a storage
/// [`DType`] plus metadata. The type declares its storage dtype from the metadata with
/// [`storage_dtype`](Self::storage_dtype) instead of accepting one from the caller. So the
/// in-memory metadata **must** hold every fact that the storage dtype depends on, even a fact that
/// the serialized metadata leaves out.
///
/// Nested nullability is part of the declaration: a nested level is nullable exactly when the type
/// accepts nulls at that level. A read compares the stored storage dtype with the declaration, as
/// [`ExtDTypePlugin::deserialize`] describes.
///
/// [`ExtDTypePlugin::deserialize`]: crate::dtype::extension::ExtDTypePlugin::deserialize
pub trait ExtVTable: 'static + Sized + Send + Sync + Clone + Debug + Eq + Hash {
    /// Associated type containing the deserialized metadata for this extension type.
    type Metadata: 'static + Send + Sync + Clone + Debug + Display + Eq + Hash;

    /// A native Rust value that represents a scalar of the extension type.
    ///
    /// The value only represents non-null values. We denote nullable values as `Option<Value>`.
    type NativeValue<'a>: Display;

    /// Returns the ID for this extension type.
    fn id(&self) -> ExtId;

    // Methods related to the extension `DType`.

    /// Serialize the metadata into a byte vector.
    fn serialize_metadata(&self, metadata: &Self::Metadata) -> VortexResult<Vec<u8>>;

    /// Deserialize the metadata from a byte slice and the storage dtype stored next to it.
    ///
    /// The storage dtype supplies the facts that the bytes leave out. The result **must not**
    /// depend on the top-level nullability of `storage_dtype`, which belongs to the extension dtype
    /// as [`storage_dtype`](Self::storage_dtype) describes. Violating this does not cause undefined
    /// behavior, but dtypes read from storage dtypes that differ only in top-level nullability then
    /// compare unequal even when nullability is ignored. This method does not need to check the
    /// rest of `storage_dtype`, because [`ExtDTypePlugin::deserialize`] compares it with the
    /// declaration.
    ///
    /// [`ExtDTypePlugin::deserialize`]: crate::dtype::extension::ExtDTypePlugin::deserialize
    fn deserialize_metadata(
        &self,
        metadata: &[u8],
        storage_dtype: &DType,
    ) -> VortexResult<Self::Metadata>;

    /// Returns the storage dtype that `metadata` declares.
    ///
    /// The metadata decides the whole storage dtype except its top-level nullability, which belongs
    /// to the extension dtype: [`ExtDType::try_with_vtable`] replaces it, so a type cannot choose
    /// it. Callers **must** only pass metadata that [`validate_metadata`](Self::validate_metadata)
    /// accepts, and an implementation may panic on other metadata.
    fn storage_dtype(metadata: &Self::Metadata) -> DType;

    /// Validates the metadata.
    ///
    /// [`ExtDType::try_with_vtable`] calls this before [`storage_dtype`](Self::storage_dtype) for
    /// every dtype that it builds, including a dtype that a read rebuilds. By default, every
    /// metadata value is valid.
    fn validate_metadata(_metadata: &Self::Metadata) -> VortexResult<()> {
        Ok(())
    }

    // Methods related to the extension scalar values.

    /// Validate the given storage value is compatible with the extension type.
    ///
    /// By default, this calls [`unpack_native()`](ExtVTable::unpack_native) and discards the
    /// result.
    ///
    /// # Errors
    ///
    /// Returns an error if the storage [`ScalarValue`] is not compatible with the extension type.
    fn validate_scalar_value(
        ext_dtype: &ExtDType<Self>,
        storage_value: &ScalarValue,
    ) -> VortexResult<()> {
        Self::unpack_native(ext_dtype, storage_value).map(|_| ())
    }

    /// Validate and unpack a native value from the storage [`ScalarValue`].
    ///
    /// The metadata satisfies [`Self::validate_metadata`], and the storage dtype is the one that it
    /// declares. [`Scalar`](crate::scalar::Scalar) construction also verifies that the storage
    /// value is compatible with the storage dtype.
    ///
    /// # Errors
    ///
    /// Returns an error if the storage [`ScalarValue`] is not compatible with the extension type.
    fn unpack_native<'a>(
        ext_dtype: &'a ExtDType<Self>,
        storage_value: &'a ScalarValue,
    ) -> VortexResult<Self::NativeValue<'a>>;
}
