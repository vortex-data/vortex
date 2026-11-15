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

/// Defines an extension type's identity, metadata, storage dtype, and scalar interpretation.
///
/// An extension dtype wraps a storage [`DType`] with metadata that gives its values their meaning.
/// [`storage_dtype`](Self::storage_dtype) derives the storage dtype from the in-memory metadata,
/// which **must** contain every fact needed for that declaration. Serializing valid metadata and
/// reconstructing it with the declared storage **must** preserve the metadata.
///
/// [`ExtDType`] applies the top-level nullability. Nested nullability is part of the type's
/// declaration: a nested level is nullable exactly when the type accepts nulls at that level.
/// Reads require the stored storage dtype to match the declaration, including nested nullability.
pub trait ExtVTable: 'static + Sized + Send + Sync + Clone + Debug + Eq + Hash {
    /// The in-memory metadata used to declare storage and interpret values.
    ///
    /// Serialized metadata may omit facts that are recorded in the stored storage dtype.
    /// [`metadata_from_storage`](Self::metadata_from_storage) reconstructs the complete metadata.
    type Metadata: 'static + Send + Sync + Clone + Debug + Display + Eq + Hash;

    /// Metadata stored alongside the storage dtype, in its in-memory form.
    ///
    /// Callers can construct this directly or decode it through [`deserialize_metadata`]. Storage
    /// supplies the remaining facts needed to reconstruct [`Self::Metadata`].
    ///
    /// [`deserialize_metadata`]: Self::deserialize_metadata
    type StoredMetadata;

    /// A native Rust value that represents a scalar of the extension type.
    ///
    /// Represents a non-null scalar value. Nullable scalar views use
    /// `Option<Self::NativeValue<'a>>`.
    type NativeValue<'a>: Display;

    /// Returns the ID for this extension type.
    fn id(&self) -> ExtId;

    // Methods related to the extension `DType`.

    /// Serializes the metadata for storage alongside the storage dtype.
    fn serialize_metadata(&self, metadata: &Self::Metadata) -> VortexResult<Vec<u8>>;

    /// Decodes the stored metadata from serialized bytes.
    fn deserialize_metadata(&self, bytes: &[u8]) -> VortexResult<Self::StoredMetadata>;

    /// Reconstructs complete metadata from the stored metadata and storage dtype.
    ///
    /// Checked constructors pass storage with a nullable top level. Implementations **must**
    /// preserve the stored metadata's semantic information and infer only facts represented by storage.
    /// The framework validates the result and compares its declaration with the supplied storage,
    /// including nested nullability, before constructing an extension dtype.
    fn metadata_from_storage(
        &self,
        stored_metadata: Self::StoredMetadata,
        storage_dtype: &DType,
    ) -> VortexResult<Self::Metadata>;

    /// Declares the storage dtype for valid metadata.
    ///
    /// The declaration determines the storage structure, element types, and nested nullability.
    /// [`ExtDType::try_with_vtable`] applies its requested top-level nullability to the result.
    /// Callers **must** pass metadata accepted by [`validate_metadata`](Self::validate_metadata).
    /// Implementations may panic on invalid metadata.
    fn storage_dtype(metadata: &Self::Metadata) -> DType;

    /// Checks that the metadata describes a valid dtype of this extension type.
    ///
    /// Dtype constructors call this before requesting the storage declaration. File reads use
    /// the same checked construction path. The default accepts every metadata value.
    fn validate_metadata(_metadata: &Self::Metadata) -> VortexResult<()> {
        Ok(())
    }

    // Methods related to the extension scalar values.

    /// Checks that the storage value belongs to the extension type.
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

    /// Validates a storage [`ScalarValue`] and unpacks its native value.
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
