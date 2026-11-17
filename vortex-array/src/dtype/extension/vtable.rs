// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::fmt::Debug;
use std::fmt::Display;
use std::hash::Hash;

use vortex_error::VortexResult;
use vortex_error::vortex_ensure_eq;

use crate::ArrayRef;
use crate::Columnar;
use crate::ExecutionCtx;
use crate::IntoArray;
use crate::dtype::Claims;
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

    /// Declares which operations give the same result on values of this dtype and on their storage.
    ///
    /// Dtype constructors check that the declared storage holds each claim. Claims **must** be
    /// independent of top-level nullability so they remain valid after a nullability change.
    /// The default is [`Claims::none`], so the type is opaque to operations that require a claim.
    ///
    /// Operations that require a claim run on the storage values and trust the claim. A claim that
    /// the storage values do not honor gives wrong results.
    ///
    /// If the metadata carries a storage dtype that can contain other extension dtypes, derive the
    /// claims from that storage with [`DType::holds`]. A reader without the plugin of a nested type
    /// then gets fewer claims instead of failing to build the dtype.
    ///
    /// Callers **must** pass metadata accepted by [`validate_metadata`](Self::validate_metadata).
    fn claims(_metadata: &Self::Metadata) -> Claims {
        Claims::none()
    }

    /// Checks that the metadata describes a valid dtype of this extension type.
    ///
    /// Dtype constructors call this before requesting the storage declaration. File reads use
    /// the same checked construction path. The default accepts every metadata value.
    fn validate_metadata(_metadata: &Self::Metadata) -> VortexResult<()> {
        Ok(())
    }

    /// Checks that each non-null storage value belongs to the extension type.
    ///
    /// The default applies [`Self::validate_scalar_value`] to each non-null value. An override
    /// **must** accept the same storage arrays as the default.
    ///
    /// # Errors
    ///
    /// Returns an error if the storage dtype differs from the declaration, reading the storage
    /// fails, or a non-null value is outside the extension type's domain.
    fn validate_array(
        ext_dtype: &ExtDType<Self>,
        storage: &ArrayRef,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<()> {
        validate_each_value(ext_dtype, storage, ctx)
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

/// The default array validator, also used as the reference for conformance checks.
pub(crate) fn validate_each_value<V: ExtVTable>(
    ext_dtype: &ExtDType<V>,
    storage: &ArrayRef,
    ctx: &mut ExecutionCtx,
) -> VortexResult<()> {
    vortex_ensure_eq!(
        ext_dtype.storage_dtype(),
        storage.dtype(),
        "{} storage must match its declaration",
        ext_dtype.id()
    );

    let canonical = match storage.clone().execute::<Columnar>(ctx)? {
        Columnar::Constant(constant) if constant.is_empty() => return Ok(()),
        Columnar::Constant(constant) => {
            return match constant.scalar().value() {
                Some(value) => ext_dtype.validate_scalar_value(value),
                None => Ok(()),
            };
        }
        Columnar::Canonical(canonical) => canonical,
    };

    let storage = canonical.into_array();
    let mut probe = storage.repeated_probe();
    for index in 0..storage.len() {
        let scalar = probe.execute_scalar(index, ctx)?;
        if let Some(value) = scalar.value() {
            ext_dtype.validate_scalar_value(value)?;
        }
    }

    Ok(())
}
