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

/// Converters between an extension's on-disk metadata bytes and the canonical Arrow JSON wire.
///
/// Each Vortex extension that maps to a canonical Arrow extension owns the codec used at the
/// Arrow boundary so [`ExtVTable`] stays Arrow-unaware in the storage path.
#[derive(Copy, Clone, Debug)]
pub struct ArrowCanonicalCodec {
    /// Convert raw extension metadata bytes into the JSON string Arrow consumers expect.
    pub to_json: fn(&[u8]) -> VortexResult<String>,
    /// Parse the JSON string Arrow consumers produce back into raw extension metadata bytes.
    pub from_json: fn(&str) -> VortexResult<Vec<u8>>,
}

/// Identifies the canonical Arrow extension this Vortex extension serializes as.
///
/// Returned by [`ExtVTable::arrow_canonical`]. The `arrow_id` is the name written into
/// `ARROW:extension:name`; the `codec` round-trips metadata bytes through Arrow's JSON wire.
#[derive(Copy, Clone, Debug)]
pub struct ArrowCanonicalAlias {
    /// The canonical Arrow extension id (e.g. `arrow.fixed_shape_tensor`).
    pub arrow_id: ExtId,
    /// Converters between Vortex on-disk metadata bytes and Arrow's JSON wire.
    pub codec: ArrowCanonicalCodec,
}

/// The public API for defining new extension types.
///
/// This is the non-object-safe trait that plugin authors implement to define a new extension type.
/// It specifies the type's identity, metadata, serialization, and validation.
pub trait ExtVTable: 'static + Sized + Send + Sync + Clone + Debug + Eq + Hash {
    /// Associated type containing the deserialized metadata for this extension type.
    type Metadata: 'static + Send + Sync + Clone + Debug + Display + Eq + Hash;

    /// A native Rust value that represents a scalar of the extension type.
    ///
    /// The value only represents non-null values. We denote nullable values as `Option<Value>`.
    type NativeValue<'a>: Display;

    /// Returns the ID for this extension type.
    fn id(&self) -> ExtId;

    /// Optional canonical Arrow extension this type serializes as at the Arrow boundary.
    ///
    /// Override to map this Vortex extension to a registered canonical Arrow extension
    /// (e.g. `arrow.fixed_shape_tensor`). The default `None` means the type round-trips
    /// through base64-encoded metadata under its own [`ExtId`].
    fn arrow_canonical(&self) -> Option<ArrowCanonicalAlias> {
        None
    }

    // Methods related to the extension `DType`.

    /// Serialize the metadata into a byte vector.
    fn serialize_metadata(&self, metadata: &Self::Metadata) -> VortexResult<Vec<u8>>;

    /// Deserialize the metadata from a byte slice.
    fn deserialize_metadata(&self, metadata: &[u8]) -> VortexResult<Self::Metadata>;

    /// Validate that the given storage type is compatible with this extension type.
    fn validate_dtype(ext_dtype: &ExtDType<Self>) -> VortexResult<()>;

    /// Can a value of `other` be implicitly widened into this type? (e.g. GeographyType might
    /// accept Point, LineString, etc.)
    ///
    /// Implementors only need to override one of `can_coerce_from` or `can_coerce_to`. We have both
    /// so that either side of the coercion can provide the logic.
    fn can_coerce_from(ext_dtype: &ExtDType<Self>, other: &DType) -> bool {
        let _ = (ext_dtype, other);
        false
    }

    /// Can this type be implicitly widened into `other`?
    ///
    /// Implementors only need to override one of `can_coerce_from` or `can_coerce_to`. We have both
    /// so that either side of the coercion can provide the logic.
    fn can_coerce_to(ext_dtype: &ExtDType<Self>, other: &DType) -> bool {
        let _ = (ext_dtype, other);
        false
    }

    /// Given two types in a Uniform context, what is their least supertype?
    ///
    /// Return None if no supertype exists.
    fn least_supertype(ext_dtype: &ExtDType<Self>, other: &DType) -> Option<DType> {
        let _ = (ext_dtype, other);
        None
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
    /// Note that [`ExtVTable::validate_dtype()`] is always called first to validate the storage
    /// [`crate::dtype::DType`], and the [`Scalar`](crate::scalar::Scalar) implementation will
    /// verify that the storage value is compatible with the storage dtype on construction.
    ///
    /// # Errors
    ///
    /// Returns an error if the storage [`ScalarValue`] is not compatible with the extension type.
    fn unpack_native<'a>(
        ext_dtype: &'a ExtDType<Self>,
        storage_value: &'a ScalarValue,
    ) -> VortexResult<Self::NativeValue<'a>>;
}
