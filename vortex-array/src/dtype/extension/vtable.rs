// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::fmt::Debug;
use std::fmt::Display;
use std::hash::Hash;

use vortex_error::VortexResult;

use crate::dtype::extension::ExtDType;
use crate::dtype::extension::ExtId;
use crate::scalar::ScalarValue;

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

    // Methods related to the extension `DType`.

    /// Serialize the metadata into a byte vector.
    fn serialize_metadata(&self, metadata: &Self::Metadata) -> VortexResult<Vec<u8>>;

    /// Deserialize the metadata from a byte slice.
    fn deserialize_metadata(&self, metadata: &[u8]) -> VortexResult<Self::Metadata>;

    /// Validate that the given storage type is compatible with this extension type.
    fn validate_dtype(ext_dtype: &ExtDType<Self>) -> VortexResult<()>;

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
