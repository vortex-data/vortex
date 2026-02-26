// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::fmt::Debug;
use std::fmt::Display;
use std::hash::Hash;

use vortex_error::VortexResult;

use crate::dtype::DType;
use crate::dtype::extension::ExtId;
use crate::scalar::ScalarValue;

/// The public API for defining new extension types.
///
/// This is the non-object-safe trait that plugin authors implement to define a new extension
/// type. It specifies the type's identity, metadata, serialization, and validation.
///
/// Vtable identity is determined by [`id()`](ExtVTable::id), not by the vtable value itself.
/// Two type-erased extension dtypes ([`ExtDTypeRef`](super::ExtDTypeRef)) are equal iff they
/// share the same ID, equal metadata, and equal storage dtype.
pub trait ExtVTable: 'static + Sized + Send + Sync + Clone {
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
    fn validate_dtype(&self, metadata: &Self::Metadata, storage_dtype: &DType) -> VortexResult<()>;

    // Methods related to the extension scalar values.

    /// Validate the given storage value is compatible with the extension type.
    ///
    /// By default, this calls [`unpack_native()`](ExtVTable::unpack_native) and discards the result.
    ///
    /// # Errors
    ///
    /// Returns an error if the storage [`ScalarValue`] is not compatible with the extension type.
    fn validate_scalar_value(
        &self,
        metadata: &Self::Metadata,
        storage_dtype: &DType,
        storage_value: &ScalarValue,
    ) -> VortexResult<()> {
        self.unpack_native(metadata, storage_dtype, storage_value)
            .map(|_| ())
    }

    /// Validate and unpack a native value from the storage [`ScalarValue`].
    ///
    /// Note that [`ExtVTable::validate_dtype()`] is always called first to validate the storage
    /// [`DType`], and the [`Scalar`](crate::scalar::Scalar) implementation will verify that the
    /// storage value is compatible with the storage dtype on construction.
    ///
    /// # Errors
    ///
    /// Returns an error if the storage [`ScalarValue`] is not compatible with the extension type.
    fn unpack_native<'a>(
        &self,
        metadata: &'a Self::Metadata,
        storage_dtype: &'a DType,
        storage_value: &'a ScalarValue,
    ) -> VortexResult<Self::NativeValue<'a>>;
}
