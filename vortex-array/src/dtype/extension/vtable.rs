// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::fmt::Debug;
use std::fmt::Display;
use std::hash::Hash;

use vortex_error::VortexResult;
use vortex_error::vortex_bail;

use crate::dtype::DType;
use crate::dtype::extension::ExtId;

/// The public API for defining new extension types.
///
/// This is the non-object-safe trait that plugin authors implement to define a new extension
/// type. It specifies the type's identity, metadata, serialization, and validation.
pub trait ExtVTable: 'static + Sized + Send + Sync + Clone + Debug + Eq + Hash {
    /// Associated type containing the deserialized metadata for this extension type
    type Metadata: 'static + Send + Sync + Clone + Debug + Display + Eq + Hash;

    /// Returns the ID for this extension type.
    fn id(&self) -> ExtId;

    /// Serialize the metadata into a byte vector.
    fn serialize(&self, metadata: &Self::Metadata) -> VortexResult<Vec<u8>> {
        _ = metadata;
        vortex_bail!(
            "Serialization not implemented for extension type {}",
            self.id()
        );
    }

    /// Deserialize the metadata from a byte slice.
    fn deserialize(&self, metadata: &[u8]) -> VortexResult<Self::Metadata> {
        _ = metadata;
        vortex_bail!(
            "Deserialization not implemented for extension type {}",
            self.id()
        );
    }

    /// Validate that the given storage type is compatible with this extension type.
    fn validate_dtype(&self, metadata: &Self::Metadata, storage_dtype: &DType) -> VortexResult<()>;
}
