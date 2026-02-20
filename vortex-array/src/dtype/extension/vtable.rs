// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::fmt::Debug;
use std::fmt::Display;
use std::hash::Hash;

use vortex_error::VortexResult;
use vortex_error::vortex_bail;

use crate::dtype::DType;
use crate::dtype::extension::ExtDType;
use crate::dtype::extension::ExtDTypeRef;
use crate::dtype::extension::ExtId;

// TODO(connor): Right now this is just a vtable for DType (it used to be called `ExtDTypeVTable`),
// need to add scalar and array functionality.
// Also need to figure out if this name makes sense?
/// The public API for defining new extension types.
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

/// A dynamic vtable for extension types, used for type-erased deserialization.
// TODO(ngates): consider renaming this to ExtDTypePlugin or similar?
pub trait DynExtVTable: 'static + Send + Sync + Debug {
    /// Returns the ID for this extension type.
    fn id(&self) -> ExtId;

    /// Deserialize an extension type from serialized metadata.
    fn deserialize(&self, data: &[u8], storage_dtype: DType) -> VortexResult<ExtDTypeRef>;
}

impl<V: ExtVTable> DynExtVTable for V {
    fn id(&self) -> ExtId {
        ExtVTable::id(self)
    }

    fn deserialize(&self, data: &[u8], storage_dtype: DType) -> VortexResult<ExtDTypeRef> {
        let metadata = ExtVTable::deserialize(self, data)?;
        Ok(ExtDType::try_with_vtable(self.clone(), metadata, storage_dtype)?.erased())
    }
}
