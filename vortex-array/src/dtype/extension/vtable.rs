// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::fmt::Debug;
use std::fmt::Display;
use std::hash::Hash;

use vortex_error::VortexResult;
use vortex_error::vortex_bail;

use crate::dtype::DType;
use crate::dtype::ExtDType;
use crate::dtype::ExtID;
use crate::dtype::extension::ExtDTypeRef;

/// The public API for defining new extension DTypes.
pub trait ExtDTypeVTable: 'static + Sized + Send + Sync + Clone + Debug + Eq + Hash {
    /// Associated type containing the deserialized metadata for this extension type
    type Metadata: 'static + Send + Sync + Clone + Debug + Display + Eq + Hash;

    /// Returns the ID for this extension type.
    fn id(&self) -> ExtID;

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
pub trait DynExtDTypeVTable: 'static + Send + Sync + Debug {
    /// Returns the ID for this extension type.
    fn id(&self) -> ExtID;

    /// Deserialize an extension type from serialized metadata.
    fn deserialize(&self, data: &[u8], storage_dtype: DType) -> VortexResult<ExtDTypeRef>;
}

impl<V: ExtDTypeVTable> DynExtDTypeVTable for V {
    fn id(&self) -> ExtID {
        ExtDTypeVTable::id(self)
    }

    fn deserialize(&self, data: &[u8], storage_dtype: DType) -> VortexResult<ExtDTypeRef> {
        let metadata = ExtDTypeVTable::deserialize(self, data)?;
        Ok(ExtDType::try_with_vtable(self.clone(), metadata, storage_dtype)?.erased())
    }
}

/// An empty metadata struct for extension dtypes that do not require any metadata.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct EmptyMetadata;
impl Display for EmptyMetadata {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "")
    }
}
