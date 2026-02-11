// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! ClickHouse LowCardinality Extension Type.
//!
//! Marker extension so the read side can reconstruct `LowCardinality(T)`.
//! The actual dictionary encoding is handled by Vortex compression.
//!
//! # Storage
//! Same as the inner type (e.g. `Utf8` for `LowCardinality(String)`).

use std::fmt::{Display, Formatter};
use std::hash::Hash;

use vortex::dtype::extension::{ExtDType, ExtDTypeVTable, ExtID};
use vortex::dtype::{DType, Nullability};
use vortex::error::{VortexResult, vortex_bail};

/// Extension type ID.
pub const LOWCARDINALITY_EXT_ID: &str = "clickhouse.lowcardinality";

/// Metadata: stores the inner ClickHouse type string for reconstruction.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct LowCardinalityMetadata {
    /// The inner ClickHouse type string, e.g. `"String"`.
    pub inner_type: String,
}

impl Display for LowCardinalityMetadata {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "LowCardinality({})", self.inner_type)
    }
}

/// The LowCardinality extension type VTable.
#[derive(Debug, Clone, Default, PartialEq, Eq, Hash)]
pub struct ClickHouseLowCardinality;

impl ClickHouseLowCardinality {
    /// Create a ClickHouse LowCardinality DType (type-erased).
    pub fn dtype(inner_type: String, storage: DType, nullability: Nullability) -> DType {
        let metadata = LowCardinalityMetadata { inner_type };
        let ext = ExtDType::try_with_vtable(Self, metadata, storage.with_nullability(nullability))
            .expect("LowCardinality storage dtype is always valid");
        DType::Extension(ext.erased())
    }

    /// Try to extract metadata from a DType.
    pub fn try_get_metadata(dtype: &DType) -> Option<LowCardinalityMetadata> {
        if let DType::Extension(ext) = dtype {
            if ext.id().as_ref() == LOWCARDINALITY_EXT_ID {
                return ext.metadata_opt::<ClickHouseLowCardinality>().cloned();
            }
        }
        None
    }

    /// Reconstruct the ClickHouse type string.
    pub fn to_clickhouse_type(metadata: &LowCardinalityMetadata) -> String {
        format!("LowCardinality({})", metadata.inner_type)
    }
}

impl ExtDTypeVTable for ClickHouseLowCardinality {
    type Metadata = LowCardinalityMetadata;

    fn id(&self) -> ExtID {
        ExtID::new_ref(LOWCARDINALITY_EXT_ID)
    }

    fn serialize(&self, metadata: &Self::Metadata) -> VortexResult<Vec<u8>> {
        Ok(metadata.inner_type.as_bytes().to_vec())
    }

    fn deserialize(&self, data: &[u8]) -> VortexResult<Self::Metadata> {
        Ok(LowCardinalityMetadata {
            inner_type: String::from_utf8_lossy(data).to_string(),
        })
    }

    fn validate_dtype(
        &self,
        _metadata: &Self::Metadata,
        _storage_dtype: &DType,
    ) -> VortexResult<()> {
        // Any storage type is valid for LowCardinality marker
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_lowcardinality_metadata_roundtrip() -> VortexResult<()> {
        let vtable = ClickHouseLowCardinality;
        let metadata = LowCardinalityMetadata {
            inner_type: "String".to_string(),
        };
        let serialized = vtable.serialize(&metadata)?;
        let deserialized = vtable.deserialize(&serialized)?;
        assert_eq!(metadata, deserialized);
        Ok(())
    }
}
