// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! ClickHouse FixedString Extension Type.
//!
//! Preserves `FixedString(N)` semantics in Vortex files.
//!
//! # Storage
//! `Utf8` — the fixed-length padding is not stored; only the actual content is kept.

use std::fmt::{Display, Formatter};
use std::hash::Hash;

use vortex::dtype::extension::{ExtDType, ExtDTypeVTable, ExtID};
use vortex::dtype::{DType, Nullability};
use vortex::error::{VortexResult, vortex_bail};

/// Extension type ID.
pub const FIXEDSTRING_EXT_ID: &str = "clickhouse.fixedstring";

/// Metadata: the N value from `FixedString(N)`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct FixedStringMetadata {
    pub n: u32,
}

impl Display for FixedStringMetadata {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "FixedString({})", self.n)
    }
}

/// The FixedString extension type VTable.
#[derive(Debug, Clone, Default, PartialEq, Eq, Hash)]
pub struct ClickHouseFixedString;

impl ClickHouseFixedString {
    /// Create a ClickHouse FixedString DType (type-erased).
    pub fn dtype(n: u32, nullability: Nullability) -> DType {
        let metadata = FixedStringMetadata { n };
        let storage = DType::Utf8(Nullability::NonNullable);
        let ext = ExtDType::try_with_vtable(Self, metadata, storage.with_nullability(nullability))
            .expect("FixedString storage dtype is always valid");
        DType::Extension(ext.erased())
    }

    /// Try to extract metadata from a DType.
    pub fn try_get_metadata(dtype: &DType) -> Option<FixedStringMetadata> {
        if let DType::Extension(ext) = dtype {
            if ext.id().as_ref() == FIXEDSTRING_EXT_ID {
                return ext.metadata_opt::<ClickHouseFixedString>().copied();
            }
        }
        None
    }

    /// Reconstruct the ClickHouse type string.
    pub fn to_clickhouse_type(metadata: &FixedStringMetadata) -> String {
        format!("FixedString({})", metadata.n)
    }
}

impl ExtDTypeVTable for ClickHouseFixedString {
    type Metadata = FixedStringMetadata;

    fn id(&self) -> ExtID {
        ExtID::new_ref(FIXEDSTRING_EXT_ID)
    }

    fn serialize(&self, metadata: &Self::Metadata) -> VortexResult<Vec<u8>> {
        Ok(metadata.n.to_le_bytes().to_vec())
    }

    fn deserialize(&self, data: &[u8]) -> VortexResult<Self::Metadata> {
        if data.len() < 4 {
            vortex_bail!("FixedString metadata too short");
        }
        let n = u32::from_le_bytes([data[0], data[1], data[2], data[3]]);
        Ok(FixedStringMetadata { n })
    }

    fn validate_dtype(
        &self,
        _metadata: &Self::Metadata,
        storage_dtype: &DType,
    ) -> VortexResult<()> {
        match storage_dtype {
            DType::Utf8(_) => Ok(()),
            _ => vortex_bail!("FixedString requires Utf8 storage, got {:?}", storage_dtype),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fixedstring_metadata_roundtrip() -> VortexResult<()> {
        let vtable = ClickHouseFixedString;
        let metadata = FixedStringMetadata { n: 16 };
        let serialized = vtable.serialize(&metadata)?;
        let deserialized = vtable.deserialize(&serialized)?;
        assert_eq!(metadata, deserialized);
        Ok(())
    }
}
