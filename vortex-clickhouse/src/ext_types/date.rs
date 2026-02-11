// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! ClickHouse Date Extension Type.
//!
//! Preserves `Date` and `Date32` semantics in Vortex files.
//!
//! # Storage
//! - `Date`: `Primitive(U16)` — days since 1970-01-01
//! - `Date32`: `Primitive(I32)` — days since 1970-01-01 (wider range)

use std::fmt::{Display, Formatter};
use std::hash::Hash;

use vortex::dtype::extension::{ExtDType, ExtDTypeVTable, ExtID};
use vortex::dtype::{DType, Nullability, PType};
use vortex::error::{VortexResult, vortex_bail};

/// Extension type ID.
pub const DATE_EXT_ID: &str = "clickhouse.date";

/// Metadata for the date extension type.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct DateMetadata {
    /// `false` for `Date` (U16), `true` for `Date32` (I32).
    pub is_date32: bool,
}

impl Display for DateMetadata {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        if self.is_date32 {
            write!(f, "Date32")
        } else {
            write!(f, "Date")
        }
    }
}

/// The Date extension type VTable.
#[derive(Debug, Clone, Default, PartialEq, Eq, Hash)]
pub struct ClickHouseDate;

impl ClickHouseDate {
    /// Create a ClickHouse Date DType (type-erased).
    pub fn dtype(is_date32: bool, nullability: Nullability) -> DType {
        let storage = if is_date32 {
            DType::Primitive(PType::I32, Nullability::NonNullable)
        } else {
            DType::Primitive(PType::U16, Nullability::NonNullable)
        };
        let metadata = DateMetadata { is_date32 };
        let ext = ExtDType::try_with_vtable(Self, metadata, storage.with_nullability(nullability))
            .expect("Date storage dtype is always valid");
        DType::Extension(ext.erased())
    }

    /// Try to extract `DateMetadata` from a DType.
    pub fn try_get_metadata(dtype: &DType) -> Option<DateMetadata> {
        if let DType::Extension(ext) = dtype {
            if ext.id().as_ref() == DATE_EXT_ID {
                return ext.metadata_opt::<ClickHouseDate>().copied();
            }
        }
        None
    }

    /// Reconstruct the ClickHouse type string.
    pub fn to_clickhouse_type(metadata: &DateMetadata) -> String {
        format!("{}", metadata)
    }
}

impl ExtDTypeVTable for ClickHouseDate {
    type Metadata = DateMetadata;

    fn id(&self) -> ExtID {
        ExtID::new_ref(DATE_EXT_ID)
    }

    fn serialize(&self, metadata: &Self::Metadata) -> VortexResult<Vec<u8>> {
        Ok(vec![metadata.is_date32 as u8])
    }

    fn deserialize(&self, data: &[u8]) -> VortexResult<Self::Metadata> {
        if data.is_empty() {
            vortex_bail!("Date metadata is empty");
        }
        Ok(DateMetadata {
            is_date32: data[0] != 0,
        })
    }

    fn validate_dtype(&self, metadata: &Self::Metadata, storage_dtype: &DType) -> VortexResult<()> {
        if metadata.is_date32 {
            match storage_dtype {
                DType::Primitive(PType::I32, _) => Ok(()),
                _ => vortex_bail!(
                    "Date32 requires Primitive(I32) storage, got {:?}",
                    storage_dtype
                ),
            }
        } else {
            match storage_dtype {
                DType::Primitive(PType::U16, _) => Ok(()),
                _ => vortex_bail!(
                    "Date requires Primitive(U16) storage, got {:?}",
                    storage_dtype
                ),
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_date_metadata_roundtrip() -> VortexResult<()> {
        let vtable = ClickHouseDate;
        for is_date32 in [false, true] {
            let metadata = DateMetadata { is_date32 };
            let serialized = vtable.serialize(&metadata)?;
            let deserialized = vtable.deserialize(&serialized)?;
            assert_eq!(metadata, deserialized);
        }
        Ok(())
    }
}
