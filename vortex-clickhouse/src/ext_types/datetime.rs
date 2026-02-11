// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! ClickHouse DateTime Extension Type.
//!
//! Preserves `DateTime`/`DateTime64` precision and timezone in Vortex files.
//!
//! # Storage
//! - `DateTime` (precision=0): `Primitive(U32)` — seconds since epoch
//! - `DateTime64(p)` (precision>0): `Primitive(I64)` — sub-second ticks since epoch

use std::fmt::{Display, Formatter};
use std::hash::Hash;

use vortex::dtype::extension::{ExtDType, ExtDTypeVTable, ExtID};
use vortex::dtype::{DType, Nullability, PType};
use vortex::error::{VortexResult, vortex_bail};

/// Extension type ID.
pub const DATETIME_EXT_ID: &str = "clickhouse.datetime";

/// Metadata for the datetime extension type.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct DateTimeMetadata {
    /// 0 for `DateTime`, 1–9 for `DateTime64(p)`.
    pub precision: u8,
    /// Optional timezone string, e.g. `"UTC"`.
    pub timezone: Option<String>,
}

impl Display for DateTimeMetadata {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        if self.precision == 0 {
            if let Some(tz) = &self.timezone {
                write!(f, "DateTime('{}')", tz)
            } else {
                write!(f, "DateTime")
            }
        } else if let Some(tz) = &self.timezone {
            write!(f, "DateTime64({}, '{}')", self.precision, tz)
        } else {
            write!(f, "DateTime64({})", self.precision)
        }
    }
}

/// The DateTime extension type VTable.
#[derive(Debug, Clone, Default, PartialEq, Eq, Hash)]
pub struct ClickHouseDateTime;

impl ClickHouseDateTime {
    /// Create a ClickHouse DateTime DType (type-erased).
    pub fn dtype(metadata: DateTimeMetadata, nullability: Nullability) -> DType {
        let storage = if metadata.precision == 0 {
            DType::Primitive(PType::U32, Nullability::NonNullable)
        } else {
            DType::Primitive(PType::I64, Nullability::NonNullable)
        };
        let ext = ExtDType::try_with_vtable(Self, metadata, storage.with_nullability(nullability))
            .expect("DateTime storage dtype is always valid");
        DType::Extension(ext.erased())
    }

    /// Try to extract `DateTimeMetadata` from a DType.
    pub fn try_get_metadata(dtype: &DType) -> Option<DateTimeMetadata> {
        if let DType::Extension(ext) = dtype {
            if ext.id().as_ref() == DATETIME_EXT_ID {
                return ext.metadata_opt::<ClickHouseDateTime>().cloned();
            }
        }
        None
    }

    /// Reconstruct the ClickHouse type string.
    pub fn to_clickhouse_type(metadata: &DateTimeMetadata) -> String {
        format!("{}", metadata)
    }
}

impl ExtDTypeVTable for ClickHouseDateTime {
    type Metadata = DateTimeMetadata;

    fn id(&self) -> ExtID {
        ExtID::new_ref(DATETIME_EXT_ID)
    }

    fn serialize(&self, metadata: &Self::Metadata) -> VortexResult<Vec<u8>> {
        // [precision: u8] [tz_len: u16] [tz_bytes...]
        let mut buf = vec![metadata.precision];
        if let Some(tz) = &metadata.timezone {
            let tz_bytes = tz.as_bytes();
            let tz_len = tz_bytes.len() as u16;
            buf.extend_from_slice(&tz_len.to_le_bytes());
            buf.extend_from_slice(tz_bytes);
        } else {
            buf.extend_from_slice(&0u16.to_le_bytes());
        }
        Ok(buf)
    }

    fn deserialize(&self, data: &[u8]) -> VortexResult<Self::Metadata> {
        if data.len() < 3 {
            vortex_bail!("DateTime metadata too short");
        }
        let precision = data[0];
        let tz_len = u16::from_le_bytes([data[1], data[2]]) as usize;
        let timezone = if tz_len > 0 {
            if data.len() < 3 + tz_len {
                vortex_bail!("Truncated DateTime metadata");
            }
            Some(String::from_utf8_lossy(&data[3..3 + tz_len]).to_string())
        } else {
            None
        };
        Ok(DateTimeMetadata {
            precision,
            timezone,
        })
    }

    fn validate_dtype(&self, metadata: &Self::Metadata, storage_dtype: &DType) -> VortexResult<()> {
        if metadata.precision == 0 {
            match storage_dtype {
                DType::Primitive(PType::U32, _) => Ok(()),
                _ => vortex_bail!(
                    "DateTime (precision 0) requires Primitive(U32) storage, got {:?}",
                    storage_dtype
                ),
            }
        } else {
            match storage_dtype {
                DType::Primitive(PType::I64, _) => Ok(()),
                _ => vortex_bail!(
                    "DateTime64 requires Primitive(I64) storage, got {:?}",
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
    fn test_datetime_metadata_roundtrip() -> VortexResult<()> {
        let vtable = ClickHouseDateTime;
        let metadata = DateTimeMetadata {
            precision: 3,
            timezone: Some("UTC".to_string()),
        };
        let serialized = vtable.serialize(&metadata)?;
        let deserialized = vtable.deserialize(&serialized)?;
        assert_eq!(metadata, deserialized);
        Ok(())
    }

    #[test]
    fn test_datetime_no_timezone() -> VortexResult<()> {
        let vtable = ClickHouseDateTime;
        let metadata = DateTimeMetadata {
            precision: 0,
            timezone: None,
        };
        let serialized = vtable.serialize(&metadata)?;
        let deserialized = vtable.deserialize(&serialized)?;
        assert_eq!(metadata, deserialized);
        Ok(())
    }

    #[test]
    fn test_datetime_display() {
        assert_eq!(
            format!(
                "{}",
                DateTimeMetadata {
                    precision: 0,
                    timezone: None
                }
            ),
            "DateTime"
        );
        assert_eq!(
            format!(
                "{}",
                DateTimeMetadata {
                    precision: 3,
                    timezone: Some("UTC".into())
                }
            ),
            "DateTime64(3, 'UTC')"
        );
    }
}
