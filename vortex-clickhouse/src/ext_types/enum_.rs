// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! ClickHouse Enum Extension Type.
//!
//! Preserves `Enum8`/`Enum16` name→value mappings in Vortex files so they can be
//! reconstructed on read when `output_clickhouse_types` is enabled.
//!
//! # Storage
//! - `Enum8`: `Primitive(I8)`
//! - `Enum16`: `Primitive(I16)`

use std::fmt::{Display, Formatter};
use std::hash::Hash;

use vortex::dtype::extension::{ExtDType, ExtDTypeVTable, ExtID};
use vortex::dtype::{DType, Nullability, PType};
use vortex::error::{VortexResult, vortex_bail};

/// Extension type ID.
pub const ENUM_EXT_ID: &str = "clickhouse.enum";

/// Whether this is an 8-bit or 16-bit enum.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum EnumSize {
    Enum8 = 8,
    Enum16 = 16,
}

/// A single enum entry: `(name, value)`.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct EnumEntry {
    pub name: String,
    pub value: i16,
}

/// Metadata for the enum extension type.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct EnumMetadata {
    pub enum_size: EnumSize,
    pub entries: Vec<EnumEntry>,
}

impl Display for EnumMetadata {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "Enum{}(", self.enum_size as u8)?;
        for (i, entry) in self.entries.iter().enumerate() {
            if i > 0 {
                write!(f, ", ")?;
            }
            write!(f, "'{}' = {}", entry.name, entry.value)?;
        }
        write!(f, ")")
    }
}

/// The Enum extension type VTable.
#[derive(Debug, Clone, Default, PartialEq, Eq, Hash)]
pub struct ClickHouseEnum;

impl ClickHouseEnum {
    /// Create a ClickHouse Enum DType (type-erased).
    pub fn dtype(metadata: EnumMetadata, nullability: Nullability) -> DType {
        let storage = match metadata.enum_size {
            EnumSize::Enum8 => DType::Primitive(PType::I8, Nullability::NonNullable),
            EnumSize::Enum16 => DType::Primitive(PType::I16, Nullability::NonNullable),
        };
        let ext = ExtDType::try_with_vtable(Self, metadata, storage.with_nullability(nullability))
            .expect("Enum storage dtype is always valid");
        DType::Extension(ext.erased())
    }

    /// Try to extract `EnumMetadata` from a DType.
    pub fn try_get_metadata(dtype: &DType) -> Option<EnumMetadata> {
        if let DType::Extension(ext) = dtype {
            if ext.id().as_ref() == ENUM_EXT_ID {
                return ext.metadata_opt::<ClickHouseEnum>().cloned();
            }
        }
        None
    }

    /// Reconstruct the ClickHouse type string, e.g. `Enum8('a' = 1, 'b' = 2)`.
    pub fn to_clickhouse_type(metadata: &EnumMetadata) -> String {
        let mut result = match metadata.enum_size {
            EnumSize::Enum8 => "Enum8(".to_string(),
            EnumSize::Enum16 => "Enum16(".to_string(),
        };
        for (i, entry) in metadata.entries.iter().enumerate() {
            if i > 0 {
                result.push_str(", ");
            }
            result.push_str(&format!("'{}' = {}", entry.name, entry.value));
        }
        result.push(')');
        result
    }
}

impl ExtDTypeVTable for ClickHouseEnum {
    type Metadata = EnumMetadata;

    fn id(&self) -> ExtID {
        ExtID::new_ref(ENUM_EXT_ID)
    }

    fn serialize(&self, metadata: &Self::Metadata) -> VortexResult<Vec<u8>> {
        // Simple binary format:
        // [enum_size: u8] [num_entries: u16] [entry...]
        // entry = [name_len: u16] [name_bytes...] [value: i16]
        let mut buf = Vec::new();
        buf.push(metadata.enum_size as u8);
        let n = metadata.entries.len() as u16;
        buf.extend_from_slice(&n.to_le_bytes());
        for entry in &metadata.entries {
            let name_bytes = entry.name.as_bytes();
            let name_len = name_bytes.len() as u16;
            buf.extend_from_slice(&name_len.to_le_bytes());
            buf.extend_from_slice(name_bytes);
            buf.extend_from_slice(&entry.value.to_le_bytes());
        }
        Ok(buf)
    }

    fn deserialize(&self, data: &[u8]) -> VortexResult<Self::Metadata> {
        if data.len() < 3 {
            vortex_bail!("Enum metadata too short");
        }
        let enum_size = match data[0] {
            8 => EnumSize::Enum8,
            16 => EnumSize::Enum16,
            other => vortex_bail!("Invalid enum size: {}", other),
        };
        let n = u16::from_le_bytes([data[1], data[2]]) as usize;
        let mut offset = 3;
        let mut entries = Vec::with_capacity(n);
        for _ in 0..n {
            if offset + 2 > data.len() {
                vortex_bail!("Truncated enum metadata");
            }
            let name_len = u16::from_le_bytes([data[offset], data[offset + 1]]) as usize;
            offset += 2;
            if offset + name_len + 2 > data.len() {
                vortex_bail!("Truncated enum metadata");
            }
            let name = String::from_utf8_lossy(&data[offset..offset + name_len]).to_string();
            offset += name_len;
            let value = i16::from_le_bytes([data[offset], data[offset + 1]]);
            offset += 2;
            entries.push(EnumEntry { name, value });
        }
        Ok(EnumMetadata { enum_size, entries })
    }

    fn validate_dtype(&self, metadata: &Self::Metadata, storage_dtype: &DType) -> VortexResult<()> {
        match (metadata.enum_size, storage_dtype) {
            (EnumSize::Enum8, DType::Primitive(PType::I8, _)) => Ok(()),
            (EnumSize::Enum16, DType::Primitive(PType::I16, _)) => Ok(()),
            _ => vortex_bail!(
                "Enum{} requires Primitive(I{}) storage, got {:?}",
                metadata.enum_size as u8,
                metadata.enum_size as u8,
                storage_dtype
            ),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_enum_metadata_roundtrip() -> VortexResult<()> {
        let vtable = ClickHouseEnum;
        let metadata = EnumMetadata {
            enum_size: EnumSize::Enum8,
            entries: vec![
                EnumEntry {
                    name: "a".into(),
                    value: 1,
                },
                EnumEntry {
                    name: "b".into(),
                    value: 2,
                },
            ],
        };
        let serialized = vtable.serialize(&metadata)?;
        let deserialized = vtable.deserialize(&serialized)?;
        assert_eq!(metadata, deserialized);
        Ok(())
    }

    #[test]
    fn test_enum_clickhouse_type_string() {
        let metadata = EnumMetadata {
            enum_size: EnumSize::Enum8,
            entries: vec![
                EnumEntry {
                    name: "hello".into(),
                    value: 1,
                },
                EnumEntry {
                    name: "world".into(),
                    value: 2,
                },
            ],
        };
        assert_eq!(
            ClickHouseEnum::to_clickhouse_type(&metadata),
            "Enum8('hello' = 1, 'world' = 2)"
        );
    }
}
