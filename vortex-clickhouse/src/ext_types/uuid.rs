// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! UUID Extension Type for ClickHouse UUID.
//!
//! ClickHouse UUID is stored as UInt128 (16 bytes), which is internally represented
//! as two uint64_t values in little-endian format.
//!
//! # Storage Layout
//! - UUID: Stored as `FixedSizeList<u8, 16>` - each value is 16 bytes
//!
//! This is consistent with how Parquet handles UUID (FIXED_LEN_BYTE_ARRAY(16))
//! and Arrow (FixedSizeBinary(16)).
//!
//! # Byte Order Note
//! ClickHouse stores UUIDs in little-endian format internally. When reading/writing,
//! the raw bytes are preserved without any byte-order conversion.
//!
//! # Example
//! ```ignore
//! use vortex_clickhouse::ext_types::UUID;
//! use vortex::dtype::Nullability;
//!
//! // Create UUID dtype
//! let uuid_dtype = UUID::dtype(Nullability::Nullable);
//! ```

use std::fmt::{Debug, Display, Formatter};
use std::hash::Hash;
use std::sync::Arc;

use vortex::dtype::extension::{ExtDType, ExtDTypeVTable, ExtID};
use vortex::dtype::{DType, Nullability, PType};
use vortex::error::{VortexResult, vortex_bail};

/// The extension type ID for ClickHouse UUID type.
pub const UUID_EXT_ID: &str = "clickhouse.uuid";

/// Byte size of UUID (128 bits = 16 bytes).
pub const UUID_BYTE_SIZE: u32 = 16;

/// Metadata for UUID extension type.
/// Currently empty as UUID has no configurable parameters.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub struct UUIDMetadata;

impl Display for UUIDMetadata {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "UUID")
    }
}

/// The UUID extension type VTable.
#[derive(Debug, Clone, Default, PartialEq, Eq, Hash)]
pub struct UUID;

impl UUID {
    /// Create a new UUID extension dtype.
    pub fn new(nullability: Nullability) -> ExtDType<Self> {
        // Storage dtype is Primitive(U8) as the underlying bytes
        let storage_dtype =
            DType::Primitive(PType::U8, Nullability::NonNullable).with_nullability(nullability);
        ExtDType::try_with_vtable(Self, UUIDMetadata, storage_dtype)
            .expect("UUID storage dtype is always valid")
    }

    /// Create the DType for UUID (as Extension type).
    pub fn dtype(nullability: Nullability) -> DType {
        DType::Extension(Self::new(nullability).erased())
    }

    /// Get the storage dtype for UUID.
    /// UUID is stored as FixedSizeList<u8, 16> (16 bytes).
    pub fn storage_dtype(nullability: Nullability) -> DType {
        DType::FixedSizeList(
            Arc::new(DType::Primitive(PType::U8, Nullability::NonNullable)),
            UUID_BYTE_SIZE,
            nullability,
        )
    }

    /// Check if a dtype represents a UUID type.
    /// This checks for either:
    /// 1. Extension type with UUID ID
    /// 2. FixedSizeList<u8, 16> (the storage format)
    pub fn is_uuid(dtype: &DType) -> bool {
        match dtype {
            DType::Extension(ext) => ext.id().as_ref() == UUID_EXT_ID,
            DType::FixedSizeList(elem, size, _) => {
                *size == UUID_BYTE_SIZE && matches!(elem.as_ref(), DType::Primitive(PType::U8, _))
            }
            _ => false,
        }
    }

    /// Returns the ClickHouse type name.
    pub const fn clickhouse_type_name() -> &'static str {
        "UUID"
    }

    /// Returns the byte size of UUID.
    pub const fn byte_size() -> u32 {
        UUID_BYTE_SIZE
    }
}

impl ExtDTypeVTable for UUID {
    type Metadata = UUIDMetadata;

    fn id(&self) -> ExtID {
        ExtID::new_ref(UUID_EXT_ID)
    }

    fn serialize(&self, _metadata: &Self::Metadata) -> VortexResult<Vec<u8>> {
        // No metadata to serialize for UUID
        Ok(vec![])
    }

    fn deserialize(&self, _data: &[u8]) -> VortexResult<Self::Metadata> {
        // No metadata to deserialize
        Ok(UUIDMetadata)
    }

    fn validate_dtype(
        &self,
        _metadata: &Self::Metadata,
        storage_dtype: &DType,
    ) -> VortexResult<()> {
        // UUID storage should be FixedSizeList<u8, 16> or Primitive(U8) for the underlying bytes
        match storage_dtype {
            DType::FixedSizeList(elem, size, _) => {
                if *size == UUID_BYTE_SIZE
                    && matches!(elem.as_ref(), DType::Primitive(PType::U8, _))
                {
                    Ok(())
                } else {
                    vortex_bail!(
                        "UUID extension requires FixedSizeList<u8, 16> storage, got FixedSizeList with size {} and elem {:?}",
                        size,
                        elem
                    )
                }
            }
            DType::Primitive(PType::U8, _) => Ok(()), // Raw bytes storage
            _ => vortex_bail!(
                "UUID extension requires FixedSizeList<u8, 16> or Primitive(U8) storage, got {:?}",
                storage_dtype
            ),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_uuid_byte_size() {
        assert_eq!(UUID::byte_size(), 16);
    }

    #[test]
    fn test_uuid_type_name() {
        assert_eq!(UUID::clickhouse_type_name(), "UUID");
    }

    #[test]
    fn test_uuid_dtype_creation() {
        let dtype = UUID::dtype(Nullability::Nullable);
        assert!(UUID::is_uuid(&dtype));

        let dtype = UUID::dtype(Nullability::NonNullable);
        assert!(UUID::is_uuid(&dtype));
    }

    #[test]
    fn test_uuid_storage_dtype() {
        let storage = UUID::storage_dtype(Nullability::Nullable);
        match storage {
            DType::FixedSizeList(elem, size, nullability) => {
                assert_eq!(size, 16);
                assert_eq!(nullability, Nullability::Nullable);
                assert!(matches!(elem.as_ref(), DType::Primitive(PType::U8, _)));
            }
            _ => panic!("Expected FixedSizeList, got {:?}", storage),
        }
    }

    #[test]
    fn test_uuid_is_uuid_detection() {
        // Test Extension type detection
        let ext_dtype = UUID::dtype(Nullability::Nullable);
        assert!(UUID::is_uuid(&ext_dtype));

        // Test FixedSizeList<u8, 16> detection (storage format)
        let fsl_dtype = DType::FixedSizeList(
            Arc::new(DType::Primitive(PType::U8, Nullability::NonNullable)),
            16,
            Nullability::Nullable,
        );
        assert!(UUID::is_uuid(&fsl_dtype));

        // Test non-UUID types
        let non_uuid = DType::Primitive(PType::U32, Nullability::Nullable);
        assert!(!UUID::is_uuid(&non_uuid));

        // Test wrong size FixedSizeList
        let wrong_size = DType::FixedSizeList(
            Arc::new(DType::Primitive(PType::U8, Nullability::NonNullable)),
            32, // Wrong size
            Nullability::Nullable,
        );
        assert!(!UUID::is_uuid(&wrong_size));
    }

    #[test]
    fn test_uuid_metadata() {
        let metadata = UUIDMetadata;
        assert_eq!(format!("{}", metadata), "UUID");
    }

    #[test]
    fn test_uuid_ext_vtable() {
        use vortex::dtype::extension::ExtDTypeVTable;

        let uuid = UUID;
        assert_eq!(uuid.id().as_ref(), UUID_EXT_ID);

        // Test serialize/deserialize metadata
        let serialized = uuid.serialize(&UUIDMetadata).unwrap();
        assert!(serialized.is_empty()); // No metadata to serialize

        let deserialized = uuid.deserialize(&[]).unwrap();
        assert_eq!(deserialized, UUIDMetadata);

        // Test validate_dtype
        let valid_dtype = DType::FixedSizeList(
            Arc::new(DType::Primitive(PType::U8, Nullability::NonNullable)),
            16,
            Nullability::Nullable,
        );
        assert!(uuid.validate_dtype(&UUIDMetadata, &valid_dtype).is_ok());

        // Invalid dtype should fail
        let invalid_dtype = DType::Primitive(PType::U32, Nullability::Nullable);
        assert!(uuid.validate_dtype(&UUIDMetadata, &invalid_dtype).is_err());
    }

    #[test]
    fn test_uuid_file_roundtrip() {
        use std::io::Write;
        use tempfile::NamedTempFile;
        use vortex::array::IntoArray;
        use vortex::array::arrays::{FixedSizeListArray, PrimitiveArray, StructArray};
        use vortex::array::stream::ArrayStreamExt;
        use vortex::array::validity::Validity;
        use vortex::buffer::Buffer;
        use vortex::dtype::FieldNames;
        use vortex::file::{OpenOptionsSessionExt, WriteOptionsSessionExt};
        use vortex::io::runtime::BlockingRuntime;

        use crate::{RUNTIME, SESSION};

        // Create test data - 2 UUIDs as FixedSizeList<u8, 16>
        // UUID format: xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx (16 bytes)
        let bytes: Vec<u8> = vec![
            // First UUID: 550e8400-e29b-41d4-a716-446655440000
            0x55, 0x0e, 0x84, 0x00, 0xe2, 0x9b, 0x41, 0xd4, 0xa7, 0x16, 0x44, 0x66, 0x55, 0x44,
            0x00, 0x00, // Second UUID: 6ba7b810-9dad-11d1-80b4-00c04fd430c8
            0x6b, 0xa7, 0xb8, 0x10, 0x9d, 0xad, 0x11, 0xd1, 0x80, 0xb4, 0x00, 0xc0, 0x4f, 0xd4,
            0x30, 0xc8,
        ];

        // Create FixedSizeList array for UUID (16 bytes per element)
        let values = PrimitiveArray::new(Buffer::<u8>::from(bytes.clone()), Validity::NonNullable);
        let fsl_array = FixedSizeListArray::try_new(
            values.into_array(),
            16, // element size in bytes
            Validity::NonNullable,
            2, // number of elements
        )
        .expect("Failed to create FixedSizeList");

        // Wrap in struct
        let field_names: Vec<Arc<str>> = vec![Arc::from("uuid_col")];
        let struct_array = StructArray::try_new(
            FieldNames::from(field_names),
            vec![fsl_array.into_array()],
            2,
            Validity::NonNullable,
        )
        .expect("Failed to create struct");

        // Write to temp file
        let temp_file = NamedTempFile::new().expect("Failed to create temp file");
        let path = temp_file.path().to_str().unwrap().to_string();

        let mut buf = Vec::new();
        (*RUNTIME).block_on(async {
            SESSION
                .write_options()
                .write(&mut buf, struct_array.to_array_stream())
                .await
                .expect("Failed to write");
        });

        // Write buffer to temp file
        let mut file = std::fs::File::create(&path).expect("Failed to create file");
        file.write_all(&buf).expect("Failed to write to file");
        file.flush().expect("Failed to flush");
        drop(file);

        // Read back and verify dtype
        let read_dtype = (*RUNTIME).block_on(async {
            let vortex_file = SESSION
                .open_options()
                .open_path(&path)
                .await
                .expect("Failed to open");
            vortex_file.dtype().clone()
        });

        // Verify the dtype is a Struct with a FixedSizeList<u8, 16> field
        if let DType::Struct(fields, _) = &read_dtype {
            let val_dtype = fields.fields().next().expect("Expected field");
            // Check that it's FixedSizeList<u8, 16> (represents UUID)
            match &val_dtype {
                DType::FixedSizeList(elem, size, _) => {
                    assert_eq!(*size, 16, "Expected size 16 for UUID");
                    assert!(
                        matches!(elem.as_ref(), DType::Primitive(PType::U8, _)),
                        "Expected Primitive(U8), got {:?}",
                        elem
                    );
                }
                _ => panic!("Expected FixedSizeList dtype, got {:?}", val_dtype),
            }
        } else {
            panic!("Expected Struct dtype, got {:?}", read_dtype);
        }
    }
}
