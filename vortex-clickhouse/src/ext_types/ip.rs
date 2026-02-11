// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! IP Address Extension Types for ClickHouse IPv4/IPv6.
//!
//! ClickHouse supports IPv4 and IPv6 address types which are stored as:
//! - IPv4: UInt32 (4 bytes, network byte order internally but stored as native UInt32)
//! - IPv6: UInt128 (16 bytes, stored as fixed-size binary)
//!
//! # Storage Layout
//! - IPv4: Stored as `Primitive(U32)` - each value is 4 bytes
//! - IPv6: Stored as `FixedSizeList<u8, 16>` - each value is 16 bytes
//!
//! This is consistent with how Parquet and Arrow handle these types.
//!
//! # Example
//! ```ignore
//! use vortex_clickhouse::ext_types::{IPAddress, IPAddressType};
//! use vortex::dtype::Nullability;
//!
//! // Create IPv4 dtype
//! let ipv4_dtype = IPAddress::dtype(IPAddressType::IPv4, Nullability::Nullable);
//!
//! // Create IPv6 dtype  
//! let ipv6_dtype = IPAddress::dtype(IPAddressType::IPv6, Nullability::Nullable);
//! ```

use std::fmt::{Debug, Display, Formatter};
use std::hash::Hash;
use std::sync::Arc;

use vortex::dtype::extension::{ExtDType, ExtDTypeVTable, ExtID};
use vortex::dtype::{DType, Nullability, PType};
use vortex::error::{VortexResult, vortex_bail};

/// The extension type ID for ClickHouse IP address types.
pub const IP_EXT_ID: &str = "clickhouse.ip";

/// The concrete IP address types supported.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum IPAddressType {
    /// IPv4 address (4 bytes, stored as UInt32)
    IPv4 = 0,
    /// IPv6 address (16 bytes, stored as 16-byte binary)
    IPv6 = 1,
}

impl IPAddressType {
    /// Returns the byte size for this IP address type.
    pub const fn byte_size(&self) -> usize {
        match self {
            IPAddressType::IPv4 => 4,
            IPAddressType::IPv6 => 16,
        }
    }

    /// Returns the ClickHouse type name.
    pub const fn clickhouse_type_name(&self) -> &'static str {
        match self {
            IPAddressType::IPv4 => "IPv4",
            IPAddressType::IPv6 => "IPv6",
        }
    }

    /// Parse from ClickHouse type name.
    pub fn from_clickhouse_type(name: &str) -> Option<Self> {
        match name {
            "IPv4" => Some(IPAddressType::IPv4),
            "IPv6" => Some(IPAddressType::IPv6),
            _ => None,
        }
    }
}

impl TryFrom<u8> for IPAddressType {
    type Error = vortex::error::VortexError;

    fn try_from(value: u8) -> Result<Self, Self::Error> {
        match value {
            0 => Ok(IPAddressType::IPv4),
            1 => Ok(IPAddressType::IPv6),
            _ => vortex_bail!("Invalid IPAddressType tag: {}", value),
        }
    }
}

impl Display for IPAddressType {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.clickhouse_type_name())
    }
}

/// Metadata for IP address extension type.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct IPAddressMetadata {
    /// The specific IP address type.
    pub ip_type: IPAddressType,
}

impl IPAddressMetadata {
    /// Create new IP address metadata.
    pub fn new(ip_type: IPAddressType) -> Self {
        Self { ip_type }
    }
}

impl Display for IPAddressMetadata {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.ip_type)
    }
}

/// The IP address extension type VTable.
#[derive(Debug, Clone, Default, PartialEq, Eq, Hash)]
pub struct IPAddress;

impl IPAddress {
    /// Create a new IP address extension dtype.
    pub fn new(ip_type: IPAddressType, nullability: Nullability) -> ExtDType<Self> {
        let metadata = IPAddressMetadata::new(ip_type);
        let storage_dtype = match ip_type {
            // IPv4: stored as UInt32 (4 bytes)
            IPAddressType::IPv4 => DType::Primitive(PType::U32, Nullability::NonNullable),
            // IPv6: stored as 16 bytes (we use U8 as storage element, will be wrapped)
            IPAddressType::IPv6 => DType::Primitive(PType::U8, Nullability::NonNullable),
        };
        ExtDType::try_with_vtable(Self, metadata, storage_dtype.with_nullability(nullability))
            .expect("IPAddress storage dtype is always valid")
    }

    /// Create an IP address DType (type-erased).
    pub fn dtype(ip_type: IPAddressType, nullability: Nullability) -> DType {
        DType::Extension(Self::new(ip_type, nullability).erased())
    }

    /// Check if a DType is an IP address extension type.
    pub fn is_ip_address(dtype: &DType) -> bool {
        if let DType::Extension(ext) = dtype {
            ext.id().as_ref() == IP_EXT_ID
        } else {
            false
        }
    }

    /// Try to extract IPAddressType from a DType.
    pub fn try_get_type(dtype: &DType) -> Option<IPAddressType> {
        if let DType::Extension(ext) = dtype {
            if ext.id().as_ref() == IP_EXT_ID {
                ext.metadata_opt::<IPAddress>().map(|m| m.ip_type)
            } else {
                None
            }
        } else {
            None
        }
    }

    /// Create DType for storing IPv4 values (as Primitive U32).
    /// This is used for actual storage in Vortex files.
    pub fn ipv4_storage_dtype(nullability: Nullability) -> DType {
        DType::Primitive(PType::U32, nullability)
    }

    /// Create DType for storing IPv6 values (as FixedSizeList<u8, 16>).
    /// This is used for actual storage in Vortex files.
    pub fn ipv6_storage_dtype(nullability: Nullability) -> DType {
        DType::FixedSizeList(
            Arc::new(DType::Primitive(PType::U8, Nullability::NonNullable)),
            16,
            nullability,
        )
    }
}

impl ExtDTypeVTable for IPAddress {
    type Metadata = IPAddressMetadata;

    fn id(&self) -> ExtID {
        ExtID::new_ref(IP_EXT_ID)
    }

    fn serialize(&self, metadata: &Self::Metadata) -> VortexResult<Vec<u8>> {
        Ok(vec![metadata.ip_type as u8])
    }

    fn deserialize(&self, data: &[u8]) -> VortexResult<Self::Metadata> {
        if data.is_empty() {
            vortex_bail!("IPAddress metadata is empty");
        }
        let ip_type = IPAddressType::try_from(data[0])?;
        Ok(IPAddressMetadata::new(ip_type))
    }

    fn validate_dtype(&self, metadata: &Self::Metadata, storage_dtype: &DType) -> VortexResult<()> {
        match metadata.ip_type {
            IPAddressType::IPv4 => {
                // IPv4 storage should be Primitive(U32)
                match storage_dtype {
                    DType::Primitive(PType::U32, _) => Ok(()),
                    _ => vortex_bail!(
                        "IPv4 extension requires Primitive(U32) storage, got {:?}",
                        storage_dtype
                    ),
                }
            }
            IPAddressType::IPv6 => {
                // IPv6 storage should be Primitive(U8) - the actual bytes
                match storage_dtype {
                    DType::Primitive(PType::U8, _) => Ok(()),
                    _ => vortex_bail!(
                        "IPv6 extension requires Primitive(U8) storage, got {:?}",
                        storage_dtype
                    ),
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ip_type_byte_size() {
        assert_eq!(IPAddressType::IPv4.byte_size(), 4);
        assert_eq!(IPAddressType::IPv6.byte_size(), 16);
    }

    #[test]
    fn test_ip_type_names() {
        assert_eq!(IPAddressType::IPv4.clickhouse_type_name(), "IPv4");
        assert_eq!(IPAddressType::IPv6.clickhouse_type_name(), "IPv6");
    }

    #[test]
    fn test_from_clickhouse_type() {
        assert_eq!(
            IPAddressType::from_clickhouse_type("IPv4"),
            Some(IPAddressType::IPv4)
        );
        assert_eq!(
            IPAddressType::from_clickhouse_type("IPv6"),
            Some(IPAddressType::IPv6)
        );
        assert_eq!(IPAddressType::from_clickhouse_type("String"), None);
    }

    #[test]
    fn test_ip_type_roundtrip() {
        for ip_type in [IPAddressType::IPv4, IPAddressType::IPv6] {
            let tag = ip_type as u8;
            let recovered = IPAddressType::try_from(tag).expect("roundtrip");
            assert_eq!(recovered, ip_type);
        }
    }

    #[test]
    fn test_ip_dtype_creation() {
        // Test IPv4 dtype
        let ipv4_dtype = IPAddress::dtype(IPAddressType::IPv4, Nullability::Nullable);
        assert!(IPAddress::is_ip_address(&ipv4_dtype));
        assert_eq!(
            IPAddress::try_get_type(&ipv4_dtype),
            Some(IPAddressType::IPv4)
        );

        // Test IPv6 dtype
        let ipv6_dtype = IPAddress::dtype(IPAddressType::IPv6, Nullability::NonNullable);
        assert!(IPAddress::is_ip_address(&ipv6_dtype));
        assert_eq!(
            IPAddress::try_get_type(&ipv6_dtype),
            Some(IPAddressType::IPv6)
        );
    }

    #[test]
    fn test_storage_dtypes() {
        // IPv4 storage dtype
        let ipv4_storage = IPAddress::ipv4_storage_dtype(Nullability::Nullable);
        assert!(matches!(ipv4_storage, DType::Primitive(PType::U32, _)));

        // IPv6 storage dtype
        let ipv6_storage = IPAddress::ipv6_storage_dtype(Nullability::Nullable);
        if let DType::FixedSizeList(elem, size, _) = ipv6_storage {
            assert_eq!(size, 16);
            assert!(matches!(elem.as_ref(), DType::Primitive(PType::U8, _)));
        } else {
            panic!("Expected FixedSizeList");
        }
    }

    #[test]
    fn test_ip_file_roundtrip() {
        use std::io::Write;
        use std::sync::Arc;
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

        // Create test data - 2 IPv4 values stored as U32
        let ipv4_values: Vec<u32> = vec![
            0x7F000001, // 127.0.0.1
            0xC0A80001, // 192.168.0.1
        ];
        let ipv4_array =
            PrimitiveArray::new(Buffer::<u32>::from(ipv4_values), Validity::NonNullable);

        // Create test data - 2 IPv6 values as FixedSizeList<u8, 16>
        let ipv6_bytes: Vec<u8> = vec![
            // ::1 (loopback)
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, // fe80::1 (link-local)
            0xfe, 0x80, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1,
        ];
        let ipv6_values =
            PrimitiveArray::new(Buffer::<u8>::from(ipv6_bytes), Validity::NonNullable);
        let ipv6_array =
            FixedSizeListArray::try_new(ipv6_values.into_array(), 16, Validity::NonNullable, 2)
                .expect("Failed to create IPv6 array");

        // Create struct with both fields
        let field_names: Vec<Arc<str>> = vec![Arc::from("ipv4"), Arc::from("ipv6")];
        let struct_array = StructArray::try_new(
            FieldNames::from(field_names),
            vec![ipv4_array.into_array(), ipv6_array.into_array()],
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

        // Verify the dtype structure
        if let DType::Struct(fields, _) = &read_dtype {
            let field_iter = fields.fields();
            let dtypes: Vec<_> = field_iter.collect();
            assert_eq!(dtypes.len(), 2);

            // IPv4 field should be Primitive(U32)
            assert!(
                matches!(&dtypes[0], DType::Primitive(PType::U32, _)),
                "Expected Primitive(U32) for IPv4, got {:?}",
                dtypes[0]
            );

            // IPv6 field should be FixedSizeList<u8, 16>
            match &dtypes[1] {
                DType::FixedSizeList(elem, size, _) => {
                    assert_eq!(*size, 16, "Expected size 16 for IPv6");
                    assert!(
                        matches!(elem.as_ref(), DType::Primitive(PType::U8, _)),
                        "Expected Primitive(U8), got {:?}",
                        elem
                    );
                }
                _ => panic!("Expected FixedSizeList for IPv6, got {:?}", dtypes[1]),
            }
        } else {
            panic!("Expected Struct dtype, got {:?}", read_dtype);
        }
    }
}
