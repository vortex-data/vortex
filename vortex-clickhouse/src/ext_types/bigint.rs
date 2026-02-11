// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! BigInt Extension Type for ClickHouse Int128/UInt128/Int256/UInt256.
//!
//! ClickHouse supports large integer types that don't have direct equivalents
//! in Vortex's primitive types. This extension type stores them as fixed-size
//! byte arrays while preserving the semantic type information.
//!
//! # Storage Layout
//! - Int128/UInt128: 16 bytes per value (little-endian)
//! - Int256/UInt256: 32 bytes per value (little-endian)
//!
//! # Example
//! ```ignore
//! use vortex_clickhouse::ext_types::{BigInt, BigIntType, BigIntMetadata};
//! use vortex::dtype::{DType, ExtDType, Nullability};
//!
//! let dtype = BigInt::dtype(BigIntType::Int128, Nullability::Nullable);
//! ```

use std::fmt::{Debug, Display, Formatter};
use std::hash::Hash;

use vortex::dtype::extension::{ExtDType, ExtDTypeVTable, ExtID};
use vortex::dtype::{DType, Nullability, PType};
use vortex::error::{VortexResult, vortex_bail};

/// The extension type ID for ClickHouse BigInt types.
pub const BIGINT_EXT_ID: &str = "clickhouse.bigint";

/// The concrete BigInt types supported.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum BigIntType {
    /// Signed 128-bit integer (16 bytes)
    Int128 = 0,
    /// Unsigned 128-bit integer (16 bytes)
    UInt128 = 1,
    /// Signed 256-bit integer (32 bytes)
    Int256 = 2,
    /// Unsigned 256-bit integer (32 bytes)
    UInt256 = 3,
}

impl BigIntType {
    /// Returns the byte size for this BigInt type.
    pub const fn byte_size(&self) -> usize {
        match self {
            BigIntType::Int128 | BigIntType::UInt128 => 16,
            BigIntType::Int256 | BigIntType::UInt256 => 32,
        }
    }

    /// Returns true if this is a signed type.
    pub const fn is_signed(&self) -> bool {
        matches!(self, BigIntType::Int128 | BigIntType::Int256)
    }

    /// Returns the ClickHouse type name.
    pub const fn clickhouse_type_name(&self) -> &'static str {
        match self {
            BigIntType::Int128 => "Int128",
            BigIntType::UInt128 => "UInt128",
            BigIntType::Int256 => "Int256",
            BigIntType::UInt256 => "UInt256",
        }
    }

    /// Parse from ClickHouse type name.
    pub fn from_clickhouse_type(name: &str) -> Option<Self> {
        match name {
            "Int128" => Some(BigIntType::Int128),
            "UInt128" => Some(BigIntType::UInt128),
            "Int256" => Some(BigIntType::Int256),
            "UInt256" => Some(BigIntType::UInt256),
            _ => None,
        }
    }
}

impl TryFrom<u8> for BigIntType {
    type Error = vortex::error::VortexError;

    fn try_from(value: u8) -> Result<Self, Self::Error> {
        match value {
            0 => Ok(BigIntType::Int128),
            1 => Ok(BigIntType::UInt128),
            2 => Ok(BigIntType::Int256),
            3 => Ok(BigIntType::UInt256),
            _ => vortex_bail!("Invalid BigIntType tag: {}", value),
        }
    }
}

impl Display for BigIntType {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.clickhouse_type_name())
    }
}

/// Metadata for BigInt extension type.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct BigIntMetadata {
    /// The specific BigInt type.
    pub bigint_type: BigIntType,
}

impl BigIntMetadata {
    /// Create new BigInt metadata.
    pub fn new(bigint_type: BigIntType) -> Self {
        Self { bigint_type }
    }
}

impl Display for BigIntMetadata {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.bigint_type)
    }
}

/// The BigInt extension type VTable.
#[derive(Debug, Clone, Default, PartialEq, Eq, Hash)]
pub struct BigInt;

impl BigInt {
    /// Create a new BigInt extension dtype.
    pub fn new(bigint_type: BigIntType, nullability: Nullability) -> ExtDType<Self> {
        let metadata = BigIntMetadata::new(bigint_type);
        // Storage: Primitive u8 array (will be wrapped in FixedSizeList by the array)
        // For extension types, we use Primitive(U8) as the storage per element
        let storage_dtype = DType::Primitive(PType::U8, Nullability::NonNullable);
        ExtDType::try_with_vtable(Self, metadata, storage_dtype.with_nullability(nullability))
            .expect("BigInt storage dtype is always valid")
    }

    /// Create a BigInt DType (type-erased).
    pub fn dtype(bigint_type: BigIntType, nullability: Nullability) -> DType {
        DType::Extension(Self::new(bigint_type, nullability).erased())
    }

    /// Check if a DType is a BigInt extension type.
    pub fn is_bigint(dtype: &DType) -> bool {
        if let DType::Extension(ext) = dtype {
            ext.id().as_ref() == BIGINT_EXT_ID
        } else {
            false
        }
    }

    /// Try to extract BigIntType from a DType.
    pub fn try_get_type(dtype: &DType) -> Option<BigIntType> {
        if let DType::Extension(ext) = dtype {
            if ext.id().as_ref() == BIGINT_EXT_ID {
                // Use the Matcher trait to get metadata
                ext.metadata_opt::<BigInt>().map(|m| m.bigint_type)
            } else {
                None
            }
        } else {
            None
        }
    }
}

impl ExtDTypeVTable for BigInt {
    type Metadata = BigIntMetadata;

    fn id(&self) -> ExtID {
        ExtID::new_ref(BIGINT_EXT_ID)
    }

    fn serialize(&self, metadata: &Self::Metadata) -> VortexResult<Vec<u8>> {
        Ok(vec![metadata.bigint_type as u8])
    }

    fn deserialize(&self, data: &[u8]) -> VortexResult<Self::Metadata> {
        if data.is_empty() {
            vortex_bail!("BigInt metadata is empty");
        }
        let bigint_type = BigIntType::try_from(data[0])?;
        Ok(BigIntMetadata::new(bigint_type))
    }

    fn validate_dtype(
        &self,
        _metadata: &Self::Metadata,
        storage_dtype: &DType,
    ) -> VortexResult<()> {
        // Storage should be Primitive(U8)
        match storage_dtype {
            DType::Primitive(PType::U8, _) => Ok(()),
            _ => vortex_bail!(
                "BigInt extension requires Primitive(U8) storage, got {:?}",
                storage_dtype
            ),
        }
    }
}

/// Helper trait for ExtDTypeRef to access BigInt metadata.
pub trait BigIntExt {
    /// Get the BigIntType if this is a BigInt extension.
    fn bigint_type(&self) -> Option<BigIntType>;

    /// Get the ClickHouse type name if this is a BigInt extension.
    fn bigint_clickhouse_type(&self) -> Option<&'static str>;
}

impl BigIntExt for vortex::dtype::extension::ExtDTypeRef {
    fn bigint_type(&self) -> Option<BigIntType> {
        if self.id().as_ref() == BIGINT_EXT_ID {
            // Use the Matcher trait to get metadata
            self.metadata_opt::<BigInt>().map(|m| m.bigint_type)
        } else {
            None
        }
    }

    fn bigint_clickhouse_type(&self) -> Option<&'static str> {
        self.bigint_type().map(|t| t.clickhouse_type_name())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bigint_type_roundtrip() {
        for bigint_type in [
            BigIntType::Int128,
            BigIntType::UInt128,
            BigIntType::Int256,
            BigIntType::UInt256,
        ] {
            let tag = bigint_type as u8;
            let roundtrip = BigIntType::try_from(tag).unwrap();
            assert_eq!(bigint_type, roundtrip);
        }
    }

    #[test]
    fn test_bigint_byte_size() {
        assert_eq!(BigIntType::Int128.byte_size(), 16);
        assert_eq!(BigIntType::UInt128.byte_size(), 16);
        assert_eq!(BigIntType::Int256.byte_size(), 32);
        assert_eq!(BigIntType::UInt256.byte_size(), 32);
    }

    #[test]
    fn test_bigint_dtype_creation() {
        let dtype = BigInt::dtype(BigIntType::Int128, Nullability::Nullable);
        assert!(BigInt::is_bigint(&dtype));

        if let DType::Extension(ext) = &dtype {
            assert_eq!(ext.id().as_ref(), BIGINT_EXT_ID);
        } else {
            panic!("Expected Extension dtype");
        }
    }

    #[test]
    fn test_clickhouse_type_names() {
        assert_eq!(BigIntType::Int128.clickhouse_type_name(), "Int128");
        assert_eq!(BigIntType::UInt128.clickhouse_type_name(), "UInt128");
        assert_eq!(BigIntType::Int256.clickhouse_type_name(), "Int256");
        assert_eq!(BigIntType::UInt256.clickhouse_type_name(), "UInt256");
    }

    #[test]
    fn test_from_clickhouse_type() {
        assert_eq!(
            BigIntType::from_clickhouse_type("Int128"),
            Some(BigIntType::Int128)
        );
        assert_eq!(
            BigIntType::from_clickhouse_type("UInt256"),
            Some(BigIntType::UInt256)
        );
        assert_eq!(BigIntType::from_clickhouse_type("String"), None);
    }

    #[test]
    fn test_bigint_file_roundtrip() {
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

        // Create test data - 2 Int128 values as FixedSizeList<u8, 16>
        let bytes: Vec<u8> = vec![
            // First Int128 value: 100 (little-endian)
            100, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            // Second Int128 value: 200 (little-endian)
            200, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        ];

        // Create FixedSizeList array for Int128 (16 bytes per element)
        let values = PrimitiveArray::new(Buffer::<u8>::from(bytes.clone()), Validity::NonNullable);
        let fsl_array = FixedSizeListArray::try_new(
            values.into_array(),
            16, // element size in bytes
            Validity::NonNullable,
            2, // number of elements
        )
        .expect("Failed to create FixedSizeList");

        // Wrap in struct
        let field_names: Vec<Arc<str>> = vec![Arc::from("val")];
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
            // Check that it's FixedSizeList<u8, 16> (represents Int128)
            match &val_dtype {
                DType::FixedSizeList(elem, size, _) => {
                    assert_eq!(*size, 16, "Expected size 16 for Int128");
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
