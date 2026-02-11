// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Big integer array exporter for ClickHouse.
//!
//! This module exports Vortex FixedSizeList arrays (used to store big integers like
//! Int128, UInt128, Int256, UInt256) to ClickHouse column buffers.
//!
//! Big integers are stored as FixedSizeList<u8, N> where N is 16 (for 128-bit) or 32 (for 256-bit).

use std::any::Any;
use std::ffi::c_void;

use vortex::array::arrays::PrimitiveArray;
use vortex::array::{Array, ArrayRef, ToCanonical};
use vortex::dtype::{DType, Nullability, PType};
use vortex::error::{VortexResult, vortex_bail};
use vortex::mask::Mask;

use super::{ColumnExporter, ExporterKind};

/// The size of each big integer in bytes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BigIntSize {
    /// 128-bit integer (16 bytes)
    Bits128 = 16,
    /// 256-bit integer (32 bytes)
    Bits256 = 32,
}

/// Exporter for big integer arrays (Int128, UInt128, Int256, UInt256).
///
/// These are stored as FixedSizeList<u8, N> in Vortex, where N is 16 or 32.
pub struct BigIntExporter {
    /// Cached flat byte slice from canonical form
    bytes: PrimitiveArray,
    /// Cached validity mask (None = non-nullable)
    validity: Option<Mask>,
    /// Current export position
    position: usize,
    /// Position at the start of last export (for validity export)
    last_export_start: usize,
    /// Number of rows exported in last export call
    last_export_count: usize,
    /// Total length of the array (number of big integers)
    len: usize,
    /// Size of each big integer
    bigint_size: BigIntSize,
    /// Whether the array is nullable
    nullable: bool,
}

impl BigIntExporter {
    /// Create a new big integer exporter for the given array.
    pub fn new(array: ArrayRef) -> VortexResult<Self> {
        let len = array.len();
        let (bigint_size, nullable) = match array.dtype() {
            DType::FixedSizeList(elem_dtype, size, nullability) => {
                // Verify the element type is u8
                if !matches!(elem_dtype.as_ref(), DType::Primitive(PType::U8, _)) {
                    vortex_bail!(
                        "BigIntExporter requires FixedSizeList<u8, N>, got FixedSizeList<{:?}, {}>",
                        elem_dtype,
                        size
                    );
                }
                let bigint_size = match *size {
                    16 => BigIntSize::Bits128,
                    32 => BigIntSize::Bits256,
                    _ => vortex_bail!(
                        "BigIntExporter requires FixedSizeList with size 16 or 32, got {}",
                        size
                    ),
                };
                (bigint_size, *nullability == Nullability::Nullable)
            }
            _ => vortex_bail!(
                "BigIntExporter requires a FixedSizeList array, got {:?}",
                array.dtype()
            ),
        };

        let canonical = array.to_fixed_size_list();
        let validity = if nullable {
            Some(canonical.validity_mask()?)
        } else {
            None
        };
        let elements = canonical.elements();
        let bytes = elements.to_primitive();

        Ok(Self {
            bytes,
            validity,
            position: 0,
            last_export_start: 0,
            last_export_count: 0,
            len,
            bigint_size,
            nullable,
        })
    }

    /// Get the size of each big integer in bytes.
    pub fn bigint_size(&self) -> BigIntSize {
        self.bigint_size
    }
}

impl ColumnExporter for BigIntExporter {
    fn kind(&self) -> ExporterKind {
        ExporterKind::BigInt
    }

    fn export(
        &mut self,
        column_ptr: *mut c_void,
        buffer_size_bytes: usize,
        max_rows: usize,
    ) -> VortexResult<usize> {
        if column_ptr.is_null() {
            vortex_bail!("column_ptr is null");
        }

        let remaining = self.len - self.position;
        let rows_to_export = remaining.min(max_rows);

        if rows_to_export == 0 {
            return Ok(0);
        }

        let byte_size = self.bigint_size as usize;
        let required_bytes = rows_to_export * byte_size;
        if buffer_size_bytes < required_bytes {
            vortex_bail!(
                "buffer too small: need {} bytes for {} BigInt({}) rows, got {}",
                required_bytes,
                rows_to_export,
                byte_size,
                buffer_size_bytes
            );
        }

        // Record the start position for validity export
        self.last_export_start = self.position;
        self.last_export_count = rows_to_export;

        let bytes = self.bytes.as_slice::<u8>();

        // Calculate byte range to export
        let start_byte = self.position * byte_size;
        let end_byte = (self.position + rows_to_export) * byte_size;
        let slice = &bytes[start_byte..end_byte];

        // Copy to destination
        let dst = column_ptr as *mut u8;
        unsafe {
            std::ptr::copy_nonoverlapping(slice.as_ptr(), dst, slice.len());
        }

        self.position += rows_to_export;
        Ok(rows_to_export)
    }

    fn element_size_bytes(&self) -> usize {
        self.bigint_size as usize
    }

    fn has_more(&self) -> bool {
        self.position < self.len
    }

    fn len(&self) -> usize {
        self.len
    }

    fn is_nullable(&self) -> bool {
        self.nullable
    }

    fn export_validity(&mut self, bitmap: *mut u8, max_rows: usize) -> VortexResult<usize> {
        super::export_validity_cached(
            bitmap,
            max_rows,
            self.validity.as_ref(),
            self.last_export_start,
            self.last_export_count,
        )
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use vortex::array::IntoArray;
    use vortex::array::arrays::{FixedSizeListArray, PrimitiveArray};
    use vortex::array::validity::Validity;
    use vortex::buffer::Buffer;

    #[test]
    fn test_bigint_exporter_128bit() {
        // Create test data: two 128-bit integers as byte arrays
        let bytes: Vec<u8> = vec![
            0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08, 0x09, 0x0a, 0x0b, 0x0c, 0x0d, 0x0e,
            0x0f, 0x10, 0x11, 0x12, 0x13, 0x14, 0x15, 0x16, 0x17, 0x18, 0x19, 0x1a, 0x1b, 0x1c,
            0x1d, 0x1e, 0x1f, 0x20,
        ];

        let values = PrimitiveArray::new(Buffer::<u8>::from(bytes.clone()), Validity::NonNullable);
        let array = FixedSizeListArray::try_new(
            values.into_array(),
            16, // 128-bit = 16 bytes
            Validity::NonNullable,
            2, // 2 elements
        )
        .expect("Failed to create FixedSizeListArray");

        let mut exporter =
            BigIntExporter::new(array.into_array()).expect("Failed to create exporter");

        assert!(exporter.has_more());
        assert_eq!(exporter.bigint_size(), BigIntSize::Bits128);

        // Export all data
        let mut output = vec![0u8; 32];
        let exported = exporter
            .export(output.as_mut_ptr() as *mut c_void, output.len(), 2)
            .expect("Export failed");

        assert_eq!(exported, 2);
        assert_eq!(output, bytes);
        assert!(!exporter.has_more());
    }

    #[test]
    fn test_bigint_exporter_256bit() {
        // Create test data: one 256-bit integer as byte array
        let bytes: Vec<u8> = (0..32).collect();

        let values = PrimitiveArray::new(Buffer::<u8>::from(bytes.clone()), Validity::NonNullable);
        let array = FixedSizeListArray::try_new(
            values.into_array(),
            32, // 256-bit = 32 bytes
            Validity::NonNullable,
            1, // 1 element
        )
        .expect("Failed to create FixedSizeListArray");

        let mut exporter =
            BigIntExporter::new(array.into_array()).expect("Failed to create exporter");

        assert!(exporter.has_more());
        assert_eq!(exporter.bigint_size(), BigIntSize::Bits256);

        // Export all data
        let mut output = vec![0u8; 32];
        let exported = exporter
            .export(output.as_mut_ptr() as *mut c_void, output.len(), 1)
            .expect("Export failed");

        assert_eq!(exported, 1);
        assert_eq!(output, bytes);
        assert!(!exporter.has_more());
    }

    #[test]
    fn test_bigint_exporter_partial() {
        // Create test data: four 128-bit integers
        let bytes: Vec<u8> = (0..64).collect();

        let values = PrimitiveArray::new(Buffer::<u8>::from(bytes.clone()), Validity::NonNullable);
        let array = FixedSizeListArray::try_new(
            values.into_array(),
            16,
            Validity::NonNullable,
            4, // 4 elements
        )
        .expect("Failed to create FixedSizeListArray");

        let mut exporter =
            BigIntExporter::new(array.into_array()).expect("Failed to create exporter");

        // Export first 2 integers
        let mut output1 = vec![0u8; 32];
        let exported1 = exporter
            .export(output1.as_mut_ptr() as *mut c_void, output1.len(), 2)
            .expect("Export failed");

        assert_eq!(exported1, 2);
        assert_eq!(output1, bytes[0..32]);
        assert!(exporter.has_more());

        // Export remaining 2 integers
        let mut output2 = vec![0u8; 32];
        let exported2 = exporter
            .export(output2.as_mut_ptr() as *mut c_void, output2.len(), 5) // request more than available
            .expect("Export failed");

        assert_eq!(exported2, 2);
        assert_eq!(output2, bytes[32..64]);
        assert!(!exporter.has_more());
    }
}
