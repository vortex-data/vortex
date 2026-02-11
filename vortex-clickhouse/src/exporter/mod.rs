// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Vortex Array exporter for ClickHouse.
//!
//! This module provides functionality to export Vortex arrays to ClickHouse's
//! column format. It handles different array encodings and attempts to use
//! zero-copy paths when possible.

pub mod bigint;
pub mod bool_;
pub mod decimal;
pub mod list;
pub mod primitive;
pub mod struct_;
pub mod varbinview;

use std::any::Any;

use vortex::array::ArrayRef;
use vortex::dtype::{DType, PType};
use vortex::error::{VortexResult, vortex_bail};
use vortex::mask::Mask;

/// Type tag identifying the concrete exporter kind.
#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExporterKind {
    /// Primitive numeric types (i8..i64, u8..u64, f32, f64)
    Primitive = 0,
    /// Utf8/Binary string types
    String = 1,
    /// Struct/Tuple types
    Struct = 2,
    /// List/Array types
    List = 3,
    /// Big integer types (Int128, UInt128, Int256, UInt256)
    BigInt = 4,
    /// Boolean type
    Bool = 5,
    /// Decimal types
    Decimal = 6,
}

pub use self::bigint::BigIntExporter;
pub use self::bool_::BoolExporter;
pub use self::decimal::DecimalExporter;
pub use self::list::ListExporter;
pub use self::primitive::PrimitiveExporter;
pub use self::struct_::StructExporter;
pub use self::varbinview::VarBinViewExporter;

/// Write a validity bitmap from a `Mask` into a caller-provided byte buffer.
///
/// The buffer is zeroed first so that trailing bits in the last byte are
/// always 0 (= null), then only the "valid" bits are set. This avoids
/// leaking uninitialized memory to the C++ consumer.
///
/// # Arguments
/// * `bitmap` - Mutable byte slice of at least `(count + 7) / 8` bytes.
/// * `validity` - The Vortex validity mask to read from.
/// * `start` - Starting row index in the validity mask.
/// * `count` - Number of rows to export.
pub(crate) fn write_validity_bitmap(
    bitmap: &mut [u8],
    validity: &Mask,
    start: usize,
    count: usize,
) {
    // Zero the buffer so trailing bits in the last byte are deterministic.
    bitmap.fill(0);

    for i in 0..count {
        if validity.value(start + i) {
            bitmap[i / 8] |= 1 << (i % 8);
        }
    }
}

/// Shared `export_validity` implementation for exporters that cache their
/// validity mask and track `last_export_start` / `last_export_count`.
///
/// This covers `PrimitiveExporter`, `VarBinViewExporter`, `BigIntExporter`,
/// `BoolExporter`, and `DecimalExporter`.
pub(crate) fn export_validity_cached(
    bitmap: *mut u8,
    max_rows: usize,
    validity: Option<&Mask>,
    last_export_start: usize,
    last_export_count: usize,
) -> VortexResult<usize> {
    if bitmap.is_null() {
        vortex_bail!("bitmap is null");
    }

    let rows_to_export = last_export_count.min(max_rows);

    if rows_to_export == 0 {
        return Ok(0);
    }

    let validity = match validity {
        Some(v) => v,
        None => vortex_bail!("export_validity called on non-nullable exporter"),
    };

    let bitmap_slice =
        unsafe { std::slice::from_raw_parts_mut(bitmap, rows_to_export.div_ceil(8)) };

    write_validity_bitmap(bitmap_slice, validity, last_export_start, rows_to_export);

    Ok(rows_to_export)
}

/// Trait for exporting Vortex arrays to ClickHouse columns.
pub trait ColumnExporter: Send {
    /// Return the kind tag for this exporter.
    fn kind(&self) -> ExporterKind;

    /// Export array data to the ClickHouse column buffer.
    ///
    /// # Arguments
    /// * `column_ptr` - Pointer to the output buffer.
    /// * `buffer_size_bytes` - Total size of the output buffer in bytes. The
    ///   implementation must verify that the buffer is large enough before
    ///   writing and return an error otherwise.
    /// * `max_rows` - Maximum number of rows to export.
    ///
    /// Returns the number of rows exported.
    fn export(
        &mut self,
        column_ptr: *mut std::ffi::c_void,
        buffer_size_bytes: usize,
        max_rows: usize,
    ) -> VortexResult<usize>;

    /// Return the number of bytes each row occupies in the export buffer.
    ///
    /// For fixed-width types this is the element width (e.g. 4 for `i32`).
    /// Variable-length exporters that do not use `export` (strings, lists)
    /// return 0 to indicate that the caller should use a specialised export
    /// path instead.
    fn element_size_bytes(&self) -> usize {
        0
    }

    /// Check if there is more data to export.
    fn has_more(&self) -> bool;

    /// Get the total number of rows in this exporter.
    fn len(&self) -> usize;

    /// Check if this exporter handles nullable data.
    fn is_nullable(&self) -> bool {
        false
    }

    /// Export validity bitmap for nullable columns.
    ///
    /// Returns the number of rows processed.
    fn export_validity(&mut self, _bitmap: *mut u8, _max_rows: usize) -> VortexResult<usize> {
        vortex_bail!("export_validity not supported for this exporter")
    }

    /// Export string data (for string/binary exporters).
    ///
    /// Returns the number of rows exported.
    fn export_strings(
        &mut self,
        _data: *mut u8,
        _lengths: *mut u32,
        _offsets: *mut u64,
        _max_rows: usize,
    ) -> VortexResult<usize> {
        vortex_bail!("export_strings not supported for this exporter")
    }

    /// Get the total size of string data for the remaining rows.
    ///
    /// This is useful for pre-allocating buffers on the C++ side.
    /// Returns (total_bytes, num_rows_remaining).
    fn string_data_size(&self) -> VortexResult<(usize, usize)> {
        vortex_bail!("string_data_size not supported for this exporter")
    }

    /// Get as Any for downcasting.
    fn as_any(&self) -> &dyn Any;

    /// Get as Any mut for downcasting.
    fn as_any_mut(&mut self) -> &mut dyn Any;
}

/// Create a new column exporter for the given Vortex array.
///
/// This factory function inspects the array's dtype and creates the appropriate
/// exporter implementation:
/// - Primitive types (i8, i16, i32, i64, u8, u16, u32, u64, f32, f64) → PrimitiveExporter
/// - Utf8/Binary types → VarBinViewExporter
/// - Struct types → StructExporter
/// - List types → ListExporter
/// - FixedSizeList<u8, 16/32> → BigIntExporter (for Int128/UInt128/Int256/UInt256)
/// - Extension types → Exporter based on storage dtype
pub fn new_exporter(array: ArrayRef) -> VortexResult<Box<dyn ColumnExporter>> {
    match array.dtype() {
        DType::Primitive(_, _) => Ok(Box::new(PrimitiveExporter::new(array)?)),
        DType::Utf8(_) | DType::Binary(_) => Ok(Box::new(VarBinViewExporter::new(array)?)),
        DType::Struct(_, _) => Ok(Box::new(StructExporter::new(array)?)),
        DType::List(_, _) => Ok(Box::new(ListExporter::new(array)?)),
        DType::FixedSizeList(elem_dtype, size, _) => {
            // Check if this is a big integer type (FixedSizeList<u8, 16/32>)
            if matches!(elem_dtype.as_ref(), DType::Primitive(PType::U8, _))
                && (*size == 16 || *size == 32)
            {
                Ok(Box::new(BigIntExporter::new(array)?))
            } else {
                Ok(Box::new(ListExporter::new(array)?))
            }
        }
        DType::Bool(_) => Ok(Box::new(BoolExporter::new(array)?)),
        DType::Null => {
            vortex_bail!("Null type arrays not supported in exporter")
        }
        DType::Extension(_) => {
            // For extension types, extract the storage array and recurse.
            // ExtensionArray wraps a storage array with the same data but
            // typed as the storage dtype (e.g., Utf8, Primitive, etc.)
            use vortex::array::ToCanonical;
            use vortex::array::arrays::ExtensionArray;
            let ext_array = array.to_extension();
            let storage = ext_array.storage().clone();
            new_exporter(storage)
        }
        DType::Decimal(_, _) => {
            // Decimal types have their own dedicated exporter
            Ok(Box::new(DecimalExporter::new(array)?))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;
    use vortex::array::IntoArray;
    use vortex::array::arrays::{PrimitiveArray, StructArray, VarBinViewArray};
    use vortex::array::validity::Validity;
    use vortex::buffer::Buffer;
    use vortex::dtype::FieldNames;

    #[test]
    fn test_new_exporter_primitive() {
        let buffer: Buffer<i64> = vec![1i64, 2, 3].into();
        let array = PrimitiveArray::new(buffer, Validity::NonNullable).into_array();

        let exporter = new_exporter(array).expect("Failed to create exporter");
        assert!(exporter.has_more());
    }

    #[test]
    fn test_new_exporter_string() {
        let array = VarBinViewArray::from_iter_str(vec!["hello", "world"]).into_array();

        let exporter = new_exporter(array).expect("Failed to create exporter");
        assert!(exporter.has_more());
    }

    #[test]
    fn test_primitive_export() {
        let buffer: Buffer<i32> = vec![10i32, 20, 30, 40, 50].into();
        let array = PrimitiveArray::new(buffer, Validity::NonNullable).into_array();

        let mut exporter = PrimitiveExporter::new(array).expect("Failed to create exporter");

        // Allocate output buffer
        let mut output = vec![0i32; 3];

        // Export first 3 elements
        let exported = exporter
            .export(
                output.as_mut_ptr() as *mut std::ffi::c_void,
                size_of_val(output.as_slice()),
                3,
            )
            .expect("Export failed");

        assert_eq!(exported, 3);
        assert_eq!(output, vec![10, 20, 30]);
        assert!(exporter.has_more());

        // Export remaining 2 elements
        let mut output2 = vec![0i32; 3];
        let exported2 = exporter
            .export(
                output2.as_mut_ptr() as *mut std::ffi::c_void,
                size_of_val(output2.as_slice()),
                3,
            )
            .expect("Export failed");

        assert_eq!(exported2, 2);
        assert_eq!(output2[0..2], vec![40, 50]);
        assert!(!exporter.has_more());
    }

    #[test]
    fn test_struct_exporter_creation() {
        let id_array = {
            let buffer: Buffer<i64> = vec![1i64, 2, 3].into();
            PrimitiveArray::new(buffer, Validity::NonNullable).into_array()
        };

        let name_array =
            VarBinViewArray::from_iter_str(vec!["Alice", "Bob", "Charlie"]).into_array();

        let field_names: Vec<Arc<str>> = vec![Arc::from("id"), Arc::from("name")];
        let struct_array = StructArray::try_new(
            FieldNames::from(field_names),
            vec![id_array, name_array],
            3,
            Validity::NonNullable,
        )
        .expect("Failed to create struct array");

        let exporter = StructExporter::new(struct_array.into_array())
            .expect("Failed to create struct exporter");

        assert_eq!(exporter.num_fields(), 2);
    }
}
