// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! VarBinView array exporter for ClickHouse (String type).
//!
//! This module exports Vortex VarBinView arrays (string/binary types) to ClickHouse column buffers.
//! It uses zero-copy `bytes_at` for direct buffer access instead of scalar extraction.

use std::any::Any;
use std::cell::Cell;
use std::ffi::c_void;

use vortex::array::arrays::VarBinViewArray;
use vortex::array::{Array, ArrayRef, ToCanonical};
use vortex::dtype::Nullability;
use vortex::error::{VortexResult, vortex_bail};
use vortex::mask::Mask;

use super::{ColumnExporter, ExporterKind};

/// Callback function type for appending strings to ClickHouse column.
///
/// The C++ side provides this callback which takes:
/// - column_ptr: Pointer to the ClickHouse column
/// - data: Pointer to string data
/// - len: Length of the string
pub type StringAppendFn = extern "C" fn(*mut c_void, *const u8, usize);

/// Exporter for variable-length binary/string arrays.
pub struct VarBinViewExporter {
    /// Cached canonical VarBinView array
    canonical: VarBinViewArray,
    /// Cached validity mask (None = non-nullable)
    validity: Option<Mask>,
    /// Current export position
    position: usize,
    /// Position at the start of last export (for validity export)
    last_export_start: usize,
    /// Number of rows exported in last export call
    last_export_count: usize,
    /// Total length of the array
    len: usize,
    /// Whether the array is nullable
    nullable: bool,
    /// Cached total data size for remaining rows (lazily computed).
    ///
    /// Uses `Cell` for interior mutability so the `&self` trait method
    /// `string_data_size` can populate the cache without `&mut self`.
    cached_data_size: Cell<Option<usize>>,
}

impl VarBinViewExporter {
    /// Create a new varbinview exporter for the given array.
    pub fn new(array: ArrayRef) -> VortexResult<Self> {
        let len = array.len();

        // Verify this is a string or binary type
        let nullable = match array.dtype() {
            vortex::dtype::DType::Utf8(nullability) => *nullability == Nullability::Nullable,
            vortex::dtype::DType::Binary(nullability) => *nullability == Nullability::Nullable,
            _ => vortex_bail!("VarBinViewExporter requires a Utf8 or Binary array"),
        };

        let canonical = array.to_varbinview();
        let validity = if nullable {
            Some(canonical.validity_mask()?)
        } else {
            None
        };

        Ok(Self {
            canonical,
            validity,
            position: 0,
            last_export_start: 0,
            last_export_count: 0,
            len,
            nullable,
            cached_data_size: Cell::new(None),
        })
    }

    /// Check if the value at the given index is valid (non-null).
    fn is_valid(&self, idx: usize) -> bool {
        match &self.validity {
            Some(mask) => mask.value(idx),
            None => true,
        }
    }

    /// Export strings using a callback function.
    ///
    /// This is the preferred method for exporting strings to ClickHouse,
    /// as it allows the C++ side to handle memory allocation.
    pub fn export_with_callback(
        &mut self,
        column_ptr: *mut c_void,
        max_rows: usize,
        append_fn: StringAppendFn,
    ) -> VortexResult<usize> {
        if column_ptr.is_null() {
            vortex_bail!("column_ptr is null");
        }

        let remaining = self.len - self.position;
        let rows_to_export = remaining.min(max_rows);

        if rows_to_export == 0 {
            return Ok(0);
        }

        for i in self.position..(self.position + rows_to_export) {
            if self.is_valid(i) {
                let bytes = self.canonical.bytes_at(i);
                let slice = bytes.as_ref();
                append_fn(column_ptr, slice.as_ptr(), slice.len());
            } else {
                append_fn(column_ptr, std::ptr::null(), 0);
            }
        }

        self.position += rows_to_export;
        // Invalidate cached data size since position changed
        self.cached_data_size.set(None);
        Ok(rows_to_export)
    }

    /// Calculate the total size of string data for the remaining rows.
    ///
    /// This method caches the result for efficiency.
    pub fn compute_total_data_size(&mut self) -> VortexResult<usize> {
        // Return cached value if available
        if let Some(size) = self.cached_data_size.get() {
            return Ok(size);
        }

        let remaining = self.len - self.position;
        if remaining == 0 {
            self.cached_data_size.set(Some(0));
            return Ok(0);
        }

        let views = self.canonical.views();
        let mut total_size = 0usize;

        for i in self.position..self.len {
            if self.is_valid(i) {
                total_size += views[i].len() as usize;
            }
        }

        self.cached_data_size.set(Some(total_size));
        Ok(total_size)
    }
}

impl ColumnExporter for VarBinViewExporter {
    fn kind(&self) -> ExporterKind {
        ExporterKind::String
    }

    fn export(
        &mut self,
        _column_ptr: *mut c_void,
        _buffer_size_bytes: usize,
        _max_rows: usize,
    ) -> VortexResult<usize> {
        // The default export method is not suitable for strings
        // because ClickHouse strings need special handling.
        // Use export_with_callback or export_strings instead.
        vortex_bail!("VarBinViewExporter::export() not supported. Use export_strings() instead.")
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

    fn export_strings(
        &mut self,
        data: *mut u8,
        lengths: *mut u32,
        offsets: *mut u64,
        max_rows: usize,
    ) -> VortexResult<usize> {
        if data.is_null() || lengths.is_null() || offsets.is_null() {
            vortex_bail!("Buffer pointers cannot be null");
        }

        let remaining = self.len - self.position;
        let rows_to_export = remaining.min(max_rows);

        if rows_to_export == 0 {
            return Ok(0);
        }

        // Record the start position for validity export
        self.last_export_start = self.position;
        self.last_export_count = rows_to_export;

        let mut current_offset: u64 = 0;

        for i in 0..rows_to_export {
            let idx = self.position + i;

            // Write offset
            unsafe {
                *offsets.add(i) = current_offset;
            }

            if self.is_valid(idx) {
                let bytes = self.canonical.bytes_at(idx);
                let slice = bytes.as_ref();
                let len = slice.len();

                // Write string/binary data
                unsafe {
                    std::ptr::copy_nonoverlapping(
                        slice.as_ptr(),
                        data.add(current_offset as usize),
                        len,
                    );
                    *lengths.add(i) = len as u32;
                }

                current_offset += len as u64;
            } else {
                unsafe {
                    *lengths.add(i) = 0;
                }
            }
        }

        self.position += rows_to_export;
        // Invalidate cached data size since position changed
        self.cached_data_size.set(None);
        Ok(rows_to_export)
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

    fn string_data_size(&self) -> VortexResult<(usize, usize)> {
        let remaining = self.len - self.position;
        if remaining == 0 {
            return Ok((0, 0));
        }

        // If we have cached value, use it
        if let Some(size) = self.cached_data_size.get() {
            return Ok((size, remaining));
        }

        // Calculate total data size from views (no scalar extraction)
        let views = self.canonical.views();
        let mut total_size = 0usize;

        for i in self.position..self.len {
            if self.is_valid(i) {
                total_size += views[i].len() as usize;
            }
        }

        self.cached_data_size.set(Some(total_size));
        Ok((total_size, remaining))
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }
}
