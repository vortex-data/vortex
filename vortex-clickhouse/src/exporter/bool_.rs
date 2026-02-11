// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Bool array exporter for ClickHouse.
//!
//! This module exports Vortex Bool arrays to ClickHouse UInt8 column buffers.
//! ClickHouse represents booleans as UInt8 (0 = false, 1 = true).

use std::any::Any;
use std::ffi::c_void;

use vortex::array::arrays::BoolArray;
use vortex::array::{Array, ArrayRef, ToCanonical};
use vortex::dtype::Nullability;
use vortex::error::{VortexResult, vortex_bail};
use vortex::mask::Mask;

use super::{ColumnExporter, ExporterKind};

/// Exporter for Bool arrays.
pub struct BoolExporter {
    /// Cached canonical bool array
    canonical: BoolArray,
    /// Cached validity mask (None = non-nullable)
    validity: Option<Mask>,
    /// Current export position
    position: usize,
    /// Position at start of last export (for validity export)
    last_export_start: usize,
    /// Number of rows exported in last export call
    last_export_count: usize,
    /// Total length of the array
    len: usize,
    /// Whether the array is nullable
    nullable: bool,
}

impl BoolExporter {
    /// Create a new bool exporter for the given array.
    pub fn new(array: ArrayRef) -> VortexResult<Self> {
        let len = array.len();

        let nullable = match array.dtype() {
            vortex::dtype::DType::Bool(nullability) => *nullability == Nullability::Nullable,
            _ => vortex_bail!("BoolExporter requires a Bool array"),
        };

        let canonical = array.to_bool();
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
        })
    }
}

impl ColumnExporter for BoolExporter {
    fn kind(&self) -> ExporterKind {
        ExporterKind::Bool
    }

    fn export(
        &mut self,
        buffer: *mut c_void,
        buffer_size_bytes: usize,
        max_rows: usize,
    ) -> VortexResult<usize> {
        if buffer.is_null() {
            vortex_bail!("buffer is null");
        }

        let remaining = self.len - self.position;
        let rows_to_export = remaining.min(max_rows);

        if rows_to_export == 0 {
            return Ok(0);
        }

        // ClickHouse booleans are exported as one byte (u8) per row.
        if buffer_size_bytes < rows_to_export {
            vortex_bail!(
                "buffer too small: need {} bytes for {} Bool rows, got {}",
                rows_to_export,
                rows_to_export,
                buffer_size_bytes
            );
        }

        // Record start position for validity export
        self.last_export_start = self.position;
        self.last_export_count = rows_to_export;

        // Export as u8 (ClickHouse boolean representation).
        // Read directly from the bit-packed buffer instead of per-element scalar_at.
        let bits = self.canonical.to_bit_buffer();
        let output_slice =
            unsafe { std::slice::from_raw_parts_mut(buffer as *mut u8, rows_to_export) };

        for i in 0..rows_to_export {
            let idx = self.position + i;
            // Null values produce 0 (false) — the validity bitmap is exported separately.
            output_slice[i] = u8::from(bits.value(idx));
        }

        self.position += rows_to_export;
        Ok(rows_to_export)
    }

    fn element_size_bytes(&self) -> usize {
        // ClickHouse represents Bool as UInt8, one byte per row.
        1
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
