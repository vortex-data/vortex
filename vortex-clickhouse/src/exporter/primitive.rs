// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Primitive array exporter for ClickHouse.
//!
//! This module exports Vortex primitive arrays (numeric types) to ClickHouse column buffers.
//! It attempts to use zero-copy paths when possible by directly copying the underlying buffer.

use std::any::Any;
use std::ffi::c_void;

use vortex::array::arrays::PrimitiveArray;
use vortex::array::{Array, ArrayRef, ToCanonical};
use vortex::dtype::{Nullability, PType};
use vortex::error::{VortexResult, vortex_bail};
use vortex::mask::Mask;

use super::{ColumnExporter, ExporterKind};

/// Exporter for primitive (numeric) arrays.
pub struct PrimitiveExporter {
    /// Cached canonical primitive array
    canonical: PrimitiveArray,
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
    /// Primitive type
    ptype: PType,
    /// Whether the array is nullable
    nullable: bool,
}

impl PrimitiveExporter {
    /// Create a new primitive exporter for the given array.
    pub fn new(array: ArrayRef) -> VortexResult<Self> {
        let len = array.len();
        let (ptype, nullable) = match array.dtype() {
            vortex::dtype::DType::Primitive(ptype, nullability) => {
                (*ptype, *nullability == Nullability::Nullable)
            }
            _ => vortex_bail!("PrimitiveExporter requires a primitive array"),
        };

        let canonical = array.to_primitive();
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
            ptype,
            nullable,
        })
    }
}

impl ColumnExporter for PrimitiveExporter {
    fn kind(&self) -> ExporterKind {
        ExporterKind::Primitive
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

        let required_bytes = rows_to_export * self.ptype.byte_width();
        if buffer_size_bytes < required_bytes {
            vortex_bail!(
                "buffer too small: need {} bytes for {} rows of {:?}, got {}",
                required_bytes,
                rows_to_export,
                self.ptype,
                buffer_size_bytes
            );
        }

        // Record the start position for validity export
        self.last_export_start = self.position;
        self.last_export_count = rows_to_export;

        // Export based on primitive type using macro
        macro_rules! export_primitive {
            ($ptype:ident, $rust_ty:ty) => {{
                let buffer = self.canonical.as_slice::<$rust_ty>();
                let start = self.position;
                let end = start + rows_to_export;
                let slice = &buffer[start..end];

                // Copy to destination
                let dst = column_ptr as *mut $rust_ty;
                unsafe {
                    std::ptr::copy_nonoverlapping(slice.as_ptr(), dst, rows_to_export);
                }
            }};
        }

        match self.ptype {
            PType::I8 => export_primitive!(I8, i8),
            PType::I16 => export_primitive!(I16, i16),
            PType::I32 => export_primitive!(I32, i32),
            PType::I64 => export_primitive!(I64, i64),
            PType::U8 => export_primitive!(U8, u8),
            PType::U16 => export_primitive!(U16, u16),
            PType::U32 => export_primitive!(U32, u32),
            PType::U64 => export_primitive!(U64, u64),
            PType::F32 => export_primitive!(F32, f32),
            PType::F64 => export_primitive!(F64, f64),
            PType::F16 => vortex_bail!("F16 export not supported"),
        }

        self.position += rows_to_export;
        Ok(rows_to_export)
    }

    fn element_size_bytes(&self) -> usize {
        self.ptype.byte_width()
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
