// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Struct array exporter for ClickHouse.
//!
//! This module exports Vortex struct arrays (tuples/records) to ClickHouse.
//! It recursively exports each field using the appropriate exporter.

use std::any::Any;
use std::ffi::c_void;

use vortex::array::{Array, ArrayRef, ToCanonical};
use vortex::error::{VortexResult, vortex_bail};

use super::{ColumnExporter, ExporterKind, new_exporter};

/// Exporter for struct (tuple) arrays.
pub struct StructExporter {
    /// Child exporters for each field
    field_exporters: Vec<Option<Box<dyn ColumnExporter>>>,
    /// Total length of the struct array
    len: usize,
    /// Whether all fields have been exported
    done: bool,
}

impl StructExporter {
    /// Create a new struct exporter for the given array.
    pub fn new(array: ArrayRef) -> VortexResult<Self> {
        // Verify this is a struct type
        match array.dtype() {
            vortex::dtype::DType::Struct(_, _) => {}
            _ => vortex_bail!("StructExporter requires a Struct array"),
        }

        let len = array.len();

        // Get struct array and create exporters for each field
        let struct_array = array.to_struct();
        let mut field_exporters = Vec::new();

        for field in struct_array.unmasked_fields().iter() {
            let exporter = new_exporter(field.clone())?;
            field_exporters.push(Some(exporter));
        }

        Ok(Self {
            field_exporters,
            len,
            done: false,
        })
    }

    /// Get the number of fields in the struct.
    pub fn num_fields(&self) -> usize {
        self.field_exporters.len()
    }

    /// Get the field exporter at the given index (borrow).
    pub fn field_exporter(&mut self, index: usize) -> Option<&mut Box<dyn ColumnExporter>> {
        self.field_exporters
            .get_mut(index)
            .and_then(|opt| opt.as_mut())
    }

    /// Take the field exporter at the given index (ownership transfer).
    /// Returns None if index is out of bounds or already taken.
    pub fn take_field_exporter(&mut self, index: usize) -> Option<Box<dyn ColumnExporter>> {
        self.field_exporters
            .get_mut(index)
            .and_then(|opt| opt.take())
    }

    /// Get the length of the struct array.
    pub fn len(&self) -> usize {
        self.len
    }

    /// Check if the struct array is empty.
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }
}

impl ColumnExporter for StructExporter {
    fn kind(&self) -> ExporterKind {
        ExporterKind::Struct
    }

    fn export(
        &mut self,
        _column_ptr: *mut c_void,
        _buffer_size_bytes: usize,
        _max_rows: usize,
    ) -> VortexResult<usize> {
        // Struct export requires exporting each field separately
        // The caller should iterate over fields using field_exporter()
        vortex_bail!(
            "StructExporter::export() not supported. Use field_exporter() to export individual fields."
        )
    }

    fn has_more(&self) -> bool {
        !self.done
            && self
                .field_exporters
                .iter()
                .any(|e| e.as_ref().map_or(false, |exp| exp.has_more()))
    }

    fn len(&self) -> usize {
        self.len
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }
}
