// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Vortex file writing for ClickHouse.
//!
//! This module implements the write path for Vortex files in ClickHouse.
//! It provides the core logic for the `VortexBlockOutputFormat` C++ class.
//!
//! # Architecture
//!
//! The writer uses a streaming architecture to avoid accumulating all data in memory:
//!
//! 1. On creation, a background write task is spawned that consumes arrays from a channel
//! 2. Each batch is sent through the channel and written incrementally
//! 3. On finalize, the channel is closed and we wait for the write task to complete
//!
//! This ensures memory usage is bounded by the channel buffer size (currently 32 batches).
//!
//! # FFI Interface
//!
//! The following C functions are exported for ClickHouse to use:
//!
//! - `vortex_writer_new` - Create a new writer
//! - `vortex_writer_free` - Free a writer
//! - `vortex_writer_add_column` - Add a column to the schema
//! - `vortex_writer_write_batch` - Write a batch of data
//! - `vortex_writer_write_string_column` - Write a string column
//! - `vortex_writer_finalize` - Finalize and flush the file

use std::ffi::{CStr, c_char, c_void};
use std::ptr;
use std::sync::Arc;

use futures::SinkExt;
use futures::StreamExt;
use futures::channel::mpsc;
use futures::channel::mpsc::Sender;
use vortex::array::arrays::{
    DecimalArray, FixedSizeListArray, ListViewArray, PrimitiveArray, StructArray, VarBinViewArray,
};
use vortex::array::builders::{ArrayBuilder, VarBinViewBuilder};
use vortex::array::stream::ArrayStreamAdapter;
use vortex::array::validity::Validity;
use vortex::array::{ArrayRef, IntoArray};
use vortex::buffer::Buffer;
use vortex::dtype::{
    DType, DecimalDType, DecimalType, FieldNames, Nullability, PType, StructFields,
};
use vortex::error::{VortexResult, vortex_bail, vortex_err};
use vortex::file::{WriteOptionsSessionExt, WriteSummary};
use vortex::io::runtime::{BlockingRuntime, Task};
use vortex::io::session::RuntimeSessionExt;

use crate::convert::dtype::clickhouse_type_to_vortex;
use crate::error::{clear_last_error, set_last_error};
use crate::{RUNTIME, SESSION};

/// Channel buffer size for streaming writes.
/// This limits memory usage to approximately this many batches in flight.
const CHANNEL_BUFFER_SIZE: usize = 32;

/// Column definition for schema building.
#[derive(Clone)]
struct ColumnDef {
    /// Column name
    name: String,
    /// Vortex dtype
    dtype: DType,
}

/// String data for a column, provided from ClickHouse side.
#[derive(Clone)]
pub struct StringColumnData {
    /// Raw string data (concatenated)
    pub data: Vec<u8>,
    /// Offsets into data for each string (length = num_rows + 1)
    pub offsets: Vec<u64>,
}

/// State of the streaming writer
enum WriterState {
    /// Schema is being built, no data written yet
    Building,
    /// Writer is active, data can be written
    Active {
        /// Channel to send arrays to the background writer
        sender: Sender<VortexResult<ArrayRef>>,
        /// Handle to the background write task
        writer_task: Task<VortexResult<WriteSummary>>,
    },
    /// Writer has been finalized
    Finalized,
}

/// Pending state for a list column being written.
struct PendingList {
    /// The list offsets (num_rows + 1 elements)
    offsets: Vec<u64>,
    /// Validity for the list itself
    validity: Validity,
    /// Number of rows
    num_rows: usize,
    /// The element array (set when elements are written)
    elements: Option<ArrayRef>,
}

/// Pending state for a struct column being written.
struct PendingStruct {
    /// Validity for the struct itself
    validity: Validity,
    /// Number of rows
    num_rows: usize,
    /// The field arrays, indexed by field_index
    fields: Vec<Option<ArrayRef>>,
    /// Number of fields expected
    num_fields: usize,
}

/// Vortex file writer that implements the write logic.
///
/// Uses a streaming architecture to avoid accumulating all data in memory.
/// Data is written incrementally through a channel to a background task.
pub struct VortexWriter {
    /// Output file path.
    output_path: String,
    /// Column definitions (schema).
    columns: Vec<ColumnDef>,
    /// Writer state machine
    state: WriterState,
    /// Total rows written.
    total_rows: usize,
    /// Pending column arrays for the current batch.
    pending_columns: Vec<Option<ArrayRef>>,
    /// Expected number of rows for current pending batch.
    pending_num_rows: usize,
    /// Pending list column writes (column_index -> PendingList)
    pending_lists: std::collections::HashMap<usize, PendingList>,
    /// Pending struct column writes (column_index -> PendingStruct)
    pending_structs: std::collections::HashMap<usize, PendingStruct>,
}

impl VortexWriter {
    /// Create a new writer for the given output path.
    pub fn new(output_path: &str) -> VortexResult<Self> {
        if output_path.is_empty() {
            vortex_bail!("Output path cannot be empty");
        }

        Ok(Self {
            output_path: output_path.to_string(),
            columns: Vec::new(),
            state: WriterState::Building,
            total_rows: 0,
            pending_columns: Vec::new(),
            pending_num_rows: 0,
            pending_lists: std::collections::HashMap::new(),
            pending_structs: std::collections::HashMap::new(),
        })
    }

    /// Add a column to the schema.
    pub fn add_column(
        &mut self,
        name: &str,
        clickhouse_type: &str,
        nullable: bool,
    ) -> VortexResult<()> {
        if !matches!(self.state, WriterState::Building) {
            vortex_bail!("Cannot add column after writing has started");
        }

        // Convert ClickHouse type to Vortex DType
        let mut dtype = clickhouse_type_to_vortex(clickhouse_type)?;

        // Adjust nullability based on the nullable parameter
        // If the ClickHouse type is already Nullable(...), respect that.
        // Otherwise, use the nullable parameter to set nullability.
        if !clickhouse_type.starts_with("Nullable(") {
            use vortex::dtype::Nullability;
            let target_nullability = if nullable {
                Nullability::Nullable
            } else {
                Nullability::NonNullable
            };
            dtype = dtype.with_nullability(target_nullability);
        }

        self.columns.push(ColumnDef {
            name: name.to_string(),
            dtype,
        });

        Ok(())
    }

    /// Get the number of columns.
    pub fn num_columns(&self) -> usize {
        self.columns.len()
    }

    /// Get the struct DType for the schema.
    fn get_struct_dtype(&self) -> DType {
        let field_names: Vec<Arc<str>> = self
            .columns
            .iter()
            .map(|c| Arc::from(c.name.as_str()))
            .collect();
        let field_dtypes: Vec<DType> = self.columns.iter().map(|c| c.dtype.clone()).collect();

        DType::Struct(
            StructFields::new(FieldNames::from(field_names), field_dtypes),
            Nullability::NonNullable,
        )
    }

    /// Start the background writer if not already started.
    fn ensure_writer_started(&mut self) -> VortexResult<()> {
        if matches!(self.state, WriterState::Building) {
            if self.columns.is_empty() {
                vortex_bail!("Cannot start writing without any columns defined");
            }

            let struct_dtype = self.get_struct_dtype();
            let output_path = self.output_path.clone();

            // Create channel for streaming data
            let (sender, receiver) = mpsc::channel(CHANNEL_BUFFER_SIZE);

            // Create array stream from channel receiver
            let array_stream = ArrayStreamAdapter::new(struct_dtype, receiver.map(|r| r));

            // Spawn background writer task
            let writer_task = SESSION.handle().spawn(async move {
                let mut file = async_fs::File::create(&output_path).await.map_err(|e| {
                    vortex_err!("Failed to create output file '{}': {}", output_path, e)
                })?;
                SESSION.write_options().write(&mut file, array_stream).await
            });

            self.state = WriterState::Active {
                sender,
                writer_task,
            };
        }
        Ok(())
    }

    /// Send an array to the background writer.
    fn send_array(&mut self, array: ArrayRef) -> VortexResult<()> {
        self.ensure_writer_started()?;

        match &mut self.state {
            WriterState::Active { sender, .. } => {
                RUNTIME
                    .block_on(sender.send(Ok(array)))
                    .map_err(|e| vortex_err!("Failed to send array to writer: {}", e))?;
                Ok(())
            }
            WriterState::Finalized => {
                vortex_bail!("Cannot write after finalization")
            }
            WriterState::Building => {
                // This shouldn't happen as ensure_writer_started was called
                vortex_bail!("Writer not started")
            }
        }
    }

    /// Begin writing a new batch with the given number of rows.
    ///
    /// This prepares the writer to accept column data via write_column_*() methods.
    pub fn begin_batch(&mut self, num_rows: usize) -> VortexResult<()> {
        if matches!(self.state, WriterState::Finalized) {
            vortex_bail!("Cannot write after finalization");
        }

        if !self.pending_columns.is_empty() {
            vortex_bail!("Previous batch not completed. Call end_batch() first.");
        }

        self.pending_columns = vec![None; self.columns.len()];
        self.pending_num_rows = num_rows;
        Ok(())
    }

    /// Write a primitive column by index.
    pub fn write_column_primitive(
        &mut self,
        column_index: usize,
        data: *const c_void,
        num_rows: usize,
    ) -> VortexResult<()> {
        self.write_column_primitive_with_validity(column_index, data, ptr::null(), num_rows)
    }

    /// Write a primitive column by index with optional validity bitmap.
    ///
    /// The validity bitmap uses ClickHouse's convention where 0 = null, 1 = valid.
    /// Each byte contains 8 validity bits in LSB order.
    pub fn write_column_primitive_with_validity(
        &mut self,
        column_index: usize,
        data: *const c_void,
        validity_bitmap: *const u8,
        num_rows: usize,
    ) -> VortexResult<()> {
        if column_index >= self.columns.len() {
            vortex_bail!("Column index {} out of bounds", column_index);
        }

        if num_rows != self.pending_num_rows {
            vortex_bail!(
                "Row count mismatch: expected {}, got {}",
                self.pending_num_rows,
                num_rows
            );
        }

        let dtype = &self.columns[column_index].dtype;
        let array = build_array_from_raw_with_validity(dtype, data, validity_bitmap, num_rows)?;
        self.pending_columns[column_index] = Some(array);
        Ok(())
    }

    /// Write a string column by index.
    ///
    /// The strings are provided as concatenated data with offsets.
    pub fn write_column_strings(
        &mut self,
        column_index: usize,
        data: *const u8,
        offsets: *const u64,
        num_rows: usize,
    ) -> VortexResult<()> {
        self.write_column_strings_with_validity(column_index, data, offsets, ptr::null(), num_rows)
    }

    /// Write a string column by index with optional validity bitmap.
    ///
    /// The strings are provided as concatenated data with offsets.
    /// The null_map uses ClickHouse's convention: one byte per row, 0 = valid, 1 = null.
    pub fn write_column_strings_with_validity(
        &mut self,
        column_index: usize,
        data: *const u8,
        offsets: *const u64,
        null_map: *const u8,
        num_rows: usize,
    ) -> VortexResult<()> {
        if column_index >= self.columns.len() {
            vortex_bail!("Column index {} out of bounds", column_index);
        }

        if num_rows != self.pending_num_rows {
            vortex_bail!(
                "Row count mismatch: expected {}, got {}",
                self.pending_num_rows,
                num_rows
            );
        }

        // Build string array from offsets
        let offsets_slice = unsafe { std::slice::from_raw_parts(offsets, num_rows + 1) };

        // Total data size is the last offset
        let total_data_len = offsets_slice[num_rows] as usize;
        let data_slice = if total_data_len > 0 {
            unsafe { std::slice::from_raw_parts(data, total_data_len) }
        } else {
            &[]
        };

        // Check if we need to handle nullability
        let dtype = &self.columns[column_index].dtype;
        let is_nullable = match dtype {
            DType::Utf8(n) | DType::Binary(n) => *n == Nullability::Nullable,
            DType::Extension(ext) => ext.storage_dtype().is_nullable(),
            _ => false,
        };

        // Check if the storage is binary (not UTF-8 text)
        let is_binary = match dtype {
            DType::Binary(_) => true,
            DType::Extension(ext) => matches!(ext.storage_dtype(), DType::Binary(_)),
            _ => false,
        };

        let array = if is_binary {
            // Binary data path: treat bytes as-is, no UTF-8 validation
            if is_nullable && !null_map.is_null() {
                let null_slice = unsafe { std::slice::from_raw_parts(null_map, num_rows) };
                let bins: Vec<Option<&[u8]>> = (0..num_rows)
                    .map(|i| {
                        if null_slice[i] != 0 {
                            None
                        } else {
                            let start = offsets_slice[i] as usize;
                            let end = offsets_slice[i + 1] as usize;
                            Some(&data_slice[start..end])
                        }
                    })
                    .collect();
                VarBinViewArray::from_iter_nullable_bin(bins).into_array()
            } else {
                let bins: Vec<&[u8]> = (0..num_rows)
                    .map(|i| {
                        let start = offsets_slice[i] as usize;
                        let end = offsets_slice[i + 1] as usize;
                        &data_slice[start..end]
                    })
                    .collect();
                VarBinViewArray::from_iter_bin(bins).into_array()
            }
        } else if is_nullable && !null_map.is_null() {
            // Build nullable string array
            let null_slice = unsafe { std::slice::from_raw_parts(null_map, num_rows) };
            let strings: Vec<Option<&str>> = (0..num_rows)
                .map(|i| {
                    // ClickHouse null map: 0 = valid, 1 = null
                    if null_slice[i] != 0 {
                        None
                    } else {
                        let start = offsets_slice[i] as usize;
                        let end = offsets_slice[i + 1] as usize;
                        Some(std::str::from_utf8(&data_slice[start..end]).unwrap_or(""))
                    }
                })
                .collect();
            VarBinViewArray::from_iter_nullable_str(strings).into_array()
        } else {
            // Build non-nullable string array
            let strings: Vec<&str> = (0..num_rows)
                .map(|i| {
                    let start = offsets_slice[i] as usize;
                    let end = offsets_slice[i + 1] as usize;
                    std::str::from_utf8(&data_slice[start..end]).unwrap_or("")
                })
                .collect();
            VarBinViewArray::from_iter_str(strings).into_array()
        };

        // Wrap in ExtensionArray if the target dtype is an Extension type
        let array = if let DType::Extension(ext) = dtype {
            use vortex::array::arrays::ExtensionArray;
            ExtensionArray::new(ext.clone(), array).into_array()
        } else {
            array
        };

        self.pending_columns[column_index] = Some(array);
        Ok(())
    }

    /// End the current batch and commit it.
    ///
    /// This sends the batch to the background writer through the channel.
    /// Memory is released as soon as the batch is sent (bounded by channel buffer).
    pub fn end_batch(&mut self) -> VortexResult<()> {
        if self.pending_columns.is_empty() {
            return Ok(());
        }

        // Check all columns are filled
        for (i, col) in self.pending_columns.iter().enumerate() {
            if col.is_none() {
                vortex_bail!("Column {} not written before end_batch()", i);
            }
        }

        let field_arrays: Vec<ArrayRef> =
            self.pending_columns.drain(..).map(|c| c.unwrap()).collect();

        let field_names: Vec<Arc<str>> = self
            .columns
            .iter()
            .map(|c| Arc::from(c.name.as_str()))
            .collect();

        let struct_array = StructArray::try_new(
            FieldNames::from(field_names),
            field_arrays,
            self.pending_num_rows,
            Validity::NonNullable,
        )?;

        // Send the batch to the background writer
        self.send_array(struct_array.into_array())?;
        self.total_rows += self.pending_num_rows;
        self.pending_num_rows = 0;

        Ok(())
    }

    // =========================================================================
    // List (Array) column writing
    // =========================================================================

    /// Write offsets for a list column. The element data is written separately.
    pub fn list_write_offsets(
        &mut self,
        column_index: usize,
        offsets: *const u64,
        null_map: *const u8,
        num_rows: usize,
    ) -> VortexResult<()> {
        if column_index >= self.columns.len() {
            vortex_bail!("Column index {} out of bounds", column_index);
        }
        if num_rows != self.pending_num_rows {
            vortex_bail!(
                "Row count mismatch: expected {}, got {}",
                self.pending_num_rows,
                num_rows
            );
        }

        let offsets_slice = unsafe { std::slice::from_raw_parts(offsets, num_rows + 1) };
        let offsets_vec: Vec<u64> = offsets_slice.to_vec();

        let dtype = &self.columns[column_index].dtype;
        let is_nullable = dtype.is_nullable();
        let validity = if is_nullable {
            null_map_to_validity(null_map, num_rows)
        } else {
            Validity::NonNullable
        };

        self.pending_lists.insert(
            column_index,
            PendingList {
                offsets: offsets_vec,
                validity,
                num_rows,
                elements: None,
            },
        );

        Ok(())
    }

    /// Write primitive element data for a list column.
    pub fn list_write_element_column(
        &mut self,
        column_index: usize,
        data: *const c_void,
        num_elements: usize,
    ) -> VortexResult<()> {
        self.list_write_element_column_nullable(column_index, data, ptr::null(), num_elements)
    }

    /// Write nullable primitive element data for a list column.
    pub fn list_write_element_column_nullable(
        &mut self,
        column_index: usize,
        data: *const c_void,
        null_map: *const u8,
        num_elements: usize,
    ) -> VortexResult<()> {
        let pending = self
            .pending_lists
            .get_mut(&column_index)
            .ok_or_else(|| vortex_err!("No pending list for column {}", column_index))?;

        // Get element dtype from the List dtype
        let elem_dtype = match &self.columns[column_index].dtype {
            DType::List(elem, _) => elem.as_ref().clone(),
            _ => vortex_bail!("Column {} is not a List type", column_index),
        };

        let array = build_array_from_raw_with_validity(&elem_dtype, data, null_map, num_elements)?;
        pending.elements = Some(array);
        Ok(())
    }

    /// Write string element data for a list column.
    pub fn list_write_element_string_column(
        &mut self,
        column_index: usize,
        data: *const u8,
        offsets: *const u64,
        num_elements: usize,
    ) -> VortexResult<()> {
        let pending = self
            .pending_lists
            .get_mut(&column_index)
            .ok_or_else(|| vortex_err!("No pending list for column {}", column_index))?;

        // Get element dtype from the List dtype to match nullability
        let elem_dtype = match &self.columns[column_index].dtype {
            DType::List(elem, _) => elem.as_ref().clone(),
            _ => DType::Utf8(Nullability::NonNullable),
        };

        // Build string array from offsets
        let offsets_slice = unsafe { std::slice::from_raw_parts(offsets, num_elements + 1) };
        let total_data_len = offsets_slice[num_elements] as usize;
        let data_slice = if total_data_len > 0 {
            unsafe { std::slice::from_raw_parts(data, total_data_len) }
        } else {
            &[]
        };

        let mut builder = VarBinViewBuilder::with_capacity(elem_dtype, num_elements);
        for i in 0..num_elements {
            let start = offsets_slice[i] as usize;
            let end = offsets_slice[i + 1] as usize;
            let s = std::str::from_utf8(&data_slice[start..end]).unwrap_or("");
            builder.append_value(s);
        }
        let array = builder.finish_into_varbinview().into_array();
        pending.elements = Some(array);
        Ok(())
    }

    /// Write nullable string element data for a list column.
    pub fn list_write_element_string_column_nullable(
        &mut self,
        column_index: usize,
        data: *const u8,
        offsets: *const u64,
        null_map: *const u8,
        num_elements: usize,
    ) -> VortexResult<()> {
        let pending = self
            .pending_lists
            .get_mut(&column_index)
            .ok_or_else(|| vortex_err!("No pending list for column {}", column_index))?;

        // Get element dtype from the List dtype to match nullability
        let elem_dtype = match &self.columns[column_index].dtype {
            DType::List(elem, _) => elem.as_ref().clone(),
            _ => DType::Utf8(Nullability::Nullable),
        };

        // Build string array from offsets with null map
        let offsets_slice = unsafe { std::slice::from_raw_parts(offsets, num_elements + 1) };
        let total_data_len = offsets_slice[num_elements] as usize;
        let data_slice = if total_data_len > 0 {
            unsafe { std::slice::from_raw_parts(data, total_data_len) }
        } else {
            &[]
        };

        let null_slice: Option<&[u8]> = if null_map.is_null() {
            None
        } else {
            Some(unsafe { std::slice::from_raw_parts(null_map, num_elements) })
        };

        let mut builder = VarBinViewBuilder::with_capacity(elem_dtype, num_elements);
        for i in 0..num_elements {
            let is_null = null_slice.map_or(false, |ns| ns[i] != 0);
            if is_null {
                builder.append_null();
            } else {
                let start = offsets_slice[i] as usize;
                let end = offsets_slice[i + 1] as usize;
                let s = std::str::from_utf8(&data_slice[start..end]).unwrap_or("");
                builder.append_value(s);
            }
        }
        let array = builder.finish_into_varbinview().into_array();
        pending.elements = Some(array);
        Ok(())
    }

    /// Finalize a list column: build the `ListViewArray` from offsets + elements.
    pub fn list_end(&mut self, column_index: usize) -> VortexResult<()> {
        let pending = self
            .pending_lists
            .remove(&column_index)
            .ok_or_else(|| vortex_err!("No pending list for column {}", column_index))?;

        let elements = pending
            .elements
            .ok_or_else(|| vortex_err!("List elements not written for column {}", column_index))?;

        // Build offsets and sizes arrays for ListViewArray
        // offsets[i] = start index of list i
        // sizes[i] = length of list i
        let num_rows = pending.num_rows;
        let mut lv_offsets: Vec<i64> = Vec::with_capacity(num_rows);
        let mut lv_sizes: Vec<i64> = Vec::with_capacity(num_rows);

        for i in 0..num_rows {
            let start = pending.offsets[i] as i64;
            let end = pending.offsets[i + 1] as i64;
            lv_offsets.push(start);
            lv_sizes.push(end - start);
        }

        let offsets_array =
            PrimitiveArray::new(Buffer::<i64>::from(lv_offsets), Validity::NonNullable)
                .into_array();
        let sizes_array =
            PrimitiveArray::new(Buffer::<i64>::from(lv_sizes), Validity::NonNullable).into_array();

        let list_array =
            ListViewArray::try_new(elements, offsets_array, sizes_array, pending.validity)?;
        self.pending_columns[column_index] = Some(list_array.into_array());
        Ok(())
    }

    // =========================================================================
    // Struct (Tuple) column writing
    // =========================================================================

    /// Extract StructFields from a column dtype.
    /// Handles both direct `Struct(fields, _)` and `List(Struct(fields, _), _)` (Map case).
    fn get_struct_fields_from_dtype(dtype: &DType) -> VortexResult<&StructFields> {
        match dtype {
            DType::Struct(fields, _) => Ok(fields),
            DType::List(elem, _) => match elem.as_ref() {
                DType::Struct(fields, _) => Ok(fields),
                _ => vortex_bail!("List element is not a Struct type"),
            },
            _ => vortex_bail!("Column is not a Struct or Map type, got {:?}", dtype),
        }
    }

    /// Begin writing a struct column.
    pub fn struct_begin(
        &mut self,
        column_index: usize,
        null_map: *const u8,
        num_rows: usize,
    ) -> VortexResult<()> {
        if column_index >= self.columns.len() {
            vortex_bail!("Column index {} out of bounds", column_index);
        }

        let dtype = &self.columns[column_index].dtype;
        let num_fields = Self::get_struct_fields_from_dtype(dtype)?.nfields();

        let is_nullable = dtype.is_nullable();
        let validity = if is_nullable {
            null_map_to_validity(null_map, num_rows)
        } else {
            Validity::NonNullable
        };

        self.pending_structs.insert(
            column_index,
            PendingStruct {
                validity,
                num_rows,
                fields: vec![None; num_fields],
                num_fields,
            },
        );

        Ok(())
    }

    /// Write a primitive field of a struct column.
    pub fn struct_write_field(
        &mut self,
        column_index: usize,
        field_index: usize,
        data: *const c_void,
        null_map: *const u8,
        num_rows: usize,
    ) -> VortexResult<()> {
        let pending = self
            .pending_structs
            .get_mut(&column_index)
            .ok_or_else(|| vortex_err!("No pending struct for column {}", column_index))?;

        if field_index >= pending.num_fields {
            vortex_bail!(
                "Field index {} out of bounds (struct has {} fields)",
                field_index,
                pending.num_fields
            );
        }

        // Get field dtype
        let field_dtype = Self::get_struct_fields_from_dtype(&self.columns[column_index].dtype)?
            .field_by_index(field_index)
            .unwrap()
            .clone();

        let array = build_array_from_raw_with_validity(&field_dtype, data, null_map, num_rows);
        pending.fields[field_index] = Some(array?);
        Ok(())
    }

    /// Write a string field of a struct column.
    pub fn struct_write_field_string(
        &mut self,
        column_index: usize,
        field_index: usize,
        data: *const u8,
        offsets: *const u64,
        null_map: *const u8,
        num_rows: usize,
    ) -> VortexResult<()> {
        let pending = self
            .pending_structs
            .get_mut(&column_index)
            .ok_or_else(|| vortex_err!("No pending struct for column {}", column_index))?;

        if field_index >= pending.num_fields {
            vortex_bail!(
                "Field index {} out of bounds (struct has {} fields)",
                field_index,
                pending.num_fields
            );
        }

        // Get field dtype
        let field_dtype = Self::get_struct_fields_from_dtype(&self.columns[column_index].dtype)?
            .field_by_index(field_index)
            .unwrap()
            .clone();

        let is_nullable = field_dtype.is_nullable();

        // Build string array
        let offsets_slice = unsafe { std::slice::from_raw_parts(offsets, num_rows + 1) };
        let total_data_len = offsets_slice[num_rows] as usize;
        let data_slice = if total_data_len > 0 {
            unsafe { std::slice::from_raw_parts(data, total_data_len) }
        } else {
            &[]
        };

        let array = if is_nullable && !null_map.is_null() {
            let null_slice = unsafe { std::slice::from_raw_parts(null_map, num_rows) };
            let strings: Vec<Option<&str>> = (0..num_rows)
                .map(|i| {
                    if null_slice[i] != 0 {
                        None
                    } else {
                        let start = offsets_slice[i] as usize;
                        let end = offsets_slice[i + 1] as usize;
                        Some(std::str::from_utf8(&data_slice[start..end]).unwrap_or(""))
                    }
                })
                .collect();
            VarBinViewArray::from_iter_nullable_str(strings).into_array()
        } else {
            let strings: Vec<&str> = (0..num_rows)
                .map(|i| {
                    let start = offsets_slice[i] as usize;
                    let end = offsets_slice[i + 1] as usize;
                    std::str::from_utf8(&data_slice[start..end]).unwrap_or("")
                })
                .collect();
            VarBinViewArray::from_iter_str(strings).into_array()
        };

        pending.fields[field_index] = Some(array);
        Ok(())
    }

    /// Finalize a struct column: build the `StructArray` from fields.
    pub fn struct_end(&mut self, column_index: usize) -> VortexResult<()> {
        let pending = self
            .pending_structs
            .remove(&column_index)
            .ok_or_else(|| vortex_err!("No pending struct for column {}", column_index))?;

        // Get field names from the dtype
        let field_names = {
            let fields = Self::get_struct_fields_from_dtype(&self.columns[column_index].dtype)?;
            let names: Vec<Arc<str>> = fields
                .names()
                .iter()
                .map(|n| Arc::from(n.as_ref()))
                .collect();
            FieldNames::from(names)
        };

        // Check all fields are written
        let mut field_arrays = Vec::with_capacity(pending.num_fields);
        for (i, field) in pending.fields.into_iter().enumerate() {
            match field {
                Some(array) => field_arrays.push(array),
                None => vortex_bail!("Struct field {} not written for column {}", i, column_index),
            }
        }

        let struct_array = StructArray::try_new(
            field_names,
            field_arrays,
            pending.num_rows,
            pending.validity,
        )?;

        // For Map columns (DType::List(Struct(...), _)), the struct is the list element,
        // so store it in the pending list's elements rather than directly as the column.
        if matches!(&self.columns[column_index].dtype, DType::List(_, _)) {
            if let Some(pending_list) = self.pending_lists.get_mut(&column_index) {
                pending_list.elements = Some(struct_array.into_array());
            } else {
                vortex_bail!(
                    "No pending list for Map column {}; list_write_offsets must be called before struct_begin",
                    column_index
                );
            }
        } else {
            self.pending_columns[column_index] = Some(struct_array.into_array());
        }
        Ok(())
    }

    /// Write a batch of data (simplified API for primitive-only columns).
    ///
    /// This method takes raw column data and constructs a Vortex struct array.
    pub fn write_batch(
        &mut self,
        column_data: &[*const c_void],
        num_rows: usize,
    ) -> VortexResult<()> {
        if matches!(self.state, WriterState::Finalized) {
            vortex_bail!("Cannot write after finalization");
        }

        if column_data.len() != self.columns.len() {
            vortex_bail!(
                "Column count mismatch: expected {}, got {}",
                self.columns.len(),
                column_data.len()
            );
        }

        if num_rows == 0 {
            return Ok(());
        }

        // Build arrays for each column
        let mut field_arrays: Vec<ArrayRef> = Vec::with_capacity(self.columns.len());

        for (i, col_def) in self.columns.iter().enumerate() {
            let data_ptr = column_data[i];
            if data_ptr.is_null() {
                vortex_bail!("Column {} data pointer is null", i);
            }

            let array = build_array_from_raw(&col_def.dtype, data_ptr, num_rows)?;
            field_arrays.push(array);
        }

        // Create struct array
        let field_names: Vec<Arc<str>> = self
            .columns
            .iter()
            .map(|c| Arc::from(c.name.as_str()))
            .collect();

        let struct_array = StructArray::try_new(
            FieldNames::from(field_names),
            field_arrays,
            num_rows,
            Validity::NonNullable,
        )?;

        // Send the batch to the background writer
        self.send_array(struct_array.into_array())?;
        self.total_rows += num_rows;

        Ok(())
    }

    /// Finalize the writer and flush all data to disk.
    ///
    /// This closes the channel and waits for the background writer to complete.
    pub fn finalize(&mut self) -> VortexResult<()> {
        // Commit any pending batch
        if !self.pending_columns.is_empty() {
            self.end_batch()?;
        }

        // Take ownership of the state
        let state = std::mem::replace(&mut self.state, WriterState::Finalized);

        match state {
            WriterState::Building => {
                // No data was written, nothing to finalize
                if self.total_rows == 0 {
                    vortex_bail!("No data to write");
                }
                Ok(())
            }
            WriterState::Active {
                sender,
                writer_task,
            } => {
                // Close the sender to signal end of stream
                drop(sender);

                // Wait for the writer task to complete
                RUNTIME.block_on(async {
                    writer_task
                        .await
                        .map_err(|e| vortex_err!("Write failed: {}", e))?;
                    Ok(())
                })
            }
            WriterState::Finalized => {
                // Already finalized, nothing to do
                Ok(())
            }
        }
    }

    /// Get the output path.
    pub fn output_path(&self) -> &str {
        &self.output_path
    }

    /// Get the total number of rows written.
    pub fn total_rows(&self) -> usize {
        self.total_rows
    }
}

/// Convert ClickHouse null map (0=valid, 1=null) to Vortex Validity.
///
/// ClickHouse uses UInt8 per row where 0 means not null, 1 means null.
/// Vortex uses a bitmask where 1 means valid, 0 means null.
fn null_map_to_validity(null_map: *const u8, num_rows: usize) -> Validity {
    if null_map.is_null() || num_rows == 0 {
        return Validity::AllValid;
    }

    let null_slice = unsafe { std::slice::from_raw_parts(null_map, num_rows) };

    // Check if all valid (all zeros) or all invalid (all ones)
    let null_count = null_slice.iter().filter(|&&v| v != 0).count();

    if null_count == 0 {
        return Validity::AllValid;
    }
    if null_count == num_rows {
        return Validity::AllInvalid;
    }

    // Build validity bitmap (invert ClickHouse's null map)
    // Vortex validity: 1 = valid, 0 = null
    // ClickHouse null map: 0 = valid, 1 = null
    let bitmap_bytes = (num_rows + 7) / 8;
    let mut validity_bitmap = vec![0u8; bitmap_bytes];

    for (i, &is_null) in null_slice.iter().enumerate() {
        if is_null == 0 {
            // Valid - set bit to 1
            let byte_idx = i / 8;
            let bit_idx = i % 8;
            validity_bitmap[byte_idx] |= 1 << bit_idx;
        }
    }

    // Create a BoolArray for validity
    use vortex::array::arrays::BoolArray;
    use vortex::buffer::{BitBuffer, ByteBuffer};

    let byte_buffer = ByteBuffer::from(validity_bitmap);
    let bit_buffer = BitBuffer::new(byte_buffer, num_rows);
    let bool_array = BoolArray::from(bit_buffer).into_array();

    Validity::Array(bool_array)
}

/// Build a Vortex array from raw data pointer (non-nullable).
fn build_array_from_raw(
    dtype: &DType,
    data: *const c_void,
    num_rows: usize,
) -> VortexResult<ArrayRef> {
    build_array_from_raw_with_validity(dtype, data, ptr::null(), num_rows)
}

/// Build a Vortex array from raw data pointer with optional validity.
fn build_array_from_raw_with_validity(
    dtype: &DType,
    data: *const c_void,
    null_map: *const u8,
    num_rows: usize,
) -> VortexResult<ArrayRef> {
    match dtype {
        DType::Primitive(ptype, nullability) => {
            let validity = if *nullability == Nullability::Nullable {
                null_map_to_validity(null_map, num_rows)
            } else {
                Validity::NonNullable
            };

            macro_rules! build_primitive {
                ($rust_ty:ty) => {{
                    let slice =
                        unsafe { std::slice::from_raw_parts(data as *const $rust_ty, num_rows) };
                    let buffer: Buffer<$rust_ty> = slice.to_vec().into();
                    PrimitiveArray::new(buffer, validity).into_array()
                }};
            }

            let array = match ptype {
                PType::I8 => build_primitive!(i8),
                PType::I16 => build_primitive!(i16),
                PType::I32 => build_primitive!(i32),
                PType::I64 => build_primitive!(i64),
                PType::U8 => build_primitive!(u8),
                PType::U16 => build_primitive!(u16),
                PType::U32 => build_primitive!(u32),
                PType::U64 => build_primitive!(u64),
                PType::F32 => build_primitive!(f32),
                PType::F64 => build_primitive!(f64),
                PType::F16 => vortex_bail!("F16 not supported"),
            };

            Ok(array)
        }
        DType::Utf8(_) | DType::Binary(_) => {
            // For strings, we need a different approach
            // The data should be an array of string pointers and lengths
            // For now, return an error
            vortex_bail!("String columns must be written using vortex_writer_write_string_column")
        }
        DType::Bool(nullability) => {
            use vortex::array::arrays::BoolArray;
            use vortex::buffer::BitBuffer;

            // Bool is stored as u8 in ClickHouse (0 = false, 1 = true)
            let slice = unsafe { std::slice::from_raw_parts(data as *const u8, num_rows) };

            let validity = if *nullability == Nullability::Nullable {
                null_map_to_validity(null_map, num_rows)
            } else {
                Validity::NonNullable
            };

            let bits = BitBuffer::from_iter(slice.iter().map(|&v| v != 0));
            Ok(BoolArray::new(bits, validity).into_array())
        }
        DType::Decimal(decimal_dtype, nullability) => {
            let validity = if *nullability == Nullability::Nullable {
                null_map_to_validity(null_map, num_rows)
            } else {
                Validity::NonNullable
            };

            // ClickHouse always sends data in its fixed-width storage format:
            //   precision 1-9   -> Int32  (4 bytes, Decimal32)
            //   precision 10-18 -> Int64  (8 bytes, Decimal64)
            //   precision 19-38 -> Int128 (16 bytes, Decimal128)
            //   precision 39-76 -> Int256 (32 bytes, Decimal256)
            // We must use the ClickHouse storage type (not the smallest possible type)
            // to correctly interpret the raw bytes from ClickHouse.
            let values_type =
                crate::exporter::decimal::clickhouse_decimal_type(decimal_dtype.precision());

            let array = build_decimal_array(data, num_rows, *decimal_dtype, values_type, validity)?;
            Ok(array)
        }
        DType::FixedSizeList(elem_dtype, size, nullability) => {
            // Verify element type is u8 (for big integers)
            if !matches!(elem_dtype.as_ref(), DType::Primitive(PType::U8, _)) {
                vortex_bail!(
                    "Only FixedSizeList<u8, N> is supported for raw write, got FixedSizeList<{:?}, {}>",
                    elem_dtype,
                    size
                );
            }

            // Only support 16 (Int128/UInt128) and 32 (Int256/UInt256) byte sizes
            if *size != 16 && *size != 32 {
                vortex_bail!(
                    "Only FixedSizeList with size 16 or 32 is supported, got {}",
                    size
                );
            }

            let validity = if *nullability == Nullability::Nullable {
                null_map_to_validity(null_map, num_rows)
            } else {
                Validity::NonNullable
            };

            // Build the flat byte array (num_rows * size bytes)
            let size_usize = *size as usize;
            let total_bytes = num_rows * size_usize;
            let slice = unsafe { std::slice::from_raw_parts(data as *const u8, total_bytes) };
            let buffer: Buffer<u8> = slice.to_vec().into();
            let values = PrimitiveArray::new(buffer, Validity::NonNullable);

            // Create the FixedSizeListArray (needs 4 args: elements, list_size, validity, len)
            let array =
                FixedSizeListArray::try_new(values.into_array(), *size, validity, num_rows)?;
            Ok(array.into_array())
        }
        DType::Extension(ext) => {
            // Build the storage array from raw data, then wrap in ExtensionArray
            use vortex::array::arrays::ExtensionArray;
            let storage_array =
                build_array_from_raw_with_validity(ext.storage_dtype(), data, null_map, num_rows)?;
            Ok(ExtensionArray::new(ext.clone(), storage_array).into_array())
        }
        _ => vortex_bail!("Unsupported dtype for raw write: {:?}", dtype),
    }
}

/// Build a DecimalArray from raw data pointer.
fn build_decimal_array(
    data: *const c_void,
    num_rows: usize,
    decimal_dtype: DecimalDType,
    values_type: DecimalType,
    validity: Validity,
) -> VortexResult<ArrayRef> {
    match values_type {
        DecimalType::I8 => {
            let slice = unsafe { std::slice::from_raw_parts(data as *const i8, num_rows) };
            let buffer: Buffer<i8> = slice.to_vec().into();
            Ok(DecimalArray::new(buffer, decimal_dtype, validity).into_array())
        }
        DecimalType::I16 => {
            let slice = unsafe { std::slice::from_raw_parts(data as *const i16, num_rows) };
            let buffer: Buffer<i16> = slice.to_vec().into();
            Ok(DecimalArray::new(buffer, decimal_dtype, validity).into_array())
        }
        DecimalType::I32 => {
            let slice = unsafe { std::slice::from_raw_parts(data as *const i32, num_rows) };
            let buffer: Buffer<i32> = slice.to_vec().into();
            Ok(DecimalArray::new(buffer, decimal_dtype, validity).into_array())
        }
        DecimalType::I64 => {
            let slice = unsafe { std::slice::from_raw_parts(data as *const i64, num_rows) };
            let buffer: Buffer<i64> = slice.to_vec().into();
            Ok(DecimalArray::new(buffer, decimal_dtype, validity).into_array())
        }
        DecimalType::I128 => {
            let slice = unsafe { std::slice::from_raw_parts(data as *const i128, num_rows) };
            let buffer: Buffer<i128> = slice.to_vec().into();
            Ok(DecimalArray::new(buffer, decimal_dtype, validity).into_array())
        }
        DecimalType::I256 => {
            use vortex::dtype::i256;
            let slice = unsafe { std::slice::from_raw_parts(data as *const i256, num_rows) };
            let buffer: Buffer<i256> = slice.to_vec().into();
            Ok(DecimalArray::new(buffer, decimal_dtype, validity).into_array())
        }
    }
}

// =============================================================================
// FFI Exports for C++
// =============================================================================

/// Create a new Vortex writer.
///
/// # Safety
/// The `path` parameter must be a valid null-terminated C string.
/// Returns NULL on error. Call `vortex_get_last_error()` for error details.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn vortex_writer_new(path: *const c_char) -> *mut VortexWriter {
    clear_last_error();

    if path.is_null() {
        set_last_error("vortex_writer_new: path is null");
        return ptr::null_mut();
    }

    let path_str = match unsafe { CStr::from_ptr(path) }.to_str() {
        Ok(s) => s,
        Err(e) => {
            set_last_error(&format!("vortex_writer_new: invalid UTF-8 in path: {}", e));
            return ptr::null_mut();
        }
    };

    match VortexWriter::new(path_str) {
        Ok(writer) => Box::into_raw(Box::new(writer)),
        Err(e) => {
            set_last_error(&format!("vortex_writer_new: {}", e));
            ptr::null_mut()
        }
    }
}

/// Free a Vortex writer.
///
/// # Safety
/// The `writer` parameter must be a valid pointer returned by `vortex_writer_new`,
/// or NULL (which is safely ignored).
#[unsafe(no_mangle)]
pub unsafe extern "C" fn vortex_writer_free(writer: *mut VortexWriter) {
    if !writer.is_null() {
        drop(unsafe { Box::from_raw(writer) });
    }
}

/// Add a column to the writer's schema.
///
/// # Safety
/// - The `writer` parameter must be a valid pointer.
/// - The `name` and `clickhouse_type` must be valid null-terminated C strings.
/// Returns 0 on success, negative error code on failure.
/// Call `vortex_get_last_error()` for error details.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn vortex_writer_add_column(
    writer: *mut VortexWriter,
    name: *const c_char,
    clickhouse_type: *const c_char,
    nullable: i32,
) -> i32 {
    clear_last_error();

    if writer.is_null() {
        set_last_error("vortex_writer_add_column: writer is null");
        return -1;
    }
    if name.is_null() {
        set_last_error("vortex_writer_add_column: name is null");
        return -1;
    }
    if clickhouse_type.is_null() {
        set_last_error("vortex_writer_add_column: clickhouse_type is null");
        return -1;
    }

    let writer = unsafe { &mut *writer };

    let name_str = match unsafe { CStr::from_ptr(name) }.to_str() {
        Ok(s) => s,
        Err(e) => {
            set_last_error(&format!(
                "vortex_writer_add_column: invalid UTF-8 in name: {}",
                e
            ));
            return -2;
        }
    };

    let type_str = match unsafe { CStr::from_ptr(clickhouse_type) }.to_str() {
        Ok(s) => s,
        Err(e) => {
            set_last_error(&format!(
                "vortex_writer_add_column: invalid UTF-8 in clickhouse_type: {}",
                e
            ));
            return -3;
        }
    };

    match writer.add_column(name_str, type_str, nullable != 0) {
        Ok(()) => 0,
        Err(e) => {
            set_last_error(&format!("vortex_writer_add_column: {}", e));
            -4
        }
    }
}

/// Begin writing a new batch with the given number of rows.
///
/// After calling this, use `vortex_writer_write_column_*` functions to write each column,
/// then call `vortex_writer_end_batch()` to commit the batch.
///
/// # Safety
/// The `writer` parameter must be a valid pointer.
/// Returns 0 on success, negative error code on failure.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn vortex_writer_begin_batch(
    writer: *mut VortexWriter,
    num_rows: usize,
) -> i32 {
    clear_last_error();

    if writer.is_null() {
        set_last_error("vortex_writer_begin_batch: writer is null");
        return -1;
    }

    let writer = unsafe { &mut *writer };

    match writer.begin_batch(num_rows) {
        Ok(()) => 0,
        Err(e) => {
            set_last_error(&format!("vortex_writer_begin_batch: {}", e));
            -2
        }
    }
}

/// Write a primitive column by index.
///
/// Must be called between `vortex_writer_begin_batch()` and `vortex_writer_end_batch()`.
///
/// # Safety
/// - The `writer` parameter must be a valid pointer.
/// - The `data` must point to an array of `num_rows` elements of the appropriate type.
/// Returns 0 on success, negative error code on failure.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn vortex_writer_write_column(
    writer: *mut VortexWriter,
    column_index: usize,
    data: *const c_void,
    num_rows: usize,
) -> i32 {
    clear_last_error();

    if writer.is_null() {
        set_last_error("vortex_writer_write_column: writer is null");
        return -1;
    }
    if data.is_null() {
        set_last_error("vortex_writer_write_column: data is null");
        return -1;
    }

    let writer = unsafe { &mut *writer };

    match writer.write_column_primitive(column_index, data, num_rows) {
        Ok(()) => 0,
        Err(e) => {
            set_last_error(&format!("vortex_writer_write_column: {}", e));
            -2
        }
    }
}

/// Write a nullable primitive column by index with validity bitmap.
///
/// The null_map uses ClickHouse's convention: one byte per row, 0 = valid, 1 = null.
/// Must be called between `vortex_writer_begin_batch()` and `vortex_writer_end_batch()`.
///
/// # Safety
/// - The `writer` parameter must be a valid pointer.
/// - The `data` must point to an array of `num_rows` elements of the appropriate type.
/// - The `null_map` must point to an array of `num_rows` bytes, or NULL for all-valid.
/// Returns 0 on success, negative error code on failure.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn vortex_writer_write_column_nullable(
    writer: *mut VortexWriter,
    column_index: usize,
    data: *const c_void,
    null_map: *const u8,
    num_rows: usize,
) -> i32 {
    clear_last_error();

    if writer.is_null() {
        set_last_error("vortex_writer_write_column_nullable: writer is null");
        return -1;
    }
    if data.is_null() {
        set_last_error("vortex_writer_write_column_nullable: data is null");
        return -1;
    }

    let writer = unsafe { &mut *writer };

    match writer.write_column_primitive_with_validity(column_index, data, null_map, num_rows) {
        Ok(()) => 0,
        Err(e) => {
            set_last_error(&format!("vortex_writer_write_column_nullable: {}", e));
            -2
        }
    }
}

/// Write a string column by index.
///
/// The strings are provided as concatenated data with offsets.
/// The `offsets` array must have `num_rows + 1` elements, where:
/// - `offsets[i]` is the start offset of string i
/// - `offsets[num_rows]` is the total data length
///
/// Must be called between `vortex_writer_begin_batch()` and `vortex_writer_end_batch()`.
///
/// # Safety
/// - The `writer` parameter must be a valid pointer.
/// - The `data` must point to the concatenated string data.
/// - The `offsets` must point to an array of `num_rows + 1` uint64_t values.
/// Returns 0 on success, negative error code on failure.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn vortex_writer_write_string_column(
    writer: *mut VortexWriter,
    column_index: usize,
    data: *const u8,
    offsets: *const u64,
    num_rows: usize,
) -> i32 {
    clear_last_error();

    if writer.is_null() {
        set_last_error("vortex_writer_write_string_column: writer is null");
        return -1;
    }
    if data.is_null() && num_rows > 0 {
        set_last_error("vortex_writer_write_string_column: data is null");
        return -1;
    }
    if offsets.is_null() {
        set_last_error("vortex_writer_write_string_column: offsets is null");
        return -1;
    }

    let writer = unsafe { &mut *writer };

    match writer.write_column_strings(column_index, data, offsets, num_rows) {
        Ok(()) => 0,
        Err(e) => {
            set_last_error(&format!("vortex_writer_write_string_column: {}", e));
            -2
        }
    }
}

/// Write a nullable string column by index with validity bitmap.
///
/// The strings are provided as concatenated data with offsets.
/// The `null_map` uses ClickHouse's convention: one byte per row, 0 = valid, 1 = null.
///
/// Must be called between `vortex_writer_begin_batch()` and `vortex_writer_end_batch()`.
///
/// # Safety
/// - The `writer` parameter must be a valid pointer.
/// - The `data` must point to the concatenated string data.
/// - The `offsets` must point to an array of `num_rows + 1` uint64_t values.
/// - The `null_map` must point to an array of `num_rows` bytes, or NULL for all-valid.
/// Returns 0 on success, negative error code on failure.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn vortex_writer_write_string_column_nullable(
    writer: *mut VortexWriter,
    column_index: usize,
    data: *const u8,
    offsets: *const u64,
    null_map: *const u8,
    num_rows: usize,
) -> i32 {
    clear_last_error();

    if writer.is_null() {
        set_last_error("vortex_writer_write_string_column_nullable: writer is null");
        return -1;
    }
    if offsets.is_null() {
        set_last_error("vortex_writer_write_string_column_nullable: offsets is null");
        return -1;
    }

    let writer = unsafe { &mut *writer };

    match writer.write_column_strings_with_validity(column_index, data, offsets, null_map, num_rows)
    {
        Ok(()) => 0,
        Err(e) => {
            set_last_error(&format!(
                "vortex_writer_write_string_column_nullable: {}",
                e
            ));
            -2
        }
    }
}

/// End the current batch and commit it.
///
/// # Safety
/// The `writer` parameter must be a valid pointer.
/// Returns 0 on success, negative error code on failure.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn vortex_writer_end_batch(writer: *mut VortexWriter) -> i32 {
    clear_last_error();

    if writer.is_null() {
        set_last_error("vortex_writer_end_batch: writer is null");
        return -1;
    }

    let writer = unsafe { &mut *writer };

    match writer.end_batch() {
        Ok(()) => 0,
        Err(e) => {
            set_last_error(&format!("vortex_writer_end_batch: {}", e));
            -2
        }
    }
}

// =============================================================================
// List (Array) Column FFI
// =============================================================================

/// Write offsets for a list (Array) column.
///
/// # Safety
/// All pointers must be valid. `offsets` must have `num_rows + 1` elements.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn vortex_writer_list_write_offsets(
    writer: *mut VortexWriter,
    column_index: usize,
    offsets: *const u64,
    null_map: *const u8,
    num_rows: usize,
) -> i32 {
    clear_last_error();
    if writer.is_null() || offsets.is_null() {
        set_last_error("vortex_writer_list_write_offsets: null pointer");
        return -1;
    }
    let writer = unsafe { &mut *writer };
    match writer.list_write_offsets(column_index, offsets, null_map, num_rows) {
        Ok(()) => 0,
        Err(e) => {
            set_last_error(&format!("vortex_writer_list_write_offsets: {}", e));
            -2
        }
    }
}

/// Write primitive element data for a list column.
///
/// # Safety
/// All pointers must be valid.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn vortex_writer_list_write_element_column(
    writer: *mut VortexWriter,
    column_index: usize,
    data: *const c_void,
    num_elements: usize,
) -> i32 {
    clear_last_error();
    if writer.is_null() {
        set_last_error("vortex_writer_list_write_element_column: null pointer");
        return -1;
    }
    let writer = unsafe { &mut *writer };
    match writer.list_write_element_column(column_index, data, num_elements) {
        Ok(()) => 0,
        Err(e) => {
            set_last_error(&format!("vortex_writer_list_write_element_column: {}", e));
            -2
        }
    }
}

/// Write string element data for a list column.
///
/// # Safety
/// All pointers must be valid.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn vortex_writer_list_write_element_string_column(
    writer: *mut VortexWriter,
    column_index: usize,
    data: *const u8,
    offsets: *const u64,
    num_elements: usize,
) -> i32 {
    clear_last_error();
    if writer.is_null() || offsets.is_null() {
        set_last_error("vortex_writer_list_write_element_string_column: null pointer");
        return -1;
    }
    let writer = unsafe { &mut *writer };
    match writer.list_write_element_string_column(column_index, data, offsets, num_elements) {
        Ok(()) => 0,
        Err(e) => {
            set_last_error(&format!(
                "vortex_writer_list_write_element_string_column: {}",
                e
            ));
            -2
        }
    }
}

/// Write nullable primitive element data for a list column.
///
/// # Safety
/// All pointers must be valid.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn vortex_writer_list_write_element_column_nullable(
    writer: *mut VortexWriter,
    column_index: usize,
    data: *const c_void,
    null_map: *const u8,
    num_elements: usize,
) -> i32 {
    clear_last_error();
    if writer.is_null() {
        set_last_error("vortex_writer_list_write_element_column_nullable: null pointer");
        return -1;
    }
    let writer = unsafe { &mut *writer };
    match writer.list_write_element_column_nullable(column_index, data, null_map, num_elements) {
        Ok(()) => 0,
        Err(e) => {
            set_last_error(&format!(
                "vortex_writer_list_write_element_column_nullable: {}",
                e
            ));
            -2
        }
    }
}

/// Write nullable string element data for a list column.
///
/// # Safety
/// All pointers must be valid.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn vortex_writer_list_write_element_string_column_nullable(
    writer: *mut VortexWriter,
    column_index: usize,
    data: *const u8,
    offsets: *const u64,
    null_map: *const u8,
    num_elements: usize,
) -> i32 {
    clear_last_error();
    if writer.is_null() || offsets.is_null() {
        set_last_error("vortex_writer_list_write_element_string_column_nullable: null pointer");
        return -1;
    }
    let writer = unsafe { &mut *writer };
    match writer.list_write_element_string_column_nullable(
        column_index,
        data,
        offsets,
        null_map,
        num_elements,
    ) {
        Ok(()) => 0,
        Err(e) => {
            set_last_error(&format!(
                "vortex_writer_list_write_element_string_column_nullable: {}",
                e
            ));
            -2
        }
    }
}

/// Finalize a list column.
///
/// # Safety
/// The `writer` parameter must be a valid pointer.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn vortex_writer_list_end(
    writer: *mut VortexWriter,
    column_index: usize,
) -> i32 {
    clear_last_error();
    if writer.is_null() {
        set_last_error("vortex_writer_list_end: null pointer");
        return -1;
    }
    let writer = unsafe { &mut *writer };
    match writer.list_end(column_index) {
        Ok(()) => 0,
        Err(e) => {
            set_last_error(&format!("vortex_writer_list_end: {}", e));
            -2
        }
    }
}

// =============================================================================
// Struct (Tuple) Column FFI
// =============================================================================

/// Begin writing a struct (Tuple) column.
///
/// # Safety
/// The `writer` parameter must be a valid pointer.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn vortex_writer_struct_begin(
    writer: *mut VortexWriter,
    column_index: usize,
    null_map: *const u8,
    num_rows: usize,
) -> i32 {
    clear_last_error();
    if writer.is_null() {
        set_last_error("vortex_writer_struct_begin: null pointer");
        return -1;
    }
    let writer = unsafe { &mut *writer };
    match writer.struct_begin(column_index, null_map, num_rows) {
        Ok(()) => 0,
        Err(e) => {
            set_last_error(&format!("vortex_writer_struct_begin: {}", e));
            -2
        }
    }
}

/// Write a primitive field of a struct column.
///
/// # Safety
/// All pointers must be valid.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn vortex_writer_struct_write_field(
    writer: *mut VortexWriter,
    column_index: usize,
    field_index: usize,
    data: *const c_void,
    null_map: *const u8,
    num_rows: usize,
) -> i32 {
    clear_last_error();
    if writer.is_null() {
        set_last_error("vortex_writer_struct_write_field: null pointer");
        return -1;
    }
    let writer = unsafe { &mut *writer };
    match writer.struct_write_field(column_index, field_index, data, null_map, num_rows) {
        Ok(()) => 0,
        Err(e) => {
            set_last_error(&format!("vortex_writer_struct_write_field: {}", e));
            -2
        }
    }
}

/// Write a string field of a struct column.
///
/// # Safety
/// All pointers must be valid.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn vortex_writer_struct_write_field_string(
    writer: *mut VortexWriter,
    column_index: usize,
    field_index: usize,
    data: *const u8,
    offsets: *const u64,
    null_map: *const u8,
    num_rows: usize,
) -> i32 {
    clear_last_error();
    if writer.is_null() || offsets.is_null() {
        set_last_error("vortex_writer_struct_write_field_string: null pointer");
        return -1;
    }
    let writer = unsafe { &mut *writer };
    match writer.struct_write_field_string(
        column_index,
        field_index,
        data,
        offsets,
        null_map,
        num_rows,
    ) {
        Ok(()) => 0,
        Err(e) => {
            set_last_error(&format!("vortex_writer_struct_write_field_string: {}", e));
            -2
        }
    }
}

/// Finalize a struct column.
///
/// # Safety
/// The `writer` parameter must be a valid pointer.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn vortex_writer_struct_end(
    writer: *mut VortexWriter,
    column_index: usize,
) -> i32 {
    clear_last_error();
    if writer.is_null() {
        set_last_error("vortex_writer_struct_end: null pointer");
        return -1;
    }
    let writer = unsafe { &mut *writer };
    match writer.struct_end(column_index) {
        Ok(()) => 0,
        Err(e) => {
            set_last_error(&format!("vortex_writer_struct_end: {}", e));
            -2
        }
    }
}

/// Write a batch of data (simplified API for primitive-only columns).
///
/// # Safety
/// - The `writer` parameter must be a valid pointer.
/// - The `data` array must contain `num_columns` valid pointers to column data.
/// - Each column data pointer must point to an array of `num_rows` elements.
/// Returns 0 on success, negative error code on failure.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn vortex_writer_write_batch(
    writer: *mut VortexWriter,
    data: *const *const c_void,
    num_columns: usize,
    num_rows: usize,
) -> i32 {
    clear_last_error();

    if writer.is_null() {
        set_last_error("vortex_writer_write_batch: writer is null");
        return -1;
    }
    if data.is_null() && num_columns > 0 {
        set_last_error("vortex_writer_write_batch: data is null");
        return -1;
    }

    let writer = unsafe { &mut *writer };

    let column_data = if num_columns > 0 {
        unsafe { std::slice::from_raw_parts(data, num_columns) }
    } else {
        &[]
    };

    match writer.write_batch(column_data, num_rows) {
        Ok(()) => 0,
        Err(e) => {
            set_last_error(&format!("vortex_writer_write_batch: {}", e));
            -2
        }
    }
}

/// Finalize the writer and flush all data.
///
/// # Safety
/// The `writer` parameter must be a valid pointer.
/// Returns 0 on success, negative error code on failure.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn vortex_writer_finalize(writer: *mut VortexWriter) -> i32 {
    clear_last_error();

    if writer.is_null() {
        set_last_error("vortex_writer_finalize: writer is null");
        return -1;
    }

    let writer = unsafe { &mut *writer };

    match writer.finalize() {
        Ok(()) => 0,
        Err(e) => {
            set_last_error(&format!("vortex_writer_finalize: {}", e));
            -2
        }
    }
}

/// Get the number of columns in the writer's schema.
///
/// # Safety
/// The `writer` parameter must be a valid pointer.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn vortex_writer_num_columns(writer: *const VortexWriter) -> usize {
    if writer.is_null() {
        return 0;
    }
    unsafe { &*writer }.num_columns()
}

/// Get the total number of rows written.
///
/// # Safety
/// The `writer` parameter must be a valid pointer.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn vortex_writer_total_rows(writer: *const VortexWriter) -> usize {
    if writer.is_null() {
        return 0;
    }
    unsafe { &*writer }.total_rows()
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use std::ffi::CString;
    use tempfile::NamedTempFile;

    #[test]
    fn test_writer_new() {
        let temp_file = NamedTempFile::new().expect("Failed to create temp file");
        let path = temp_file.path().to_string_lossy().to_string();

        let writer = VortexWriter::new(&path).expect("Failed to create writer");
        assert_eq!(writer.output_path(), path);
        assert_eq!(writer.num_columns(), 0);
    }

    #[test]
    fn test_writer_new_empty_path() {
        let result = VortexWriter::new("");
        assert!(result.is_err());
    }

    #[test]
    fn test_writer_add_column() {
        let temp_file = NamedTempFile::new().expect("Failed to create temp file");
        let path = temp_file.path().to_string_lossy().to_string();

        let mut writer = VortexWriter::new(&path).expect("Failed to create writer");
        writer
            .add_column("id", "Int64", false)
            .expect("Failed to add column");
        writer
            .add_column("value", "Float64", false)
            .expect("Failed to add column");

        assert_eq!(writer.num_columns(), 2);
    }

    #[test]
    fn test_writer_ffi_new_free() {
        let temp_file = NamedTempFile::new().expect("Failed to create temp file");
        let path = temp_file.path().to_string_lossy().to_string();
        let c_path = CString::new(path).expect("CString failed");

        let writer = unsafe { vortex_writer_new(c_path.as_ptr()) };
        assert!(!writer.is_null());

        unsafe { vortex_writer_free(writer) };
    }

    #[test]
    fn test_writer_ffi_null_path() {
        let writer = unsafe { vortex_writer_new(ptr::null()) };
        assert!(writer.is_null());
    }

    #[test]
    fn test_writer_ffi_add_column() {
        let temp_file = NamedTempFile::new().expect("Failed to create temp file");
        let path = temp_file.path().to_string_lossy().to_string();
        let c_path = CString::new(path).expect("CString failed");

        let writer = unsafe { vortex_writer_new(c_path.as_ptr()) };
        assert!(!writer.is_null());

        let name = CString::new("id").unwrap();
        let ch_type = CString::new("Int64").unwrap();

        let result =
            unsafe { vortex_writer_add_column(writer, name.as_ptr(), ch_type.as_ptr(), 0) };
        assert_eq!(result, 0);

        let num_cols = unsafe { vortex_writer_num_columns(writer) };
        assert_eq!(num_cols, 1);

        unsafe { vortex_writer_free(writer) };
    }

    #[test]
    fn test_writer_write_and_finalize() {
        let temp_file = NamedTempFile::new().expect("Failed to create temp file");
        let path = temp_file.path().to_string_lossy().to_string();

        let mut writer = VortexWriter::new(&path).expect("Failed to create writer");
        writer
            .add_column("id", "Int64", false)
            .expect("Failed to add column");

        // Create test data
        let data: Vec<i64> = vec![1, 2, 3, 4, 5];
        let data_ptr = data.as_ptr() as *const c_void;

        writer
            .write_batch(&[data_ptr], 5)
            .expect("Failed to write batch");

        assert_eq!(writer.total_rows(), 5);

        writer.finalize().expect("Failed to finalize");

        // Verify the file was created
        assert!(std::path::Path::new(&path).exists());
    }

    #[test]
    fn test_writer_string_column() {
        let temp_file = NamedTempFile::new().expect("Failed to create temp file");
        let path = temp_file.path().to_string_lossy().to_string();

        let mut writer = VortexWriter::new(&path).expect("Failed to create writer");
        writer
            .add_column("id", "Int64", false)
            .expect("Failed to add column");
        writer
            .add_column("name", "String", false)
            .expect("Failed to add column");

        // Begin batch
        writer.begin_batch(3).expect("Failed to begin batch");

        // Write id column
        let ids: Vec<i64> = vec![1, 2, 3];
        writer
            .write_column_primitive(0, ids.as_ptr() as *const c_void, 3)
            .expect("Failed to write id column");

        // Write string column
        let strings = "AliceBobCharlie";
        let offsets: Vec<u64> = vec![0, 5, 8, 15]; // Alice(5), Bob(3), Charlie(7)
        writer
            .write_column_strings(1, strings.as_ptr(), offsets.as_ptr(), 3)
            .expect("Failed to write string column");

        // End batch
        writer.end_batch().expect("Failed to end batch");

        // Finalize
        writer.finalize().expect("Failed to finalize");

        // Verify the file was created
        assert!(std::path::Path::new(&path).exists());
    }

    #[test]
    fn test_writer_multiple_batches() {
        let temp_file = NamedTempFile::new().expect("Failed to create temp file");
        let path = temp_file.path().to_string_lossy().to_string();

        let mut writer = VortexWriter::new(&path).expect("Failed to create writer");
        writer
            .add_column("id", "Int64", false)
            .expect("Failed to add column");
        writer
            .add_column("value", "Float64", false)
            .expect("Failed to add column");

        // Write first batch (3 rows)
        let ids1: Vec<i64> = vec![1, 2, 3];
        let values1: Vec<f64> = vec![1.1, 2.2, 3.3];
        writer.begin_batch(3).expect("Failed to begin batch 1");
        writer
            .write_column_primitive(0, ids1.as_ptr() as *const c_void, 3)
            .expect("Failed to write id column");
        writer
            .write_column_primitive(1, values1.as_ptr() as *const c_void, 3)
            .expect("Failed to write value column");
        writer.end_batch().expect("Failed to end batch 1");

        // Write second batch (2 rows)
        let ids2: Vec<i64> = vec![4, 5];
        let values2: Vec<f64> = vec![4.4, 5.5];
        writer.begin_batch(2).expect("Failed to begin batch 2");
        writer
            .write_column_primitive(0, ids2.as_ptr() as *const c_void, 2)
            .expect("Failed to write id column");
        writer
            .write_column_primitive(1, values2.as_ptr() as *const c_void, 2)
            .expect("Failed to write value column");
        writer.end_batch().expect("Failed to end batch 2");

        // Write third batch (4 rows)
        let ids3: Vec<i64> = vec![6, 7, 8, 9];
        let values3: Vec<f64> = vec![6.6, 7.7, 8.8, 9.9];
        writer.begin_batch(4).expect("Failed to begin batch 3");
        writer
            .write_column_primitive(0, ids3.as_ptr() as *const c_void, 4)
            .expect("Failed to write id column");
        writer
            .write_column_primitive(1, values3.as_ptr() as *const c_void, 4)
            .expect("Failed to write value column");
        writer.end_batch().expect("Failed to end batch 3");

        // Verify total rows before finalize
        assert_eq!(writer.total_rows(), 9); // 3 + 2 + 4 = 9

        // Finalize
        writer.finalize().expect("Failed to finalize");

        // Verify the file was created
        assert!(std::path::Path::new(&path).exists());

        // Verify file is not empty
        let metadata = std::fs::metadata(&path).expect("Failed to get file metadata");
        assert!(metadata.len() > 0, "File should not be empty");
    }

    // =============================================================================
    // Edge Case Tests - Boundary Conditions
    // =============================================================================

    #[test]
    fn test_writer_empty_batch() {
        // Test writing a batch with 0 rows
        let temp_file = NamedTempFile::new().expect("Failed to create temp file");
        let path = temp_file.path().to_string_lossy().to_string();

        let mut writer = VortexWriter::new(&path).expect("Failed to create writer");
        writer
            .add_column("id", "Int64", false)
            .expect("Failed to add column");

        // Write empty batch (should be a no-op)
        let data: Vec<i64> = vec![];
        let data_ptr = data.as_ptr() as *const c_void;
        writer
            .write_batch(&[data_ptr], 0)
            .expect("Empty batch should succeed");

        assert_eq!(writer.total_rows(), 0);

        // Now write actual data
        let data2: Vec<i64> = vec![1, 2, 3];
        writer
            .write_batch(&[data2.as_ptr() as *const c_void], 3)
            .expect("Failed to write batch");

        writer.finalize().expect("Failed to finalize");
        assert!(std::path::Path::new(&path).exists());
    }

    #[test]
    fn test_writer_single_row_batch() {
        // Test writing a batch with exactly 1 row
        let temp_file = NamedTempFile::new().expect("Failed to create temp file");
        let path = temp_file.path().to_string_lossy().to_string();

        let mut writer = VortexWriter::new(&path).expect("Failed to create writer");
        writer
            .add_column("id", "Int64", false)
            .expect("Failed to add column");

        let data: Vec<i64> = vec![42];
        writer
            .write_batch(&[data.as_ptr() as *const c_void], 1)
            .expect("Failed to write single row batch");

        assert_eq!(writer.total_rows(), 1);
        writer.finalize().expect("Failed to finalize");
    }

    #[test]
    fn test_writer_large_batch() {
        // Test writing a large batch (stress test)
        let temp_file = NamedTempFile::new().expect("Failed to create temp file");
        let path = temp_file.path().to_string_lossy().to_string();

        let mut writer = VortexWriter::new(&path).expect("Failed to create writer");
        writer
            .add_column("id", "Int64", false)
            .expect("Failed to add column");

        // 1 million rows
        let num_rows = 1_000_000;
        let data: Vec<i64> = (0..num_rows as i64).collect();
        writer
            .write_batch(&[data.as_ptr() as *const c_void], num_rows)
            .expect("Failed to write large batch");

        assert_eq!(writer.total_rows(), num_rows);
        writer.finalize().expect("Failed to finalize");

        let metadata = std::fs::metadata(&path).expect("Failed to get file metadata");
        assert!(metadata.len() > 0, "File should not be empty");
    }

    #[test]
    fn test_writer_many_small_batches() {
        // Test writing many small batches (tests channel backpressure)
        let temp_file = NamedTempFile::new().expect("Failed to create temp file");
        let path = temp_file.path().to_string_lossy().to_string();

        let mut writer = VortexWriter::new(&path).expect("Failed to create writer");
        writer
            .add_column("id", "Int64", false)
            .expect("Failed to add column");

        // Write 100 batches of 10 rows each (exceeds channel buffer of 32)
        let num_batches = 100;
        let rows_per_batch = 10;

        for i in 0..num_batches {
            let start = (i * rows_per_batch) as i64;
            let data: Vec<i64> = (start..start + rows_per_batch as i64).collect();
            writer
                .write_batch(&[data.as_ptr() as *const c_void], rows_per_batch)
                .expect(&format!("Failed to write batch {}", i));
        }

        assert_eq!(writer.total_rows(), num_batches * rows_per_batch);
        writer.finalize().expect("Failed to finalize");
    }

    #[test]
    fn test_writer_finalize_no_data() {
        // Test finalizing without writing any data
        let temp_file = NamedTempFile::new().expect("Failed to create temp file");
        let path = temp_file.path().to_string_lossy().to_string();

        let mut writer = VortexWriter::new(&path).expect("Failed to create writer");
        writer
            .add_column("id", "Int64", false)
            .expect("Failed to add column");

        // Finalize without writing any data should fail
        let result = writer.finalize();
        assert!(result.is_err(), "Finalize with no data should fail");
    }

    #[test]
    fn test_writer_double_finalize() {
        // Test calling finalize twice
        let temp_file = NamedTempFile::new().expect("Failed to create temp file");
        let path = temp_file.path().to_string_lossy().to_string();

        let mut writer = VortexWriter::new(&path).expect("Failed to create writer");
        writer
            .add_column("id", "Int64", false)
            .expect("Failed to add column");

        let data: Vec<i64> = vec![1, 2, 3];
        writer
            .write_batch(&[data.as_ptr() as *const c_void], 3)
            .expect("Failed to write batch");

        writer.finalize().expect("First finalize should succeed");

        // Second finalize should succeed (idempotent)
        let result = writer.finalize();
        assert!(result.is_ok(), "Second finalize should be idempotent");
    }

    #[test]
    fn test_writer_write_after_finalize() {
        // Test writing after finalization
        let temp_file = NamedTempFile::new().expect("Failed to create temp file");
        let path = temp_file.path().to_string_lossy().to_string();

        let mut writer = VortexWriter::new(&path).expect("Failed to create writer");
        writer
            .add_column("id", "Int64", false)
            .expect("Failed to add column");

        let data: Vec<i64> = vec![1, 2, 3];
        writer
            .write_batch(&[data.as_ptr() as *const c_void], 3)
            .expect("Failed to write batch");

        writer.finalize().expect("Failed to finalize");

        // Writing after finalize should fail
        let result = writer.write_batch(&[data.as_ptr() as *const c_void], 3);
        assert!(result.is_err(), "Write after finalize should fail");
    }

    #[test]
    fn test_writer_add_column_after_write() {
        // Test adding column after data has been written
        let temp_file = NamedTempFile::new().expect("Failed to create temp file");
        let path = temp_file.path().to_string_lossy().to_string();

        let mut writer = VortexWriter::new(&path).expect("Failed to create writer");
        writer
            .add_column("id", "Int64", false)
            .expect("Failed to add column");

        let data: Vec<i64> = vec![1, 2, 3];
        writer
            .write_batch(&[data.as_ptr() as *const c_void], 3)
            .expect("Failed to write batch");

        // Adding column after writing should fail
        let result = writer.add_column("value", "Float64", false);
        assert!(result.is_err(), "Add column after writing should fail");
    }

    #[test]
    fn test_writer_column_count_mismatch() {
        // Test writing with wrong number of columns
        let temp_file = NamedTempFile::new().expect("Failed to create temp file");
        let path = temp_file.path().to_string_lossy().to_string();

        let mut writer = VortexWriter::new(&path).expect("Failed to create writer");
        writer
            .add_column("id", "Int64", false)
            .expect("Failed to add column");
        writer
            .add_column("value", "Float64", false)
            .expect("Failed to add column");

        // Try to write with only one column (should fail)
        let data: Vec<i64> = vec![1, 2, 3];
        let result = writer.write_batch(&[data.as_ptr() as *const c_void], 3);
        assert!(result.is_err(), "Column count mismatch should fail");
    }

    #[test]
    fn test_writer_incomplete_batch() {
        // Test begin_batch without end_batch, then finalize
        let temp_file = NamedTempFile::new().expect("Failed to create temp file");
        let path = temp_file.path().to_string_lossy().to_string();

        let mut writer = VortexWriter::new(&path).expect("Failed to create writer");
        writer
            .add_column("id", "Int64", false)
            .expect("Failed to add column");
        writer
            .add_column("name", "String", false)
            .expect("Failed to add column");

        // Begin batch but only write one column
        writer.begin_batch(3).expect("Failed to begin batch");

        let ids: Vec<i64> = vec![1, 2, 3];
        writer
            .write_column_primitive(0, ids.as_ptr() as *const c_void, 3)
            .expect("Failed to write id column");

        // Try to end batch without writing all columns - should fail
        let result = writer.end_batch();
        assert!(result.is_err(), "Incomplete batch should fail on end_batch");
    }

    #[test]
    fn test_writer_begin_batch_twice() {
        // Test calling begin_batch twice without end_batch
        let temp_file = NamedTempFile::new().expect("Failed to create temp file");
        let path = temp_file.path().to_string_lossy().to_string();

        let mut writer = VortexWriter::new(&path).expect("Failed to create writer");
        writer
            .add_column("id", "Int64", false)
            .expect("Failed to add column");

        writer
            .begin_batch(3)
            .expect("First begin_batch should succeed");

        // Second begin_batch without end_batch should fail
        let result = writer.begin_batch(3);
        assert!(result.is_err(), "Second begin_batch should fail");
    }

    #[test]
    fn test_writer_column_index_out_of_bounds() {
        // Test writing to invalid column index
        let temp_file = NamedTempFile::new().expect("Failed to create temp file");
        let path = temp_file.path().to_string_lossy().to_string();

        let mut writer = VortexWriter::new(&path).expect("Failed to create writer");
        writer
            .add_column("id", "Int64", false)
            .expect("Failed to add column");

        writer.begin_batch(3).expect("Failed to begin batch");

        let data: Vec<i64> = vec![1, 2, 3];
        // Try to write to column index 5 (only have 1 column)
        let result = writer.write_column_primitive(5, data.as_ptr() as *const c_void, 3);
        assert!(result.is_err(), "Out of bounds column index should fail");
    }

    #[test]
    fn test_writer_row_count_mismatch_in_batch() {
        // Test writing column with different row count than begin_batch specified
        let temp_file = NamedTempFile::new().expect("Failed to create temp file");
        let path = temp_file.path().to_string_lossy().to_string();

        let mut writer = VortexWriter::new(&path).expect("Failed to create writer");
        writer
            .add_column("id", "Int64", false)
            .expect("Failed to add column");

        writer.begin_batch(3).expect("Failed to begin batch");

        let data: Vec<i64> = vec![1, 2, 3, 4, 5]; // 5 rows instead of 3
        let result = writer.write_column_primitive(0, data.as_ptr() as *const c_void, 5);
        assert!(result.is_err(), "Row count mismatch should fail");
    }

    #[test]
    fn test_writer_empty_string_column() {
        // Test writing empty strings
        let temp_file = NamedTempFile::new().expect("Failed to create temp file");
        let path = temp_file.path().to_string_lossy().to_string();

        let mut writer = VortexWriter::new(&path).expect("Failed to create writer");
        writer
            .add_column("name", "String", false)
            .expect("Failed to add column");

        writer.begin_batch(3).expect("Failed to begin batch");

        // Three empty strings
        let strings = "";
        let offsets: Vec<u64> = vec![0, 0, 0, 0];
        writer
            .write_column_strings(0, strings.as_ptr(), offsets.as_ptr(), 3)
            .expect("Failed to write empty strings");

        writer.end_batch().expect("Failed to end batch");
        writer.finalize().expect("Failed to finalize");
    }

    #[test]
    fn test_writer_mixed_empty_and_nonempty_strings() {
        // Test writing mix of empty and non-empty strings
        let temp_file = NamedTempFile::new().expect("Failed to create temp file");
        let path = temp_file.path().to_string_lossy().to_string();

        let mut writer = VortexWriter::new(&path).expect("Failed to create writer");
        writer
            .add_column("name", "String", false)
            .expect("Failed to add column");

        writer.begin_batch(4).expect("Failed to begin batch");

        // "Hello", "", "World", ""
        let strings = "HelloWorld";
        let offsets: Vec<u64> = vec![0, 5, 5, 10, 10]; // Hello(5), empty, World(5), empty
        writer
            .write_column_strings(0, strings.as_ptr(), offsets.as_ptr(), 4)
            .expect("Failed to write mixed strings");

        writer.end_batch().expect("Failed to end batch");
        writer.finalize().expect("Failed to finalize");
    }

    #[test]
    fn test_writer_all_data_types() {
        // Test all supported primitive types
        let temp_file = NamedTempFile::new().expect("Failed to create temp file");
        let path = temp_file.path().to_string_lossy().to_string();

        let mut writer = VortexWriter::new(&path).expect("Failed to create writer");

        // Add columns of various types
        writer.add_column("i8", "Int8", false).expect("Failed");
        writer.add_column("i16", "Int16", false).expect("Failed");
        writer.add_column("i32", "Int32", false).expect("Failed");
        writer.add_column("i64", "Int64", false).expect("Failed");
        writer.add_column("u8", "UInt8", false).expect("Failed");
        writer.add_column("u16", "UInt16", false).expect("Failed");
        writer.add_column("u32", "UInt32", false).expect("Failed");
        writer.add_column("u64", "UInt64", false).expect("Failed");
        writer.add_column("f32", "Float32", false).expect("Failed");
        writer.add_column("f64", "Float64", false).expect("Failed");

        writer.begin_batch(2).expect("Failed to begin batch");

        let i8_data: Vec<i8> = vec![1, 2];
        let i16_data: Vec<i16> = vec![100, 200];
        let i32_data: Vec<i32> = vec![1000, 2000];
        let i64_data: Vec<i64> = vec![10000, 20000];
        let u8_data: Vec<u8> = vec![1, 2];
        let u16_data: Vec<u16> = vec![100, 200];
        let u32_data: Vec<u32> = vec![1000, 2000];
        let u64_data: Vec<u64> = vec![10000, 20000];
        let f32_data: Vec<f32> = vec![1.5, 2.5];
        let f64_data: Vec<f64> = vec![1.5, 2.5];

        writer
            .write_column_primitive(0, i8_data.as_ptr() as *const c_void, 2)
            .expect("Failed");
        writer
            .write_column_primitive(1, i16_data.as_ptr() as *const c_void, 2)
            .expect("Failed");
        writer
            .write_column_primitive(2, i32_data.as_ptr() as *const c_void, 2)
            .expect("Failed");
        writer
            .write_column_primitive(3, i64_data.as_ptr() as *const c_void, 2)
            .expect("Failed");
        writer
            .write_column_primitive(4, u8_data.as_ptr() as *const c_void, 2)
            .expect("Failed");
        writer
            .write_column_primitive(5, u16_data.as_ptr() as *const c_void, 2)
            .expect("Failed");
        writer
            .write_column_primitive(6, u32_data.as_ptr() as *const c_void, 2)
            .expect("Failed");
        writer
            .write_column_primitive(7, u64_data.as_ptr() as *const c_void, 2)
            .expect("Failed");
        writer
            .write_column_primitive(8, f32_data.as_ptr() as *const c_void, 2)
            .expect("Failed");
        writer
            .write_column_primitive(9, f64_data.as_ptr() as *const c_void, 2)
            .expect("Failed");

        writer.end_batch().expect("Failed to end batch");
        writer.finalize().expect("Failed to finalize");
    }

    /// Test that low-precision Decimal types (precision 1-4) are correctly handled
    /// on the write path. ClickHouse always sends Decimal32 as 4-byte Int32, even
    /// for low precisions like Decimal(3, 2). The write path must use the ClickHouse
    /// storage width (i32) rather than the smallest possible width (i16 for precision 3).
    #[test]
    fn test_decimal_low_precision_write_read_roundtrip() {
        use crate::exporter::ColumnExporter;
        use crate::exporter::decimal::DecimalExporter;

        // Simulate what ClickHouse does for Decimal(3, 2):
        // It sends raw i32 values (unscaled): 1.23 -> 123, -4.56 -> -456, 9.99 -> 999
        let clickhouse_data: Vec<i32> = vec![123, -456, 999];
        let num_rows = clickhouse_data.len();

        // Parse the type string the same way the writer does
        let dtype =
            clickhouse_type_to_vortex("Decimal(3, 2)").expect("Failed to parse Decimal(3, 2)");

        // Build array from raw data - this is the critical function being tested
        let array = build_array_from_raw_with_validity(
            &dtype,
            clickhouse_data.as_ptr() as *const c_void,
            ptr::null(),
            num_rows,
        )
        .expect("Failed to build decimal array from raw data");

        // Now export back to ClickHouse format via DecimalExporter
        let mut exporter = DecimalExporter::new(array).expect("Failed to create exporter");

        let mut output = vec![0i32; num_rows];
        let exported = exporter
            .export(
                output.as_mut_ptr() as *mut c_void,
                size_of_val(output.as_slice()),
                num_rows,
            )
            .expect("Export failed");

        assert_eq!(exported, num_rows);
        assert_eq!(output, vec![123, -456, 999]);
    }

    /// Same test for precision 1-2 (would map to I8 with smallest_decimal_value_type).
    #[test]
    fn test_decimal_precision_2_write_read_roundtrip() {
        use crate::exporter::ColumnExporter;
        use crate::exporter::decimal::DecimalExporter;

        // Decimal(2, 1): ClickHouse sends raw i32 values: 1.2 -> 12, -9.9 -> -99
        let clickhouse_data: Vec<i32> = vec![12, -99];
        let num_rows = clickhouse_data.len();

        let dtype =
            clickhouse_type_to_vortex("Decimal(2, 1)").expect("Failed to parse Decimal(2, 1)");

        let array = build_array_from_raw_with_validity(
            &dtype,
            clickhouse_data.as_ptr() as *const c_void,
            ptr::null(),
            num_rows,
        )
        .expect("Failed to build decimal array from raw data");

        let mut exporter = DecimalExporter::new(array).expect("Failed to create exporter");

        let mut output = vec![0i32; num_rows];
        let exported = exporter
            .export(
                output.as_mut_ptr() as *mut c_void,
                size_of_val(output.as_slice()),
                num_rows,
            )
            .expect("Export failed");

        assert_eq!(exported, num_rows);
        assert_eq!(output, vec![12, -99]);
    }

    #[test]
    fn test_writer_ffi_workflow_complete() {
        // Complete FFI workflow test
        let temp_file = NamedTempFile::new().expect("Failed to create temp file");
        let path = temp_file.path().to_string_lossy().to_string();
        let c_path = CString::new(path.clone()).expect("CString failed");

        unsafe {
            // Create writer
            let writer = vortex_writer_new(c_path.as_ptr());
            assert!(!writer.is_null());

            // Add columns
            let name_col = CString::new("id").unwrap();
            let type_col = CString::new("Int64").unwrap();
            let result = vortex_writer_add_column(writer, name_col.as_ptr(), type_col.as_ptr(), 0);
            assert_eq!(result, 0);

            // Begin batch
            let result = vortex_writer_begin_batch(writer, 3);
            assert_eq!(result, 0);

            // Write column
            let data: Vec<i64> = vec![1, 2, 3];
            let result = vortex_writer_write_column(writer, 0, data.as_ptr() as *const c_void, 3);
            assert_eq!(result, 0);

            // End batch
            let result = vortex_writer_end_batch(writer);
            assert_eq!(result, 0);

            // Verify row count
            let rows = vortex_writer_total_rows(writer);
            assert_eq!(rows, 3);

            // Finalize
            let result = vortex_writer_finalize(writer);
            assert_eq!(result, 0);

            // Free
            vortex_writer_free(writer);
        }

        assert!(std::path::Path::new(&path).exists());
    }
}
