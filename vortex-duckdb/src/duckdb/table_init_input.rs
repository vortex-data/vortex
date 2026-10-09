// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::fmt::Debug;
use std::fmt::Formatter;
use std::fmt::Result;

use crate::cpp;
use crate::duckdb::LogicalType;
use crate::duckdb::LogicalTypeRef;
use crate::duckdb::TableFilterSet;
use crate::duckdb::TableFilterSetRef;

/// A struct field DuckDB asked the scan to read instead of the whole column.
#[derive(Debug, Clone, Copy)]
pub struct StructExtract<'a> {
    /// Struct child indexes, outermost first.
    pub indexes: &'a [u64],
    /// The type DuckDB expects the scan to emit for this field, which is the
    /// cast target when the query casts the extracted field itself.
    pub datatype: &'a LogicalTypeRef,
}

/// The struct extracts DuckDB pushed into the scan, one entry per column id.
#[derive(Debug, Clone, Copy, Default)]
pub struct StructExtracts<'a> {
    pub(crate) indexes: &'a [u64],
    pub(crate) offsets: &'a [usize],
    pub(crate) datatypes: &'a [cpp::duckdb_logical_type],
}

impl<'a> StructExtracts<'a> {
    /// The extract for the column at `column`, if DuckDB asked for one.
    pub fn get(&self, column: usize) -> Option<StructExtract<'a>> {
        let start = usize::try_from(*self.offsets.get(column)?).ok()?;
        let end = usize::try_from(*self.offsets.get(column + 1)?).ok()?;
        if start == end {
            return None;
        }

        let indexes = self.indexes.get(start..end)?;
        let datatype = unsafe { LogicalType::borrow(*self.datatypes.get(column)?) };
        Some(StructExtract { indexes, datatype })
    }
}

/// Borrows a C++ array as a slice, reading a null pointer as an empty array.
///
/// # Safety
///
/// `ptr` must point to `len` initialized values that stay alive for `'a`.
unsafe fn borrow_array<'a, T>(ptr: *const T, len: usize) -> &'a [T] {
    if ptr.is_null() {
        return &[];
    }
    unsafe { std::slice::from_raw_parts(ptr, len) }
}

pub struct TableInitInput<'a> {
    pub input: &'a cpp::duckdb_vx_tfunc_init_input,
}

impl Debug for TableInitInput<'_> {
    fn fmt(&self, f: &mut Formatter<'_>) -> Result {
        f.debug_struct("TableInitInput")
            .field("column_ids", &self.column_ids())
            .field("projection_ids", &self.projection_ids())
            .field("table_filter_set", &self.table_filter_set())
            .finish()
    }
}

impl<'a> TableInitInput<'a> {
    pub fn new(input: &'a cpp::duckdb_vx_tfunc_init_input) -> Self {
        Self { input }
    }

    pub fn column_ids(&self) -> &[u64] {
        unsafe { std::slice::from_raw_parts(self.input.column_ids, self.input.column_ids_count) }
    }

    pub fn projection_ids(&self) -> &[u64] {
        if self.input.projection_ids_count == 0 {
            // from_raw_parts requires a non-null pointer. C++'s empty vector
            // may have a null pointer.
            return &[];
        }
        unsafe {
            std::slice::from_raw_parts(self.input.projection_ids, self.input.projection_ids_count)
        }
    }

    /// Returns the table filter set for the table function.
    pub fn table_filter_set(&self) -> Option<&TableFilterSetRef> {
        let ptr = self.input.filters;
        if ptr.is_null() {
            None
        } else {
            Some(unsafe { TableFilterSet::borrow(ptr) })
        }
    }

    /// The struct extracts DuckDB pushed into the scan, one per column id.
    ///
    /// The paths and types are borrowed from the bind data and only valid for
    /// the duration of the `init_global` call.
    pub fn struct_extracts(&self) -> StructExtracts<'_> {
        // The offsets array carries one entry per column id plus the end offset.
        let offsets_len = if self.input.column_extract_offsets.is_null() {
            0
        } else {
            self.input.column_ids_count + 1
        };

        StructExtracts {
            indexes: unsafe {
                borrow_array(
                    self.input.column_extract_indexes,
                    self.input.column_extract_indexes_count,
                )
            },
            offsets: unsafe {
                borrow_array(self.input.column_extract_offsets, offsets_len)
            },
            datatypes: unsafe {
                borrow_array(
                    self.input.column_extract_types,
                    self.input.column_extract_types_count,
                )
            },
        }
    }
}
