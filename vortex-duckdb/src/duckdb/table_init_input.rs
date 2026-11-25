// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::ffi::CStr;
use std::fmt::Debug;
use std::fmt::Formatter;
use std::fmt::Result;
use std::num::NonZeroUsize;

use vortex::error::VortexResult;
use vortex::error::vortex_err;

use crate::cpp;
use crate::duckdb::TableFilterSet;
use crate::duckdb::TableFilterSetRef;

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

    /// The number of threads DuckDB runs, at least one.
    pub fn max_threads(&self) -> NonZeroUsize {
        usize::try_from(self.input.max_threads)
            .ok()
            .and_then(NonZeroUsize::new)
            .unwrap_or(NonZeroUsize::MIN)
    }

    /// The number of files the scan reads, at least one.
    pub fn file_count(&self) -> NonZeroUsize {
        usize::try_from(self.input.file_count)
            .ok()
            .and_then(NonZeroUsize::new)
            .unwrap_or(NonZeroUsize::MIN)
    }

    /// The paths of the files the scan reads.
    pub fn file_paths(&self) -> VortexResult<Vec<&str>> {
        if self.input.file_count == 0 {
            return Ok(Vec::new());
        }
        let count = usize::try_from(self.input.file_count)?;
        // SAFETY: DuckDB passes `file_count` paths, each a NUL-terminated string that outlives
        // the call that gives this input.
        let paths = unsafe { std::slice::from_raw_parts(self.input.file_paths, count) };
        paths
            .iter()
            .map(|&path| {
                // SAFETY: as above.
                let path = unsafe { CStr::from_ptr(path) };
                path.to_str()
                    .map_err(|err| vortex_err!("file path is not UTF-8: {err}"))
            })
            .collect()
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
}
