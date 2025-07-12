// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::ffi::CStr;
use std::ptr;

use vortex::error::{VortexError, VortexExpect, vortex_bail};

use crate::cpp::{duckdb_logical_type, duckdb_vx_error};
use crate::duckdb::{LogicalType, Vector};
use crate::{cpp, wrapper};

wrapper!(
    DataChunk,
    cpp::duckdb_data_chunk,
    cpp::duckdb_destroy_data_chunk
);

impl DataChunk {
    /// Create a new data chunk using a list of logical dtypes
    pub fn new(column_types: impl AsRef<[LogicalType]>) -> DataChunk {
        let mut ptrs = column_types
            .as_ref()
            .iter()
            .map(|x| x.as_ptr())
            .collect::<Vec<duckdb_logical_type>>();

        let ptr = unsafe { cpp::duckdb_create_data_chunk(ptrs.as_mut_ptr(), ptrs.len() as _) };
        unsafe { DataChunk::own(ptr) }
    }

    /// Returns the column count of the data chunk.
    pub fn column_count(&self) -> usize {
        usize::try_from(unsafe { cpp::duckdb_data_chunk_get_column_count(self.as_ptr()) })
            .vortex_expect("Column count exceeds usize")
    }

    /// Set the length of the data chunk.
    pub fn set_len(&mut self, len: usize) {
        unsafe { cpp::duckdb_data_chunk_set_size(self.as_ptr(), len as _) }
    }

    /// Returns the vector at the specified column index.
    pub fn get_vector(&self, idx: usize) -> Vector {
        unsafe { Vector::borrow(cpp::duckdb_data_chunk_get_vector(self.as_ptr(), idx as _)) }
    }

    pub fn len(&self) -> u64 {
        unsafe { cpp::duckdb_data_chunk_get_size(self.ptr) }
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

impl TryFrom<&DataChunk> for String {
    type Error = VortexError;

    fn try_from(value: &DataChunk) -> Result<Self, Self::Error> {
        let mut err: duckdb_vx_error = ptr::null_mut();
        #[cfg(debug_assertions)]
        unsafe {
            cpp::duckdb_data_chunk_verify(value.as_ptr(), &mut err);
            if !err.is_null() {
                vortex_bail!(
                    "{}",
                    CStr::from_ptr(cpp::duckdb_vx_error_value(err)).to_string_lossy()
                )
            }
        };
        let debug = unsafe { cpp::duckdb_data_chunk_to_string(value.as_ptr(), &mut err) };
        if !err.is_null() {
            vortex_bail!("{}", unsafe {
                CStr::from_ptr(cpp::duckdb_vx_error_value(err)).to_string_lossy()
            })
        }
        let string = unsafe { CStr::from_ptr(debug).to_string_lossy() }.to_string();
        unsafe { cpp::duckdb_free(debug.cast_mut().cast()) };
        Ok(string)
    }
}
