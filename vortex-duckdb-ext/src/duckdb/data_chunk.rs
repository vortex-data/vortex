use std::fmt::{Debug, Formatter};

use vortex::error::VortexExpect;

use crate::duckdb::Vector;
use crate::{cpp, wrapper};

wrapper!(
    DataChunk,
    cpp::duckdb_data_chunk,
    cpp::duckdb_destroy_data_chunk
);

impl DataChunk {
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
}

impl Debug for DataChunk {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        let debug = unsafe { cpp::duckdb_data_chunk_to_string(self.as_ptr()) };
        write!(f, "{}", unsafe {
            std::ffi::CStr::from_ptr(debug).to_string_lossy()
        })?;
        unsafe { cpp::duckdb_free(debug.cast_mut().cast()) };
        Ok(())
    }
}
