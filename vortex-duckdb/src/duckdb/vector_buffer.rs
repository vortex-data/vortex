// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use crate::cpp;
use crate::duckdb::Data;
use crate::lifetime_wrapper;

// A wrapped buffer that give duckdb a strong reference to a vortex buffer.

lifetime_wrapper!(
    VectorBuffer,
    cpp::duckdb_vx_vector_buffer,
    cpp::duckdb_vx_vector_buffer_destroy
);

impl VectorBuffer {
    pub fn new<T>(data: T) -> Self {
        let data = Data::from(Box::new(data));
        unsafe { Self::own(cpp::duckdb_vx_vector_buffer_create(data.as_ptr())) }
    }

    /// Create a VectorBuffer that keeps `data` alive but whose `DataPtr()` returns
    /// `data_ptr` instead of a pointer to `data`. This is needed when the keep-alive
    /// object (e.g. a `ByteBuffer`) is not the raw data itself.
    ///
    /// # Safety
    ///
    /// `data_ptr` must remain valid for as long as `data` is alive.
    pub unsafe fn with_data_ptr<T>(data: T, data_ptr: *const u8) -> Self {
        let data = Data::from(Box::new(data));
        unsafe {
            Self::own(cpp::duckdb_vx_vector_buffer_create_with_data_ptr(
                data.as_ptr(),
                data_ptr as *mut std::ffi::c_void,
            ))
        }
    }
}
