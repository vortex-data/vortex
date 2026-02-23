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
}
