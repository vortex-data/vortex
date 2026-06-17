// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

mod bind_input;
mod client_context;
mod config;
mod connection;
mod data;
mod data_chunk;
mod database;
mod ddb_string;
mod expr;
mod logical_type;
mod macro_;
mod query_result;
mod reusable_dict;
mod scalar_function;
mod selection_vector;
mod string_map;
mod table_filter;
mod table_init_input;
mod value;
mod vector;
mod vector_buffer;

use std::ffi::c_void;
use std::ptr;

pub use bind_input::*;
pub use client_context::*;
pub use config::*;
pub use connection::*;
pub use data::*;
pub use data_chunk::*;
pub use database::*;
pub use ddb_string::*;
pub use expr::*;
pub use logical_type::*;
pub use query_result::*;
pub use reusable_dict::*;
pub use scalar_function::*;
pub use selection_vector::*;
pub use string_map::*;
pub use table_filter::*;
pub use table_init_input::*;
pub use value::*;
pub use vector::*;
pub use vector_buffer::*;
use vortex::error::VortexResult;

use crate::cpp;

/// Try to execute a Rust function, or else return a null pointer and set the error.
pub(crate) fn try_or_null<T>(
    error_out: *mut cpp::duckdb_vx_error,
    function: impl FnOnce() -> VortexResult<*mut T>,
) -> *mut T {
    match function() {
        Ok(value) => {
            unsafe { error_out.write(ptr::null_mut()) };
            value
        }
        Err(err) => {
            // Set the error in the bind result.
            let msg = err.to_string();
            unsafe { error_out.write(cpp::duckdb_vx_error_create(msg.as_ptr().cast(), msg.len())) };
            ptr::null_mut::<T>()
        }
    }
}

/// Try to execute a Rust function, or else return the default value and set the error.
pub(crate) fn try_or<T: Default>(
    error_out: *mut cpp::duckdb_vx_error,
    function: impl FnOnce() -> VortexResult<T>,
) -> T {
    match function() {
        Ok(value) => {
            unsafe { error_out.write(ptr::null_mut()) };
            value
        }
        Err(err) => {
            // Set the error in the bind result.
            let msg = err.to_string();
            unsafe { error_out.write(cpp::duckdb_vx_error_create(msg.as_ptr().cast(), msg.len())) };
            T::default()
        }
    }
}

/// Creates a function that drops a `Box<T>` when called.
extern "C-unwind" fn drop_boxed<T>(ptr: *mut c_void) {
    // Safety: We assume that the pointer is valid and points to a Box<T>.
    // The caller is responsible for ensuring that the pointer is valid.
    if !ptr.is_null() {
        drop(unsafe { Box::from_raw(ptr.cast::<T>()) })
    }
}
