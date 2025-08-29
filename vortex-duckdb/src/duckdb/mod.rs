// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

mod client_context;
mod config;
mod connection;
mod copy_function;
mod data;
mod data_chunk;
mod database;
mod expr;
pub mod footer_cache;
mod logical_type;
mod object_cache;
mod query_result;
mod scalar_function;
mod selection_vector;
mod string;
mod table_filter;
mod table_function;
mod value;
mod vector;

use std::ffi::c_void;
use std::ptr;

pub use client_context::*;
pub use config::*;
pub use connection::*;
pub use copy_function::*;
pub use data::*;
pub use data_chunk::*;
pub use database::*;
pub use expr::*;
pub use logical_type::*;
pub use object_cache::*;
pub use query_result::*;
pub use scalar_function::*;
pub use selection_vector::*;
pub use string::*;
pub use table_filter::*;
pub use table_function::*;
pub use value::*;
pub use vector::*;
use vortex::error::VortexResult;

use crate::cpp;

#[macro_export]
macro_rules! duckdb_try {
    // Pattern: duckdb_try!(function_call)
    ($call:expr) => {
        if $call != $crate::cpp::duckdb_state::DuckDBSuccess {
            vortex::error::vortex_bail!("DuckDB operation failed");
        }
    };

    // Pattern: duckdb_try!(function_call, "error message")
    ($call:expr, $msg:expr) => {
        if $call != $crate::cpp::duckdb_state::DuckDBSuccess {
            vortex::error::vortex_bail!($msg);
        }
    };

    // Pattern: duckdb_try!(function_call, "error message with {}", args...)
    ($call:expr, $msg:expr, $($args:expr),+) => {
        if $call != $crate::cpp::duckdb_state::DuckDBSuccess {
            vortex::error::vortex_bail!($msg, $($args),+);
        }
    };
}

#[macro_export]
macro_rules! wrapper {
    ($(#[$meta:meta])* $Name:ident, $ffi_type:ty, $destructor:expr) => {
        $(#[$meta])*
        pub struct $Name {
            ptr: $ffi_type,
            owned: bool,
        }

        #[allow(dead_code)]
        impl $Name {
            /// Takes ownership of the memory. The Rust wrapper becomes
            /// responsible for calling the destructor when dropped.
            pub unsafe fn own(ptr: $ffi_type) -> Self {
                if ptr.is_null() {
                    vortex::error::vortex_panic!("Attempted to create a wrapper from a null pointer");
                }
                Self { ptr, owned: true }
            }

            /// Borrows the pointer without taking ownership.
            pub unsafe fn borrow(ptr: $ffi_type) -> Self {
                if ptr.is_null() {
                    vortex::error::vortex_panic!("Attempted to create a wrapper from a null pointer");
                }
                Self { ptr, owned: false }
            }

            /// Returns the raw pointer.
            pub fn as_ptr(&self) -> $ffi_type {
                self.ptr
            }

            /// Release ownership and return the raw pointer.
            pub fn into_ptr(mut self) -> $ffi_type {
                assert!(self.owned, "Cannot take ownership of unowned ptr");
                self.owned = false; // Prevent destructor from being called
                self.ptr
            }
        }

        impl Drop for $Name {
            fn drop(&mut self) {
                if self.owned {
                    let destructor = $destructor;
                    #[allow(unused_unsafe)]
                    unsafe { destructor(&mut self.ptr) }
                }
            }
        }
    };
}

/// Try to execute a Rust function, or else return a null pointer and set the error.
#[inline]
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
#[inline]
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
