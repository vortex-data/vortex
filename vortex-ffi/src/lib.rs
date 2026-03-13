// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

#![allow(clippy::missing_safety_doc)]
#![deny(missing_docs)]

//! Native interface to Vortex arrays, types, files and streams.

mod array;
mod array_iterator;
mod binary;
mod data_source;
mod dtype;
mod error;
mod expression;
mod file;
mod log;
mod macros;
mod ptype;
mod scan;
mod session;
mod sink;
mod string;
mod struct_fields;

use std::ffi::CStr;
use std::ffi::c_char;
use std::ffi::c_int;
use std::sync::LazyLock;

// TODO hack for duckdb exporter
pub use array::vx_array;
pub use log::vx_log_level;
use vortex::io::runtime::current::CurrentThreadRuntime;

#[cfg(all(feature = "mimalloc", not(miri)))]
#[global_allocator]
static GLOBAL: mimalloc::MiMalloc = mimalloc::MiMalloc;

/// A shared runtime for all FFI operations.
// TODO(ngates): also create a CurrentThreadPool to manage background worker threads.
static RUNTIME: LazyLock<CurrentThreadRuntime> = LazyLock::new(CurrentThreadRuntime::new);

pub(crate) unsafe fn to_string(ptr: *const c_char) -> String {
    let c_str = unsafe { CStr::from_ptr(ptr) };
    c_str.to_string_lossy().into_owned()
}

pub(crate) unsafe fn to_string_vec(ptr: *const *const c_char, len: c_int) -> Vec<String> {
    (0..len)
        .map(|i| unsafe { to_string(*ptr.offset(i as isize)) })
        .collect()
}
