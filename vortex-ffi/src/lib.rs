// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

#![allow(clippy::missing_safety_doc)]
#![deny(missing_docs)]

//! Native interface to Vortex arrays, types, files and streams.

mod array;
mod array_iterator;
mod dtype;
mod error;
mod file;
mod log;
mod macros;
mod ptype;
mod session;
mod sink;
mod string;
mod struct_fields;

use std::ffi::{c_char, c_int, CStr};

pub use log::vx_log_level;

#[cfg(all(feature = "mimalloc", not(miri)))]
#[global_allocator]
static GLOBAL: mimalloc::MiMalloc = mimalloc::MiMalloc;

pub(crate) unsafe fn to_string(ptr: *const c_char) -> String {
    let c_str = unsafe { CStr::from_ptr(ptr) };
    c_str.to_string_lossy().into_owned()
}

pub(crate) unsafe fn to_string_vec(ptr: *const *const c_char, len: c_int) -> Vec<String> {
    (0..len)
        .map(|i| unsafe { to_string(*ptr.offset(i as isize)) })
        .collect()
}
