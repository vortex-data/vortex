#![allow(unsafe_op_in_unsafe_fn, clippy::missing_safety_doc, clippy::panic)]

//! Native interface to Vortex arrays, types, files and streams.

mod array;
mod dtype;
#[cfg(feature = "duckdb")]
mod duckdb;
mod file;
mod log;
mod stream;

use std::cell::LazyCell;
use std::ffi::{CStr, c_char, c_int};

use tokio::runtime::{Builder, Runtime};
use vortex::error::VortexExpect;

thread_local! {
    static RUNTIME: LazyCell<Runtime> = LazyCell::new(|| {
        // Using a new_multi_thread runtime since a current local runtime has a deadlock.
        Builder::new_multi_thread()
            .enable_all()
            .build()
            .vortex_expect("building runtime")
    });
}

pub(crate) unsafe fn to_string(ptr: *const c_char) -> String {
    let c_str = CStr::from_ptr(ptr);
    c_str.to_string_lossy().into_owned()
}

pub(crate) unsafe fn to_string_vec(ptr: *const *const c_char, len: c_int) -> Vec<String> {
    (0..len)
        .map(|i| to_string(*ptr.offset(i as isize)))
        .collect()
}
