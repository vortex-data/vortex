#![allow(unsafe_op_in_unsafe_fn, clippy::missing_safety_doc, clippy::panic)]

//! Native interface to Vortex arrays, types, files and streams.

pub mod array;
pub mod dtype;
pub mod file;
pub mod stream;

use std::ffi::{CStr, c_char};
use std::sync::LazyLock;

use tokio::runtime::{Builder, Runtime};
use vortex::error::VortexExpect;

// Shared Tokio runtime for all of the async operations in this package.
static RUNTIME: LazyLock<Runtime> = LazyLock::new(|| {
    Builder::new_multi_thread()
        .enable_all()
        .build()
        .vortex_expect("Failed to build Tokio runtime")
});

pub(crate) unsafe fn to_string(ptr: *const c_char) -> String {
    let c_str = CStr::from_ptr(ptr);
    c_str.to_string_lossy().into_owned()
}
