#![allow(clippy::missing_safety_doc)]
#![deny(missing_docs)]

//! Native interface to Vortex arrays, types, files and streams.

mod array;
mod array_iterator;
mod dtype;
#[cfg(feature = "duckdb")]
mod duckdb;
mod error;
mod file;
mod log;
mod session;
mod sink;

use std::ffi::{CStr, c_char, c_int};
use std::sync::LazyLock;

pub use log::vx_log_level;
use tokio::runtime;
use tokio::runtime::Runtime;
use vortex::error::VortexExpect;

#[cfg(all(feature = "mimalloc", not(miri)))]
#[global_allocator]
static GLOBAL: mimalloc::MiMalloc = mimalloc::MiMalloc;

static RUNTIME: LazyLock<Runtime> = LazyLock::new(|| {
    runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
        .vortex_expect("Cannot start runtime")
});

pub(crate) unsafe fn to_string(ptr: *const c_char) -> String {
    let c_str = unsafe { CStr::from_ptr(ptr) };
    c_str.to_string_lossy().into_owned()
}

pub(crate) unsafe fn to_string_vec(ptr: *const *const c_char, len: c_int) -> Vec<String> {
    (0..len)
        .map(|i| unsafe { to_string(*ptr.offset(i as isize)) })
        .collect()
}

/// Define a native FFI type that wraps an [`std::sync::Arc<T>`] type.
#[macro_export]
macro_rules! arc_wrapper {
    ($(#[$meta:meta])* $T:ty, $ffi_ident:ident) => {
        paste::paste! {
            $(#[$meta])*
            #[allow(non_camel_case_types)]
            pub struct $ffi_ident(std::sync::Arc<$T>);

            impl $ffi_ident {
                /// Extract a borrowed reference from a const pointer.
                pub(crate) fn as_ref<'a>(ptr: *const $ffi_ident) -> &'a std::sync::Arc<$T> {
                    &unsafe { ptr.as_ref() }
                        .vortex_expect("null pointer")
                        .0
                }

                /// Extract an owned reference from a mutable pointer to a `vx_array`.
                pub(crate) fn into_arc(ptr: *mut $ffi_ident) -> std::sync::Arc<$T>{
                    if ptr.is_null() {
                        vortex_panic!("null pointer");
                    }
                    unsafe { Box::from_raw(ptr) }.0
                }

                /// Convert a Rust object into a raw pointer.
                pub(crate) fn from(obj: std::sync::Arc<$T>) -> *mut $ffi_ident {
                    Box::into_raw(Box::new($ffi_ident(obj)))
                }
            }

            #[doc = "Free an owned [`" $ffi_ident "`] object."]
            pub extern "C-unwind" fn [<$ffi_ident _free>](ptr: *mut $ffi_ident) {
                drop($ffi_ident::into_arc(ptr))
            }
        }
    };
}
