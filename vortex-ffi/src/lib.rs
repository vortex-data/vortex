#![allow(clippy::missing_safety_doc)]
#![deny(missing_docs)]

//! Native interface to Vortex arrays, types, files and streams.

mod array;
mod array_iterator;
mod dtype;
mod dtype_struct;
#[cfg(feature = "duckdb")]
mod duckdb;
mod error;
mod file;
mod log;
mod ptype;
mod session;
mod sink;
mod string;

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

/// Define a native FFI type that wraps an [`std::sync::Arc<T>`] type with unsized T.
///
/// To solve the problem of dynamic traits using fat pointers, we box the `Arc<T>` and pass a
/// pointer to the heap-allocated arc struct.
///
/// Note: Box::into_raw produces mut pointers, since in theory you can mutate the contents of the
/// box. In practice though, our box contains an arc, and the callers of this macro tend to deal
/// in arcs, and so const pointers make more sense.
#[macro_export]
macro_rules! arc_dyn_wrapper {
    ($(#[$meta:meta])* $T:ty, $ffi_ident:ident) => {
        paste::paste! {
            $(#[$meta])*
            #[allow(non_camel_case_types)]
            pub struct $ffi_ident(std::sync::Arc<$T>);

            #[allow(dead_code)]
            impl $ffi_ident {
                /// Wrap an owned object into a raw pointer.
                pub(crate) fn new(obj: std::sync::Arc<$T>) -> *const $ffi_ident {
                    // For unsized types, we need to box the Arc.
                    Box::into_raw(Box::new($ffi_ident(obj))).cast_const()
                }

                /// Wrap a borrowed object into a raw pointer.
                pub(crate) fn new_ref(obj: &std::sync::Arc<$T>) -> *const $ffi_ident {
                    obj as *const std::sync::Arc<$T> as *const $ffi_ident
                }

                /// Extract a borrowed reference from a const pointer.
                pub(crate) fn as_ref<'a>(ptr: *const $ffi_ident) -> &'a std::sync::Arc<$T> {
                    use vortex::error::VortexExpect;
                    &unsafe { ptr.as_ref() }
                        .vortex_expect("null pointer")
                        .0
                }

                /// Extract an owned reference from a const pointer.
                pub(crate) fn into_arc(ptr: *const $ffi_ident) -> std::sync::Arc<$T>{
                    if ptr.is_null() {
                        vortex::error::vortex_panic!("null pointer");
                    }
                    unsafe { Box::from_raw(ptr.cast_mut()) }.0
                }
            }

            #[doc = r" Clone a borrowed [`" $ffi_ident "`], returning an owned [`" $ffi_ident "`].\n\n"]
            #[doc = r" Must be released with [`" $ffi_ident "_free`]."]
            #[unsafe(no_mangle)]
            pub unsafe extern "C-unwind" fn [<$ffi_ident _clone>](ptr: *const $ffi_ident) -> *const $ffi_ident {
                if ptr.is_null() {
                    vortex::error::vortex_panic!("null pointer");
                }

                $ffi_ident::new($ffi_ident::as_ref(ptr).clone())
            }

            #[doc = r" Free an owned [`" $ffi_ident "`] object."]
            #[unsafe(no_mangle)]
            pub unsafe extern "C-unwind" fn [<$ffi_ident _free>](ptr: *const $ffi_ident) {
                if ptr.is_null() {
                    vortex::error::vortex_panic!("null pointer");
                }
                drop($ffi_ident::into_arc(ptr))
            }
        }
    };
}

/// Define a native FFI type that uses an [`std::sync::Arc`] wrapper.
#[macro_export]
macro_rules! arc_wrapper {
    ($(#[$meta:meta])* $T:ty, $ffi_ident:ident) => {
        paste::paste! {
            $(#[$meta])*
            #[allow(non_camel_case_types)]
            pub struct $ffi_ident($T);

            #[allow(dead_code)]
            impl $ffi_ident {
                /// Wrap an owned object into a raw pointer.
                pub(crate) fn new(obj: std::sync::Arc<$T>) -> *const $ffi_ident {
                    std::sync::Arc::into_raw(obj).cast()
                }

                /// Wrap a borrowed object into a raw pointer.
                pub(crate) fn new_ref(obj: &$T) -> *const $ffi_ident {
                    obj as *const $T as *const $ffi_ident
                }

                /// Extract a borrowed reference from a const pointer.
                pub(crate) fn as_ref<'a>(ptr: *const $ffi_ident) -> &'a $T {
                    use vortex::error::VortexExpect;
                    &unsafe { ptr.as_ref() }
                        .vortex_expect("null pointer")
                        .0
                }

                /// Extract an owned reference.
                pub(crate) fn into_arc(ptr: *const $ffi_ident) -> std::sync::Arc<$T>{
                    if ptr.is_null() {
                        vortex::error::vortex_panic!("null pointer");
                    }
                    unsafe { std::sync::Arc::from_raw(ptr.cast::<$T>()) }
                }
            }

            #[doc = r" Clone a borrowed [`" $ffi_ident "`], returning an owned [`" $ffi_ident "`].\n\n"]
            #[doc = r" Must be released with [`" $ffi_ident "_free`]."]
            #[unsafe(no_mangle)]
            pub unsafe extern "C-unwind" fn [<$ffi_ident _clone>](ptr: *const $ffi_ident) -> *const $ffi_ident {
                if ptr.is_null() {
                    vortex::error::vortex_panic!("null pointer");
                }
                unsafe { std::sync::Arc::increment_strong_count(ptr) };
                ptr
            }

            #[doc = r" Free an owned [`" $ffi_ident "`] object."]
            #[unsafe(no_mangle)]
            pub unsafe extern "C-unwind" fn [<$ffi_ident _free>](ptr: *const $ffi_ident) {
                if ptr.is_null() {
                    vortex::error::vortex_panic!("null pointer");
                }
                unsafe { std::sync::Arc::decrement_strong_count(ptr) };
            }
        }
    };
}

/// Define a native FFI type that wraps an [`Box<T>`] type with unsized T.
///
/// To solve the problem of dynamic traits using fat pointers, we box the `Box<T>` a second time.
#[macro_export]
macro_rules! box_dyn_wrapper {
    ($(#[$meta:meta])* $T:ty, $ffi_ident:ident) => {
        paste::paste! {
            $(#[$meta])*
            #[allow(non_camel_case_types)]
            pub struct $ffi_ident(Box<$T>);

            #[allow(dead_code)]
            impl $ffi_ident {
                /// Wrap an owned object into a raw pointer.
                pub(crate) fn new(obj: Box<$T>) -> *mut $ffi_ident {
                    // For unsized types, we need to box the Arc.
                    Box::into_raw(Box::new($ffi_ident(obj)))
                }

                /// Wrap a borrowed object into a raw pointer.
                pub(crate) fn new_ref(obj: &$T) -> *const $ffi_ident {
                    obj as *const $T as *const $ffi_ident
                }

                /// Extract a borrowed reference from a const pointer.
                pub(crate) fn as_ref<'a>(ptr: *const $ffi_ident) -> &'a $T {
                    use vortex::error::VortexExpect;
                    unsafe { ptr.as_ref() }
                        .vortex_expect("null pointer")
                        .0
                        .as_ref()
                }

                /// Extract a borrowed mutable reference from a mut pointer.
                pub(crate) fn as_mut<'a>(ptr: *mut $ffi_ident) -> &'a mut $T {
                    use vortex::error::VortexExpect;
                    unsafe { ptr.as_mut() }
                        .vortex_expect("null pointer")
                        .0
                        .as_mut()
                }

                /// Extract an owned reference from a mutable pointer.
                pub(crate) fn into_box(ptr: *mut $ffi_ident) -> Box<$T>{
                    if ptr.is_null() {
                        vortex::error::vortex_panic!("null pointer");
                    }
                    unsafe { Box::from_raw(ptr) }.0
                }
            }

            #[doc = r" Free an owned [`" $ffi_ident "`] object."]
            #[unsafe(no_mangle)]
            pub unsafe extern "C-unwind" fn [<$ffi_ident _free>](ptr: *mut $ffi_ident) {
                if ptr.is_null() {
                    vortex::error::vortex_panic!("null pointer");
                }
                drop($ffi_ident::into_box(ptr))
            }
        }
    };
}

/// Define a native FFI type that uses a [`Box`] wrapper.
#[macro_export]
macro_rules! box_wrapper {
    ($(#[$meta:meta])* $T:ty, $ffi_ident:ident) => {
        paste::paste! {
            $(#[$meta])*
            #[allow(non_camel_case_types)]
            pub struct $ffi_ident($T);

            #[allow(dead_code)]
            impl $ffi_ident {
                /// Wrap an owned object into a raw pointer.
                pub(crate) fn new(obj: Box<$T>) -> *mut $ffi_ident {
                    Box::into_raw(obj).cast()
                }

                /// Wrap a borrowed object into a raw pointer.
                pub(crate) fn new_ref(obj: &$T) -> *const $ffi_ident {
                    obj as *const $T as *const $ffi_ident
                }

                /// Extract a borrowed reference from a const pointer.
                pub(crate) fn as_ref<'a>(ptr: *const $ffi_ident) -> &'a $T {
                    use vortex::error::VortexExpect;
                    &unsafe { ptr.as_ref() }
                        .vortex_expect("null pointer")
                        .0
                }

                /// Extract a borrowed mutable reference from a mut pointer.
                pub(crate) fn as_mut<'a>(ptr: *mut $ffi_ident) -> &'a mut $T {
                    use vortex::error::VortexExpect;
                    &mut unsafe { ptr.as_mut() }
                        .vortex_expect("null pointer")
                        .0
                }

                /// Extract an owned reference.
                pub(crate) fn into_box(ptr: *mut $ffi_ident) -> Box<$T>{
                    if ptr.is_null() {
                        vortex::error::vortex_panic!("null pointer");
                    }
                    unsafe { Box::from_raw(ptr.cast::<$T>()) }
                }
            }

            #[doc = r" Free an owned [`" $ffi_ident "`] object."]
            #[unsafe(no_mangle)]
            pub unsafe extern "C-unwind" fn [<$ffi_ident _free>](ptr: *mut $ffi_ident) {
                if ptr.is_null() {
                    vortex::error::vortex_panic!("null pointer");
                }
                std::mem::drop(unsafe { Box::from_raw(ptr.cast::<$T>()) })
            }
        }
    };
}
