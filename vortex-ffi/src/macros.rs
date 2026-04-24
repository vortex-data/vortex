// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Macros for defining FFI wrapper types.
//!
//! These macros make it easy to wrap Rust types in a way that can be used in C FFI contexts.
//! They provide _safer_ accessors and constructors for the wrapped types, ensuring consistency
//! in our APIs and reducing boilerplate code.
//!
//! There are four macros provided:
//! - [`arc_wrapper!`]: Wraps a type in an `Arc<T>` for shared ownership.
//! - [`arc_dyn_wrapper!`]: Wraps a type in an `Arc<dyn T>` for shared ownership, allowing for dynamic
//!   dispatch of unsized types (like trait objects).
//! - [`box_wrapper!`]: Wraps a type in a `Box<T>` for single ownership.
//! - [`box_dyn_wrapper!`]: Wraps a type in a `Box<dyn T>` for single ownership, allowing for dynamic
//!   dispatch of unsized types (like trait objects).
//!
//! Similarly to Rust, `Box` can be chosen to provide single ownership and mutability semantics,
//! while `Arc` can be used to provide shared ownership (with corresponding immutability).
//!
//! Each macro provides a `free` function, and the `Arc` variants also provide a `clone` function.
//!
//! Converting between the raw pointer and the wrapped type is done using the generated `new`,
//! `new_ref`, `as_ref`, `as_mut`, `into_box`, and `into_arc` methods, which internally check for
//! null pointers.
//!
//! ## Internals
//!
//! Dynamic traits in Rust use fat pointers, which means that the size of the type is too large to
//! fit into a single C pointer. There are various ways to handle this, but the most common and the
//! one used here is to box the dynamic type, e.g. `Box<Box<dyn Trait>>` or `Box<Arc<dyn Trait>>`.
//!
//! These types therefore have slightly more overhead than their non-dynamic counterparts, but
//! in practice, any other approach would similarly involve two heap allocations (for example a
//! C-style vtable).
//!
//! ## Safety
//!
//! These macros don't require any Send + Sync bounds. We could try and define this behaviour here
//! to make it clearer, but for now, it's important to just be careful when documenting the thread
//! safety of the functions that use these types.
//!

/// Define a native FFI type that wraps an [`std::sync::Arc<T>`] type with unsized T.
///
/// To solve the problem of dynamic traits using fat pointers, we box the `Arc<T>` a second time.
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
                    // TODO(joe): propagate this error up instead of expecting
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
                     std::sync::Arc::into_raw(obj).cast::<$ffi_ident>()
                }

                /// Wrap a borrowed object into a raw pointer.
                pub(crate) fn new_ref(obj: &$T) -> *const $ffi_ident {
                    obj as *const $T as *const $ffi_ident
                }

                /// Extract a borrowed reference from a const pointer.
                pub(crate) fn as_ref(ptr: *const $ffi_ident) -> &'static $T {
                    use vortex::error::VortexExpect;
                    // TODO(joe): propagate this error up instead of expecting
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
            #[expect(non_camel_case_types)]
            pub struct $ffi_ident(Box<$T>);

            #[expect(dead_code)]
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
                    // TODO(joe): propagate this error up instead of expecting
                    unsafe { ptr.as_ref() }
                        .vortex_expect("null pointer")
                        .0
                        .as_ref()
                }

                /// Extract a borrowed mutable reference from a mut pointer.
                pub(crate) fn as_mut<'a>(ptr: *mut $ffi_ident) -> &'a mut $T {
                    use vortex::error::VortexExpect;
                    // TODO(joe): propagate this error up instead of expecting
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
            #[expect(non_camel_case_types)]
            pub struct $ffi_ident($T);

            #[expect(dead_code)]
            impl $ffi_ident {
                /// Wrap an owned object into a raw pointer.
                pub(crate) fn new_box(obj: Box<$T>) -> *mut $ffi_ident {
                    Box::into_raw(obj).cast::<$ffi_ident>()
                }

                /// Wrap an owned object into a raw pointer.
                pub(crate) fn new(obj: $T) -> *mut $ffi_ident {
                    Box::into_raw(Box::new(obj)).cast::<$ffi_ident>()
                }

                /// Wrap a borrowed object into a raw pointer.
                pub(crate) fn new_ref(obj: &$T) -> *const $ffi_ident {
                    obj as *const $T as *const $ffi_ident
                }

                /// Extract a borrowed reference from a const pointer.
                pub(crate) fn as_ref<'a>(ptr: *const $ffi_ident) -> &'a $T {
                    use vortex::error::VortexExpect;
                    // TODO(joe): propagate this error up instead of expecting
                    &unsafe { ptr.as_ref() }
                        .vortex_expect("null pointer")
                        .0
                }

                /// Extract a borrowed mutable reference from a mut pointer.
                pub(crate) fn as_mut<'a>(ptr: *mut $ffi_ident) -> &'a mut $T {
                    use vortex::error::VortexExpect;
                    // TODO(joe): propagate this error up instead of expecting
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
