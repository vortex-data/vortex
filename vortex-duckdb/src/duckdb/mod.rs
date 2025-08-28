// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

mod client_context;
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
macro_rules! lifetime_wrapper {
    // Full variant with all types (Owned + Ref + MutRef)
    ($(#[$meta:meta])* $Name:ident, $ffi_type:ty, $destructor:expr) => {
        $crate::lifetime_wrapper!($(#[$meta])* $Name, $ffi_type, $destructor, [owned, ref, mut_ref]);
    };
    
    // Selective variant - specify which types to generate
    ($(#[$meta:meta])* $Name:ident, $ffi_type:ty, $destructor:expr, [$($variant:ident),*]) => {
        $crate::lifetime_wrapper_impl!($(#[$meta])* $Name, $ffi_type, $destructor, [$($variant),*]);
    };
}

#[macro_export]
macro_rules! lifetime_wrapper_impl {
    ($(#[$meta:meta])* $Name:ident, $ffi_type:ty, $destructor:expr, [$($variant:ident),*]) => {
        $crate::lifetime_wrapper_generate_owned!($(#[$meta])* $Name, $ffi_type, $destructor, [$($variant),*]);
        $crate::lifetime_wrapper_generate_ref!($(#[$meta])* $Name, $ffi_type, [$($variant),*]);
        $crate::lifetime_wrapper_generate_mut_ref!($(#[$meta])* $Name, $ffi_type, [$($variant),*]);
    };
}

// Generate owned type
#[macro_export]
macro_rules! lifetime_wrapper_generate_owned {
    ($(#[$meta:meta])* $Name:ident, $ffi_type:ty, $destructor:expr, [owned $(, $rest:ident)*]) => {
        // Owned version that manages the FFI pointer's lifetime
        $(#[$meta])*
        pub struct $Name {
            ptr: $ffi_type,
            owned: bool,
        }

        #[allow(dead_code)]
        impl $Name {
            /// Takes ownership of the memory. The Rust wrapper becomes
            /// responsible for calling the destructor when dropped.
            /// This is the only constructor for the owned variant.
            pub unsafe fn own(ptr: $ffi_type) -> Self {
                if ptr.is_null() {
                    vortex::error::vortex_panic!("Attempted to create a wrapper from a null pointer");
                }
                Self { ptr, owned: true }
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

            $crate::lifetime_wrapper_owned_methods!($Name, [$($rest),*]);
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
    // Skip if 'owned' is not in the list
    ($(#[$meta:meta])* $Name:ident, $ffi_type:ty, $destructor:expr, [$($other:ident),*]) => {};
}

// Generate methods for owned type based on what other variants are available
#[macro_export]
macro_rules! lifetime_wrapper_owned_methods {
    ($Name:ident, [ref $(, $rest:ident)*]) => {
        paste::paste! {
            /// Convert to a borrowed reference with explicit lifetime
            pub fn as_ref(&self) -> [<$Name Ref>]<'_> {
                [<$Name Ref>] {
                    ptr: self.ptr,
                    _lifetime: std::marker::PhantomData,
                }
            }
        }
        $crate::lifetime_wrapper_owned_methods!($Name, [$($rest),*]);
    };
    ($Name:ident, [mut_ref $(, $rest:ident)*]) => {
        paste::paste! {
            /// Convert to a mutable borrowed reference with explicit lifetime
            pub fn as_mut_ref(&mut self) -> [<$Name MutRef>]<'_> {
                [<$Name MutRef>] {
                    ptr: self.ptr,
                    _lifetime: std::marker::PhantomData,
                }
            }
        }
        $crate::lifetime_wrapper_owned_methods!($Name, [$($rest),*]);
    };
    ($Name:ident, [$first:ident $(, $rest:ident)*]) => {
        $crate::lifetime_wrapper_owned_methods!($Name, [$($rest),*]);
    };
    ($Name:ident, []) => {};
}

// Generate ref type
#[macro_export]
macro_rules! lifetime_wrapper_generate_ref {
    ($(#[$meta:meta])* $Name:ident, $ffi_type:ty, [ref $(, $rest:ident)*]) => {
        paste::paste! {
            // Borrowed version with explicit lifetime parameter
            $(#[$meta])*
            pub struct [<$Name Ref>]<'a> {
                ptr: $ffi_type,
                _lifetime: std::marker::PhantomData<&'a ()>,
            }

            #[allow(dead_code)]
            impl<'a> [<$Name Ref>]<'a> {
                /// Borrows the pointer without taking ownership.
                /// This is the only constructor for the ref variant.
                pub unsafe fn borrow(ptr: $ffi_type) -> Self {
                    if ptr.is_null() {
                        vortex::error::vortex_panic!("Attempted to create a wrapper ref from a null pointer");
                    }
                    Self {
                        ptr,
                        _lifetime: std::marker::PhantomData,
                    }
                }

                /// Returns the raw pointer.
                pub fn as_ptr(&self) -> $ffi_type {
                    self.ptr
                }
            }
            
            // Ref versions can be Copy since they're just borrowed pointers
            impl<'a> Copy for [<$Name Ref>]<'a> {}
            impl<'a> Clone for [<$Name Ref>]<'a> {
                fn clone(&self) -> Self {
                    *self
                }
            }
        }
    };
    // Skip if 'ref' is not in the list
    ($(#[$meta:meta])* $Name:ident, $ffi_type:ty, [$($other:ident),*]) => {};
}

// Generate mut_ref type
#[macro_export]
macro_rules! lifetime_wrapper_generate_mut_ref {
    ($(#[$meta:meta])* $Name:ident, $ffi_type:ty, [mut_ref $(, $rest:ident)*]) => {
        paste::paste! {
            // Mutable borrowed version with explicit lifetime parameter
            $(#[$meta])*
            pub struct [<$Name MutRef>]<'a> {
                ptr: $ffi_type,
                _lifetime: std::marker::PhantomData<&'a mut ()>,
            }

            #[allow(dead_code)]
            impl<'a> [<$Name MutRef>]<'a> {
                /// Borrows the pointer mutably without taking ownership.
                /// This is the only constructor for the mut_ref variant.
                pub unsafe fn borrow_mut(ptr: $ffi_type) -> Self {
                    if ptr.is_null() {
                        vortex::error::vortex_panic!("Attempted to create a wrapper mut ref from a null pointer");
                    }
                    Self {
                        ptr,
                        _lifetime: std::marker::PhantomData,
                    }
                }

                /// Returns the raw pointer.
                pub fn as_ptr(&self) -> $ffi_type {
                    self.ptr
                }

                $crate::lifetime_wrapper_mut_ref_methods!($Name, [$($rest),*]);
            }
            
            // Check if we should implement Deref to Ref
            $crate::lifetime_wrapper_mut_ref_deref!($Name, [$($rest),*]);
        }
    };
    // Skip if 'mut_ref' is not in the list
    ($(#[$meta:meta])* $Name:ident, $ffi_type:ty, [$($other:ident),*]) => {};
}

// Generate methods for mut_ref type based on what other variants are available
#[macro_export]
macro_rules! lifetime_wrapper_mut_ref_methods {
    ($Name:ident, [ref $(, $rest:ident)*]) => {
        paste::paste! {
            /// Convert to immutable reference
            pub fn as_ref(&self) -> [<$Name Ref>]<'a> {
                [<$Name Ref>] {
                    ptr: self.ptr,
                    _lifetime: std::marker::PhantomData,
                }
            }
        }
        $crate::lifetime_wrapper_mut_ref_methods!($Name, [$($rest),*]);
    };
    ($Name:ident, [$first:ident $(, $rest:ident)*]) => {
        $crate::lifetime_wrapper_mut_ref_methods!($Name, [$($rest),*]);
    };
    ($Name:ident, []) => {};
}

// Generate Deref implementation for MutRef -> Ref if both exist
#[macro_export]
macro_rules! lifetime_wrapper_mut_ref_deref {
    ($Name:ident, [ref $(, $rest:ident)*]) => {
        paste::paste! {
            // MutRef can deref to Ref for convenient immutable access
            impl<'a> std::ops::Deref for [<$Name MutRef>]<'a> {
                type Target = [<$Name Ref>]<'a>;
                
                fn deref(&self) -> &Self::Target {
                    // SAFETY: We can transmute MutRef to Ref because:
                    // 1. They have the same layout (same ptr field)
                    // 2. MutRef's lifetime is more restrictive (&'a mut -> &'a)
                    // 3. This is the same as &mut T -> &T coercion
                    unsafe { std::mem::transmute(self) }
                }
            }
        }
    };
    // Skip if 'ref' is not in the list
    ($Name:ident, [$first:ident $(, $rest:ident)*]) => {
        $crate::lifetime_wrapper_mut_ref_deref!($Name, [$($rest),*]);
    };
    ($Name:ident, []) => {};
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
            /// For compatibility with existing code.
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
