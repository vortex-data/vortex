// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

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

/// Generates two FFI pointer wrapper types using the opaque-pointee pattern:
///
/// - `$Name` — an opaque ZST pointee type, never constructed directly.
///   Used only behind pointers/references (`&$Name`, `&mut $Name`).
///
/// - `Owned$Name` — an owned handle that calls the destructor on drop.
///   Derefs to `&$Name` / `&mut $Name`.
#[macro_export]
macro_rules! lifetime_wrapper {
    // Accept old variant-list syntax for backward compat (variant list is ignored).
    ($(#[$meta:meta])* $Name:ident, $ffi_type:ty, $destructor:expr, [$($variant:ident),*]) => {
        $crate::lifetime_wrapper!($(#[$meta])* $Name, $ffi_type, $destructor);
    };
    // Main form: generates opaque $Name + Owned$Name.
    ($(#[$meta:meta])* $Name:ident, $ffi_type:ty, $destructor:expr) => {
        paste::paste! {
            // --- Opaque pointee (never constructed, used as &$Name / &mut $Name) ---

            $(#[$meta])*
            #[allow(dead_code)]
            pub struct $Name(());

            #[allow(dead_code)]
            impl $Name {
                /// Borrows the pointer as an immutable reference with explicit lifetime.
                ///
                /// # Safety
                ///
                /// The pointer must be valid for the lifetime `'a` and must not be null.
                pub unsafe fn borrow<'a>(ptr: $ffi_type) -> &'a Self {
                    if ptr.is_null() {
                        vortex::error::vortex_panic!(
                            "Attempted to borrow from a null pointer"
                        );
                    }
                    unsafe { &*(ptr as *const Self) }
                }

                /// Borrows the pointer as a mutable reference with explicit lifetime.
                ///
                /// # Safety
                ///
                /// The pointer must be valid for the lifetime `'a`, must not be null,
                /// and no other references to the same pointer must exist.
                pub unsafe fn borrow_mut<'a>(ptr: $ffi_type) -> &'a mut Self {
                    if ptr.is_null() {
                        vortex::error::vortex_panic!(
                            "Attempted to borrow_mut from a null pointer"
                        );
                    }
                    unsafe { &mut *(ptr as *mut Self) }
                }

                /// Returns the raw FFI pointer.
                pub fn as_ptr(&self) -> $ffi_type {
                    (self as *const Self).cast_mut().cast()
                }
            }

            // --- Owned handle (calls destructor on drop) ---

            $(#[$meta])*
            #[allow(dead_code)]
            pub struct [<Owned $Name>]($ffi_type);

            #[allow(dead_code)]
            impl [<Owned $Name>] {
                /// Takes ownership of the pointer. The owned handle becomes
                /// responsible for calling the destructor when dropped.
                ///
                /// # Safety
                ///
                /// The pointer must be valid and the caller must transfer ownership.
                pub unsafe fn own(ptr: $ffi_type) -> Self {
                    if ptr.is_null() {
                        vortex::error::vortex_panic!(
                            "Attempted to create an owned wrapper from a null pointer"
                        );
                    }
                    Self(ptr)
                }

                /// Releases ownership and returns the raw pointer without
                /// calling the destructor.
                pub fn into_ptr(self) -> $ffi_type {
                    let this = std::mem::ManuallyDrop::new(self);
                    this.0
                }
            }

            impl std::ops::Deref for [<Owned $Name>] {
                type Target = $Name;

                fn deref(&self) -> &Self::Target {
                    // SAFETY: The opaque $Name is a ZST and the pointer is valid
                    // for the lifetime of the owned handle.
                    unsafe { &*(self.0 as *const $Name) }
                }
            }

            impl std::ops::DerefMut for [<Owned $Name>] {
                fn deref_mut(&mut self) -> &mut Self::Target {
                    // SAFETY: The opaque $Name is a ZST and the pointer is valid
                    // for the lifetime of the owned handle. We have &mut self so
                    // exclusive access is guaranteed.
                    unsafe { &mut *(self.0 as *mut $Name) }
                }
            }

            impl Drop for [<Owned $Name>] {
                fn drop(&mut self) {
                    let destructor = $destructor;
                    #[allow(unused_unsafe)]
                    unsafe {
                        destructor(&mut self.0)
                    }
                }
            }
        }
    };
}
