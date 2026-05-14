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
/// - `$Name` — an owned handle that calls the destructor on drop.
///   Derefs to `&${Name}Ref` / `&mut ${Name}Ref`.
///
/// - `${Name}Ref` — an opaque ZST pointee type, never constructed directly.
///   Used only behind pointers/references (`&${Name}Ref`, `&mut ${Name}Ref`).
#[macro_export]
macro_rules! lifetime_wrapper {
    // Main form: generates owned $Name + opaque ${Name}Ref.
    ($(#[$meta:meta])* $Name:ident, $ffi_type:ty, $destructor:expr) => {
        paste::paste! {
            // --- Opaque pointee (never constructed, used as &${Name}Ref / &mut ${Name}Ref) ---

            $(#[$meta])*
            #[allow(dead_code)]
            pub struct [<$Name Ref>](());

            #[allow(dead_code)]
            impl [<$Name Ref>] {
                /// Returns the raw FFI pointer.
                pub fn as_ptr(&self) -> $ffi_type {
                    (self as *const Self).cast_mut().cast()
                }
            }

            $(#[$meta])*
            #[allow(dead_code)]
            pub struct $Name(std::ptr::NonNull<std::ffi::c_void>);

            #[allow(dead_code)]
            impl $Name {
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
                    Self(unsafe { std::ptr::NonNull::new_unchecked(ptr.cast()) })
                }

                /// Borrows the pointer as an immutable reference with explicit lifetime.
                ///
                /// # Safety
                ///
                /// The pointer must be valid for the lifetime `'a` and must not be null.
                pub unsafe fn borrow<'a>(ptr: $ffi_type) -> &'a [<$Name Ref>] {
                    if ptr.is_null() {
                        vortex::error::vortex_panic!(
                            "Attempted to borrow from a null pointer"
                        );
                    }
                    unsafe { &*(ptr as *const [<$Name Ref>]) }
                }

                /// Borrows the pointer as a mutable reference with explicit lifetime.
                ///
                /// # Safety
                ///
                /// The pointer must be valid for the lifetime `'a`, must not be null,
                /// and no other references to the same pointer must exist.
                pub unsafe fn borrow_mut<'a>(ptr: $ffi_type) -> &'a mut [<$Name Ref>] {
                    if ptr.is_null() {
                        vortex::error::vortex_panic!(
                            "Attempted to borrow_mut from a null pointer"
                        );
                    }
                    unsafe { &mut *(ptr as *mut [<$Name Ref>]) }
                }

                /// Releases ownership and returns the raw pointer without
                /// calling the destructor.
                pub fn into_ptr(self) -> $ffi_type {
                    let this = std::mem::ManuallyDrop::new(self);
                    (*this).0.as_ptr().cast()
                }
            }

            impl std::ops::Deref for $Name {
                type Target = [<$Name Ref>];

                fn deref(&self) -> &Self::Target {
                    // SAFETY: The opaque [<$Name Ref>] is a ZST and the pointer is valid
                    // for the lifetime of the owned handle.
                    unsafe { &*(self.0.as_ptr() as *const [<$Name Ref>]) }
                }
            }

            impl std::ops::DerefMut for $Name {
                fn deref_mut(&mut self) -> &mut Self::Target {
                    // SAFETY: The opaque [<$Name Ref>] is a ZST and the pointer is valid
                    // for the lifetime of the owned handle. We have &mut self so
                    // exclusive access is guaranteed.
                    unsafe { &mut *(self.0.as_ptr() as *mut [<$Name Ref>]) }
                }
            }

            impl Drop for $Name {
                fn drop(&mut self) {
                    let destructor = $destructor;
                    let mut ptr: $ffi_type = self.0.as_ptr().cast();
                    #[allow(unused_unsafe)]
                    unsafe {
                        destructor(&mut ptr)
                    }
                }
            }
        }
    };
}
