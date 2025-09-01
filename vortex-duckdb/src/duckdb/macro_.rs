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
        // Step 1: Generate all type definitions first
        $crate::lifetime_wrapper_generate_owned_type!($(#[$meta])* $Name, $ffi_type, [$($variant),*]);
        $crate::lifetime_wrapper_generate_ref_type!($(#[$meta])* $Name, $ffi_type, [$($variant),*]);
        $crate::lifetime_wrapper_generate_mut_ref_type!($(#[$meta])* $Name, $ffi_type, [$($variant),*]);

        // Step 2: Generate all implementations that may reference other types
        $crate::lifetime_wrapper_generate_owned_impl!($(#[$meta])* $Name, $ffi_type, $destructor, [$($variant),*]);
        $crate::lifetime_wrapper_generate_ref_impl!($(#[$meta])* $Name, $ffi_type, [$($variant),*]);
        $crate::lifetime_wrapper_generate_mut_ref_impl!($(#[$meta])* $Name, $ffi_type, [$($variant),*]);

        // Step 3: Generate cross-type relationships (Deref, etc.)
        $crate::lifetime_wrapper_generate_relationships!($(#[$meta])* $Name, $ffi_type, [$($variant),*]);
    };
}

// Generate owned type definition only
#[macro_export]
macro_rules! lifetime_wrapper_generate_owned_type {
    ($(#[$meta:meta])* $Name:ident, $ffi_type:ty, [owned $(, $rest:ident)*]) => {
        // Owned version that manages the FFI pointer's lifetime
        $(#[$meta])*
        pub struct $Name {
            ptr: $ffi_type,
            owned: bool,
        }
    };
    // Skip if 'owned' is not in the list
    ($(#[$meta:meta])* $Name:ident, $ffi_type:ty, [$($other:ident),*]) => {};
}

// Generate owned implementation
#[macro_export]
macro_rules! lifetime_wrapper_generate_owned_impl {
    ($(#[$meta:meta])* $Name:ident, $ffi_type:ty, $destructor:expr,[owned $(, $rest:ident)*]) => {
        #[allow(dead_code)]
        impl $Name {
            /// Takes ownership of the memory. The Rust wrapper becomes
            /// responsible for calling the destructor when dropped.
            /// This is the only constructor for the owned variant.
            pub unsafe fn own(ptr: $ffi_type) -> Self {
                if ptr.is_null() {
                    vortex::error::vortex_panic!(
                        "Attempted to create a wrapper from a null pointer"
                    );
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
        }

        impl Drop for $Name {
            fn drop(&mut self) {
                if self.owned {
                    let destructor = $destructor;
                    #[allow(unused_unsafe)]
                    unsafe {
                        destructor(&mut self.ptr)
                    }
                }
            }
        }
    };
    // Skip if 'owned' is not in the list
    ($(#[$meta:meta])* $Name:ident, $ffi_type:ty, $destructor:expr,[$($other:ident),*]) => {};
}

// Generate ref type definition only
#[macro_export]
macro_rules! lifetime_wrapper_generate_ref_type {
    ($(#[$meta:meta])* $Name:ident, $ffi_type:ty, [ref $(, $rest:ident)*]) => {
        paste::paste! {
            // Borrowed version with explicit lifetime parameter
            $(#[$meta])*
            pub struct [<$Name Ref>]<'a> {
                ptr: $ffi_type,
                _lifetime: std::marker::PhantomData<&'a ()>,
            }
        }
    };
    ($(#[$meta:meta])* $Name:ident, $ffi_type:ty, [owned, ref $(, $rest:ident)*]) => {
        paste::paste! {
            // Borrowed version with explicit lifetime parameter
            $(#[$meta])*
            pub struct [<$Name Ref>]<'a> {
                ptr: $ffi_type,
                _lifetime: std::marker::PhantomData<&'a ()>,
            }
        }
    };
    // Skip if 'ref' is not in the list
    ($(#[$meta:meta])* $Name:ident, $ffi_type:ty, [$($other:ident),*]) => {};
}

// Generate ref implementation
#[macro_export]
macro_rules! lifetime_wrapper_generate_ref_impl {
    ($(#[$meta:meta])* $Name:ident, $ffi_type:ty, [ref $(, $rest:ident)*]) => {
        paste::paste! {
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
    ($(#[$meta:meta])* $Name:ident, $ffi_type:ty, [owned, ref $(, $rest:ident)*]) => {
        paste::paste! {
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

// Generate mut_ref type definition only
#[macro_export]
macro_rules! lifetime_wrapper_generate_mut_ref_type {
    ($(#[$meta:meta])* $Name:ident, $ffi_type:ty, [mut_ref $(, $rest:ident)*]) => {
        paste::paste! {
            // Mutable borrowed version with explicit lifetime parameter
            $(#[$meta])*
            pub struct [<$Name MutRef>]<'a> {
                ptr: $ffi_type,
                _lifetime: std::marker::PhantomData<&'a mut ()>,
            }
        }
    };
    // Skip if 'mut_ref' is not in the list
    ($(#[$meta:meta])* $Name:ident, $ffi_type:ty, [$($other:ident),*]) => {};
}

// Generate mut_ref implementation
#[macro_export]
macro_rules! lifetime_wrapper_generate_mut_ref_impl {
    ($(#[$meta:meta])* $Name:ident, $ffi_type:ty, [mut_ref $(, $rest:ident)*]) => {
        paste::paste! {
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
            }
        }
    };
    // Skip if 'mut_ref' is not in the list
    ($(#[$meta:meta])* $Name:ident, $ffi_type:ty, [$($other:ident),*]) => {};
}

// Generate cross-type relationships (Deref, conversion methods, etc.)
#[macro_export]
macro_rules! lifetime_wrapper_generate_relationships {
    ($(#[$meta:meta])* $Name:ident, $ffi_type:ty, [$($variant:ident),*]) => {
        // Generate Deref from MutRef to Ref if both exist
        $crate::lifetime_wrapper_mut_ref_deref_if_both_exist!($Name, [$($variant),*]);

        // Generate owned methods that reference other types
        $crate::lifetime_wrapper_owned_cross_methods!($Name, [$($variant),*]);
    };
}

// Generate methods for owned type that reference other types
#[macro_export]
macro_rules! lifetime_wrapper_owned_cross_methods {
    ($Name:ident, [$($variant:ident),*]) => {
        $crate::lifetime_wrapper_generate_as_ref_if_needed!($Name, [$($variant),*]);
        $crate::lifetime_wrapper_generate_as_mut_ref_if_needed!($Name, [$($variant),*]);
    };
}

// Generate as_ref method if both owned and ref exist
#[macro_export]
macro_rules! lifetime_wrapper_generate_as_ref_if_needed {
    ($Name:ident, [owned, ref $(, $rest:ident)*]) => {
        paste::paste! {
            impl $Name {
                /// Convert to a borrowed reference with explicit lifetime
                pub fn as_ref(&self) -> [<$Name Ref>]<'_> {
                    [<$Name Ref>] {
                        ptr: self.ptr,
                        _lifetime: std::marker::PhantomData,
                    }
                }
            }
        }
    };
    ($Name:ident, [ref, owned $(, $rest:ident)*]) => {
        paste::paste! {
            impl $Name {
                /// Convert to a borrowed reference with explicit lifetime
                pub fn as_ref(&self) -> [<$Name Ref>]<'_> {
                    [<$Name Ref>] {
                        ptr: self.ptr,
                        _lifetime: std::marker::PhantomData,
                    }
                }
            }
        }
    };
    ($Name:ident, [$first:ident $(, $rest:ident)*]) => {
        $crate::lifetime_wrapper_generate_as_ref_if_needed!($Name, [$($rest),*]);
    };
    ($Name:ident, []) => {};
}

// Generate as_mut_ref method if both owned and mut_ref exist
#[macro_export]
macro_rules! lifetime_wrapper_generate_as_mut_ref_if_needed {
    ($Name:ident, [owned, mut_ref $(, $rest:ident)*]) => {
        paste::paste! {
            impl $Name {
                /// Convert to a mutable borrowed reference with explicit lifetime
                pub fn as_mut_ref(&mut self) -> [<$Name MutRef>]<'_> {
                    [<$Name MutRef>] {
                        ptr: self.ptr,
                        _lifetime: std::marker::PhantomData,
                    }
                }
            }
        }
    };
    ($Name:ident, [mut_ref, owned $(, $rest:ident)*]) => {
        paste::paste! {
            impl $Name {
                /// Convert to a mutable borrowed reference with explicit lifetime
                pub fn as_mut_ref(&mut self) -> [<$Name MutRef>]<'_> {
                    [<$Name MutRef>] {
                        ptr: self.ptr,
                        _lifetime: std::marker::PhantomData,
                    }
                }
            }
        }
    };
    ($Name:ident, [$first:ident $(, $rest:ident)*]) => {
        $crate::lifetime_wrapper_generate_as_mut_ref_if_needed!($Name, [$($rest),*]);
    };
    ($Name:ident, []) => {};
}

// Generate Deref implementation for MutRef -> Ref if both exist
#[macro_export]
macro_rules! lifetime_wrapper_mut_ref_deref_if_both_exist {
    ($Name:ident, [mut_ref, ref $(, $rest:ident)*]) => {
        $crate::lifetime_wrapper_generate_deref!($Name);
    };
    ($Name:ident, [ref, mut_ref $(, $rest:ident)*]) => {
        $crate::lifetime_wrapper_generate_deref!($Name);
    };
    ($Name:ident, [mut_ref $(, $rest:ident)*]) => {
        $crate::lifetime_wrapper_mut_ref_deref_if_both_exist!($Name, [$($rest),*]);
    };
    ($Name:ident, [ref $(, $rest:ident)*]) => {
        $crate::lifetime_wrapper_mut_ref_deref_if_both_exist!($Name, [$($rest),*]);
    };
    ($Name:ident, [$first:ident $(, $rest:ident)*]) => {
        $crate::lifetime_wrapper_mut_ref_deref_if_both_exist!($Name, [$($rest),*]);
    };
    ($Name:ident, []) => {};
}

// Actually generate the Deref implementation
#[macro_export]
macro_rules! lifetime_wrapper_generate_deref {
    ($Name:ident) => {
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
}

// TODO(joe): replace with lifetime_wrapper!
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
