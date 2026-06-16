// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Error type for the Diplomat-based Vortex FFI.
//!
//! In the hand-written C ABI, fallible functions took an `error_out: *mut *mut vx_error`
//! out-parameter and used the `try_or` / `try_or_default` helpers to write an owned
//! `vx_error` on failure (and clear it to null on success). Diplomat does not use
//! out-parameters for errors. Instead, a fallible function returns
//! `Result<T, Box<VortexFfiError>>`, and Diplomat maps that `Result` onto the idiomatic
//! error-handling construct of each target language: exceptions in C++/JS/Python/Dart and
//! a `Result` in Kotlin. This module therefore only needs to define the opaque error type
//! and a way to read its message back out.

#[diplomat::bridge]
pub mod ffi {
    use std::fmt::Write;

    use diplomat_runtime::DiplomatWrite;

    /// An error returned by a fallible Vortex FFI function.
    ///
    /// This replaces the `vx_error` out-parameter type from the hand-written C ABI. It is
    /// produced when a Diplomat method returns `Err(..)`, and surfaces in the target language
    /// as an exception (C++/JS/Python/Dart) or `Result` (Kotlin).
    #[diplomat::opaque]
    pub struct VortexFfiError(pub(crate) String);

    impl VortexFfiError {
        /// Construct an error from a message. Intended for internal use when bridging a
        /// `vortex::error::VortexError` into the FFI boundary.
        pub(crate) fn new(message: impl Into<String>) -> Box<VortexFfiError> {
            Box::new(VortexFfiError(message.into()))
        }

        /// The human-readable error message.
        #[diplomat::attr(auto, getter)]
        pub fn message(&self, out: &mut DiplomatWrite) {
            // Infallible: writing to a DiplomatWrite cannot fail.
            let _ = write!(out, "{}", self.0);
        }
    }
}

/// Convert any `vortex::error::VortexError` into an owned FFI error.
///
/// This is the Diplomat analogue of `write_error` from the C ABI: rather than writing into an
/// out-parameter, it produces the boxed opaque error that a `Result` will carry.
impl From<vortex::error::VortexError> for Box<ffi::VortexFfiError> {
    fn from(err: vortex::error::VortexError) -> Self {
        ffi::VortexFfiError::new(err.to_string())
    }
}
