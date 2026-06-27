// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! A minimal, allocation- and formatting-free error type for guest kernels.
//!
//! Kernels deliberately avoid `vortex-error` (which pulls in `jiff`, `prost`, and `arrow-schema`,
//! and the `core::fmt` formatting machinery) to keep the compiled `.wasm` small. A [`GuestError`]
//! carries only a `&'static str`, so error paths need no allocation or formatting.

/// A guest-side error carrying a static description. No allocation, no formatting.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GuestError(pub &'static str);

impl GuestError {
    /// Construct an error from a static message.
    pub const fn new(message: &'static str) -> Self {
        Self(message)
    }

    /// The static error message.
    pub const fn message(self) -> &'static str {
        self.0
    }
}

/// Result alias for guest operations.
pub type GuestResult<T> = Result<T, GuestError>;

/// Return early with a [`GuestError`] built from a static string.
#[macro_export]
macro_rules! guest_bail {
    ($msg:literal) => {
        return ::core::result::Result::Err($crate::GuestError::new($msg))
    };
}

/// Return a [`GuestError`] unless a condition holds.
#[macro_export]
macro_rules! guest_ensure {
    ($cond:expr, $msg:literal) => {
        if !($cond) {
            return ::core::result::Result::Err($crate::GuestError::new($msg));
        }
    };
}
