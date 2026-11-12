// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! CUDA driver errors and status conversion.

use std::ffi::CStr;
use std::ffi::c_char;
use std::fmt;

use crate::sys;
use crate::sys::CUresult;

/// A failed CUDA driver call.
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub struct DriverError(pub CUresult);

impl DriverError {
    /// The driver's name for the error, such as `CUDA_ERROR_INVALID_VALUE`.
    pub fn error_name(&self) -> Result<&'static CStr, DriverError> {
        self.describe(sys::cuGetErrorName)
    }

    /// The driver's description of the error.
    pub fn error_string(&self) -> Result<&'static CStr, DriverError> {
        self.describe(sys::cuGetErrorString)
    }

    /// Returns the static error name or description written by `describe`.
    fn describe(
        &self,
        describe: unsafe fn(CUresult, *mut *const c_char) -> CUresult,
    ) -> Result<&'static CStr, DriverError> {
        // Errors are formatted even when the driver library is missing.
        if sys::driver_load_error().is_some() {
            return Err(*self);
        }
        let mut text = std::ptr::null();
        // SAFETY: The driver writes a pointer to a static string.
        unsafe { describe(self.0, &raw mut text) }.result()?;
        // SAFETY: Successful calls return a NUL-terminated static string.
        Ok(unsafe { CStr::from_ptr(text) })
    }
}

impl fmt::Debug for DriverError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if let Some(reason) = sys::driver_load_error() {
            return write!(
                f,
                "DriverError({}, CUDA driver unavailable: {reason})",
                self.0.0
            );
        }
        match self.error_name() {
            Ok(name) => write!(f, "DriverError({}, {})", self.0.0, name.to_string_lossy()),
            Err(_) => write!(f, "DriverError({})", self.0.0),
        }
    }
}

impl fmt::Display for DriverError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if let Some(reason) = sys::driver_load_error() {
            return write!(f, "CUDA driver unavailable: {reason}");
        }
        match (self.error_name(), self.error_string()) {
            (Ok(name), Ok(description)) => write!(
                f,
                "{}: {}",
                name.to_string_lossy(),
                description.to_string_lossy()
            ),
            _ => write!(f, "CUDA driver error {}", self.0.0),
        }
    }
}

impl std::error::Error for DriverError {}

/// Reports a destructor or secondary error that cannot be returned to the caller.
pub(crate) fn log_error(operation: &'static str, error: DriverError) {
    tracing::error!(%error, operation, "CUDA operation failed");
}

impl CUresult {
    /// `Ok` for [`CUresult::CUDA_SUCCESS`], otherwise the error.
    #[inline]
    pub fn result(self) -> Result<(), DriverError> {
        if self == CUresult::CUDA_SUCCESS {
            Ok(())
        } else {
            Err(DriverError(self))
        }
    }
}
