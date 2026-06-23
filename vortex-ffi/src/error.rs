// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::any::Any;
use std::panic::AssertUnwindSafe;
use std::panic::catch_unwind;
use std::ptr;
use std::sync::Arc;

use vortex::error::VortexResult;

use crate::box_wrapper;
use crate::string::vx_string;

pub(crate) struct VortexError {
    message: Arc<str>,
}

box_wrapper!(
    /// The error structure populated by fallible Vortex C functions.
    VortexError,
    vx_error
);

/// Create an owned Vortex FFI error from a message.
pub(crate) fn vx_error_new(message: &str) -> *mut vx_error {
    vx_error::new(VortexError {
        message: message.into(),
    })
}

/// Write an error message to `error` which has not been populated before.
/// A null `error` pointer discards the message.
pub(crate) fn write_error(error: *mut *mut vx_error, message: &str) {
    if error.is_null() {
        return;
    }
    unsafe { error.write(vx_error_new(message)) };
}

/// Clear `*error_out` to null unless `error_out` itself is null.
fn clear_error(error_out: *mut *mut vx_error) {
    if error_out.is_null() {
        return;
    }
    unsafe { error_out.write(ptr::null_mut()) };
}

/// Convert a panic payload into the message stored in an FFI error.
fn panic_message(payload: &(dyn Any + Send)) -> String {
    if let Some(message) = payload.downcast_ref::<&str>() {
        format!("panic in Vortex FFI function: {message}")
    } else if let Some(message) = payload.downcast_ref::<String>() {
        format!("panic in Vortex FFI function: {message}")
    } else {
        "panic in Vortex FFI function".to_string()
    }
}

#[inline]
pub fn try_or_default<T: Default>(
    error_out: *mut *mut vx_error,
    function: impl FnOnce() -> VortexResult<T>,
) -> T {
    match catch_unwind(AssertUnwindSafe(function)) {
        Ok(Ok(value)) => {
            clear_error(error_out);
            value
        }
        Ok(Err(err)) => {
            write_error(error_out, &err.to_string());
            T::default()
        }
        Err(payload) => {
            write_error(error_out, &panic_message(payload.as_ref()));
            T::default()
        }
    }
}

/// Run `function`, returning its value on success and `error_value` on failure.
///
/// `error_out` may be null, in which case error details are discarded. When it is non-null,
/// `*error_out` is cleared to null on success and set to an owned `vx_error` on failure.
pub fn try_or<T>(
    error_out: *mut *mut vx_error,
    error_value: T,
    function: impl FnOnce() -> VortexResult<T>,
) -> T {
    match catch_unwind(AssertUnwindSafe(function)) {
        Ok(Ok(value)) => {
            clear_error(error_out);
            value
        }
        Ok(Err(err)) => {
            write_error(error_out, &err.to_string());
            error_value
        }
        Err(payload) => {
            write_error(error_out, &panic_message(payload.as_ref()));
            error_value
        }
    }
}

/// Returns the error message from the given Vortex error.
///
/// The returned pointer is valid as long as the error is valid.
/// Do NOT free the returned string pointer - it shares the lifetime of the error.
#[unsafe(no_mangle)]
pub unsafe extern "C-unwind" fn vx_error_get_message(error: *const vx_error) -> *const vx_string {
    vx_string::new_ref(&vx_error::as_ref(error).message)
}

#[cfg(test)]
mod tests {
    use std::ptr;

    use vortex::error::vortex_err;

    use super::*;
    use crate::error::vx_error_free;

    #[test]
    fn test_try_or_null_error_out() {
        // A null error_out must be tolerated on both the success and failure paths.
        assert_eq!(try_or(ptr::null_mut(), -1, || Ok(42)), 42);
        assert_eq!(try_or(ptr::null_mut(), -1, || Err(vortex_err!("boom"))), -1);
    }

    #[test]
    fn test_try_or_default_null_error_out() {
        assert_eq!(try_or_default(ptr::null_mut(), || Ok(42)), 42);
        assert_eq!(
            try_or_default::<i32>(ptr::null_mut(), || Err(vortex_err!("boom"))),
            0
        );
    }

    #[test]
    fn test_try_or_writes_and_clears_error_out() {
        let mut error: *mut vx_error = ptr::null_mut();

        assert_eq!(try_or(&raw mut error, -1, || Err(vortex_err!("boom"))), -1);
        assert!(!error.is_null());
        unsafe { vx_error_free(error) };

        assert_eq!(try_or(&raw mut error, -1, || Ok(42)), 42);
        assert!(error.is_null());
    }

    #[test]
    fn test_try_or_catches_panic() {
        let mut error: *mut vx_error = ptr::null_mut();

        assert_eq!(try_or(&raw mut error, -1, || panic!("boom")), -1);
        assert!(!error.is_null());
        let message = unsafe { vx_error_get_message(error) };
        assert_eq!(
            vx_string::as_ref(message).as_ref(),
            "panic in Vortex FFI function: boom"
        );
        unsafe { vx_error_free(error) };
    }
}
