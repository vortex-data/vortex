// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::any::Any;
use std::panic::AssertUnwindSafe;
use std::panic::catch_unwind;
use std::ptr;
use std::sync::Arc;

use vortex::error::VortexError;
use vortex::error::VortexResult;

use crate::box_wrapper;
use crate::string::vx_string;

/// Error category for vx_error.
#[repr(C)]
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
#[expect(non_camel_case_types)]
pub enum vx_error_code {
    /// All other errors
    VX_ERROR_CODE_OTHER = 0,
    /// Index out of bounds
    VX_ERROR_CODE_OUT_OF_BOUNDS = 1,
    /// Compute kernel execute error
    VX_ERROR_CODE_COMPUTE = 2,
    /// An invalid argument was provided.
    VX_ERROR_CODE_INVALID_ARGUMENT = 3,
    /// Serialization/deserialization error
    VX_ERROR_CODE_SERIALIZATION = 4,
    /// Unimplemented function
    VX_ERROR_CODE_NOT_IMPLEMENTED = 5,
    /// Type mismatch
    VX_ERROR_CODE_MISMATCHED_TYPES = 6,
    /// Assertion failed
    VX_ERROR_CODE_ASSERTION_FAILED = 7,
    /// IO error
    VX_ERROR_CODE_IO = 8,
    /// Panic inside FFI
    VX_ERROR_CODE_PANIC = 9,
}

fn error_code(error: &VortexError) -> vx_error_code {
    match error {
        VortexError::OutOfBounds(..) => vx_error_code::VX_ERROR_CODE_OUT_OF_BOUNDS,
        VortexError::Compute(..) => vx_error_code::VX_ERROR_CODE_COMPUTE,
        VortexError::InvalidArgument(..) => vx_error_code::VX_ERROR_CODE_INVALID_ARGUMENT,
        VortexError::Serde(..) => vx_error_code::VX_ERROR_CODE_SERIALIZATION,
        VortexError::NotImplemented(..) => vx_error_code::VX_ERROR_CODE_NOT_IMPLEMENTED,
        VortexError::MismatchedTypes(..) => vx_error_code::VX_ERROR_CODE_MISMATCHED_TYPES,
        VortexError::AssertionFailed(..) => vx_error_code::VX_ERROR_CODE_ASSERTION_FAILED,
        VortexError::Io(..) => vx_error_code::VX_ERROR_CODE_IO,
        VortexError::Context(_, inner) => error_code(inner),
        VortexError::Shared(inner) => error_code(inner),
        _ => vx_error_code::VX_ERROR_CODE_OTHER,
    }
}

pub(crate) struct VortexFFIError {
    message: Arc<str>,
    code: vx_error_code,
}

box_wrapper!(
    /// The error structure populated by fallible Vortex C functions.
    VortexFFIError,
    vx_error
);

fn vx_error_new_with_code(message: &str, code: vx_error_code) -> *mut vx_error {
    vx_error::new(VortexFFIError {
        message: message.into(),
        code,
    })
}

/// Write an error message to `error` which has not been populated before.
/// A null `error` pointer discards the message.
pub(crate) fn write_error(error: *mut *mut vx_error, message: &str) {
    write_error_with_code(error, message, vx_error_code::VX_ERROR_CODE_OTHER);
}

fn write_error_with_code(error: *mut *mut vx_error, message: &str, code: vx_error_code) {
    if error.is_null() {
        return;
    }
    unsafe { error.write(vx_error_new_with_code(message, code)) };
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
            write_error_with_code(error_out, &err.to_string(), error_code(&err));
            T::default()
        }
        Err(payload) => {
            write_error_with_code(
                error_out,
                &panic_message(payload.as_ref()),
                vx_error_code::VX_ERROR_CODE_PANIC,
            );
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
            write_error_with_code(error_out, &err.to_string(), error_code(&err));
            error_value
        }
        Err(payload) => {
            write_error_with_code(
                error_out,
                &panic_message(payload.as_ref()),
                vx_error_code::VX_ERROR_CODE_PANIC,
            );
            error_value
        }
    }
}

/// Return an error message for this error
#[unsafe(no_mangle)]
pub unsafe extern "C-unwind" fn vx_error_get_message(error: *const vx_error) -> *const vx_string {
    vx_string::new(Arc::clone(&vx_error::as_ref(error).message))
}

/// Return category code for "error".
#[unsafe(no_mangle)]
pub unsafe extern "C-unwind" fn vx_error_get_code(error: *const vx_error) -> vx_error_code {
    vx_error::as_ref(error).code
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
        unsafe { crate::string::vx_string_free(message) };
        unsafe { vx_error_free(error) };
    }

    #[test]
    fn test_error_codes() {
        let mut error: *mut vx_error = ptr::null_mut();

        assert_eq!(
            try_or(&raw mut error, -1, || Err::<i32, _>(vortex_err!(
                OutOfBounds: 5, 0, 3
            ))),
            -1
        );
        assert_eq!(
            unsafe { vx_error_get_code(error) },
            vx_error_code::VX_ERROR_CODE_OUT_OF_BOUNDS
        );
        unsafe { vx_error_free(error) };

        assert_eq!(try_or(&raw mut error, -1, || panic!("panic")), -1);
        assert_eq!(
            unsafe { vx_error_get_code(error) },
            vx_error_code::VX_ERROR_CODE_PANIC
        );
        unsafe { vx_error_free(error) };
    }
}
