use std::ffi::{CString, c_char, c_int};
use std::ptr;

use vortex::error::{VortexExpect, VortexResult};

/// The error structure populated by fallible Vortex C functions.
#[allow(non_camel_case_types)]
pub struct vx_error {
    code: c_int,
    message: CString,
}

#[inline]
pub fn try_or<T>(
    error_out: *mut *mut vx_error,
    on_err: T,
    function: impl FnOnce() -> VortexResult<T>,
) -> T {
    match function() {
        Ok(value) => {
            unsafe { error_out.write(ptr::null_mut()) };
            value
        }
        Err(err) => {
            #[allow(clippy::expect_used)]
            let c_string = CString::new(err.to_string()).expect("Error string contains null byte");
            let error = Box::new(vx_error {
                code: -1,
                message: c_string,
            });
            unsafe { error_out.write(Box::into_raw(error)) };
            on_err
        }
    }
}

/// Return the integer error code from the given Vortex error.
#[unsafe(no_mangle)]
pub unsafe extern "C-unwind" fn vx_error_get_code(error: *mut vx_error) -> c_int {
    unsafe { error.as_ref() }.vortex_expect("error null").code
}

/// Passes out an unowned reference to the error message from the given Vortex error.
/// Return value is the length of the message string.
#[unsafe(no_mangle)]
pub unsafe extern "C-unwind" fn vx_error_get_message(error: *mut vx_error) -> *const c_char {
    unsafe { error.as_ref() }
        .vortex_expect("error null")
        .message
        .as_ptr()
}

#[unsafe(no_mangle)]
pub unsafe extern "C-unwind" fn vx_error_free(error: *mut vx_error) {
    drop(unsafe { Box::from_raw(error) })
}
