use std::ptr;

use vortex::error::VortexResult;

use crate::string::{vx_string, vx_string_free};

/// The error structure populated by fallible Vortex C functions.
// NOTE(ngates): our errors are passed back out as opaque structs, so while we currently alias
// `vx_string`, we could change this to a different type in the future without breaking the API.
#[allow(non_camel_case_types)]
#[allow(dead_code)]
pub struct vx_error(vx_string);

#[inline]
pub fn try_or<T>(
    error_out: *mut *const vx_error,
    on_err: T,
    function: impl FnOnce() -> VortexResult<T>,
) -> T {
    match function() {
        Ok(value) => {
            unsafe { error_out.write(ptr::null_mut()) };
            value
        }
        Err(err) => {
            let err = vx_string::new(err.to_string().into()).cast::<vx_error>();
            unsafe { error_out.write(err) };
            on_err
        }
    }
}

/// Returns a borrowed reference to the error message from the given Vortex error.
#[unsafe(no_mangle)]
pub unsafe extern "C-unwind" fn vx_error_get_message(error: *const vx_error) -> *const vx_string {
    error.cast()
}

#[unsafe(no_mangle)]
pub unsafe extern "C-unwind" fn vx_error_free(error: *const vx_error) {
    unsafe { vx_string_free(error.cast::<vx_string>()) };
}
