use std::ffi::{c_char, c_int};
use std::ptr;

use vortex::error::VortexResult;

#[repr(C)]
pub struct vx_error {
    pub code: c_int,
    pub message: *const c_char,
}

pub fn try_or<T>(
    error: *mut *mut vx_error,
    default_value: T,
    mut function: impl FnMut() -> VortexResult<T>,
) -> T {
    match function() {
        Ok(value) => {
            unsafe { error.write(ptr::null_mut()) };
            value
        }
        Err(err) => {
            #[allow(clippy::expect_used)]
            let c_string =
                std::ffi::CString::new(err.to_string()).expect("Failed to create CString");
            unsafe {
                error.write(
                    Box::into_raw(Box::new(vx_error {
                        code: -1,
                        message: c_string.into_raw(),
                    }))
                    .cast(),
                )
            };
            default_value
        }
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "C-unwind" fn vx_error_free(error: *mut vx_error) {
    drop(unsafe { Box::from_raw(error) })
}
