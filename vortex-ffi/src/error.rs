use std::ffi::{c_char, c_int};
use std::ptr;

use vortex::error::VortexResult;

#[repr(C)]
pub struct FFIError {
    pub code: c_int,
    pub message: *const c_char,
}

pub unsafe fn into_return_mut<T, V>(
    result: VortexResult<T>,
    to_result: impl Fn(T) -> V,
    default: V,
    error: *mut *mut FFIError,
) -> V {
    match result {
        Ok(file) => {
            error.write(ptr::null_mut());
            to_result(file)
        }
        Err(err) => {
            let c_string =
                std::ffi::CString::new(err.to_string()).expect("Failed to create CString");
            unsafe {
                error.write(
                    Box::into_raw(Box::new(FFIError {
                        code: -1,
                        message: c_string.into_raw(),
                    }))
                    .cast(),
                )
            };
            default
        }
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn FFIError_free(error: *mut FFIError) {
    drop(unsafe { Box::from_raw(error) })
}
