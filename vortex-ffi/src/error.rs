use std::ffi::{CStr, c_char, c_int};
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
    error: *mut *const FFIError,
) -> V {
    match result {
        Ok(file) => Box::into_raw(Box::new(file)),
        Err(err) => {
            let cstr: CStr = err.into();
            unsafe {
                error.write(
                    Box::into_raw(Box::new(FFIError {
                        code: -1,
                        message: cstr.as_ptr(),
                    }))
                    .cast(),
                )
            };
            ptr::null_mut()
        }
    }
}

#[unsafe(no_mangle)]
pub fn free_error(error: *mut FFIError) {
    drop(unsafe { Box::from_raw(error) })
}
