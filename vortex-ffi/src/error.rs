use std::ffi::{c_char, c_int};
use std::ptr;

use vortex::error::VortexResult;

#[repr(C)]
pub struct FFIError {
    pub code: c_int,
    pub message: *const c_char,
}

pub fn try_or<F, ValueT>(error: *mut *mut FFIError, default_value: ValueT, function: F) -> ValueT
where
    F: Fn() -> VortexResult<ValueT>,
{
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
                    Box::into_raw(Box::new(FFIError {
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
pub unsafe extern "C" fn FFIError_free(error: *mut FFIError) {
    drop(unsafe { Box::from_raw(error) })
}
