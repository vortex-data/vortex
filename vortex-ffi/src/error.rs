use std::ffi::{c_char, c_int};
use std::ptr;

use vortex::error::VortexResult;

#[repr(C)]
pub struct VXError {
    pub code: c_int,
    pub message: *const c_char,
}

pub unsafe fn into_c_error<V>(result: VortexResult<V>, default: V, error: *mut *mut VXError) -> V {
    map_into_c_error(result, |r| r, default, error)
}

pub unsafe fn map_into_c_error<T, V>(
    result: VortexResult<T>,
    to_result: impl Fn(T) -> V,
    default: V,
    error: *mut *mut VXError,
) -> V {
    match result {
        Ok(file) => {
            error.write(ptr::null_mut());
            to_result(file)
        }
        Err(err) => {
            #[allow(clippy::expect_used)]
            let c_string =
                std::ffi::CString::new(err.to_string()).expect("Failed to create CString");
            unsafe {
                error.write(
                    Box::into_raw(Box::new(VXError {
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
pub unsafe extern "C" fn vx_error_free(error: *mut VXError) {
    drop(unsafe { Box::from_raw(error) })
}
