// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

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

#[inline]
pub fn try_or_default<T: Default>(
    error_out: *mut *mut vx_error,
    function: impl FnOnce() -> VortexResult<T>,
) -> T {
    match function() {
        Ok(value) => {
            unsafe { error_out.write(ptr::null_mut()) };
            value
        }
        Err(err) => {
            let err = vx_error::new(Box::new(VortexError {
                message: err.to_string().into(),
            }));
            unsafe { error_out.write(err) };
            T::default()
        }
    }
}

/// Returns a borrowed reference to the error message from the given Vortex error.
#[unsafe(no_mangle)]
pub unsafe extern "C-unwind" fn vx_error_get_message(error: *const vx_error) -> *const vx_string {
    vx_string::new_ref(&vx_error::as_ref(error).message)
}
