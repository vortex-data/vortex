// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::ffi::{CStr, c_char};
use std::slice;

use vortex::error::{VortexExpect, vortex_err};

use crate::arc_dyn_wrapper;

arc_dyn_wrapper!(
    /// Strings for use within Vortex.
    str,
    vx_string
);

impl vx_string {
    #[allow(dead_code)]
    pub(crate) fn as_str<'a>(ptr: *const vx_string) -> &'a str {
        unsafe {
            str::from_utf8_unchecked(slice::from_raw_parts(
                vx_string_ptr(ptr).cast(),
                vx_string_len(ptr),
            ))
        }
    }
}

/// Create a new Vortex UTF-8 string by copying from a pointer and length.
#[unsafe(no_mangle)]
pub unsafe extern "C-unwind" fn vx_string_new(ptr: *const c_char, len: usize) -> *const vx_string {
    let slice = unsafe { slice::from_raw_parts(ptr.cast(), len) };
    let string = String::from_utf8(slice.to_vec())
        .map_err(|e| vortex_err!("Invalid UTF-8 string {e}"))
        .vortex_expect("Invalid UTF-8 string");
    vx_string::new(string.into())
}

/// Create a new Vortex UTF-8 string by copying from a null-terminated C-style string.
#[unsafe(no_mangle)]
pub unsafe extern "C-unwind" fn vx_string_new_from_cstr(ptr: *const c_char) -> *const vx_string {
    let string = unsafe { CStr::from_ptr(ptr) }
        .to_str()
        .vortex_expect("Invalid UTF-8 string");
    vx_string::new(string.into())
}

/// Return the length of the string in bytes.
#[unsafe(no_mangle)]
pub unsafe extern "C-unwind" fn vx_string_len(ptr: *const vx_string) -> usize {
    vx_string::as_ref(ptr).len()
}

/// Return the pointer to the string data.
#[unsafe(no_mangle)]
pub unsafe extern "C-unwind" fn vx_string_ptr(ptr: *const vx_string) -> *const c_char {
    vx_string::as_ref(ptr).as_ptr().cast()
}
