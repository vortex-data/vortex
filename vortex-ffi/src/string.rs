// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::ffi::CStr;
use std::ffi::c_char;
use std::slice;

use vortex::error::VortexExpect;
use vortex::error::vortex_err;

use crate::arc_dyn_wrapper;

arc_dyn_wrapper!(
    /// Strings for use within Vortex.
    str,
    vx_string
);

impl vx_string {
    #[allow(dead_code)]
    pub(crate) fn as_str(ptr: *const vx_string) -> &'static str {
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
        .map_err(|e| vortex_err!("invalid utf-8: {e}"))
        .vortex_expect("CString creation should succeed");
    vx_string::new(string.into())
}

/// Create a new Vortex UTF-8 string by copying from a null-terminated C-style string.
#[unsafe(no_mangle)]
pub unsafe extern "C-unwind" fn vx_string_new_from_cstr(ptr: *const c_char) -> *const vx_string {
    let string = unsafe { CStr::from_ptr(ptr) }
        .to_str()
        .map_err(|e| vortex_err!("invalid utf-8: {e}"))
        .vortex_expect("CString creation should succeed");
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

#[cfg(test)]
mod tests {
    use std::ffi::CString;

    use super::*;

    #[test]
    fn test_string_new() {
        unsafe {
            let test_str = "hello world";
            let ptr = test_str.as_ptr().cast();
            let len = test_str.len();

            let vx_str = vx_string_new(ptr, len);
            assert_eq!(vx_string_len(vx_str), 11);
            assert_eq!(vx_string::as_str(vx_str), "hello world");

            vx_string_free(vx_str);
        }
    }

    #[test]
    fn test_string_new_from_cstr() {
        unsafe {
            let c_string = CString::new("test string").unwrap();
            let vx_str = vx_string_new_from_cstr(c_string.as_ptr());

            assert_eq!(vx_string_len(vx_str), 11);
            assert_eq!(vx_string::as_str(vx_str), "test string");

            vx_string_free(vx_str);
        }
    }

    #[test]
    fn test_string_ptr() {
        unsafe {
            let test_str = "testing";
            let vx_str = vx_string::new(test_str.into());

            let ptr = vx_string_ptr(vx_str);
            let len = vx_string_len(vx_str);

            let slice = slice::from_raw_parts(ptr.cast::<u8>(), len);
            let recovered = str::from_utf8_unchecked(slice);
            assert_eq!(recovered, "testing");

            vx_string_free(vx_str);
        }
    }

    #[test]
    fn test_empty_string() {
        unsafe {
            let empty = "";
            let ptr = empty.as_ptr().cast();
            let vx_str = vx_string_new(ptr, 0);

            assert_eq!(vx_string_len(vx_str), 0);
            assert_eq!(vx_string::as_str(vx_str), "");

            vx_string_free(vx_str);
        }
    }

    #[test]
    fn test_unicode_string() {
        unsafe {
            let unicode_str = "Hello 世界 🌍";
            let ptr = unicode_str.as_ptr().cast();
            let len = unicode_str.len();

            let vx_str = vx_string_new(ptr, len);
            assert_eq!(vx_string_len(vx_str), unicode_str.len());
            assert_eq!(vx_string::as_str(vx_str), unicode_str);

            vx_string_free(vx_str);
        }
    }
}
