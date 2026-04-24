// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::ffi::c_char;
use std::slice;

use crate::arc_dyn_wrapper;

arc_dyn_wrapper!(
    /// Strings for use within Vortex.
    [u8],
    vx_binary
);

impl vx_binary {
    #[allow(dead_code)]
    pub(crate) fn as_slice(ptr: *const vx_binary) -> &'static [u8] {
        unsafe { slice::from_raw_parts(vx_binary_ptr(ptr).cast(), vx_binary_len(ptr)) }
    }
}

/// Create a new Vortex UTF-8 string by copying from a pointer and length.
#[unsafe(no_mangle)]
pub unsafe extern "C-unwind" fn vx_binary_new(ptr: *const c_char, len: usize) -> *const vx_binary {
    let slice = unsafe { slice::from_raw_parts(ptr.cast(), len) };
    vx_binary::new(slice.into())
}

/// Return the length of the string in bytes.
#[unsafe(no_mangle)]
pub unsafe extern "C-unwind" fn vx_binary_len(ptr: *const vx_binary) -> usize {
    vx_binary::as_ref(ptr).len()
}

/// Return the pointer to the string data.
#[unsafe(no_mangle)]
pub unsafe extern "C-unwind" fn vx_binary_ptr(ptr: *const vx_binary) -> *const c_char {
    vx_binary::as_ref(ptr).as_ptr().cast()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_string_new() {
        unsafe {
            let test_str = "hello world";
            let ptr = test_str.as_ptr().cast();
            let len = test_str.len();

            let vx_str = vx_binary_new(ptr, len);
            assert_eq!(vx_binary_len(vx_str), 11);
            assert_eq!(vx_binary::as_slice(vx_str), "hello world".as_bytes());

            vx_binary_free(vx_str);
        }
    }

    #[test]
    fn test_string_ptr() {
        unsafe {
            let test_str = "testing".as_bytes();
            let vx_str = vx_binary::new(test_str.into());

            let ptr = vx_binary_ptr(vx_str);
            let len = vx_binary_len(vx_str);

            let slice = slice::from_raw_parts(ptr.cast::<u8>(), len);
            assert_eq!(slice, "testing".as_bytes());

            vx_binary_free(vx_str);
        }
    }

    #[test]
    fn test_empty_string() {
        unsafe {
            let empty = "";
            let ptr = empty.as_ptr().cast();
            let vx_str = vx_binary_new(ptr, 0);

            assert_eq!(vx_binary_len(vx_str), 0);
            assert_eq!(vx_binary::as_slice(vx_str), "".as_bytes());

            vx_binary_free(vx_str);
        }
    }

    #[test]
    fn test_unicode_string() {
        unsafe {
            let unicode_str = "Hello 世界 🌍";
            let ptr = unicode_str.as_ptr().cast();
            let len = unicode_str.len();

            let vx_str = vx_binary_new(ptr, len);
            assert_eq!(vx_binary_len(vx_str), unicode_str.len());
            assert_eq!(vx_binary::as_slice(vx_str), unicode_str.as_bytes());

            vx_binary_free(vx_str);
        }
    }
}
