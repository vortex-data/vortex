// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::ffi::{CStr, c_char};
use std::fmt::{Debug, Display, Formatter};
use std::str::Utf8Error;

use crate::cpp::*;

/// Wraps a heap allocated DuckDB string.
pub struct String {
    ptr: *const c_char,
}

impl String {
    pub fn from_ptr(ptr: *const c_char) -> Self {
        String { ptr }
    }

    pub fn as_cstr(&self) -> &CStr {
        unsafe { CStr::from_ptr(self.ptr) }
    }

    pub fn to_str(&self) -> Result<&str, Utf8Error> {
        self.as_cstr().to_str()
    }
}

impl Debug for String {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:?}", self.as_cstr())
    }
}

impl Display for String {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.as_cstr().to_string_lossy())
    }
}

impl Drop for String {
    fn drop(&mut self) {
        unsafe { duckdb_free(self.ptr.cast_mut().cast()) };
    }
}

/// Safely convert a C string pointer to a Rust String using length-based copying
/// This is more efficient than null-terminated string copying and safer than CStr
pub unsafe fn c_string_to_rust_string(
    str_ptr: *mut std::os::raw::c_char,
) -> Option<std::string::String> {
    if str_ptr.is_null() {
        return None;
    }

    let len = unsafe { duckdb_vx_c_string_length(str_ptr) };
    if len == 0 {
        unsafe { duckdb_vx_free_string(str_ptr) };
        return Some(std::string::String::new());
    }

    let slice = unsafe { std::slice::from_raw_parts(str_ptr as *const u8, len as usize) };
    let result = std::string::String::from_utf8_lossy(slice).into_owned();
    unsafe { duckdb_vx_free_string(str_ptr) };
    Some(result)
}

/// Wrapper for duckdb_vx_string that provides safe access to C++ std::string
pub struct VxString {
    ptr: duckdb_vx_string,
}

impl VxString {
    /// Create a VxString from a duckdb_vx_string pointer (takes ownership)
    pub unsafe fn from_raw(ptr: duckdb_vx_string) -> Option<Self> {
        if ptr.is_null() {
            None
        } else {
            Some(VxString { ptr })
        }
    }

    /// Get the length of the string
    pub fn len(&self) -> usize {
        unsafe { duckdb_vx_string_length(self.ptr) as usize }
    }

    /// Check if the string is empty
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Get the string data as a C string pointer
    pub fn as_ptr(&self) -> *const std::os::raw::c_char {
        unsafe { duckdb_vx_string_data(self.ptr) }
    }

    /// Convert to Rust String
    pub fn to_string(&self) -> std::string::String {
        if self.ptr.is_null() {
            return std::string::String::new();
        }

        let len = self.len();
        if len == 0 {
            return std::string::String::new();
        }

        let c_str = self.as_ptr();
        let slice = unsafe { std::slice::from_raw_parts(c_str as *const u8, len) };
        std::string::String::from_utf8_lossy(slice).into_owned()
    }
}

impl Drop for VxString {
    fn drop(&mut self) {
        if !self.ptr.is_null() {
            unsafe { duckdb_vx_string_free(self.ptr) };
        }
    }
}

impl Display for VxString {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.to_string())
    }
}

impl Debug for VxString {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "VxString(\"{}\")", self.to_string())
    }
}
