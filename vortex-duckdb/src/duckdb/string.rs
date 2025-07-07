// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::ffi::{CStr, c_char};
use std::fmt::{Debug, Display, Formatter};
use std::str::Utf8Error;

use crate::cpp;

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
        unsafe { cpp::duckdb_free(self.ptr.cast_mut().cast()) };
    }
}
