// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::ffi::CStr;
use std::fmt::Display;
use std::fmt::Formatter;

use vortex::dtype::FieldName;
use vortex::error::vortex_err;

use crate::cpp::duckdb_free;
use crate::wrapper;

wrapper!(
    #[derive(Debug)]
    DDBString,
    *mut std::ffi::c_char,
    |ptr: *mut std::ffi::c_char| {
        unsafe { CStr::from_ptr(ptr) }
            .to_str()
            .map_err(|e| vortex_err!("Failed to convert C string to str: {e}"))
            .expect("DuckDB string should be valid UTF-8")
    },
    |ptr: &mut *mut std::ffi::c_char| unsafe { duckdb_free((*ptr).cast()) }
);

impl Display for DDBString {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.as_ref())
    }
}

impl AsRef<str> for DDBString {
    fn as_ref(&self) -> &str {
        // SAFETY: The string have been validated on construction.
        unsafe { str::from_utf8_unchecked(CStr::from_ptr(self.ptr).to_bytes()) }
    }
}

impl PartialEq for DDBString {
    fn eq(&self, other: &Self) -> bool {
        self.as_ref() == other.as_ref()
    }
}

impl PartialEq<str> for DDBString {
    fn eq(&self, other: &str) -> bool {
        self.as_ref() == other
    }
}

impl From<DDBString> for FieldName {
    fn from(value: DDBString) -> Self {
        FieldName::from(value.as_ref())
    }
}
