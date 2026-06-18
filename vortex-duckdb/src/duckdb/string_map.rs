// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::ffi::CString;

use crate::cpp;

// String map lifetime is managed by C++ code
crate::lifetime_wrapper!(DuckdbStringMap, cpp::duckdb_vx_string_map, |_| {});
impl DuckdbStringMapRef {
    pub fn push(&mut self, key: &str, value: &str) {
        let key = CString::new(key).unwrap_or_else(|_| CString::default());
        let value = CString::new(value).unwrap_or_else(|_| CString::default());
        unsafe {
            cpp::duckdb_vx_string_map_insert(self.as_ptr(), key.as_ptr(), value.as_ptr());
        }
    }
}
