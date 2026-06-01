// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use crate::cpp;
use crate::duckdb::LogicalTypeRef;
use crate::duckdb::Value;
use crate::lifetime_wrapper;

lifetime_wrapper!(BindInput, cpp::duckdb_vx_tfunc_bind_input, |_| {});

impl BindInputRef {
    /// Returns the parameter at the given index.
    pub fn get_parameter(&self, index: usize) -> Option<Value> {
        let value_ptr =
            unsafe { cpp::duckdb_vx_tfunc_bind_input_get_parameter(self.as_ptr(), index as _) };
        if value_ptr.is_null() {
            None
        } else {
            Some(unsafe { Value::own(value_ptr) })
        }
    }
}

lifetime_wrapper!(BindResult, cpp::duckdb_vx_tfunc_bind_result, |_| {});

impl BindResultRef {
    pub fn add_result_column(&self, name: &str, logical_type: &LogicalTypeRef) {
        unsafe {
            cpp::duckdb_vx_tfunc_bind_result_add_column(
                self.as_ptr(),
                name.as_ptr().cast(),
                name.len() as _,
                logical_type.as_ptr(),
            )
        }
    }
}
