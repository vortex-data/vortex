// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use crate::cpp;
use crate::duckdb::LogicalTypeRef;
use crate::lifetime_wrapper;

lifetime_wrapper!(BindResult, cpp::duckdb_bind_result, |_| {});

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
