// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex::error::VortexExpect;
use vortex::error::vortex_err;

use crate::cpp;
use crate::duckdb::LogicalType;
use crate::duckdb::LogicalTypeRef;
use crate::lifetime_wrapper;

lifetime_wrapper!(ScalarFunction, cpp::duckdb_vx_sfunc, |_| {});

impl ScalarFunctionRef {
    pub fn name(&self) -> &str {
        unsafe {
            let name_ptr = cpp::duckdb_vx_sfunc_name(self.as_ptr());
            std::ffi::CStr::from_ptr(name_ptr)
                .to_str()
                .map_err(|e| vortex_err!("invalid utf-8: {e}"))
                .vortex_expect("scalar function name should be valid UTF-8")
        }
    }

    pub fn return_type(&self) -> &LogicalTypeRef {
        unsafe { LogicalType::borrow(cpp::duckdb_vx_sfunc_return_type(self.as_ptr())) }
    }
}
