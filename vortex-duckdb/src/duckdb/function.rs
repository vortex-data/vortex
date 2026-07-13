// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex::error::VortexExpect;
use vortex::error::vortex_err;

use crate::cpp;
use crate::lifetime_wrapper;

lifetime_wrapper!(ScalarFunction, cpp::duckdb_vx_sfunc, |_| {});
lifetime_wrapper!(AggregateFunction, cpp::duckdb_vx_agg_func, |_| {});

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
}

impl AggregateFunctionRef {
    pub fn name(&self) -> &str {
        unsafe {
            let name_ptr = cpp::duckdb_vx_agg_func_name(self.as_ptr());
            std::ffi::CStr::from_ptr(name_ptr)
                .to_str()
                .map_err(|e| vortex_err!("invalid utf-8: {e}"))
                .vortex_expect("aggregate function name should be valid UTF-8")
        }
    }
}
