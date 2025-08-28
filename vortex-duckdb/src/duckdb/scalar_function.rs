// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex::error::{VortexUnwrap, vortex_err};

use crate::duckdb::LogicalType;
use crate::{cpp, wrapper};

wrapper!(ScalarFunction, cpp::duckdb_vx_sfunc, |_| {});

impl ScalarFunction {
    pub fn name(&self) -> &str {
        unsafe {
            let name_ptr = cpp::duckdb_vx_sfunc_name(self.as_ptr());
            std::ffi::CStr::from_ptr(name_ptr)
                .to_str()
                .map_err(|e| vortex_err!("invalid utf-8: {e}"))
                .vortex_unwrap()
        }
    }

    pub fn return_type(&self) -> LogicalType {
        unsafe { LogicalType::borrow(cpp::duckdb_vx_sfunc_return_type(self.as_ptr())) }
    }
}
