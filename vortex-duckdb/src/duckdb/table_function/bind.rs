// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::ffi::CStr;

use vortex::error::vortex_err;

use crate::duckdb::data::Data;
use crate::duckdb::{LogicalType, TableFunction, Value, try_or_null};
use crate::{cpp, wrapper};

/// The native bind callback for a table function.
pub(crate) unsafe extern "C" fn bind_callback<T: TableFunction>(
    bind_input: cpp::duckdb_vx_tfunc_bind_input,
    bind_result: cpp::duckdb_vx_tfunc_bind_result,
    error_out: *mut cpp::duckdb_vx_error,
) -> cpp::duckdb_vx_data {
    let bind_input = unsafe { BindInput::own(bind_input) };
    let mut bind_result = unsafe { BindResult::own(bind_result) };

    try_or_null(error_out, || {
        let bind_data = T::bind(&bind_input, &mut bind_result)?;
        Ok(Data::from(Box::new(bind_data)).as_ptr())
    })
}

/// The native copy callback for bind data.
pub(crate) unsafe extern "C" fn bind_data_clone_callback<T: TableFunction>(
    bind_data: *const std::ffi::c_void,
    error_out: *mut cpp::duckdb_vx_error,
) -> cpp::duckdb_vx_data {
    try_or_null(error_out, || {
        let bind_data = unsafe {
            (bind_data as *const T::BindData)
                .as_ref()
                .ok_or(vortex_err!("bind_data is nullptr"))?
        };
        let copied_data = bind_data.clone();
        Ok(Data::from(Box::new(copied_data)).as_ptr())
    })
}

wrapper!(BindInput, cpp::duckdb_vx_tfunc_bind_input, |_| {});

impl BindInput {
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

    /// Returns the named parameter with the given name, if it exists.
    pub fn get_named_parameter(&self, name: &CStr) -> Option<Value> {
        let value_ptr = unsafe {
            cpp::duckdb_vx_tfunc_bind_input_get_named_parameter(self.as_ptr(), name.as_ptr())
        };
        if value_ptr.is_null() {
            None
        } else {
            Some(unsafe { Value::own(value_ptr) })
        }
    }

    /// Returns the number of parameters bound to this function.
    pub fn parameter_count(&self) -> usize {
        unsafe { cpp::duckdb_vx_tfunc_bind_input_get_parameter_count(self.as_ptr()) as usize }
    }
}

wrapper!(BindResult, cpp::duckdb_vx_tfunc_bind_result, |_| {});

impl BindResult {
    pub fn add_result_column(&self, name: &str, logical_type: &LogicalType) {
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
