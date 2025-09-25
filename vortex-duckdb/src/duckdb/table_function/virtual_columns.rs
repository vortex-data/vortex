// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::ffi::c_void;

use vortex::error::VortexExpect;

use crate::duckdb::{LogicalType, TableFunction};
use crate::{cpp, wrapper};

/// Native callback for the get_virtual_columns function.
pub(crate) unsafe extern "C-unwind" fn get_virtual_columns_callback<T: TableFunction>(
    bind_data: *mut c_void,
    result: cpp::duckdb_vx_tfunc_virtual_cols_result,
) {
    let bind_data =
        unsafe { bind_data.cast::<T::BindData>().as_ref() }.vortex_expect("bind_data null pointer");
    let mut result = unsafe { VirtualColumnsResult::borrow(result) };

    T::virtual_columns(bind_data, &mut result);
}

wrapper!(
    VirtualColumnsResult,
    cpp::duckdb_vx_tfunc_virtual_cols_result,
    |_| {}
);

impl VirtualColumnsResult {
    pub fn register(&self, column_idx: u64, name: &str, logical_type: &LogicalType) {
        unsafe {
            cpp::duckdb_vx_tfunc_virtual_columns_push(
                self.as_ptr(),
                column_idx as _,
                name.as_ptr().cast(),
                name.len() as _,
                logical_type.as_ptr(),
            )
        }
    }
}
