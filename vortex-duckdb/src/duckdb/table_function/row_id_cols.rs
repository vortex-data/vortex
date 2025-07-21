// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use crate::duckdb::TableFunction;
use crate::{cpp, wrapper};
use std::ffi::c_void;
use vortex::error::VortexExpect;

/// Native callback for the get_row_id_columns function.
pub(crate) unsafe extern "C" fn get_row_id_columns_callback<T: TableFunction>(
    bind_data: *mut c_void,
    result: cpp::duckdb_vx_tfunc_row_id_cols_result,
) {
    let bind_data =
        unsafe { bind_data.cast::<T::BindData>().as_ref() }.vortex_expect("bind_data null pointer");
    let mut result = unsafe { RowIdColsResult::borrow(result) };
    T::row_id_columns(bind_data, &mut result);
}

wrapper!(
    RowIdColsResult,
    cpp::duckdb_vx_tfunc_row_id_cols_result,
    |_| {}
);

impl RowIdColsResult {
    pub fn push(&self, column_idx: u64) {
        unsafe { cpp::duckdb_vx_tfunc_row_id_cols_push(self.as_ptr(), column_idx as _) }
    }
}
