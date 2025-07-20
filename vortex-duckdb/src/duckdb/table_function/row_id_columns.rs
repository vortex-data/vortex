// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use crate::cpp;
use crate::duckdb::TableFunction;
use std::ffi::c_void;
use vortex::error::VortexExpect;

/// Native callback for the get_row_id_columns function.
///
/// For now, we only support returning a single column index. This is sufficient for our use-case
/// and avoids dealing with allocating vectors across the FFI boundary.
pub(crate) unsafe extern "C" fn get_row_id_columns_callback<T: TableFunction>(
    bind_data: *mut c_void,
    col_idx_out: *mut cpp::idx_t,
) -> bool {
    let bind_data =
        unsafe { bind_data.cast::<T::BindData>().as_mut() }.vortex_expect("bind_data null pointer");

    match T::row_id_columns(bind_data) {
        None => false,
        Some(col_idx) => {
            unsafe { col_idx_out.write(col_idx as _) };
            true
        }
    }
}
