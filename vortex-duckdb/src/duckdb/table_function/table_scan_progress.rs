// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex::error::VortexExpect;

use crate::duckdb::TableFunction;

pub(crate) unsafe extern "C-unwind" fn table_scan_progress_callback<T: TableFunction>(
    ctx: crate::cpp::duckdb_client_context,
    bind_data: *mut ::std::os::raw::c_void,
    global_state: *mut ::std::os::raw::c_void,
) -> f64 {
    let ctx = unsafe { crate::duckdb::ClientContext::borrow(ctx) };
    let bind_data =
        unsafe { bind_data.cast::<T::BindData>().as_mut() }.vortex_expect("bind_data null pointer");
    let global_state = unsafe { global_state.cast::<T::GlobalState>().as_mut() }
        .vortex_expect("global_init_data null pointer");
    T::table_scan_progress(ctx, bind_data, global_state)
}
