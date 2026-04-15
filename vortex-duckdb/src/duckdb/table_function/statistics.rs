// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::ffi::c_void;
use std::ptr;

use vortex::error::VortexExpect;

use crate::cpp;
use crate::duckdb::ClientContext;
use crate::duckdb::TableFunction;

pub(crate) unsafe extern "C-unwind" fn statistics<T: TableFunction>(
    ctx: cpp::duckdb_client_context,
    bind_data: *const c_void,
    column_index: usize,
    stats_out: *mut cpp::duckdb_column_statistics,
) -> bool {
    let stats_out = unsafe { &mut *stats_out };
    let client_context = unsafe { ClientContext::borrow(ctx) };
    let bind_data =
        unsafe { bind_data.cast::<T::BindData>().as_ref() }.vortex_expect("bind_data null pointer");
    let Some(stats) = T::statistics(client_context, bind_data, column_index) else {
        return false;
    };
    stats_out.min = stats.min.map_or(ptr::null_mut(), |v| v.into_ptr());
    stats_out.max = stats.max.map_or(ptr::null_mut(), |v| v.into_ptr());
    stats_out.max_string_length = stats.max_string_length;
    stats_out.has_null = stats.has_null;
    true
}
