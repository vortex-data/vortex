// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use crate::cpp::idx_t;
use crate::duckdb::TableFunction;
use std::ffi::c_void;
use vortex::error::VortexExpect;

/// Native callback for the cardinality estimate of a table function.
pub(crate) unsafe extern "C-unwind" fn get_partition_data_callback<T: TableFunction>(
    bind_data: *const c_void,
    global_init_data: *mut c_void,
    local_init_data: *mut c_void,
) -> idx_t {
    let bind_data =
        unsafe { bind_data.cast::<T::BindData>().as_ref() }.vortex_expect("bind_data null pointer");
    let global_init_data = unsafe { global_init_data.cast::<T::GlobalState>().as_mut() }
        .vortex_expect("global_init_data null pointer");
    let local_init_data = unsafe { local_init_data.cast::<T::LocalState>().as_mut() }
        .vortex_expect("local_init_data null pointer");

    T::partition_data(bind_data, global_init_data, local_init_data)
}
