// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::ffi::CStr;
use std::os::raw::{c_char, c_ulong, c_void};

use bitvec::macros::internal::funty::Fundamental;
use itertools::Itertools;
use vortex::error::VortexExpect;

use crate::cpp;
use crate::cpp::{
    duckdb_data_chunk, duckdb_logical_type, duckdb_vx_copy_func_bind_input, duckdb_vx_error,
};
use crate::duckdb::{CopyFunction, Data, DataChunk, LogicalType, try_or, try_or_null};

pub(crate) unsafe extern "C" fn bind_callback<T: CopyFunction>(
    // TODO(joe): pass this into T::bind(..)
    _input: duckdb_vx_copy_func_bind_input,
    column_names: *const *const c_char,
    column_name_count: c_ulong,
    column_types: *const duckdb_logical_type,
    column_type_count: c_ulong,
    error_out: *mut duckdb_vx_error,
) -> cpp::duckdb_vx_data {
    let column_names =
        unsafe { std::slice::from_raw_parts(column_names, column_name_count.as_usize()) }
            .iter()
            .map(|name| {
                unsafe { CStr::from_ptr(name.cast()) }
                    .to_string_lossy()
                    .into_owned()
            })
            .collect_vec();

    let column_types =
        unsafe { std::slice::from_raw_parts(column_types, column_type_count.as_usize()) }
            .iter()
            .map(|c| unsafe { LogicalType::borrow(c.cast()) })
            .collect_vec();

    try_or_null(error_out, || {
        let bind_data = T::bind(column_names, column_types)?;
        Ok(Data::from(Box::new(bind_data)).as_ptr())
    })
}

pub(crate) unsafe extern "C" fn global_callback<T: CopyFunction>(
    bind_data: *const c_void,
    file_path: *const c_char,
    error_out: *mut duckdb_vx_error,
) -> cpp::duckdb_vx_data {
    let file_path = unsafe { CStr::from_ptr(file_path) }
        .to_string_lossy()
        .into_owned();
    let bind_data = unsafe { bind_data.cast::<T::BindData>().as_ref() }
        .vortex_expect("global_init_data null pointer");
    try_or_null(error_out, || {
        let bind_data = T::init_global(bind_data, file_path)?;
        Ok(Data::from(Box::new(bind_data)).as_ptr())
    })
}

pub(crate) unsafe extern "C" fn local_callback<T: CopyFunction>(
    bind_data: *const c_void,
    error_out: *mut duckdb_vx_error,
) -> cpp::duckdb_vx_data {
    let bind_data =
        unsafe { bind_data.cast::<T::BindData>().as_ref() }.vortex_expect("bind_data null pointer");
    try_or_null(error_out, || {
        let bind_data = T::init_local(bind_data)?;
        Ok(Data::from(Box::new(bind_data)).as_ptr())
    })
}

pub(crate) unsafe extern "C" fn copy_to_sink_callback<T: CopyFunction>(
    bind_data: *const c_void,
    global_data: *mut c_void,
    local_data: *mut c_void,
    data_chunk: duckdb_data_chunk,
    error_out: *mut duckdb_vx_error,
) {
    let bind_data =
        unsafe { bind_data.cast::<T::BindData>().as_ref() }.vortex_expect("bind_data null pointer");
    let global_data = unsafe { global_data.cast::<T::GlobalState>().as_mut() }
        .vortex_expect("bind_data null pointer");
    let local_data = unsafe { local_data.cast::<T::LocalState>().as_mut() }
        .vortex_expect("bind_data null pointer");

    try_or(error_out, || {
        T::copy_to_sink(bind_data, global_data, local_data, &mut unsafe {
            DataChunk::borrow(data_chunk)
        })?;
        Ok(())
    })
}

pub(crate) unsafe extern "C" fn copy_to_finalize_callback<T: CopyFunction>(
    bind_data: *const c_void,
    global_data: *mut c_void,
    error_out: *mut duckdb_vx_error,
) {
    let bind_data =
        unsafe { bind_data.cast::<T::BindData>().as_ref() }.vortex_expect("bind_data null pointer");
    let global_data = unsafe { global_data.cast::<T::GlobalState>().as_mut() }
        .vortex_expect("bind_data null pointer");

    try_or(error_out, || {
        T::copy_to_finalize(bind_data, global_data)?;
        Ok(())
    })
}
