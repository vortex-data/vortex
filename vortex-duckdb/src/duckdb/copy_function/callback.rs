// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::ffi::CStr;
use std::os::raw::c_char;
use std::os::raw::c_ulong;
use std::os::raw::c_void;

use num_traits::AsPrimitive;
use vortex::error::VortexExpect;

use crate::cpp;
use crate::cpp::duckdb_data_chunk;
use crate::cpp::duckdb_logical_type;
use crate::cpp::duckdb_vx_copy_func_bind_input;
use crate::cpp::duckdb_vx_error;
use crate::duckdb::ClientContext;
use crate::duckdb::CopyFunction;
use crate::duckdb::Data;
use crate::duckdb::DataChunk;
use crate::duckdb::LogicalType;
use crate::duckdb::try_or;
use crate::duckdb::try_or_null;

pub(crate) unsafe extern "C-unwind" fn bind_callback<T: CopyFunction>(
    // TODO(joe): pass this into T::bind(..)
    _input: duckdb_vx_copy_func_bind_input,
    column_names: *const *const c_char,
    column_name_count: c_ulong,
    column_types: *const duckdb_logical_type,
    column_type_count: c_ulong,
    error_out: *mut duckdb_vx_error,
) -> cpp::duckdb_vx_data {
    let column_names = unsafe { std::slice::from_raw_parts(column_names, column_name_count.as_()) }
        .iter()
        .map(|name| {
            unsafe { CStr::from_ptr(name.cast()) }
                .to_string_lossy()
                .into_owned()
        })
        .collect();

    let column_types = unsafe { std::slice::from_raw_parts(column_types, column_type_count.as_()) }
        .iter()
        .map(|c| unsafe { LogicalType::borrow(*c) })
        .collect();

    try_or_null(error_out, || {
        let bind_data = T::bind(column_names, column_types)?;
        Ok(Data::from(Box::new(bind_data)).as_ptr())
    })
}

pub(crate) unsafe extern "C-unwind" fn global_callback<T: CopyFunction>(
    client_context: cpp::duckdb_client_context,
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
        let ctx = unsafe { ClientContext::borrow(client_context) };
        let bind_data = T::init_global(ctx, bind_data, file_path)?;
        Ok(Data::from(Box::new(bind_data)).as_ptr())
    })
}

pub(crate) unsafe extern "C-unwind" fn local_callback<T: CopyFunction>(
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

pub(crate) unsafe extern "C-unwind" fn copy_to_sink_callback<T: CopyFunction>(
    bind_data: *const c_void,
    global_data: *mut c_void,
    local_data: *mut c_void,
    data_chunk: duckdb_data_chunk,
    error_out: *mut duckdb_vx_error,
) {
    let bind_data =
        unsafe { bind_data.cast::<T::BindData>().as_ref() }.vortex_expect("bind_data null pointer");
    let global_data = unsafe { global_data.cast::<T::GlobalState>().as_ref() }
        .vortex_expect("bind_data null pointer");
    let local_data = unsafe { local_data.cast::<T::LocalState>().as_mut() }
        .vortex_expect("bind_data null pointer");

    try_or(error_out, || {
        T::copy_to_sink(bind_data, global_data, local_data, unsafe {
            DataChunk::borrow_mut(data_chunk)
        })?;
        Ok(())
    })
}

pub(crate) unsafe extern "C-unwind" fn copy_to_finalize_callback<T: CopyFunction>(
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
