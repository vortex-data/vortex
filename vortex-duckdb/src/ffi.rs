// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::ffi::CStr;
use std::ffi::c_char;
use std::ffi::c_void;
use std::ptr;

use num_traits::AsPrimitive;
use vortex::error::VortexExpect;
use vortex::error::vortex_err;

use crate::convert::can_push_expression;
use crate::copy::CopyFunctionBind;
use crate::copy::CopyFunctionGlobal;
use crate::copy::CopyPreparedBatch;
use crate::copy::copy_to_bind;
use crate::copy::copy_to_finalize;
use crate::copy::copy_to_initialize_global;
use crate::copy::copy_to_sink;
use crate::copy::flush_batch;
use crate::copy::prepare_batch_push;
use crate::cpp;
use crate::duckdb::AggregatePushdownInput;
use crate::duckdb::BindResult;
use crate::duckdb::Data;
use crate::duckdb::DataChunk;
use crate::duckdb::DuckdbStringMap;
use crate::duckdb::Expression;
use crate::duckdb::LogicalType;
use crate::duckdb::LogicalTypeRef;
use crate::duckdb::TableInitInput;
use crate::duckdb::try_or;
use crate::duckdb::try_or_null;
use crate::file_reader::File;
use crate::file_reader::reader_bind;
use crate::file_reader::reader_get_progress_in_file;
use crate::file_reader::reader_get_statistics;
use crate::file_reader::reader_initialize;
use crate::file_reader::reader_open;
use crate::file_reader::reader_scan;
use crate::file_reader::reader_try_initialize_scan;
use crate::table_function::BindState;
use crate::table_function::Cardinality;
use crate::table_function::GlobalState;
use crate::table_function::LocalState;
use crate::table_function::cardinality;
use crate::table_function::finalize_scan;
use crate::table_function::finish_reading;
use crate::table_function::init_global;
use crate::table_function::init_local;
use crate::table_function::pushdown_complex_filter;
use crate::table_function::pushdown_projection_aggregates;
use crate::table_function::pushdown_projection_expression;
use crate::table_function::to_string;

#[unsafe(no_mangle)]
unsafe extern "C-unwind" fn duckdb_table_function_to_string(
    bind: *const c_void,
    map: cpp::duckdb_vx_string_map,
) {
    let bind = unsafe { bind.cast::<BindState>().as_ref() }.vortex_expect("null pointer");
    let map = unsafe { DuckdbStringMap::borrow_mut(map) };
    to_string(bind, map);
}

#[unsafe(no_mangle)]
unsafe extern "C-unwind" fn duckdb_table_function_pushdown_complex_filter(
    bind: *mut c_void,
    expr: cpp::duckdb_vx_expr,
    error: *mut cpp::duckdb_vx_error,
) -> bool {
    let bind = unsafe { bind.cast::<BindState>().as_mut() }.vortex_expect("null pointer");
    let expr = unsafe { Expression::borrow(expr) };
    try_or(error, || pushdown_complex_filter(bind, expr))
}

#[unsafe(no_mangle)]
unsafe extern "C-unwind" fn duckdb_table_function_pushdown_projection_expression(
    bind: *mut c_void,
    expr: cpp::duckdb_vx_expr,
    column_id: usize,
    error: *mut cpp::duckdb_vx_error,
) -> bool {
    let bind = unsafe { bind.cast::<BindState>().as_mut() }.vortex_expect("null pointer");
    let expr = unsafe { Expression::borrow(expr) };
    try_or(error, || {
        pushdown_projection_expression(bind, expr, column_id)
    })
}

#[unsafe(no_mangle)]
pub unsafe extern "C-unwind" fn duckdb_table_function_pushdown_projection_aggregates(
    bind: *mut c_void,
    input: cpp::duckdb_vx_agg_input,
    error: *mut cpp::duckdb_vx_error,
) -> bool {
    let bind = unsafe { bind.cast::<BindState>().as_mut() }.vortex_expect("null pointer");
    let input = unsafe { AggregatePushdownInput::borrow(input) };
    try_or(error, || pushdown_projection_aggregates(bind, input))
}

#[unsafe(no_mangle)]
pub unsafe extern "C-unwind" fn duckdb_table_function_pushdown_expression(
    expr: cpp::duckdb_vx_expr,
) -> bool {
    can_push_expression(unsafe { Expression::borrow(expr) })
}

#[unsafe(no_mangle)]
pub unsafe extern "C-unwind" fn duckdb_table_function_cardinality(
    bind: *const c_void,
    file_count: u64,
    stats: *mut cpp::duckdb_vx_node_statistics,
) {
    let bind = unsafe { bind.cast::<BindState>().as_ref() }.vortex_expect("null pointer");
    let stats = unsafe { stats.as_mut() }.vortex_expect("null pointer");

    match cardinality(bind, file_count) {
        Cardinality::Exact(c) => {
            stats.has_estimated_cardinality = true;
            stats.estimated_cardinality = c as _;
            stats.has_max_cardinality = true;
            stats.max_cardinality = c as _;
        }
        Cardinality::Estimate(c) => {
            stats.has_estimated_cardinality = true;
            stats.estimated_cardinality = c as _;
        }
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "C-unwind" fn duckdb_table_function_init_global(
    init_input: *const cpp::duckdb_vx_tfunc_init_input,
    error: *mut cpp::duckdb_vx_error,
) -> cpp::duckdb_vx_data {
    let init_input =
        TableInitInput::new(unsafe { init_input.as_ref() }.vortex_expect("null pointer"));

    match init_global(&init_input) {
        Ok(init_data) => Data::from(Box::new(init_data)).as_ptr(),
        Err(e) => {
            // Set the error in the error output.
            let msg = e.to_string();
            unsafe { error.write(cpp::duckdb_vx_error_create(msg.as_ptr().cast(), msg.len())) };
            ptr::null_mut::<cpp::duckdb_vx_data_>().cast()
        }
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "C-unwind" fn duckdb_table_function_init_local(
    bind: *const c_void,
    global: *const c_void,
) -> cpp::duckdb_vx_data {
    let bind = unsafe { bind.cast::<BindState>().as_ref() }.vortex_expect("null pointer");
    let global = unsafe { global.cast::<GlobalState>().as_ref() }.vortex_expect("null pointer");
    let local = init_local(bind, global);
    Data::from(Box::new(local)).as_ptr()
}

#[unsafe(no_mangle)]
pub unsafe extern "C-unwind" fn duckdb_reader_bind(
    first_file: *const c_void,
    result: cpp::duckdb_bind_result,
    error_out: *mut cpp::duckdb_vx_error,
) -> cpp::duckdb_vx_data {
    let first_file = unsafe { first_file.cast::<File>().as_ref() }.vortex_expect("null pointer");
    let mut result = unsafe { BindResult::own(result) };

    try_or_null(error_out, || {
        let bind_data = reader_bind(first_file, &mut result)?;
        Ok(Data::from(Box::new(bind_data)).as_ptr())
    })
}

#[unsafe(no_mangle)]
pub unsafe extern "C-unwind" fn duckdb_reader_open(
    file_path: *const c_char,
    file_path_len: usize,
    error: *mut cpp::duckdb_vx_error,
) -> cpp::duckdb_vx_data {
    let path = unsafe { std::slice::from_raw_parts(file_path.cast::<u8>(), file_path_len) };

    try_or_null(error, || {
        let path = str::from_utf8(path).map_err(|_| vortex_err!("invalid utf-8"))?;
        let file = reader_open(path)?;
        Ok(Data::from(Box::new(file)).as_ptr())
    })
}

#[unsafe(no_mangle)]
pub unsafe extern "C-unwind" fn duckdb_reader_get_statistics(
    file: *const c_void,
    column_name: *const c_char,
    column_name_len: usize,
    stats_out: *mut cpp::duckdb_column_statistics,
) -> bool {
    let file = unsafe { file.cast::<File>().as_ref() }.vortex_expect("null pointer");
    let name_bytes =
        unsafe { std::slice::from_raw_parts(column_name.cast::<u8>(), column_name_len) };
    let column_name = String::from_utf8_lossy(name_bytes);

    let Some(stats) = reader_get_statistics(file, &column_name) else {
        return false;
    };
    let stats_out = unsafe { &mut *stats_out };
    stats_out.min = stats.min.map_or(ptr::null_mut(), |v| v.into_ptr());
    stats_out.max = stats.max.map_or(ptr::null_mut(), |v| v.into_ptr());
    stats_out.max_string_length = stats.max_string_length;
    stats_out.has_null = stats.has_null;
    stats_out.type_ = stats.logical_type.into_ptr();
    true
}

#[unsafe(no_mangle)]
pub unsafe extern "C-unwind" fn duckdb_reader_initialize(
    global: *const c_void,
    file: *mut c_void,
    error: *mut cpp::duckdb_vx_error,
) -> bool {
    let global = unsafe { global.cast::<GlobalState>().as_ref() }.vortex_expect("null pointer");
    let file = unsafe { file.cast::<File>().as_mut() }.vortex_expect("null pointer");
    try_or(error, || reader_initialize(file, global))
}

#[unsafe(no_mangle)]
pub unsafe extern "C-unwind" fn duckdb_reader_bind_column_type(
    bind: *const c_void,
    index: usize,
) -> cpp::duckdb_logical_type {
    let bind = unsafe { bind.cast::<BindState>().as_ref() }.vortex_expect("null pointer");
    bind.columns[index].logical_type.as_ptr()
}

#[unsafe(no_mangle)]
pub unsafe extern "C-unwind" fn duckdb_reader_is_aggregate(bind: *const c_void) -> bool {
    let bind = unsafe { bind.cast::<BindState>().as_ref() }.vortex_expect("null pointer");
    !bind.aggregates.is_empty()
}

#[unsafe(no_mangle)]
pub unsafe extern "C-unwind" fn duckdb_reader_try_initialize_scan(
    local: *mut c_void,
    file: *mut c_void,
) -> bool {
    let file = unsafe { file.cast::<File>().as_mut() }.vortex_expect("null pointer");
    let local = unsafe { local.cast::<LocalState>().as_mut() }.vortex_expect("null pointer");
    reader_try_initialize_scan(file, local)
}

#[unsafe(no_mangle)]
pub unsafe extern "C-unwind" fn duckdb_reader_scan(
    file: *const c_void,
    global: *const c_void,
    local: *mut c_void,
    chunk: cpp::duckdb_data_chunk,
    error: *mut cpp::duckdb_vx_error,
) -> bool {
    let file = unsafe { file.cast::<File>().as_ref() }.vortex_expect("null pointer");
    let global = unsafe { global.cast::<GlobalState>().as_ref() }.vortex_expect("null pointer");
    let local = unsafe { local.cast::<LocalState>().as_mut() }.vortex_expect("null pointer");
    let chunk = unsafe { DataChunk::borrow_mut(chunk) };
    try_or(error, || reader_scan(file, global, local, chunk))
}

#[unsafe(no_mangle)]
pub unsafe extern "C-unwind" fn duckdb_reader_get_progress_in_file(file: *const c_void) -> f64 {
    let file = unsafe { file.cast::<File>().as_ref() }.vortex_expect("null pointer");
    reader_get_progress_in_file(file)
}

#[unsafe(no_mangle)]
pub unsafe extern "C-unwind" fn duckdb_reader_finalize_scan(
    global: *const c_void,
    chunk: cpp::duckdb_data_chunk,
    error: *mut cpp::duckdb_vx_error,
) -> bool {
    let global = unsafe { global.cast::<GlobalState>().as_ref() }.vortex_expect("null pointer");
    let chunk = unsafe { DataChunk::borrow_mut(chunk) };
    try_or(error, || finalize_scan(global, chunk))
}

#[unsafe(no_mangle)]
pub unsafe extern "C-unwind" fn duckdb_reader_finish_reading(
    global: *const c_void,
    local: *mut c_void,
) {
    let global = unsafe { global.cast::<GlobalState>().as_ref() }.vortex_expect("null pointer");
    let local = unsafe { local.cast::<LocalState>().as_mut() }.vortex_expect("null pointer");
    finish_reading(global, local);
}

#[unsafe(no_mangle)]
pub unsafe extern "C-unwind" fn duckdb_table_function_bind_data_clone(
    bind: *const c_void,
) -> cpp::duckdb_vx_data {
    let bind = unsafe { bind.cast::<BindState>().as_ref() }.vortex_expect("null pointer");
    let copied_data = bind.clone();
    Data::from(Box::new(copied_data)).as_ptr()
}

#[unsafe(no_mangle)]
pub unsafe extern "C-unwind" fn duckdb_copy_function_copy_to_bind(
    column_names: *const *const c_char,
    column_name_count: usize,
    column_types: *const cpp::duckdb_logical_type,
    column_type_count: usize,
    error_out: *mut cpp::duckdb_vx_error,
) -> cpp::duckdb_vx_data {
    let column_names: Vec<String> =
        unsafe { std::slice::from_raw_parts(column_names, column_name_count.as_()) }
            .iter()
            .map(|name| {
                unsafe { CStr::from_ptr(name.cast()) }
                    .to_string_lossy()
                    .into_owned()
            })
            .collect();

    let column_types: Vec<&LogicalTypeRef> =
        unsafe { std::slice::from_raw_parts(column_types, column_type_count.as_()) }
            .iter()
            .map(|c| unsafe { LogicalType::borrow(*c) })
            .collect();

    try_or_null(error_out, || {
        let bind_data = copy_to_bind(&column_names, &column_types)?;
        Ok(Data::from(Box::new(bind_data)).as_ptr())
    })
}

#[unsafe(no_mangle)]
pub unsafe extern "C-unwind" fn duckdb_copy_function_copy_to_initialize_global(
    bind_data: *const c_void,
    file_path: *const c_char,
    error_out: *mut cpp::duckdb_vx_error,
) -> cpp::duckdb_vx_data {
    let file_path = unsafe { CStr::from_ptr(file_path) }
        .to_string_lossy()
        .into_owned();
    let bind_data = unsafe { bind_data.cast::<CopyFunctionBind>().as_ref() }
        .vortex_expect("bind_data null pointer");
    try_or_null(error_out, || {
        let bind_data = copy_to_initialize_global(bind_data, file_path)?;
        Ok(Data::from(Box::new(bind_data)).as_ptr())
    })
}

#[unsafe(no_mangle)]
pub unsafe extern "C-unwind" fn duckdb_copy_function_copy_to_sink(
    bind_data: *const c_void,
    global_data: *const c_void,
    data_chunk: cpp::duckdb_data_chunk,
    error_out: *mut cpp::duckdb_vx_error,
) {
    let bind_data = unsafe { bind_data.cast::<CopyFunctionBind>().as_ref() }
        .vortex_expect("bind_data null pointer");
    let global_data = unsafe { global_data.cast::<CopyFunctionGlobal>().as_ref() }
        .vortex_expect("bind_data null pointer");
    let data_chunk = unsafe { DataChunk::borrow_mut(data_chunk) };
    try_or(error_out, || {
        copy_to_sink(bind_data, global_data, data_chunk)
    })
}

#[unsafe(no_mangle)]
pub unsafe extern "C-unwind" fn duckdb_copy_function_copy_to_finalize(
    global_data: *mut c_void,
    error_out: *mut cpp::duckdb_vx_error,
) {
    let global_data = unsafe { global_data.cast::<CopyFunctionGlobal>().as_mut() }
        .vortex_expect("bind_data null pointer");
    try_or(error_out, || copy_to_finalize(global_data))
}

#[unsafe(no_mangle)]
pub unsafe extern "C-unwind" fn duckdb_copy_function_prepare_batch_new() -> cpp::duckdb_vx_data {
    Data::from(Box::new(CopyPreparedBatch::default())).as_ptr()
}

#[unsafe(no_mangle)]
pub unsafe extern "C-unwind" fn duckdb_copy_function_prepare_batch_push(
    bind: *const c_void,
    batch: *mut c_void,
    chunk: cpp::duckdb_data_chunk,
    error: *mut cpp::duckdb_vx_error,
) {
    let bind = unsafe { bind.cast::<CopyFunctionBind>().as_ref() }.vortex_expect("null pointer");
    let batch = unsafe { batch.cast::<CopyPreparedBatch>().as_mut() }.vortex_expect("null pointer");
    let chunk = unsafe { DataChunk::borrow_mut(chunk) };
    try_or(error, || prepare_batch_push(bind, batch, chunk))
}

#[unsafe(no_mangle)]
pub unsafe extern "C-unwind" fn duckdb_copy_function_flush_batch(
    global: *const c_void,
    batch: *const c_void,
    error: *mut cpp::duckdb_vx_error,
) {
    let global =
        unsafe { global.cast::<CopyFunctionGlobal>().as_ref() }.vortex_expect("null pointer");
    let batch = unsafe { batch.cast::<CopyPreparedBatch>().as_ref() }.vortex_expect("null pointer");
    try_or(error, || flush_batch(global, batch))
}
