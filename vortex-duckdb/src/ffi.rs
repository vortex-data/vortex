// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::ffi::CStr;
use std::ffi::c_char;
use std::ffi::c_void;
use std::ptr;

use num_traits::AsPrimitive;
use vortex::error::VortexExpect;

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
use crate::file_reader::FileReader;
use crate::file_reader::FileScan;
use crate::file_reader::file_has_work;
use crate::file_reader::file_open;
use crate::file_reader::file_progress;
use crate::file_reader::file_scan;
use crate::file_reader::file_schema;
use crate::file_reader::file_should_skip;
use crate::file_reader::file_start_scan;
use crate::file_reader::file_statistics;
use crate::table_function::Cardinality;
use crate::table_function::TableFunctionBind;
use crate::table_function::TableFunctionGlobal;
use crate::table_function::TableFunctionLocal;
use crate::table_function::bind;
use crate::table_function::bind_schema;
use crate::table_function::cardinality;
use crate::table_function::finalize_scan;
use crate::table_function::init_global;
use crate::table_function::init_local;
use crate::table_function::pushdown_complex_filter;
use crate::table_function::pushdown_projection_aggregates;
use crate::table_function::pushdown_projection_expression;
use crate::table_function::to_string;

#[unsafe(no_mangle)]
unsafe extern "C-unwind" fn duckdb_table_function_to_string(
    bind_data: *const c_void,
    map: cpp::duckdb_vx_string_map,
) {
    let bind_data = unsafe { bind_data.cast::<TableFunctionBind>().as_ref() }
        .vortex_expect("bind_data null pointer");
    let map = unsafe { DuckdbStringMap::borrow_mut(map) };
    to_string(bind_data, map);
}

#[unsafe(no_mangle)]
unsafe extern "C-unwind" fn duckdb_table_function_pushdown_complex_filter(
    bind_data: *mut c_void,
    expr: cpp::duckdb_vx_expr,
    error_out: *mut cpp::duckdb_vx_error,
) -> bool {
    let bind_data = unsafe { bind_data.cast::<TableFunctionBind>().as_mut() }
        .vortex_expect("bind_data null pointer");
    let expr = unsafe { Expression::borrow(expr) };
    try_or(error_out, || pushdown_complex_filter(bind_data, expr))
}

#[unsafe(no_mangle)]
unsafe extern "C-unwind" fn duckdb_table_function_pushdown_projection_expression(
    bind_data: *mut c_void,
    expr: cpp::duckdb_vx_expr,
    column_id: usize,
    error_out: *mut cpp::duckdb_vx_error,
) -> bool {
    let bind_data = unsafe { bind_data.cast::<TableFunctionBind>().as_mut() }
        .vortex_expect("bind_data null pointer");
    let expr = unsafe { Expression::borrow(expr) };
    try_or(error_out, || {
        pushdown_projection_expression(bind_data, expr, column_id)
    })
}

#[unsafe(no_mangle)]
pub unsafe extern "C-unwind" fn duckdb_table_function_pushdown_projection_aggregates(
    bind_data: *mut c_void,
    input: cpp::duckdb_vx_agg_input,
    error_out: *mut cpp::duckdb_vx_error,
) -> bool {
    let bind_data = unsafe { bind_data.cast::<TableFunctionBind>().as_mut() }
        .vortex_expect("bind_data null pointer");
    let input = unsafe { AggregatePushdownInput::borrow(input) };
    try_or(error_out, || {
        pushdown_projection_aggregates(bind_data, input)
    })
}

#[unsafe(no_mangle)]
pub unsafe extern "C-unwind" fn duckdb_table_function_pushdown_expression(
    expr: cpp::duckdb_vx_expr,
) -> bool {
    can_push_expression(unsafe { Expression::borrow(expr) })
}

#[unsafe(no_mangle)]
pub unsafe extern "C-unwind" fn duckdb_table_function_cardinality(
    bind_data: *const c_void,
    file_count: u64,
    node_stats_out: *mut cpp::duckdb_vx_node_statistics,
) {
    let bind_data = unsafe { bind_data.cast::<TableFunctionBind>().as_ref() }
        .vortex_expect("bind_data null pointer");
    let node_stats =
        unsafe { node_stats_out.as_mut() }.vortex_expect("node_stats_out null pointer");

    match cardinality(bind_data, file_count) {
        Cardinality::Exact(c) => {
            node_stats.has_estimated_cardinality = true;
            node_stats.estimated_cardinality = c as _;
            node_stats.has_max_cardinality = true;
            node_stats.max_cardinality = c as _;
        }
        Cardinality::Estimate(c) => {
            node_stats.has_estimated_cardinality = true;
            node_stats.estimated_cardinality = c as _;
        }
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "C-unwind" fn duckdb_table_function_init_global(
    init_input: *const cpp::duckdb_vx_tfunc_init_input,
    error_out: *mut cpp::duckdb_vx_error,
) -> cpp::duckdb_vx_data {
    let init_input = TableInitInput::new(
        unsafe { init_input.as_ref() }.vortex_expect("init_input null pointer"),
    );

    match init_global(&init_input) {
        Ok(init_data) => Data::from(Box::new(init_data)).as_ptr(),
        Err(e) => {
            // Set the error in the error output.
            let msg = e.to_string();
            unsafe { error_out.write(cpp::duckdb_vx_error_create(msg.as_ptr().cast(), msg.len())) };
            ptr::null_mut::<cpp::duckdb_vx_data_>().cast()
        }
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "C-unwind" fn duckdb_table_function_init_local(
    bind_data: *const c_void,
    global_init_data: *mut c_void,
) -> cpp::duckdb_vx_data {
    let bind_data = unsafe { bind_data.cast::<TableFunctionBind>().as_ref() }
        .vortex_expect("bind_data null pointer");
    let global_init_data = unsafe { global_init_data.cast::<TableFunctionGlobal>().as_ref() }
        .vortex_expect("global_init_data null pointer");

    let init_data = init_local(bind_data, global_init_data);
    Data::from(Box::new(init_data)).as_ptr()
}

#[unsafe(no_mangle)]
pub unsafe extern "C-unwind" fn duckdb_table_function_bind(
    first_file: *const c_void,
    error_out: *mut cpp::duckdb_vx_error,
) -> cpp::duckdb_vx_data {
    let first_file =
        unsafe { first_file.cast::<FileReader>().as_ref() }.vortex_expect("file null pointer");

    try_or_null(error_out, || {
        let bind_data = bind(first_file)?;
        Ok(Data::from(Box::new(bind_data)).as_ptr())
    })
}

#[unsafe(no_mangle)]
pub unsafe extern "C-unwind" fn duckdb_table_function_bind_schema(
    bind_data: *const c_void,
    schema_result: cpp::duckdb_vx_tfunc_bind_result,
) {
    let bind_data = unsafe { bind_data.cast::<TableFunctionBind>().as_ref() }
        .vortex_expect("bind_data null pointer");
    let schema_result = unsafe { BindResult::borrow_mut(schema_result) };
    bind_schema(bind_data, schema_result);
}

#[unsafe(no_mangle)]
pub unsafe extern "C-unwind" fn duckdb_table_function_file_open(
    file_path: *const c_char,
    file_path_len: usize,
    file_index: u64,
    error_out: *mut cpp::duckdb_vx_error,
) -> cpp::duckdb_vx_data {
    let path_bytes = unsafe { std::slice::from_raw_parts(file_path.cast::<u8>(), file_path_len) };
    let file_path = String::from_utf8_lossy(path_bytes).into_owned();

    try_or_null(error_out, || {
        let file = file_open(&file_path, file_index)?;
        Ok(Data::from(Box::new(file)).as_ptr())
    })
}

#[unsafe(no_mangle)]
pub unsafe extern "C-unwind" fn duckdb_table_function_file_schema(
    file: *const c_void,
    schema_result: cpp::duckdb_vx_tfunc_bind_result,
    error_out: *mut cpp::duckdb_vx_error,
) {
    let file = unsafe { file.cast::<FileReader>().as_ref() }.vortex_expect("file null pointer");
    let schema_result = unsafe { BindResult::borrow_mut(schema_result) };
    try_or(error_out, || file_schema(file, schema_result))
}

#[unsafe(no_mangle)]
pub unsafe extern "C-unwind" fn duckdb_table_function_file_statistics(
    file: *const c_void,
    column_name: *const c_char,
    column_name_len: usize,
    stats_out: *mut cpp::duckdb_column_statistics,
) -> bool {
    let file = unsafe { file.cast::<FileReader>().as_ref() }.vortex_expect("file null pointer");
    let name_bytes =
        unsafe { std::slice::from_raw_parts(column_name.cast::<u8>(), column_name_len) };
    let column_name = String::from_utf8_lossy(name_bytes);

    let Some(stats) = file_statistics(file, &column_name) else {
        return false;
    };
    let stats_out = unsafe { &mut *stats_out };
    stats_out.min = stats.min.map_or(ptr::null_mut(), |v| v.into_ptr());
    stats_out.max = stats.max.map_or(ptr::null_mut(), |v| v.into_ptr());
    stats_out.max_string_length = stats.max_string_length;
    stats_out.has_null = stats.has_null;
    true
}

#[unsafe(no_mangle)]
pub unsafe extern "C-unwind" fn duckdb_table_function_file_should_skip(
    global_init_data: *const c_void,
    file: *const c_void,
    error_out: *mut cpp::duckdb_vx_error,
) -> bool {
    let global_init_data = unsafe { global_init_data.cast::<TableFunctionGlobal>().as_ref() }
        .vortex_expect("global_init_data null pointer");
    let file = unsafe { file.cast::<FileReader>().as_ref() }.vortex_expect("file null pointer");
    try_or(error_out, || file_should_skip(global_init_data, file))
}

#[unsafe(no_mangle)]
pub unsafe extern "C-unwind" fn duckdb_table_function_file_start_scan(
    bind_data: *const c_void,
    global_init_data: *mut c_void,
    file: *const c_void,
    column_ids: *const u64,
    column_ids_count: usize,
    filters: cpp::duckdb_vx_table_filter_set,
    error_out: *mut cpp::duckdb_vx_error,
) -> cpp::duckdb_vx_data {
    let bind_data = unsafe { bind_data.cast::<TableFunctionBind>().as_ref() }
        .vortex_expect("bind_data null pointer");
    let file = unsafe { file.cast::<FileReader>().as_ref() }.vortex_expect("file null pointer");
    let column_ids = if column_ids_count == 0 {
        &[]
    } else {
        unsafe { std::slice::from_raw_parts(column_ids, column_ids_count) }
    };

    let global_init_data = unsafe { global_init_data.cast::<TableFunctionGlobal>().as_ref() }
        .vortex_expect("global_init_data null pointer");
    try_or_null(error_out, || {
        let scan = file_start_scan(bind_data, global_init_data, file, column_ids, filters)?;
        Ok(Data::from(Box::new(scan)).as_ptr())
    })
}

#[unsafe(no_mangle)]
pub unsafe extern "C-unwind" fn duckdb_table_function_file_has_work(
    local_init_data: *const c_void,
) -> bool {
    let local = unsafe { local_init_data.cast::<TableFunctionLocal>().as_ref() }
        .vortex_expect("local_init_data null pointer");
    file_has_work(local)
}

#[unsafe(no_mangle)]
pub unsafe extern "C-unwind" fn duckdb_table_function_file_scan(
    file_scan_data: *const c_void,
    global_init_data: *mut c_void,
    local_init_data: *mut c_void,
    output: cpp::duckdb_data_chunk,
    error_out: *mut cpp::duckdb_vx_error,
) {
    let scan = unsafe { file_scan_data.cast::<FileScan>().as_ref() }
        .vortex_expect("file_scan null pointer");
    let global_init_data = unsafe { global_init_data.cast::<TableFunctionGlobal>().as_ref() }
        .vortex_expect("global_init_data null pointer");
    let local_init_data = unsafe { local_init_data.cast::<TableFunctionLocal>().as_mut() }
        .vortex_expect("local_init_data null pointer");
    let data_chunk = unsafe { DataChunk::borrow_mut(output) };
    try_or(error_out, || {
        file_scan(scan, global_init_data, local_init_data, data_chunk)
    })
}

#[unsafe(no_mangle)]
pub unsafe extern "C-unwind" fn duckdb_table_function_file_progress(
    file_scan_data: *const c_void,
) -> f64 {
    let scan = unsafe { file_scan_data.cast::<FileScan>().as_ref() }
        .vortex_expect("file_scan null pointer");
    file_progress(scan)
}

#[unsafe(no_mangle)]
pub unsafe extern "C-unwind" fn duckdb_table_function_finalize_scan(
    global_init_data: *mut c_void,
    output: cpp::duckdb_data_chunk,
    error_out: *mut cpp::duckdb_vx_error,
) -> bool {
    let global_init_data = unsafe { global_init_data.cast::<TableFunctionGlobal>().as_ref() }
        .vortex_expect("global_init_data null pointer");
    let data_chunk = unsafe { DataChunk::borrow_mut(output) };
    try_or(error_out, || finalize_scan(global_init_data, data_chunk))
}

#[unsafe(no_mangle)]
pub unsafe extern "C-unwind" fn duckdb_table_function_bind_data_clone(
    bind_data: *const c_void,
) -> cpp::duckdb_vx_data {
    let bind_data = unsafe { bind_data.cast::<TableFunctionBind>().as_ref() }
        .vortex_expect("bind_data null pointer");
    let copied_data = bind_data.clone();
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
