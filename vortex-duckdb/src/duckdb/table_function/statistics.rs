// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::ffi::c_void;
use std::ptr;

use vortex::error::VortexExpect;
use vortex::scalar::Scalar;

use crate::convert::ToDuckDBScalar;
use crate::cpp::duckdb_client_context;
use crate::cpp::{self};
use crate::duckdb::ClientContext;
use crate::duckdb::TableFunction;
use crate::duckdb::Value;

pub(crate) unsafe extern "C-unwind" fn statistics<T: TableFunction>(
    ctx: duckdb_client_context,
    bind_data: *const c_void,
    column_index: usize,
    stats_out: *mut cpp::duckdb_column_statistics,
) {
    let stats_out = unsafe { &mut *stats_out };
    let client_context = unsafe { ClientContext::borrow(ctx) };
    let bind_data =
        unsafe { bind_data.cast::<T::BindData>().as_ref() }.vortex_expect("bind_data null pointer");
    let stats_ref = T::statistics(client_context, bind_data, column_index);
    let dtype = &stats_ref.dtype;

    // By definition dtype matches the value, so use vortex_expect
    if let Some(ref value) = stats_ref.min {
        let value = Scalar::try_new(dtype.clone(), Some(value.clone()))
            .vortex_expect("scalar dtype and value are incompatible")
            .try_to_duckdb_scalar()
            .vortex_expect("can't convert Scalar to duckdb Value");
        stats_out.min = Value::into_ptr(value);
    } else {
        stats_out.min = ptr::null_mut();
    }

    if let Some(ref value) = stats_ref.max {
        let value = Scalar::try_new(dtype.clone(), Some(value.clone()))
            .vortex_expect("scalar dtype and value are incompatible")
            .try_to_duckdb_scalar()
            .vortex_expect("can't convert Scalar to duckdb Value");
        stats_out.max = Value::into_ptr(value);
    } else {
        stats_out.max = ptr::null_mut();
    }

    stats_out.max_string_length = stats_ref
        .max_string_length
        .map_or(0, |len| (1u64 << 63) | (len as u64));
}
