// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::ffi::c_void;

use vortex::error::VortexExpect;

use crate::cpp;
use crate::duckdb::expr::Expression;
use crate::duckdb::{TableFunction, try_or};

/// Native callback for the global initialization of a table function.
pub(crate) unsafe extern "C" fn pushdown_complex_filter_callback<T: TableFunction>(
    bind_data: *mut c_void,
    expr: cpp::duckdb_vx_expr,
    error_out: *mut cpp::duckdb_vx_error,
) -> bool {
    let bind_data =
        unsafe { bind_data.cast::<T::BindData>().as_mut() }.vortex_expect("bind_data null pointer");
    let expr = unsafe { Expression::borrow(expr) };
    try_or(error_out, || T::pushdown_complex_filter(bind_data, &expr))
}
