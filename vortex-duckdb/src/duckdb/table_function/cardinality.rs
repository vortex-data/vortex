// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::ffi::c_void;

use vortex::error::VortexExpect;

use crate::cpp;
use crate::duckdb::{Cardinality, TableFunction};

/// Native callback for the cardinality estimate of a table function.
pub(crate) unsafe extern "C-unwind" fn cardinality_callback<T: TableFunction>(
    bind_data: *mut c_void,
    node_stats_out: *mut cpp::duckdb_vx_node_statistics,
) {
    let bind_data =
        unsafe { bind_data.cast::<T::BindData>().as_mut() }.vortex_expect("bind_data null pointer");
    let node_stats =
        unsafe { node_stats_out.as_mut() }.vortex_expect("node_stats_out null pointer");

    match T::cardinality(bind_data) {
        Cardinality::Unknown => {}
        Cardinality::Estimate(c) => {
            node_stats.has_estimated_cardinality = true;
            node_stats.estimated_cardinality = c as _;
        }
        Cardinality::Maximum(c) => {
            node_stats.has_max_cardinality = true;
            node_stats.max_cardinality = c as _;
            node_stats.has_estimated_cardinality = true;
            node_stats.estimated_cardinality = c as _;
        }
    }
}
