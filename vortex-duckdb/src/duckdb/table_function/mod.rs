// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::ffi::{CStr, CString, c_void};
use std::fmt::Debug;
use std::ptr;

use vortex::error::{VortexExpect, VortexResult};
mod bind;
mod init;
mod pushdown_complex_filter;

pub use bind::*;
pub use init::*;

use crate::duckdb::LogicalType;
use crate::duckdb::connection::Connection;
use crate::duckdb::data_chunk::DataChunk;
use crate::duckdb::expr::Expression;
use crate::duckdb::table_function::pushdown_complex_filter::pushdown_complex_filter_callback;
use crate::{cpp, duckdb_try};

/// A trait that defines the supported operations for a table function in DuckDB.
///
/// This trait does not yet cover the full C++ API, see table_function.hpp.
pub trait TableFunction: Sized + Debug {
    type BindData: Send + Clone;
    type GlobalState: Send + Sync;
    type LocalState;

    /// Whether the table function supports projection pushdown.
    /// If not supported a projection will be added that filters out unused columns.
    const PROJECTION_PUSHDOWN: bool = false;

    /// Whether the table function supports filter pushdown.
    /// If not supported a filter will be added that applies the table filter directly.
    const FILTER_PUSHDOWN: bool = false;

    /// Whether the table function can immediately prune out filter columns that are unused
    /// in the remainder of the query plan.
    /// e.g. "SELECT i FROM tbl WHERE j = 42;"
    ///   - j does not need to leave the table function at all.
    const FILTER_PRUNE: bool = false;

    /// Returns the parameters of the table function.
    fn parameters() -> Vec<LogicalType> {
        // By default, we don't have any parameters.
        vec![]
    }

    /// Returns the named parameters of the table function, if any.
    fn named_parameters() -> Vec<(CString, LogicalType)> {
        // By default, we don't have any named parameters.
        vec![]
    }

    /// This function is used for determining the schema of a table producing function and
    /// returning bind data.
    fn bind(input: &BindInput, result: &mut BindResult) -> VortexResult<Self::BindData>;

    /// The function is called during query execution and is responsible for producing the output
    fn scan(
        bind_data: &Self::BindData,
        init_local: &mut Self::LocalState,
        init_global: &mut Self::GlobalState,
        chunk: &mut DataChunk,
    ) -> VortexResult<()>;

    /// Initialize the global operator state of the function.
    ///
    /// The global operator state is used to keep track of the progress in the table function and
    /// is shared between all threads working on the table function.
    fn init_global(input: &TableInitInput<Self>) -> VortexResult<Self::GlobalState>;

    /// Initialize the local operator state of the function.
    ///
    /// The local operator state is used to keep track of the progress in the table function and
    /// is thread-local.
    fn init_local(
        init: &TableInitInput<Self>,
        global: &mut Self::GlobalState,
    ) -> VortexResult<Self::LocalState>;

    /// Pushes down a filter expression to the table function.
    ///
    /// Returns `true` if the filter was successfully pushed down (and stored on the bind data),
    /// or `false` if the filter could not be pushed down. In which case, the filter will be
    /// applied later in the query plan.
    fn pushdown_complex_filter(
        _bind_data: &mut Self::BindData,
        _expr: &Expression,
    ) -> VortexResult<bool> {
        Ok(false)
    }

    // TODO(ngates): there are many more callbacks that can be configured.
}

impl Connection {
    pub fn register_table_function<T: TableFunction>(&self, name: &CStr) -> VortexResult<()> {
        // Set up the parameters.
        let parameters = T::parameters();
        let parameter_ptrs = parameters
            .iter()
            .map(|logical_type| logical_type.as_ptr())
            .collect::<Vec<_>>();

        let param_names = T::named_parameters();
        let (param_names_ptrs, param_types_ptr) = param_names
            .into_iter()
            .map(|(name, logical_type)| (name.as_ptr(), logical_type.as_ptr()))
            .unzip::<_, _, Vec<_>, Vec<_>>();

        let vtab = cpp::duckdb_vx_tfunc_vtab_t {
            name: name.as_ptr(),
            parameters: parameter_ptrs.as_ptr(),
            parameter_count: parameters.len() as _,
            named_parameter_names: param_names_ptrs.as_ptr(),
            named_parameter_types: param_types_ptr.as_ptr(),
            named_parameter_count: param_names_ptrs.len() as _,
            bind: Some(bind_callback::<T>),
            bind_data_clone: Some(bind_data_clone_callback::<T>),
            init_global: Some(init_global_callback::<T>),
            init_local: Some(init_local_callback::<T>),
            function: Some(function::<T>),
            statistics: ptr::null_mut::<c_void>(),
            cardinality: ptr::null_mut::<c_void>(),
            pushdown_complex_filter: Some(pushdown_complex_filter_callback::<T>),
            pushdown_expression: ptr::null_mut::<c_void>(),
            table_scan_progress: ptr::null_mut::<c_void>(),
            projection_pushdown: T::PROJECTION_PUSHDOWN,
            filter_pushdown: T::FILTER_PUSHDOWN,
            filter_prune: T::FILTER_PRUNE,
            sampling_pushdown: false,
            late_materialization: false,
        };

        duckdb_try!(
            unsafe { cpp::duckdb_vx_tfunc_register(self.as_ptr(), &raw const vtab) },
            "Failed to register table function '{}'",
            name.to_string_lossy()
        );

        Ok(())
    }
}

/// The native function callback for a table function.
unsafe extern "C" fn function<T: TableFunction>(
    bind_data: *const c_void,
    global_init_data: *mut c_void,
    local_init_data: *mut c_void,
    output: cpp::duckdb_data_chunk,
    error_out: *mut cpp::duckdb_vx_error,
) {
    let bind_data = unsafe { &*(bind_data as *const T::BindData) };
    let global_init_data = unsafe { global_init_data.cast::<T::GlobalState>().as_mut() }
        .vortex_expect("global_init_data null pointer");
    let local_init_data = unsafe { local_init_data.cast::<T::LocalState>().as_mut() }
        .vortex_expect("local_init_data null pointer");
    let mut data_chunk = unsafe { DataChunk::borrow(output) };

    match T::scan(
        bind_data,
        local_init_data,
        global_init_data,
        &mut data_chunk,
    ) {
        Ok(()) => {
            // The data chunk is already filled by the function.
            // No need to do anything here.
        }
        Err(e) => unsafe {
            error_out.write(cpp::duckdb_vx_error_create(
                e.to_string().as_ptr().cast(),
                e.to_string().len(),
            ));
        },
    }
}
