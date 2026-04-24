// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::ffi::CStr;
use std::ffi::CString;
use std::ffi::c_void;
use std::fmt::Debug;

use vortex::error::VortexExpect;
use vortex::error::VortexResult;
mod bind;
mod cardinality;
mod init;
mod partition;
mod pushdown_complex_filter;
mod statistics;
mod table_scan_progress;

pub use bind::*;
pub use init::*;

use crate::cpp;
use crate::cpp::duckdb_client_context;
use crate::duckdb::ClientContext;
use crate::duckdb::DataChunk;
use crate::duckdb::DatabaseRef;
use crate::duckdb::LogicalType;
use crate::duckdb::Value;
use crate::duckdb::client_context::ClientContextRef;
use crate::duckdb::data_chunk::DataChunkRef;
use crate::duckdb::expr::ExpressionRef;
use crate::duckdb::table_function::cardinality::cardinality_callback;
use crate::duckdb::table_function::partition::get_partition_data_callback;
use crate::duckdb::table_function::pushdown_complex_filter::pushdown_complex_filter_callback;
use crate::duckdb::table_function::statistics::statistics;
use crate::duckdb::table_function::table_scan_progress::table_scan_progress_callback;
use crate::duckdb_try;

#[derive(Debug, Default)]
pub struct ColumnStatistics {
    pub min: Option<Value>,
    pub max: Option<Value>,
    pub max_string_length: u64,
    pub has_null: bool,
}

// String map lifetime is managed by C++ code
crate::lifetime_wrapper!(DuckdbStringMap, cpp::duckdb_vx_string_map, |_| {});
impl DuckdbStringMapRef {
    pub fn push(&mut self, key: &str, value: &str) {
        let key = CString::new(key).unwrap_or_else(|_| CString::default());
        let value = CString::new(value).unwrap_or_else(|_| CString::default());
        unsafe {
            cpp::duckdb_vx_string_map_insert(self.as_ptr(), key.as_ptr(), value.as_ptr());
        }
    }
}

/// A trait that defines the supported operations for a table function in DuckDB.
///
/// This trait does not yet cover the full C++ API, see table_function.hpp.
pub trait TableFunction: Sized + Debug {
    type BindData: Send + Clone;
    type GlobalState: Send + Sync;
    type LocalState;

    /// Returns the parameters of the table function.
    fn parameters() -> Vec<LogicalType> {
        // By default, we don't have any parameters.
        vec![]
    }

    /// This function is used for determining the schema of a table producing function and
    /// returning bind data.
    fn bind(
        client_context: &ClientContextRef,
        input: &BindInputRef,
        result: &mut BindResultRef,
    ) -> VortexResult<Self::BindData>;

    /// Report column statistics for a file or collections of files e.g.
    /// registered as a VIEW.
    fn statistics(
        client_context: &ClientContextRef,
        bind_data: &Self::BindData,
        column_index: usize,
    ) -> Option<ColumnStatistics>;

    /// The function is called during query execution and is responsible for producing the output
    fn scan(
        client_context: &ClientContextRef,
        bind_data: &Self::BindData,
        init_local: &mut Self::LocalState,
        init_global: &Self::GlobalState,
        chunk: &mut DataChunkRef,
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
        global: &Self::GlobalState,
    ) -> VortexResult<Self::LocalState>;

    /// Return table scanning progress from 0. to 100.
    fn table_scan_progress(
        client_context: &ClientContextRef,
        bind_data: &Self::BindData,
        global_state: &Self::GlobalState,
    ) -> f64;

    /// Pushes down a filter expression to the table function.
    ///
    /// Returns `true` if the filter was successfully pushed down (and stored on the bind data),
    /// or `false` if the filter could not be pushed down. In which case, the filter will be
    /// applied later in the query plan.
    fn pushdown_complex_filter(
        _bind_data: &mut Self::BindData,
        _expr: &ExpressionRef,
    ) -> VortexResult<bool> {
        Ok(false)
    }

    /// Returns the cardinality estimate of the table function.
    fn cardinality(_bind_data: &Self::BindData) -> Cardinality {
        Cardinality::Unknown
    }

    /// Returns the idx of the current partition being processed by a local threa.
    /// This *must* be globally unique.
    fn partition_data(
        _bind_data: &Self::BindData,
        _global_init_data: &Self::GlobalState,
        _local_init_data: &mut Self::LocalState,
    ) -> VortexResult<u64>;

    /// Returns a vector of key-value pairs for EXPLAIN output
    fn to_string(bind_data: &Self::BindData, map: &mut DuckdbStringMapRef);
}

#[derive(Debug)]
pub enum Cardinality {
    /// Completely unknown cardinality.
    Unknown,
    /// An estimate of the number of rows that will be returned by the table function.
    Estimate(u64),
    /// Will not return more than this number of rows.
    Maximum(u64),
}

impl DatabaseRef {
    pub fn register_table_function<T: TableFunction>(&self, name: &CStr) -> VortexResult<()> {
        // Set up the parameters.
        let parameters = T::parameters();
        let parameter_ptrs = parameters
            .iter()
            .map(|logical_type| logical_type.as_ptr())
            .collect::<Vec<_>>();

        let vtab = cpp::duckdb_vx_tfunc_vtab_t {
            name: name.as_ptr(),
            parameters: parameter_ptrs.as_ptr(),
            parameter_count: parameters.len() as _,
            bind: Some(bind_callback::<T>),
            bind_data_clone: Some(bind_data_clone_callback::<T>),
            init_global: Some(init_global_callback::<T>),
            init_local: Some(init_local_callback::<T>),
            function: Some(function::<T>),
            statistics: Some(statistics::<T>),
            cardinality: Some(cardinality_callback::<T>),
            pushdown_complex_filter: Some(pushdown_complex_filter_callback::<T>),
            to_string: Some(to_string_callback::<T>),
            table_scan_progress: Some(table_scan_progress_callback::<T>),
            get_partition_data: Some(get_partition_data_callback::<T>),
        };

        duckdb_try!(
            unsafe { cpp::duckdb_vx_tfunc_register(self.as_ptr(), &raw const vtab) },
            "Failed to register table function '{}'",
            name.to_string_lossy()
        );

        Ok(())
    }
}

unsafe extern "C-unwind" fn to_string_callback<T: TableFunction>(
    bind_data: *mut c_void,
    map: cpp::duckdb_vx_string_map,
) {
    let bind_data = unsafe { &*(bind_data as *const T::BindData) };
    let map = unsafe { DuckdbStringMap::borrow_mut(map) };
    T::to_string(bind_data, map);
}

/// The native function callback for a table function.
unsafe extern "C-unwind" fn function<T: TableFunction>(
    duckdb_client_context: duckdb_client_context,
    bind_data: *const c_void,
    global_init_data: *mut c_void,
    local_init_data: *mut c_void,
    output: cpp::duckdb_data_chunk,
    error_out: *mut cpp::duckdb_vx_error,
) {
    let client_context = unsafe { ClientContext::borrow(duckdb_client_context) };
    let bind_data = unsafe { &*(bind_data as *const T::BindData) };
    let global_init_data = unsafe { global_init_data.cast::<T::GlobalState>().as_ref() }
        .vortex_expect("global_init_data null pointer");
    let local_init_data = unsafe { local_init_data.cast::<T::LocalState>().as_mut() }
        .vortex_expect("local_init_data null pointer");
    let data_chunk = unsafe { DataChunk::borrow_mut(output) };

    match T::scan(
        client_context,
        bind_data,
        local_init_data,
        global_init_data,
        data_chunk,
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
