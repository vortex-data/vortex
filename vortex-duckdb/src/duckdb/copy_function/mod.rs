// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

mod callback;

use std::ffi::CStr;
use std::fmt::Debug;

use vortex::error::VortexExpect;
use vortex::error::VortexResult;

use crate::Connection;
use crate::cpp;
use crate::duckdb::DataChunk;
use crate::duckdb::LogicalType;
use crate::duckdb::copy_function::callback::bind_callback;
use crate::duckdb::copy_function::callback::copy_to_finalize_callback;
use crate::duckdb::copy_function::callback::copy_to_sink_callback;
use crate::duckdb::copy_function::callback::global_callback;
use crate::duckdb::copy_function::callback::local_callback;
use crate::duckdb_try;

pub trait CopyFunction: Sized + Debug {
    type BindData: Send;
    type GlobalState: Send + Sync;
    type LocalState;

    /// This function is used for determining the schema of a file produced by the function.
    fn bind(
        column_names: Vec<String>,
        column_types: Vec<LogicalType>,
    ) -> VortexResult<Self::BindData>;

    /// The function is called during query execution and is responsible for consuming the output
    fn copy_to_sink(
        bind_data: &Self::BindData,
        init_global: &mut Self::GlobalState,
        init_local: &mut Self::LocalState,
        chunk: &mut DataChunk,
    ) -> VortexResult<()>;

    fn copy_to_finalize(
        bind_data: &Self::BindData,
        init_global: &mut Self::GlobalState,
    ) -> VortexResult<()>;

    /// Initialize the global operator state of the function.
    ///
    /// The global operator state is used to keep track of the progress in the copy function and
    /// is shared between all threads working on the copy function.
    fn init_global(
        bind_data: &Self::BindData,
        file_path: String,
    ) -> VortexResult<Self::GlobalState>;

    /// Initialize the local operator state of the function.
    ///
    /// The local operator state is used to keep track of the progress in the copy function and
    /// is thread-local.
    fn init_local(bind: &Self::BindData) -> VortexResult<Self::LocalState>;

    // TODO(joe): there are many more callbacks that can be configured.
}

impl Connection {
    pub fn register_copy_function<T: CopyFunction>(
        &self,
        name: &CStr,
        extension: &CStr,
    ) -> VortexResult<()> {
        let vtab: &mut cpp::duckdb_vx_copy_func_vtab_t =
            unsafe { cpp::get_vtab_one().as_mut() }.vortex_expect("copy vtab cannot be null");

        vtab.name = name.as_ptr();
        vtab.extension = extension.as_ptr();
        vtab.bind = Some(bind_callback::<T>);
        vtab.init_global = Some(global_callback::<T>);
        vtab.init_local = Some(local_callback::<T>);
        vtab.copy_to_sink = Some(copy_to_sink_callback::<T>);
        vtab.copy_to_finalize = Some(copy_to_finalize_callback::<T>);

        duckdb_try!(
            unsafe { cpp::duckdb_vx_copy_func_register_vtab_one(self.as_ptr()) },
            "Failed to register copy function '{}'",
            name.to_string_lossy()
        );

        Ok(())
    }
}
