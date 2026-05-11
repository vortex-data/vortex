// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::ffi::c_void;
use std::fmt::Debug;
use std::fmt::Formatter;
use std::ptr;

use vortex::error::VortexExpect;
use vortex::error::VortexResult;
use vortex::error::vortex_bail;

use crate::cpp;
use crate::duckdb::ClientContext;
use crate::duckdb::ClientContextRef;
use crate::duckdb::Data;
use crate::duckdb::TableFilterSet;
use crate::duckdb::TableFilterSetRef;
use crate::duckdb::TableFunction;

/// Native callback for the global initialization of a table function.
pub(crate) unsafe extern "C-unwind" fn init_global_callback<T: TableFunction>(
    init_input: *const cpp::duckdb_vx_tfunc_init_input,
    error_out: *mut cpp::duckdb_vx_error,
) -> cpp::duckdb_vx_data {
    let init_input = TableInitInput::new(
        unsafe { init_input.as_ref() }.vortex_expect("init_input null pointer"),
    );

    match T::init_global(&init_input) {
        Ok(init_data) => Data::from(Box::new(init_data)).as_ptr(),
        Err(e) => {
            // Set the error in the error output.
            let msg = e.to_string();
            unsafe { error_out.write(cpp::duckdb_vx_error_create(msg.as_ptr().cast(), msg.len())) };
            ptr::null_mut::<cpp::duckdb_vx_data_>().cast()
        }
    }
}

/// Native callback for the local initialization of a table function.
pub(crate) unsafe extern "C-unwind" fn init_local_callback<T: TableFunction>(
    global_init_data: *mut c_void,
) -> cpp::duckdb_vx_data {
    let global_init_data = unsafe { global_init_data.cast::<T::GlobalState>().as_ref() }
        .vortex_expect("global_init_data null pointer");

    let init_data = T::init_local(global_init_data);
    Data::from(Box::new(init_data)).as_ptr()
}

/// A typed wrapper for the input to a table function's initialization.
pub struct TableInitInput<'a, T: TableFunction> {
    input: &'a cpp::duckdb_vx_tfunc_init_input,
    phantom: std::marker::PhantomData<T>,
}

impl<T: TableFunction> Debug for TableInitInput<'_, T> {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("TableInitInput")
            .field("table_function", &std::any::type_name::<T>())
            .field("column_ids", &self.column_ids())
            .field("projection_ids", &self.projection_ids())
            .field("table_filter_set", &self.table_filter_set())
            .finish()
    }
}

impl<'a, T: TableFunction> TableInitInput<'a, T> {
    fn new(input: &'a cpp::duckdb_vx_tfunc_init_input) -> Self {
        Self {
            input,
            phantom: std::marker::PhantomData,
        }
    }

    /// Returns the bind data for the table function.
    pub fn bind_data(&self) -> &T::BindData {
        unsafe { &*self.input.bind_data.cast::<T::BindData>() }
    }

    pub fn column_ids(&self) -> &[u64] {
        unsafe { std::slice::from_raw_parts(self.input.column_ids, self.input.column_ids_count) }
    }

    pub fn projection_ids(&self) -> Option<&[u64]> {
        // Passed pointer is std::vector's .data(). However, C++ doesn't
        // guarantee an empty vector's pointer is nullptr so we need to check
        // both conditions
        if self.input.projection_ids.is_null() || self.input.projection_ids_count == 0 {
            return None;
        }
        Some(unsafe {
            std::slice::from_raw_parts(self.input.projection_ids, self.input.projection_ids_count)
        })
    }

    /// Returns the table filter set for the table function.
    pub fn table_filter_set(&self) -> Option<&TableFilterSetRef> {
        let ptr = self.input.filters;
        if ptr.is_null() {
            None
        } else {
            Some(unsafe { TableFilterSet::borrow(ptr) })
        }
    }

    /// Returns the object cache from the client context for the table function.
    pub fn client_context(&self) -> VortexResult<&ClientContextRef> {
        unsafe {
            if self.input.client_context.is_null() {
                vortex_bail!("Client context is null");
            }
            Ok(ClientContext::borrow(self.input.client_context))
        }
    }
}
