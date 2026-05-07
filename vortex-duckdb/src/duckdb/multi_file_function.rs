// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Rust-side wrapper for DuckDB's `MultiFileFunction<OP>` template.
//!
//! Lets a table-function-like type plug into DuckDB's native multi-file machinery
//! (file globbing, virtual columns, hive partitioning, COPY support, etc.) by
//! supplying only what's format-specific: how to open a file, how to read its
//! schema, and how to scan a chunk. The cross-file orchestration is owned by
//! DuckDB.
//!
//! The trait pair mirrors the DuckDB C++ surface:
//!   - [`MultiFileFunction`] ↔ `MultiFileReaderInterface`
//!   - [`BaseFileReader`]    ↔ `BaseFileReader`
//!
//! Both are non-object-safe with associated types so each implementation gets
//! statically-monomorphised callbacks (no per-call dyn dispatch).

use std::ffi::CStr;
use std::ffi::CString;
use std::fmt::Debug;
use std::ptr;
use std::slice;

use vortex::error::VortexExpect;
use vortex::error::VortexResult;
use vortex::error::vortex_err;

use crate::cpp;
use crate::duckdb::ClientContext;
use crate::duckdb::ClientContextRef;
use crate::duckdb::ColumnStatistics;
use crate::duckdb::DataChunk;
use crate::duckdb::DataChunkRef;
use crate::duckdb::DatabaseRef;
use crate::duckdb::LogicalTypeRef;
use crate::duckdb::try_or;
use crate::duckdb_try;

/// A table function backed by DuckDB's `MultiFileFunction<OP>` template.
///
/// Implementors describe a per-format reader; DuckDB owns the cross-file
/// orchestration (globbing, parallelism, virtual columns).
pub trait MultiFileFunction: Sized + Debug {
    /// Per-format options collected from the `TABLE(...)` named parameters.
    /// For minimal implementations this can be a unit struct.
    type ReaderOptions: Send + Sync;

    /// Bind-time data, populated from options and (the schema of) the first
    /// file. Must be `Send` because DuckDB may move it across threads.
    type BindData: Send;

    /// Global state for one query invocation. Shared across worker threads.
    type GlobalState: Send + Sync;

    /// Per-thread local state.
    type LocalState;

    /// Per-file reader. Created when DuckDB first opens a file, dropped when
    /// scanning of that file finishes.
    type Reader: BaseFileReader<GlobalState = Self::GlobalState, LocalState = Self::LocalState>;

    /// Construct default options. Called once per bind.
    fn create_options(ctx: &ClientContextRef) -> VortexResult<Self::ReaderOptions>;

    /// Build bind data from options. Takes ownership of the options struct.
    fn initialize_bind_data(options: Self::ReaderOptions) -> VortexResult<Self::BindData>;

    /// Populate the result schema. DuckDB picks the first file in the file list
    /// to bind against; the implementation should open it (cheaply, metadata-
    /// only if possible) and append columns to `schema`.
    fn bind_reader(
        ctx: &ClientContextRef,
        bind_data: &Self::BindData,
        first_file: &str,
        schema: &mut SchemaBuilder,
    ) -> VortexResult<()>;

    /// Initialize global state for one query.
    fn init_global(
        ctx: &ClientContextRef,
        bind_data: &Self::BindData,
    ) -> VortexResult<Self::GlobalState>;

    /// Initialize per-thread state.
    fn init_local(global: &Self::GlobalState) -> Self::LocalState;

    /// Open a per-file reader. Called once per file, on the thread that won the
    /// race to open it.
    fn create_reader(
        ctx: &ClientContextRef,
        global: &Self::GlobalState,
        bind_data: &Self::BindData,
        file_path: &str,
        file_idx: usize,
    ) -> VortexResult<Self::Reader>;
}

/// Per-file reader contract. Implementations are owned by DuckDB once handed
/// off via [`MultiFileFunction::create_reader`] and dropped when scanning of
/// that file completes.
///
/// Note: this trait is intentionally not `Send`. DuckDB's `MultiFileFunction`
/// guarantees one-thread-at-a-time access to a given reader (it threads them
/// through `MultiFileLocalState` and acquires per-file locks on transitions);
/// requiring `Send` here would force `BaseFileReader` impls to wrap their
/// scan iterators in synchronization primitives unnecessarily.
pub trait BaseFileReader {
    type GlobalState;
    type LocalState;

    /// Set up scan state for the next batch. Called under DuckDB's per-file
    /// lock; should not block on I/O. Return `false` once exhausted.
    fn try_initialize_scan(
        &mut self,
        global: &Self::GlobalState,
        local: &mut Self::LocalState,
    ) -> VortexResult<bool>;

    /// Produce the next batch into `chunk`. Setting `chunk` to size 0 signals
    /// end-of-file; otherwise non-empty implies more may follow.
    fn scan(
        &mut self,
        global: &Self::GlobalState,
        local: &mut Self::LocalState,
        chunk: &mut DataChunkRef,
    ) -> VortexResult<()>;

    /// Per-column statistics by name. Default returns `None`.
    fn get_statistics(&self, _name: &str) -> Option<ColumnStatistics> {
        None
    }

    /// Scan progress within this file in `[0.0, 100.0]`. Default `0.0`.
    fn progress_in_file(&self) -> f64 {
        0.0
    }
}

/// Append-only schema builder passed to [`MultiFileFunction::bind_reader`].
///
/// Wraps the C++ `vector<string>` / `vector<LogicalType>` pair via
/// `duckdb_vx_mff_schema_writer_add_column`.
pub struct SchemaBuilder {
    raw: cpp::duckdb_vx_mff_schema_writer,
}

impl SchemaBuilder {
    /// Append `(name, type)` to the result schema.
    pub fn add_column(&mut self, name: &str, logical_type: &LogicalTypeRef) {
        unsafe {
            cpp::duckdb_vx_mff_schema_writer_add_column(
                self.raw,
                name.as_ptr().cast(),
                name.len(),
                logical_type.as_ptr(),
            );
        }
    }
}

impl DatabaseRef {
    /// Register `T` as a multi-file table function on this database under
    /// `name`.
    ///
    /// The vtable is statically derived from `T` and copied into a C++
    /// `TableFunctionInfo` owned by the catalog; `T` itself is never instanced.
    pub fn register_multi_file_function<T: MultiFileFunction>(
        &self,
        name: &CStr,
    ) -> VortexResult<()> {
        let vtab = cpp::duckdb_vx_mff_vtab_t {
            name: name.as_ptr(),
            create_options: Some(create_options::<T>),
            free_options: Some(free_options::<T>),
            initialize_bind_data: Some(initialize_bind_data::<T>),
            free_bind_data: Some(free_bind_data::<T>),
            bind_reader: Some(bind_reader::<T>),
            init_global: Some(init_global::<T>),
            free_global: Some(free_global::<T>),
            init_local: Some(init_local::<T>),
            free_local: Some(free_local::<T>),
            create_reader: Some(create_reader::<T>),
            free_reader: Some(free_reader::<T>),
            try_initialize_scan: Some(try_initialize_scan::<T>),
            scan: Some(scan::<T>),
            get_statistics: Some(get_statistics::<T>),
            progress_in_file: Some(progress_in_file::<T>),
        };

        duckdb_try!(
            unsafe { cpp::duckdb_vx_mff_register(self.as_ptr(), &raw const vtab) },
            "Failed to register multi-file function '{}'",
            name.to_string_lossy()
        );

        Ok(())
    }
}

// ---------------------------------------------------------------------------
// FFI shim: each callback boxes/unboxes the trait's associated type and
// dispatches to the corresponding trait method.
// ---------------------------------------------------------------------------

unsafe extern "C-unwind" fn create_options<T: MultiFileFunction>(
    ctx: cpp::duckdb_client_context,
    error_out: *mut cpp::duckdb_vx_error,
) -> cpp::duckdb_vx_mff_options {
    let ctx = unsafe { ClientContext::borrow(ctx) };
    try_or(error_out, || {
        let opts = T::create_options(ctx)?;
        Ok(Box::into_raw(Box::new(opts)).cast())
    })
}

unsafe extern "C-unwind" fn free_options<T: MultiFileFunction>(opts: cpp::duckdb_vx_mff_options) {
    if !opts.is_null() {
        drop(unsafe { Box::from_raw(opts.cast::<T::ReaderOptions>()) });
    }
}

unsafe extern "C-unwind" fn initialize_bind_data<T: MultiFileFunction>(
    opts: cpp::duckdb_vx_mff_options,
    error_out: *mut cpp::duckdb_vx_error,
) -> cpp::duckdb_vx_mff_bind_data {
    let opts = unsafe { Box::from_raw(opts.cast::<T::ReaderOptions>()) };
    try_or(error_out, || {
        let bind_data = T::initialize_bind_data(*opts)?;
        Ok(Box::into_raw(Box::new(bind_data)).cast())
    })
}

unsafe extern "C-unwind" fn free_bind_data<T: MultiFileFunction>(
    bind_data: cpp::duckdb_vx_mff_bind_data,
) {
    if !bind_data.is_null() {
        drop(unsafe { Box::from_raw(bind_data.cast::<T::BindData>()) });
    }
}

unsafe extern "C-unwind" fn bind_reader<T: MultiFileFunction>(
    ctx: cpp::duckdb_client_context,
    bind_data: cpp::duckdb_vx_mff_bind_data,
    file_path: *const std::os::raw::c_char,
    path_len: usize,
    schema_writer: cpp::duckdb_vx_mff_schema_writer,
    error_out: *mut cpp::duckdb_vx_error,
) {
    let ctx = unsafe { ClientContext::borrow(ctx) };
    let bind_data =
        unsafe { bind_data.cast::<T::BindData>().as_ref() }.vortex_expect("bind_data null");
    let mut builder = SchemaBuilder { raw: schema_writer };
    try_or(error_out, || {
        let path_bytes = unsafe { slice::from_raw_parts(file_path.cast::<u8>(), path_len) };
        let path = std::str::from_utf8(path_bytes)
            .map_err(|e| vortex_err!("file path is not UTF-8: {e}"))?;
        T::bind_reader(ctx, bind_data, path, &mut builder)
    })
}

unsafe extern "C-unwind" fn init_global<T: MultiFileFunction>(
    ctx: cpp::duckdb_client_context,
    bind_data: cpp::duckdb_vx_mff_bind_data,
    error_out: *mut cpp::duckdb_vx_error,
) -> cpp::duckdb_vx_mff_global {
    let ctx = unsafe { ClientContext::borrow(ctx) };
    let bind_data =
        unsafe { bind_data.cast::<T::BindData>().as_ref() }.vortex_expect("bind_data null");
    try_or(error_out, || {
        let global = T::init_global(ctx, bind_data)?;
        Ok(Box::into_raw(Box::new(global)).cast())
    })
}

unsafe extern "C-unwind" fn free_global<T: MultiFileFunction>(global: cpp::duckdb_vx_mff_global) {
    if !global.is_null() {
        drop(unsafe { Box::from_raw(global.cast::<T::GlobalState>()) });
    }
}

unsafe extern "C-unwind" fn init_local<T: MultiFileFunction>(
    global: cpp::duckdb_vx_mff_global,
) -> cpp::duckdb_vx_mff_local {
    let global = unsafe { global.cast::<T::GlobalState>().as_ref() }.vortex_expect("global null");
    let local = T::init_local(global);
    Box::into_raw(Box::new(local)).cast()
}

unsafe extern "C-unwind" fn free_local<T: MultiFileFunction>(local: cpp::duckdb_vx_mff_local) {
    if !local.is_null() {
        drop(unsafe { Box::from_raw(local.cast::<T::LocalState>()) });
    }
}

#[allow(clippy::too_many_arguments)]
unsafe extern "C-unwind" fn create_reader<T: MultiFileFunction>(
    ctx: cpp::duckdb_client_context,
    global: cpp::duckdb_vx_mff_global,
    bind_data: cpp::duckdb_vx_mff_bind_data,
    file_path: *const std::os::raw::c_char,
    path_len: usize,
    file_idx: usize,
    error_out: *mut cpp::duckdb_vx_error,
) -> cpp::duckdb_vx_mff_reader {
    let ctx = unsafe { ClientContext::borrow(ctx) };
    let global = unsafe { global.cast::<T::GlobalState>().as_ref() }.vortex_expect("global null");
    let bind_data =
        unsafe { bind_data.cast::<T::BindData>().as_ref() }.vortex_expect("bind_data null");
    try_or(error_out, || {
        let path_bytes = unsafe { slice::from_raw_parts(file_path.cast::<u8>(), path_len) };
        let path = std::str::from_utf8(path_bytes)
            .map_err(|e| vortex_err!("file path is not UTF-8: {e}"))?;
        let reader = T::create_reader(ctx, global, bind_data, path, file_idx)?;
        Ok(Box::into_raw(Box::new(reader)).cast())
    })
}

unsafe extern "C-unwind" fn free_reader<T: MultiFileFunction>(reader: cpp::duckdb_vx_mff_reader) {
    if !reader.is_null() {
        drop(unsafe { Box::from_raw(reader.cast::<T::Reader>()) });
    }
}

unsafe extern "C-unwind" fn try_initialize_scan<T: MultiFileFunction>(
    reader: cpp::duckdb_vx_mff_reader,
    global: cpp::duckdb_vx_mff_global,
    local: cpp::duckdb_vx_mff_local,
    error_out: *mut cpp::duckdb_vx_error,
) -> bool {
    let reader = unsafe { reader.cast::<T::Reader>().as_mut() }.vortex_expect("reader null");
    let global = unsafe { global.cast::<T::GlobalState>().as_ref() }.vortex_expect("global null");
    let local = unsafe { local.cast::<T::LocalState>().as_mut() }.vortex_expect("local null");
    try_or(error_out, || reader.try_initialize_scan(global, local))
}

unsafe extern "C-unwind" fn scan<T: MultiFileFunction>(
    reader: cpp::duckdb_vx_mff_reader,
    global: cpp::duckdb_vx_mff_global,
    local: cpp::duckdb_vx_mff_local,
    chunk: cpp::duckdb_data_chunk,
    error_out: *mut cpp::duckdb_vx_error,
) -> bool {
    let reader = unsafe { reader.cast::<T::Reader>().as_mut() }.vortex_expect("reader null");
    let global = unsafe { global.cast::<T::GlobalState>().as_ref() }.vortex_expect("global null");
    let local = unsafe { local.cast::<T::LocalState>().as_mut() }.vortex_expect("local null");
    let chunk_ref = unsafe { DataChunk::borrow_mut(chunk) };
    match reader.scan(global, local, chunk_ref) {
        Ok(()) => {
            unsafe { error_out.write(ptr::null_mut()) };
            true
        }
        Err(err) => {
            let msg = err.to_string();
            unsafe { error_out.write(cpp::duckdb_vx_error_create(msg.as_ptr().cast(), msg.len())) };
            false
        }
    }
}

unsafe extern "C-unwind" fn get_statistics<T: MultiFileFunction>(
    reader: cpp::duckdb_vx_mff_reader,
    name: *const std::os::raw::c_char,
    name_len: usize,
    stats_out: *mut cpp::duckdb_column_statistics,
) -> bool {
    let reader = unsafe { reader.cast::<T::Reader>().as_ref() }.vortex_expect("reader null");
    let name = unsafe { slice::from_raw_parts(name.cast::<u8>(), name_len) };
    let Ok(name) = std::str::from_utf8(name) else {
        return false;
    };
    let Some(stats) = reader.get_statistics(name) else {
        return false;
    };
    let out = unsafe { &mut *stats_out };
    out.min = stats.min.map_or(ptr::null_mut(), |v| v.into_ptr());
    out.max = stats.max.map_or(ptr::null_mut(), |v| v.into_ptr());
    out.max_string_length = stats.max_string_length;
    out.has_null = stats.has_null;
    true
}

unsafe extern "C-unwind" fn progress_in_file<T: MultiFileFunction>(
    reader: cpp::duckdb_vx_mff_reader,
) -> f64 {
    let reader = unsafe { reader.cast::<T::Reader>().as_ref() }.vortex_expect("reader null");
    reader.progress_in_file()
}

// ---------------------------------------------------------------------------
// Helpers used by Phase 5 (concrete implementations).
// ---------------------------------------------------------------------------

/// Build a `CStr` literal-equivalent at runtime. Convenient for type names
/// passed to `LogicalType` / DuckDB FFI.
#[allow(dead_code)]
pub(crate) fn cstring(s: &str) -> CString {
    CString::new(s).unwrap_or_else(|_| CString::default())
}
