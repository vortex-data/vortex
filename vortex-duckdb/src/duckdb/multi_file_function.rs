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
//! The wrapper is generic over one [`MultiFileFunction`] implementation so each
//! registered function gets statically-monomorphised callbacks (no per-call dyn
//! dispatch).
//!
//! Callback lifecycle, matching DuckDB's `MultiFileFunction` / Parquet reader
//! model:
//!
//! 1. Bind: [`MultiFileFunction::create_options`],
//!    [`MultiFileFunction::initialize_bind_data`], and
//!    [`MultiFileFunction::bind_reader`] run once to collect options, bind-time
//!    state, and schema.
//! 2. Query init: [`MultiFileFunction::init_global`] and
//!    [`MultiFileFunction::init_local`] create per-query and per-worker state.
//! 3. File open: [`MultiFileFunction::create_reader`] is called when DuckDB
//!    decides to open a file. DuckDB does not hold the global multi-file
//!    scheduling mutex while opening; it switches to a per-file mutex so other
//!    workers can wait for that specific reader.
//! 4. Reader preparation: [`BaseFileReader::prepare_reader`] maps projection
//!    and filters onto the opened reader. It happens once per reader before any
//!    scan assignment for that reader.
//! 5. Scan assignment: [`BaseFileReader::try_initialize_scan`] is called while
//!    DuckDB holds its global multi-file scheduling mutex. This must be a cheap
//!    claim of one independent unit of work, e.g. a row group or row range. It
//!    must not perform I/O, block on async work, or construct expensive scan
//!    pipelines.
//! 6. Scan execution: DuckDB releases the scheduling mutex before calling its
//!    `PrepareScan` hook, exposed here as [`BaseFileReader::prepare_scan`].
//!    Reader implementations should build per-assignment local scan state
//!    there, then [`BaseFileReader::scan`] drains that local state into chunks.

use std::ffi::CStr;
use std::ffi::CString;
use std::fmt::Debug;
use std::ptr;
use std::slice;

use vortex::error::VortexExpect;
use vortex::error::VortexResult;
use vortex::error::vortex_err;

use crate::cpp;
use crate::duckdb::Cardinality;
use crate::duckdb::ClientContext;
use crate::duckdb::ClientContextRef;
use crate::duckdb::ColumnStatistics;
use crate::duckdb::DataChunk;
use crate::duckdb::DataChunkRef;
use crate::duckdb::DatabaseRef;
use crate::duckdb::DuckdbStringMap;
use crate::duckdb::DuckdbStringMapRef;
use crate::duckdb::ExpressionRef;
use crate::duckdb::LogicalTypeRef;
use crate::duckdb::TableFilterSet;
use crate::duckdb::TableFilterSetRef;
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
    type BindData: Clone + Send;

    /// Global state for one query invocation. Shared across worker threads.
    type GlobalState: Send + Sync;

    /// Per-thread local state.
    type LocalState;

    /// Per-file reader. Created when DuckDB first opens a file, dropped when
    /// scanning of that file finishes. DuckDB stores it in a shared pointer and
    /// may call read-only scan callbacks from multiple workers, so shared
    /// callbacks must be thread-safe.
    type Reader: BaseFileReader<Self::GlobalState, Self::LocalState> + Sync;

    /// Whether DuckDB may pass pushed table filters to
    /// [`BaseFileReader::prepare_reader`].
    const FILTER_PUSHDOWN: bool = false;

    /// Whether DuckDB may omit filter-only columns from final table-scan
    /// output.
    ///
    /// Only meaningful when [`Self::FILTER_PUSHDOWN`] is true.
    const FILTER_PRUNE: bool = false;

    /// Construct default options. Called once per bind.
    fn create_options(ctx: &ClientContextRef) -> VortexResult<Self::ReaderOptions>;

    /// Push a complex filter expression into bind data.
    ///
    /// Returning `true` tells DuckDB the filter is handled exactly and may be
    /// removed from the remaining plan. Returning `false` leaves it for DuckDB
    /// to apply above the scan or turn into a regular table filter.
    fn pushdown_complex_filter(
        bind_data: &mut Self::BindData,
        expr: &ExpressionRef,
    ) -> VortexResult<bool> {
        let _ = (bind_data, expr);
        Ok(false)
    }

    /// Build bind data from options. Takes ownership of the options struct.
    fn initialize_bind_data(options: Self::ReaderOptions) -> VortexResult<Self::BindData>;

    /// Populate the result schema. DuckDB picks the first file in the file list
    /// to bind against; the implementation should open it (cheaply, metadata-
    /// only if possible), record any bind-time metadata it needs, and append
    /// columns to `schema`.
    fn bind_reader(
        ctx: &ClientContextRef,
        bind_data: &mut Self::BindData,
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
    /// race to open it. DuckDB has dropped the global multi-file scheduling
    /// mutex before this call, but holds a per-file mutex for this reader. It is
    /// reasonable to open file metadata here; do not do per-scan or per-split
    /// work here because projection/filter state is not fully prepared yet.
    fn create_reader(
        ctx: &ClientContextRef,
        global: &Self::GlobalState,
        bind_data: &Self::BindData,
        file_path: &str,
        file_idx: usize,
    ) -> VortexResult<Self::Reader>;

    /// Estimated cardinality across `file_count` files. Default returns
    /// [`Cardinality::Unknown`] (DuckDB falls back to its own heuristic).
    fn cardinality(_bind_data: &Self::BindData, _file_count: usize) -> Cardinality {
        Cardinality::Unknown
    }

    /// Exact partition statistics for a file, if already available cheaply.
    ///
    /// DuckDB uses these to fold aggregates such as `COUNT(*)` during
    /// optimization. Returning `None` leaves the scan plan unchanged.
    fn partition_stats(
        _ctx: &ClientContextRef,
        _bind_data: &Self::BindData,
        _file_path: &str,
    ) -> VortexResult<Option<PartitionStats>> {
        Ok(None)
    }

    /// Per-column statistics available from bind-time metadata. Default
    /// returns `None`.
    fn statistics(_bind_data: &Self::BindData, _name: &str) -> Option<ColumnStatistics> {
        None
    }

    /// Populate the bind-time EXPLAIN map with key/value pairs (typical keys:
    /// `Function`, `Files`, `Projection`, `Filters`). Default no-op.
    fn to_string(_bind_data: &Self::BindData, _map: &mut DuckdbStringMapRef) {}
}

/// Exact per-file partition statistics exposed to DuckDB's optimizer.
#[derive(Clone, Copy, Debug)]
pub struct PartitionStats {
    /// Exact number of rows in this file.
    pub row_count: u64,
}

/// A column DuckDB asks a [`BaseFileReader`] to produce in the intermediate
/// scan chunk.
#[derive(Clone, Copy, Debug)]
pub struct ProjectedColumn<'a> {
    /// Column name in the file-local scan chunk.
    pub name: &'a str,
    /// DuckDB column id. Physical columns use a file-local id; virtual columns
    /// use DuckDB's global virtual column id.
    pub column_id: u64,
    /// True when this column is one of DuckDB's virtual columns.
    pub is_virtual: bool,
    /// True when DuckDB's final output expressions reference this column.
    ///
    /// False columns are filter-only: the reader may use them for pushed filter
    /// evaluation, but does not need to materialize them into the scan chunk.
    pub is_projected: bool,
}

/// Per-file reader contract. Implementations are owned by DuckDB once handed
/// off via [`MultiFileFunction::create_reader`] and dropped when scanning of
/// that file completes.
///
/// DuckDB calls [`Self::try_initialize_scan`] while holding its global
/// multi-file lock. That method should claim one independent unit of scan work
/// and store only its descriptor in `LocalState`. [`Self::prepare_scan`] then
/// initializes actual per-worker state outside that lock. [`Self::scan`] drains
/// only that local state and may overlap with later
/// [`Self::try_initialize_scan`] calls on the same reader.
pub trait BaseFileReader<GlobalState, LocalState> {
    /// Configure projection and filter pushdown. Called once after the reader
    /// is created and before any [`Self::try_initialize_scan`] call.
    /// `projection` is the ordered list of intermediate scan columns DuckDB
    /// allocated for this reader. Filter-only columns have
    /// [`ProjectedColumn::is_projected`] set to false. `filters` carries any
    /// filters DuckDB pushed down for this scan.
    ///
    /// Default: no-op (reader scans all columns, no filter pushdown).
    fn prepare_reader(
        &mut self,
        projection: &[ProjectedColumn<'_>],
        filters: Option<&TableFilterSetRef>,
    ) -> VortexResult<()> {
        let _ = (projection, filters);
        Ok(())
    }

    /// Set up scan state for the next batch. Called under DuckDB's global
    /// multi-file scheduling lock; this should only claim work into `local`.
    /// Do not open readers, call `block_on`, or construct scan iterators here.
    /// Return `false` once exhausted.
    fn try_initialize_scan(
        &self,
        global: &GlobalState,
        local: &mut LocalState,
    ) -> VortexResult<bool>;

    /// Initialize local scan state for the work claimed by
    /// [`Self::try_initialize_scan`]. DuckDB calls this outside its global
    /// multi-file scheduling lock, so implementations may open per-split
    /// iterators, block on async setup, or build scan pipelines here.
    ///
    /// Default: no-op.
    fn prepare_scan(&self, global: &GlobalState, local: &mut LocalState) -> VortexResult<()> {
        let _ = (global, local);
        Ok(())
    }

    /// Produce the next batch into `chunk`. Setting `chunk` to size 0 signals
    /// end-of-assignment; otherwise non-empty implies more may follow. This is
    /// called outside DuckDB's global multi-file scheduling lock after
    /// [`Self::prepare_scan`].
    fn scan(
        &self,
        global: &GlobalState,
        local: &mut LocalState,
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
            filter_pushdown: T::FILTER_PUSHDOWN,
            filter_prune: T::FILTER_PRUNE,
            pushdown_complex_filter: Some(pushdown_complex_filter::<T>),
            create_options: Some(create_options::<T>),
            free_options: Some(free_options::<T>),
            initialize_bind_data: Some(initialize_bind_data::<T>),
            clone_bind_data: Some(clone_bind_data::<T>),
            free_bind_data: Some(free_bind_data::<T>),
            bind_reader: Some(bind_reader::<T>),
            init_global: Some(init_global::<T>),
            free_global: Some(free_global::<T>),
            init_local: Some(init_local::<T>),
            free_local: Some(free_local::<T>),
            create_reader: Some(create_reader::<T>),
            free_reader: Some(free_reader::<T>),
            prepare_reader: Some(prepare_reader::<T>),
            try_initialize_scan: Some(try_initialize_scan::<T>),
            prepare_scan: Some(prepare_scan::<T>),
            scan: Some(scan::<T>),
            statistics: Some(statistics::<T>),
            get_statistics: Some(get_statistics::<T>),
            progress_in_file: Some(progress_in_file::<T>),
            cardinality: Some(cardinality::<T>),
            partition_stats: Some(partition_stats::<T>),
            to_string: Some(to_string::<T>),
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

unsafe extern "C-unwind" fn clone_bind_data<T: MultiFileFunction>(
    bind_data: cpp::duckdb_vx_mff_bind_data,
    error_out: *mut cpp::duckdb_vx_error,
) -> cpp::duckdb_vx_mff_bind_data {
    let bind_data =
        unsafe { bind_data.cast::<T::BindData>().as_ref() }.vortex_expect("bind_data null");
    try_or(error_out, || {
        Ok(Box::into_raw(Box::new(bind_data.clone())).cast())
    })
}

unsafe extern "C-unwind" fn pushdown_complex_filter<T: MultiFileFunction>(
    bind_data: cpp::duckdb_vx_mff_bind_data,
    expr: cpp::duckdb_vx_expr,
    error_out: *mut cpp::duckdb_vx_error,
) -> bool {
    let bind_data =
        unsafe { bind_data.cast::<T::BindData>().as_mut() }.vortex_expect("bind_data null");
    let expr = unsafe { crate::duckdb::Expression::borrow(expr) };
    try_or(error_out, || T::pushdown_complex_filter(bind_data, expr))
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
        unsafe { bind_data.cast::<T::BindData>().as_mut() }.vortex_expect("bind_data null");
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

unsafe extern "C-unwind" fn prepare_reader<T: MultiFileFunction>(
    reader: cpp::duckdb_vx_mff_reader,
    projection: *const cpp::duckdb_vx_mff_column,
    projection_count: usize,
    filters: cpp::duckdb_vx_table_filter_set,
    error_out: *mut cpp::duckdb_vx_error,
) {
    let reader = unsafe { reader.cast::<T::Reader>().as_mut() }.vortex_expect("reader null");
    let filter_ref = if filters.is_null() {
        None
    } else {
        Some(unsafe { TableFilterSet::borrow(filters) })
    };
    try_or(error_out, || {
        // Materialize column metadata with &str borrows scoped to this call.
        let mut projected_columns = Vec::with_capacity(projection_count);
        for i in 0..projection_count {
            let col = unsafe { &*projection.add(i) };
            let bytes = unsafe { slice::from_raw_parts(col.name.cast::<u8>(), col.name_len) };
            let name = std::str::from_utf8(bytes)
                .map_err(|e| vortex_err!("projection column name not UTF-8: {e}"))?;
            projected_columns.push(ProjectedColumn {
                name,
                column_id: col.column_id,
                is_virtual: col.is_virtual,
                is_projected: col.is_projected,
            });
        }
        reader.prepare_reader(&projected_columns, filter_ref)
    });
}

unsafe extern "C-unwind" fn try_initialize_scan<T: MultiFileFunction>(
    reader: cpp::duckdb_vx_mff_reader,
    global: cpp::duckdb_vx_mff_global,
    local: cpp::duckdb_vx_mff_local,
    error_out: *mut cpp::duckdb_vx_error,
) -> bool {
    let reader = unsafe { reader.cast::<T::Reader>().as_ref() }.vortex_expect("reader null");
    let global = unsafe { global.cast::<T::GlobalState>().as_ref() }.vortex_expect("global null");
    let local = unsafe { local.cast::<T::LocalState>().as_mut() }.vortex_expect("local null");
    try_or(error_out, || reader.try_initialize_scan(global, local))
}

unsafe extern "C-unwind" fn prepare_scan<T: MultiFileFunction>(
    reader: cpp::duckdb_vx_mff_reader,
    global: cpp::duckdb_vx_mff_global,
    local: cpp::duckdb_vx_mff_local,
    error_out: *mut cpp::duckdb_vx_error,
) {
    let reader = unsafe { reader.cast::<T::Reader>().as_ref() }.vortex_expect("reader null");
    let global = unsafe { global.cast::<T::GlobalState>().as_ref() }.vortex_expect("global null");
    let local = unsafe { local.cast::<T::LocalState>().as_mut() }.vortex_expect("local null");
    try_or(error_out, || reader.prepare_scan(global, local))
}

unsafe extern "C-unwind" fn scan<T: MultiFileFunction>(
    reader: cpp::duckdb_vx_mff_reader,
    global: cpp::duckdb_vx_mff_global,
    local: cpp::duckdb_vx_mff_local,
    chunk: cpp::duckdb_data_chunk,
    error_out: *mut cpp::duckdb_vx_error,
) -> bool {
    let reader = unsafe { reader.cast::<T::Reader>().as_ref() }.vortex_expect("reader null");
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
    write_column_statistics(stats_out, stats);
    true
}

unsafe extern "C-unwind" fn statistics<T: MultiFileFunction>(
    bind_data: cpp::duckdb_vx_mff_bind_data,
    name: *const std::os::raw::c_char,
    name_len: usize,
    stats_out: *mut cpp::duckdb_column_statistics,
) -> bool {
    let bind_data =
        unsafe { bind_data.cast::<T::BindData>().as_ref() }.vortex_expect("bind_data null");
    let name = unsafe { slice::from_raw_parts(name.cast::<u8>(), name_len) };
    let Ok(name) = std::str::from_utf8(name) else {
        return false;
    };
    let Some(stats) = T::statistics(bind_data, name) else {
        return false;
    };
    write_column_statistics(stats_out, stats);
    true
}

fn write_column_statistics(stats_out: *mut cpp::duckdb_column_statistics, stats: ColumnStatistics) {
    let out = unsafe { &mut *stats_out };
    out.min = stats.min.map_or(ptr::null_mut(), |v| v.into_ptr());
    out.max = stats.max.map_or(ptr::null_mut(), |v| v.into_ptr());
    out.max_string_length = stats.max_string_length;
    out.has_null = stats.has_null;
}

unsafe extern "C-unwind" fn progress_in_file<T: MultiFileFunction>(
    reader: cpp::duckdb_vx_mff_reader,
) -> f64 {
    let reader = unsafe { reader.cast::<T::Reader>().as_ref() }.vortex_expect("reader null");
    reader.progress_in_file()
}

unsafe extern "C-unwind" fn cardinality<T: MultiFileFunction>(
    bind_data: cpp::duckdb_vx_mff_bind_data,
    file_count: usize,
    out: *mut cpp::duckdb_vx_node_statistics,
) -> bool {
    let bind_data =
        unsafe { bind_data.cast::<T::BindData>().as_ref() }.vortex_expect("bind_data null");
    let out = unsafe { &mut *out };
    match T::cardinality(bind_data, file_count) {
        Cardinality::Unknown => false,
        Cardinality::Estimate(c) => {
            out.has_estimated_cardinality = true;
            out.estimated_cardinality = c;
            true
        }
        Cardinality::Maximum(c) => {
            out.has_max_cardinality = true;
            out.max_cardinality = c;
            out.has_estimated_cardinality = true;
            out.estimated_cardinality = c;
            true
        }
    }
}

unsafe extern "C-unwind" fn partition_stats<T: MultiFileFunction>(
    ctx: cpp::duckdb_client_context,
    bind_data: cpp::duckdb_vx_mff_bind_data,
    file_path: *const std::os::raw::c_char,
    path_len: usize,
    out: *mut cpp::duckdb_vx_mff_partition_stats,
    error_out: *mut cpp::duckdb_vx_error,
) -> bool {
    let ctx = unsafe { ClientContext::borrow(ctx) };
    let bind_data =
        unsafe { bind_data.cast::<T::BindData>().as_ref() }.vortex_expect("bind_data null");
    try_or(error_out, || {
        let path_bytes = unsafe { slice::from_raw_parts(file_path.cast::<u8>(), path_len) };
        let path = std::str::from_utf8(path_bytes)
            .map_err(|e| vortex_err!("file path is not UTF-8: {e}"))?;
        let Some(stats) = T::partition_stats(ctx, bind_data, path)? else {
            return Ok(false);
        };
        let out = unsafe { &mut *out };
        out.row_count = stats.row_count;
        Ok(true)
    })
}

unsafe extern "C-unwind" fn to_string<T: MultiFileFunction>(
    bind_data: cpp::duckdb_vx_mff_bind_data,
    map: cpp::duckdb_vx_string_map,
) {
    let bind_data =
        unsafe { bind_data.cast::<T::BindData>().as_ref() }.vortex_expect("bind_data null");
    let map = unsafe { DuckdbStringMap::borrow_mut(map) };
    T::to_string(bind_data, map);
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
