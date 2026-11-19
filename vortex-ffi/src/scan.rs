// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use core::slice;
use std::ffi::c_int;
use std::num::NonZeroUsize;
use std::ops::Range;
use std::ptr;
use std::sync::Arc;

use arrow_array::RecordBatch;
use arrow_array::cast::AsArray;
use arrow_array::ffi_stream::FFI_ArrowArrayStream;
use arrow_schema::ArrowError;
use arrow_schema::Field;
use futures::StreamExt;
use vortex::array::ArrayRef;
use vortex::array::ExecutionCtx;
use vortex::array::VortexSessionExecute;
use vortex::array::expr::stats::Precision;
use vortex::array::stream::SendableArrayStream;
use vortex::buffer::Buffer;
use vortex::error::VortexResult;
use vortex::error::vortex_bail;
use vortex::error::vortex_ensure;
use vortex::expr::root;
use vortex::io::runtime::BlockingRuntime;
use vortex::layout::scan::arrow::RecordBatchIteratorAdapter;
use vortex::scan::DataSource;
use vortex::scan::ScanPoolRef;
use vortex::scan::ScanRef;
use vortex::scan::ScanRequest;
use vortex::scan::execute_with_workers;
use vortex::scan::selection::Selection;
use vortex::scan::strict_sorted_buffer::StrictSortedBuffer;
use vortex_arrow::ArrowSessionExt;
use vortex_arrow::ExecuteArrowExt;

use crate::POOL;
use crate::RUNTIME;
use crate::array::vx_array;
use crate::box_wrapper;
use crate::data_source::vx_data_source;
use crate::dtype::vx_dtype;
use crate::error::try_or;
use crate::error::try_or_default;
use crate::error::vx_error;
use crate::expression::vx_expression;
use crate::session::vx_session;

box_wrapper!(
    /// A planned scan of a vx_data_source with projections and filters.
    ///
    /// A scan does no I/O until it runs. Either open it into a vx_scan_pool with vx_scan_open,
    /// cut it into independent pieces with vx_scan_split, or run it on the runtime's threads with
    /// vx_scan_execute. Each of these consumes the scan.
    ScanRef,
    vx_scan
);

box_wrapper!(
    /// An opened scan whose work any number of workers share.
    ///
    /// Each unit of work is read by exactly one worker, and all workers of a pool share its row
    /// limit. vx_scan_pool_worker may be called from any thread, at any time.
    ScanPoolRef,
    vx_scan_pool
);

box_wrapper!(
    /// A stream of a scan's arrays.
    ///
    /// A worker of a vx_scan_pool does its work on the thread that calls vx_scan_worker_next (or
    /// reads its Arrow stream); it never runs on other threads. Drive each worker from one thread
    /// at a time.
    Option<SendableArrayStream>,
    vx_scan_worker
);

/// Consume an owned worker pointer for layered FFI crates and return its Vortex array stream.
///
/// Drive the stream on [`crate::ffi_runtime`].
///
/// # Safety
///
/// `worker` must be a non-null owned worker handle created by `vortex-ffi`. This function
/// consumes the handle; callers must not use or free it after calling this function.
pub unsafe fn vx_scan_worker_into_array_stream(
    worker: *mut vx_scan_worker,
) -> VortexResult<SendableArrayStream> {
    vortex_ensure!(!worker.is_null(), "null vx_scan_worker");
    match *vx_scan_worker::into_box(worker) {
        Some(stream) => Ok(stream),
        None => vortex_bail!("worker already exhausted"),
    }
}

// We parse Selection from vx_scan_selection[_include], so we don't need
// to instantiate VX_SELECTION_* items directly.
#[repr(C)]
#[allow(dead_code)]
#[cfg_attr(test, derive(Default))]
pub enum vx_scan_selection_include {
    #[cfg_attr(test, default)]
    VX_SELECTION_INCLUDE_ALL = 0,
    /// Include rows at the indices in vx_scan_selection.idx.
    VX_SELECTION_INCLUDE_RANGE = 1,
    /// Exclude rows at the indices in vx_scan_selection.idx.
    VX_SELECTION_EXCLUDE_RANGE = 2,
}

/// Scan row selection.
/// "idx" is copied while calling vx_data_source_scan and can be freed after.
#[repr(C)]
#[cfg_attr(test, derive(Default))]
pub struct vx_scan_selection {
    /// Used only when "include" is not VX_SELECTION_INCLUDE_ALL.
    /// If set, must point to an array of len "idx_len" row_indices.
    pub idx: *const u64,
    /// Used only when "include" is not VX_SELECTION_INCLUDE_ALL
    pub idx_len: usize,
    pub include: vx_scan_selection_include,
}

/// Scan options. All fields are optional. To return everything,
/// zero-initialize this struct.
#[repr(C)]
#[cfg_attr(test, derive(Default))]
pub struct vx_scan_options {
    /// What columns to return. NULL means all columns.
    pub projection: *const vx_expression,
    /// Predicate expression. NULL means no filter.
    pub filter: *const vx_expression,
    /// Row range [begin, end). Setting row_range_begin and row_range_end to 0
    /// means no limit.
    pub row_range_begin: u64,
    pub row_range_end: u64,
    /// Row-index filter applied after row_range.
    pub selection: vx_scan_selection,
    /// Maximum number of rows to return. 0 means no limit.
    pub limit: u64,
    /// If true, return in storage order.
    pub ordered: bool,
}

#[repr(C)]
#[cfg_attr(test, derive(Debug, PartialEq, Eq, Default))]
pub enum vx_estimate_type {
    /// No estimate is available.
    #[cfg_attr(test, default)]
    VX_ESTIMATE_UNKNOWN = 0,
    /// The value in vx_estimate.estimate is exact.
    VX_ESTIMATE_EXACT = 1,
    /// The value in vx_estimate.estimate is an upper bound.
    VX_ESTIMATE_INEXACT = 2,
}

/// An estimate of a count, such as the number of rows a scan returns.
#[repr(C)]
#[cfg_attr(test, derive(Default))]
pub struct vx_estimate {
    pub r#type: vx_estimate_type,
    /// Set only when "type" is not VX_ESTIMATE_UNKNOWN.
    pub estimate: u64,
}

fn scan_request(opts: *const vx_scan_options) -> VortexResult<ScanRequest> {
    if opts.is_null() {
        return Ok(ScanRequest::default());
    }
    let opts = unsafe { &*opts };

    let projection = if opts.projection.is_null() {
        root()
    } else {
        vx_expression::as_ref(opts.projection).clone()
    };

    let filter = if opts.filter.is_null() {
        None
    } else {
        Some(vx_expression::as_ref(opts.filter).clone())
    };

    let selection = &opts.selection;
    let selection = match selection.include {
        vx_scan_selection_include::VX_SELECTION_INCLUDE_ALL => Selection::All,
        vx_scan_selection_include::VX_SELECTION_INCLUDE_RANGE => {
            vortex_ensure!(!selection.idx.is_null());
            let buf = unsafe { slice::from_raw_parts(selection.idx, selection.idx_len) };
            let buf = Buffer::copy_from(buf);
            Selection::IncludeByIndex(StrictSortedBuffer::try_new(buf)?)
        }
        vx_scan_selection_include::VX_SELECTION_EXCLUDE_RANGE => {
            vortex_ensure!(!selection.idx.is_null());
            let buf = unsafe { slice::from_raw_parts(selection.idx, selection.idx_len) };
            let buf = Buffer::copy_from(buf);
            Selection::ExcludeByIndex(StrictSortedBuffer::try_new(buf)?)
        }
    };

    let ordered = opts.ordered;

    let start = opts.row_range_begin;
    let end = opts.row_range_end;
    let row_range = (start > 0 || end > 0).then_some(Range { start, end });

    let limit = (opts.limit != 0).then_some(opts.limit);

    Ok(ScanRequest {
        projection,
        filter,
        row_range,
        selection,
        ordered,
        limit,
    })
}

fn write_estimate<T: Into<u64>>(estimate: Precision<T>, out: &mut vx_estimate) {
    match estimate {
        Precision::Exact(value) => {
            out.r#type = vx_estimate_type::VX_ESTIMATE_EXACT;
            out.estimate = value.into();
        }
        Precision::Inexact(value) => {
            out.r#type = vx_estimate_type::VX_ESTIMATE_INEXACT;
            out.estimate = value.into();
        }
        Precision::Absent => {
            out.r#type = vx_estimate_type::VX_ESTIMATE_UNKNOWN;
        }
    }
}

/// Plan a scan of a data source.
///
/// Planning reads metadata only. "options" may be NULL, in which case all rows
/// and columns are returned.
///
/// Returns NULL and writes an error to "*err" on failure.
#[unsafe(no_mangle)]
pub unsafe extern "C-unwind" fn vx_data_source_scan(
    data_source: *const vx_data_source,
    options: *const vx_scan_options,
    err: *mut *mut vx_error,
) -> *mut vx_scan {
    try_or(err, ptr::null_mut(), || {
        let request = scan_request(options)?;
        let scan = RUNTIME.block_on(vx_data_source::as_ref(data_source).scan(request))?;
        Ok(vx_scan::new(scan))
    })
}

/// Return the dtype of the arrays the scan returns.
/// On error returns NULL and sets "err".
#[unsafe(no_mangle)]
pub unsafe extern "C-unwind" fn vx_scan_dtype(
    scan: *const vx_scan,
    err: *mut *mut vx_error,
) -> *const vx_dtype {
    try_or(err, ptr::null(), || {
        Ok(vx_dtype::new(vx_scan::as_ref(scan).dtype().clone()))
    })
}

/// Write an estimate of the number of rows the scan returns to "*count".
///
/// On success, returns 0.
/// On error, returns 1 and sets "err".
#[unsafe(no_mangle)]
pub unsafe extern "C-unwind" fn vx_scan_row_count(
    scan: *const vx_scan,
    count: *mut vx_estimate,
    err: *mut *mut vx_error,
) -> c_int {
    try_or(err, 1, || {
        vortex_ensure!(!count.is_null(), "null count");
        write_estimate(vx_scan::as_ref(scan).row_count(), unsafe { &mut *count });
        Ok(0)
    })
}

/// Cut a scan into at most "n" independent scans, in row order.
///
/// Consumes "scan", even on error. Writes the pieces to "out", which must point
/// to an array of at least "n" vx_scan pointers, and returns how many it wrote.
/// Together the pieces read exactly what "scan" reads. A scan that cannot be cut
/// (for example one with a limit) is returned as a single piece. Splitting reads
/// metadata only.
///
/// On error returns 0 and sets "err".
#[unsafe(no_mangle)]
pub unsafe extern "C-unwind" fn vx_scan_split(
    scan: *mut vx_scan,
    n: usize,
    out: *mut *mut vx_scan,
    err: *mut *mut vx_error,
) -> usize {
    let scan = *vx_scan::into_box(scan);
    try_or(err, 0, || {
        vortex_ensure!(!out.is_null(), "null output array");
        let Some(n) = NonZeroUsize::new(n) else {
            vortex_bail!("cannot split a scan into zero pieces");
        };
        let pieces = scan.split(n)?;
        vortex_ensure!(pieces.len() <= n.get(), "scan split into too many pieces");
        let out = unsafe { slice::from_raw_parts_mut(out, n.get()) };
        let count = pieces.len();
        for (slot, piece) in out.iter_mut().zip(pieces) {
            *slot = vx_scan::new(piece);
        }
        Ok(count)
    })
}

/// Open a scan for execution.
///
/// Consumes "scan", even on error. Opening is cheap: all I/O happens when the
/// pool's workers are driven.
///
/// On error returns NULL and sets "err".
#[unsafe(no_mangle)]
pub unsafe extern "C-unwind" fn vx_scan_open(
    scan: *mut vx_scan,
    err: *mut *mut vx_error,
) -> *mut vx_scan_pool {
    let scan = *vx_scan::into_box(scan);
    try_or(err, ptr::null_mut(), || Ok(vx_scan_pool::new(scan.open()?)))
}

/// Run a scan on the runtime's threads and return its arrays as one stream.
///
/// Consumes "scan", even on error. Runs "workers" workers as tasks on the shared
/// runtime, which makes progress on the threads set with
/// vx_runtime_set_worker_threads and on any thread inside a Vortex call. "workers"
/// of 0 runs one worker per runtime thread, or one if the runtime has none. An
/// ordered scan returns its rows in row order.
///
/// Read the returned stream with vx_scan_worker_next or
/// vx_scan_worker_scan_arrow from one thread at a time.
///
/// On error returns NULL and sets "err".
#[unsafe(no_mangle)]
pub unsafe extern "C-unwind" fn vx_scan_execute(
    scan: *mut vx_scan,
    workers: usize,
    err: *mut *mut vx_error,
) -> *mut vx_scan_worker {
    let scan = *vx_scan::into_box(scan);
    try_or(err, ptr::null_mut(), || {
        let workers = NonZeroUsize::new(workers)
            .unwrap_or_else(|| NonZeroUsize::new(POOL.worker_count()).unwrap_or(NonZeroUsize::MIN));
        let stream = execute_with_workers(scan, workers, &RUNTIME.handle())?;
        Ok(vx_scan_worker::new(Some(stream)))
    })
}

/// Return true once every unit of the pool's work has been handed to a worker.
///
/// Workers may still be producing arrays. A host uses this to stop adding
/// threads to a pool that has nothing left to hand out.
#[unsafe(no_mangle)]
pub unsafe extern "C-unwind" fn vx_scan_pool_is_exhausted(pool: *const vx_scan_pool) -> bool {
    vx_scan_pool::as_ref(pool).is_exhausted()
}

/// Create a new worker of a pool.
///
/// Thread-safe: any number of threads may create workers of the same pool at
/// any time. The worker keeps the pool's work alive, so the pool may be freed
/// before its workers. A worker ends once the pool has no work left for it.
///
/// On error returns NULL and sets "err".
#[unsafe(no_mangle)]
pub unsafe extern "C-unwind" fn vx_scan_pool_worker(
    pool: *const vx_scan_pool,
    err: *mut *mut vx_error,
) -> *mut vx_scan_worker {
    try_or(err, ptr::null_mut(), || {
        let pool = Arc::clone(vx_scan_pool::as_ref(pool));
        Ok(vx_scan_worker::new(Some(pool.worker())))
    })
}

/// Return the next array of a worker.
///
/// Blocks the calling thread, which does the worker's work.
/// On success returns an array.
/// When the worker is exhausted returns NULL without setting "err", and keeps
/// doing so on later calls.
/// On error returns NULL and sets "err".
///
/// Not thread-safe: drive each worker from one thread at a time.
#[unsafe(no_mangle)]
pub unsafe extern "C-unwind" fn vx_scan_worker_next(
    worker: *mut vx_scan_worker,
    err: *mut *mut vx_error,
) -> *const vx_array {
    let worker = vx_scan_worker::as_mut(worker);
    try_or_default(err, || {
        let Some(stream) = worker.as_mut() else {
            return Ok(ptr::null());
        };
        match RUNTIME.block_on(stream.next()) {
            Some(array) => Ok(vx_array::new(array?)),
            None => {
                *worker = None;
                Ok(ptr::null())
            }
        }
    })
}

/// Consume a worker into an ArrowArrayStream.
///
/// Consumes "worker", even on error: do not free it after calling this function.
/// Each call to the stream's get_next blocks the calling thread, which does the
/// worker's work.
///
/// On success, sets "stream" and returns 0.
/// On error, sets "err" and returns 1.
#[unsafe(no_mangle)]
pub unsafe extern "C-unwind" fn vx_scan_worker_scan_arrow(
    session: *const vx_session,
    worker: *mut vx_scan_worker,
    stream: *mut FFI_ArrowArrayStream,
    err: *mut *mut vx_error,
) -> c_int {
    let worker = *vx_scan_worker::into_box(worker);
    try_or(err, 1, || {
        vortex_ensure!(!stream.is_null(), "null ArrowArrayStream");
        let Some(array_stream) = worker else {
            vortex_bail!("Can't consume worker into ArrowArrayStream: worker already exhausted");
        };
        let dtype = array_stream.dtype();

        let session = vx_session::as_ref(session);

        let schema = session.arrow().to_arrow_schema(dtype)?;
        let schema = Arc::new(schema);
        let target = Field::new_struct("", schema.fields().clone(), false);

        let on_chunk = move |chunk: VortexResult<ArrayRef>| -> VortexResult<RecordBatch> {
            let chunk: ArrayRef = chunk?;
            let mut ctx: ExecutionCtx = session.create_execution_ctx();
            let arrow = chunk.execute_arrow(Some(&target), &mut ctx)?;
            Ok(RecordBatch::from(arrow.as_struct().clone()))
        };

        let iter = RUNTIME
            .block_on_stream(array_stream)
            .map(on_chunk)
            .map(|result| result.map_err(|e| ArrowError::ExternalError(Box::new(e))));

        let reader = RecordBatchIteratorAdapter::new(iter, schema);
        let arrow_stream = FFI_ArrowArrayStream::new(Box::new(reader));
        unsafe {
            ptr::write(stream, arrow_stream);
        };
        Ok(0)
    })
}

// Object store error: Generic LocalFileSystem error: Unable to convert
// URL "file:///C:%255CWindows%255CSystemTemp%255C.tmpRXzX38" to filesystem path
// https://github.com/servo/rust-url/issues/1077
#[cfg(not(windows))]
#[cfg(test)]
mod tests {
    use std::ptr;
    use std::sync::LazyLock;

    use tempfile::NamedTempFile;
    use vortex::VortexSessionDefault;
    use vortex::array::IntoArray;
    use vortex::array::arrays::PrimitiveArray;
    use vortex::array::arrays::StructArray;
    use vortex::array::arrays::VarBinView;
    use vortex::array::arrays::VarBinViewArray;
    use vortex::array::validity::Validity;
    use vortex::session::VortexSession;
    use vortex_array::VortexSessionExecute;
    use vortex_array::arrays::struct_::StructArrayExt;
    use vortex_array::assert_arrays_eq;

    use crate::array::vx_array;
    use crate::array::vx_array_free;
    use crate::array::vx_array_get_field;
    use crate::data_source::vx_data_source_free;
    use crate::data_source::vx_data_source_new;
    use crate::data_source::vx_data_source_options;
    use crate::dtype::vx_dtype;
    use crate::dtype::vx_dtype_free;
    use crate::expression::vx_binary_operator;
    use crate::expression::vx_expression_binary;
    use crate::expression::vx_expression_free;
    use crate::expression::vx_expression_get_item;
    use crate::expression::vx_expression_literal;
    use crate::expression::vx_expression_root;
    use crate::scalar::vx_scalar_free;
    use crate::scalar::vx_scalar_new_u64;
    use crate::data_source::vx_data_source;
    use crate::scan::vx_data_source_scan;
    use crate::scan::vx_estimate;
    use crate::scan::vx_estimate_type;
    use crate::scan::vx_scan;
    use crate::scan::vx_scan_execute;
    use crate::scan::vx_scan_free;
    use crate::scan::vx_scan_open;
    use crate::scan::vx_scan_options;
    use crate::scan::vx_scan_pool_free;
    use crate::scan::vx_scan_pool_is_exhausted;
    use crate::scan::vx_scan_pool_worker;
    use crate::scan::vx_scan_row_count;
    use crate::scan::vx_scan_split;
    use crate::scan::vx_scan_worker;
    use crate::scan::vx_scan_worker_free;
    use crate::scan::vx_scan_worker_next;
    use crate::scan::vx_scan_selection;
    use crate::scan::vx_scan_selection_include;
    use crate::session::vx_session;
    use crate::session::vx_session_free;
    use crate::session::vx_session_new;
    use crate::sink::vx_array_sink_close;
    use crate::sink::vx_array_sink_open_file;
    use crate::sink::vx_array_sink_push;
    use crate::string::vx_view;
    use crate::tests::SAMPLE_ROWS;
    use crate::tests::assert_error;
    use crate::tests::assert_no_error;
    use crate::tests::write_sample;

    static SESSION: LazyLock<VortexSession> = LazyLock::new(VortexSession::empty);

    /// Open a data source over a fresh sample file.
    unsafe fn sample_source() -> (
        *const vx_session,
        NamedTempFile,
        StructArray,
        *const vx_data_source,
    ) {
        unsafe {
            let session = vx_session_new();
            let (sample, struct_array) = write_sample(session);
            let path = vx_view::from_str(sample.path().to_str().unwrap());
            let ds_options = vx_data_source_options {
                paths: &raw const path,
                paths_len: 1,
                ..Default::default()
            };

            let mut error = ptr::null_mut();
            let ds = vx_data_source_new(session, &raw const ds_options, &raw mut error);
            assert_no_error(error);
            assert!(!ds.is_null());
            (session, sample, struct_array, ds)
        }
    }

    unsafe fn plan(ds: *const vx_data_source, options: *const vx_scan_options) -> *mut vx_scan {
        unsafe {
            let mut error = ptr::null_mut();
            let scan = vx_data_source_scan(ds, options, &raw mut error);
            assert_no_error(error);
            assert!(!scan.is_null());
            scan
        }
    }

    /// Drain a worker, returning the arrays it yields. Frees the worker.
    unsafe fn drain(worker: *mut vx_scan_worker) -> Vec<*const vx_array> {
        unsafe {
            let mut error = ptr::null_mut();
            let mut arrays = Vec::new();
            loop {
                let array = vx_scan_worker_next(worker, &raw mut error);
                assert_no_error(error);
                if array.is_null() {
                    break;
                }
                arrays.push(array);
            }
            // An exhausted worker keeps returning NULL.
            assert!(vx_scan_worker_next(worker, &raw mut error).is_null());
            assert_no_error(error);
            vx_scan_worker_free(worker);
            arrays
        }
    }

    /// The `age` column (row index) of each array, concatenated.
    unsafe fn ages(arrays: &[*const vx_array]) -> Vec<u64> {
        let mut ctx = SESSION.create_execution_ctx();
        let mut out = Vec::new();
        for &ptr in arrays {
            let age = vx_array::as_ref(ptr)
                .clone()
                .execute::<StructArray>(&mut ctx)
                .unwrap()
                .unmasked_field_by_name("age")
                .unwrap()
                .clone()
                .execute::<PrimitiveArray>(&mut ctx)
                .unwrap();
            out.extend_from_slice(age.as_slice::<u64>());
            unsafe { vx_array_free(ptr) };
        }
        out
    }

    /// Perform a scan with options over a sample file, return owned read array and
    /// original generated array for the sample file.
    fn scan(options: *const vx_scan_options) -> (*const vx_array, StructArray) {
        unsafe {
            let (session, _sample, struct_array, ds) = sample_source();
            let scan = plan(ds, options);

            let mut error = ptr::null_mut();
            let pool = vx_scan_open(scan, &raw mut error);
            assert_no_error(error);
            assert!(!pool.is_null());

            let worker = vx_scan_pool_worker(pool, &raw mut error);
            assert_no_error(error);
            assert!(!worker.is_null());

            let mut arrays = drain(worker);
            assert_eq!(arrays.len(), 1);
            assert!(vx_scan_pool_is_exhausted(pool));

            vx_scan_pool_free(pool);
            vx_data_source_free(ds);
            vx_session_free(session);

            (arrays.remove(0), struct_array)
        }
    }

    fn expected_rows(source: &StructArray, indices: &[usize]) -> StructArray {
        let names = source.unmasked_field_by_name("name").unwrap();
        let names = names.as_::<VarBinView>();
        StructArray::try_new(
            ["age", "height", "name"].into(),
            vec![
                PrimitiveArray::from_option_iter(indices.iter().map(|&i| Some(i as u64)))
                    .into_array(),
                PrimitiveArray::from_option_iter(indices.iter().map(|&i| Some(200 * i as u64)))
                    .into_array(),
                VarBinViewArray::from_iter_str(
                    indices
                        .iter()
                        .map(|&i| String::from_utf8(names.bytes_at(i).to_vec()).unwrap()),
                )
                .into_array(),
            ],
            indices.len(),
            Validity::NonNullable,
        )
        .unwrap()
    }

    #[test]
    #[cfg_attr(miri, ignore)]
    fn test_no_options() {
        let mut ctx = SESSION.create_execution_ctx();
        let (array, struct_array) = scan(ptr::null());
        assert_arrays_eq!(vx_array::as_ref(array), struct_array, &mut ctx);
        unsafe { vx_array_free(array) };
    }

    #[test]
    #[cfg_attr(miri, ignore)]
    fn test_project_all() {
        let mut ctx = SESSION.create_execution_ctx();
        let opts = vx_scan_options::default();
        let (array, struct_array) = scan(&raw const opts);
        assert_arrays_eq!(vx_array::as_ref(array), struct_array, &mut ctx);
        unsafe { vx_array_free(array) };
    }

    /// C callers read fields with `vx_array_get_field`, which needs a struct rather than the
    /// lazy pack expression a layout reader builds for the projection.
    #[test]
    #[cfg_attr(miri, ignore)]
    fn test_scan_returns_struct() {
        let mut ctx = SESSION.create_execution_ctx();
        let strings = VarBinViewArray::from_iter_nullable_str([Some("inlined"), Some("y")]);
        let struct_array = StructArray::try_new(
            ["s"].into(),
            vec![strings.clone().into_array()],
            strings.len(),
            Validity::NonNullable,
        )
        .unwrap();
        unsafe {
            let session = vx_session_new();
            let file = NamedTempFile::new().unwrap();
            let path = vx_view::from_str(file.path().to_str().unwrap());
            let dtype = vx_dtype::new(struct_array.dtype().clone());
            let mut error = ptr::null_mut();
            let sink = vx_array_sink_open_file(session, path, dtype, &raw mut error);
            assert_no_error(error);
            let input = vx_array::new(struct_array.into_array());
            vx_array_sink_push(sink, input, &raw mut error);
            assert_no_error(error);
            vx_array_sink_close(sink, &raw mut error);
            assert_no_error(error);
            vx_array_free(input);
            vx_dtype_free(dtype);

            let ds_options = vx_data_source_options {
                paths: &raw const path,
                paths_len: 1,
                ..Default::default()
            };
            let ds = vx_data_source_new(session, &raw const ds_options, &raw mut error);
            assert_no_error(error);
            let scan = plan(ds, ptr::null());
            let worker = vx_scan_execute(scan, 1, &raw mut error);
            assert_no_error(error);
            let array = vx_scan_worker_next(worker, &raw mut error);
            assert_no_error(error);

            let field = vx_array_get_field(array, 0, &raw mut error);
            assert_no_error(error);
            assert_arrays_eq!(vx_array::as_ref(field), strings, &mut ctx);

            vx_array_free(field);
            vx_array_free(array);
            vx_scan_worker_free(worker);
            vx_data_source_free(ds);
            vx_session_free(session);
        }
    }

    #[test]
    #[cfg_attr(miri, ignore)]
    fn test_project_single_field() {
        unsafe {
            let mut ctx = SESSION.create_execution_ctx();
            let root = vx_expression_root();
            let mut opts = vx_scan_options::default();

            for field in ["age", "height", "name"] {
                let field_expr = vx_expression_get_item(vx_view::from_str(field), root);
                assert!(!field_expr.is_null());
                opts.projection = field_expr;
                let (array, struct_array) = scan(&raw const opts);
                assert_arrays_eq!(
                    vx_array::as_ref(array),
                    struct_array.unmasked_field_by_name(field).unwrap(),
                    &mut ctx
                );
                vx_array_free(array);
                vx_expression_free(field_expr);
            }
            vx_expression_free(root);
        }
    }

    #[test]
    #[cfg_attr(miri, ignore)]
    fn test_project_sum() {
        let session = VortexSession::default();
        let mut ctx = session.create_execution_ctx();
        unsafe {
            let root = vx_expression_root();
            let mut opts = vx_scan_options::default();

            let expr_age = vx_expression_get_item(vx_view::from_str("age"), root);
            let expr_height = vx_expression_get_item(vx_view::from_str("height"), root);
            let expr_sum =
                vx_expression_binary(vx_binary_operator::VX_OPERATOR_ADD, expr_age, expr_height);

            opts.projection = expr_sum;
            let (array, _) = scan(&raw const opts);
            let expected =
                PrimitiveArray::from_option_iter((0..SAMPLE_ROWS as u64).map(|i| Some(201 * i)));
            assert_arrays_eq!(vx_array::as_ref(array), expected, &mut ctx);
            {
                let array = vx_array::as_ref(array);
                let stats = array.statistics();
                assert!(stats.compute_is_sorted(&mut ctx).unwrap());
                assert_eq!(stats.compute_min(&mut ctx), Some(0));
                assert_eq!(
                    stats.compute_max(&mut ctx),
                    Some(200 * (SAMPLE_ROWS - 1) + 199)
                );
            }
            vx_array_free(array);

            vx_expression_free(expr_age);
            vx_expression_free(expr_height);
            vx_expression_free(expr_sum);
            vx_expression_free(root);
        }
    }

    #[test]
    #[cfg_attr(miri, ignore)]
    fn test_filter() {
        unsafe {
            let root = vx_expression_root();
            let age_expr = vx_expression_get_item(vx_view::from_str("age"), root);
            let value = vx_scalar_new_u64(100, false);
            let mut error = ptr::null_mut();
            let lit_100 = vx_expression_literal(value, &raw mut error);
            assert_no_error(error);
            vx_scalar_free(value);
            let filter =
                vx_expression_binary(vx_binary_operator::VX_OPERATOR_GTE, age_expr, lit_100);

            let opts = vx_scan_options {
                filter,
                ..Default::default()
            };
            let (array, source) = scan(&raw const opts);
            let mut ctx = SESSION.create_execution_ctx();
            let expected = expected_rows(&source, &(100..200).collect::<Vec<_>>());
            assert_arrays_eq!(vx_array::as_ref(array), expected, &mut ctx);

            vx_array_free(array);
            vx_expression_free(filter);
            vx_expression_free(age_expr);
            vx_expression_free(lit_100);
            vx_expression_free(root);
        }
    }

    #[test]
    #[cfg_attr(miri, ignore)]
    fn test_filter_project() {
        unsafe {
            let root = vx_expression_root();
            let age_expr = vx_expression_get_item(vx_view::from_str("age"), root);
            let value = vx_scalar_new_u64(100, false);
            let mut error = ptr::null_mut();
            let lit_100 = vx_expression_literal(value, &raw mut error);
            assert_no_error(error);
            vx_scalar_free(value);
            let filter =
                vx_expression_binary(vx_binary_operator::VX_OPERATOR_GTE, age_expr, lit_100);
            let projection = vx_expression_get_item(vx_view::from_str("age"), root);

            let opts = vx_scan_options {
                projection,
                filter,
                ..Default::default()
            };
            let (array, _) = scan(&raw const opts);
            let mut ctx = SESSION.create_execution_ctx();
            let expected = PrimitiveArray::from_option_iter((100u64..200).map(Some));
            assert_arrays_eq!(vx_array::as_ref(array), expected, &mut ctx);

            vx_array_free(array);
            vx_expression_free(filter);
            vx_expression_free(age_expr);
            vx_expression_free(lit_100);
            vx_expression_free(projection);
            vx_expression_free(root);
        }
    }

    #[test]
    #[cfg_attr(miri, ignore)]
    fn test_row_range() {
        let opts = vx_scan_options {
            row_range_begin: 50,
            row_range_end: 100,
            ..Default::default()
        };
        let (array, source) = scan(&raw const opts);
        let mut ctx = SESSION.create_execution_ctx();
        let expected = expected_rows(&source, &(50..100).collect::<Vec<_>>());
        assert_arrays_eq!(vx_array::as_ref(array), expected, &mut ctx);
        unsafe { vx_array_free(array) };
    }

    #[test]
    #[cfg_attr(miri, ignore)]
    fn test_selection() {
        let indices = [0u64, 50, 100, 150, 199];
        let opts = vx_scan_options {
            selection: vx_scan_selection {
                idx: indices.as_ptr(),
                idx_len: indices.len(),
                include: vx_scan_selection_include::VX_SELECTION_INCLUDE_RANGE,
            },
            ..Default::default()
        };
        let (array, source) = scan(&raw const opts);
        let mut ctx = SESSION.create_execution_ctx();
        let expected = expected_rows(&source, &[0, 50, 100, 150, 199]);
        assert_arrays_eq!(vx_array::as_ref(array), expected, &mut ctx);
        unsafe { vx_array_free(array) };
    }

    #[test]
    #[cfg_attr(miri, ignore)]
    fn test_limit() {
        let opts = vx_scan_options {
            limit: 50,
            ..Default::default()
        };
        let (array, _) = scan(&raw const opts);
        assert_eq!(vx_array::as_ref(array).len(), 50);
        unsafe { vx_array_free(array) };
    }

    #[test]
    #[cfg_attr(miri, ignore)]
    fn test_ordered() {
        let mut ctx = SESSION.create_execution_ctx();
        let opts = vx_scan_options {
            ordered: true,
            ..Default::default()
        };
        let (array, struct_array) = scan(&raw const opts);
        assert_arrays_eq!(vx_array::as_ref(array), struct_array, &mut ctx);
        unsafe { vx_array_free(array) };
    }

    #[test]
    #[cfg_attr(miri, ignore)]
    fn test_row_count() {
        unsafe {
            let (session, _sample, _, ds) = sample_source();
            let scan = plan(ds, ptr::null());

            let mut error = ptr::null_mut();
            let mut count = vx_estimate::default();
            let result = vx_scan_row_count(scan, &raw mut count, &raw mut error);
            assert_no_error(error);
            assert_eq!(result, 0);
            assert_eq!(count.r#type, vx_estimate_type::VX_ESTIMATE_EXACT);
            assert_eq!(count.estimate, SAMPLE_ROWS as u64);

            vx_scan_free(scan);
            vx_data_source_free(ds);
            vx_session_free(session);
        }
    }

    #[test]
    #[cfg_attr(miri, ignore)]
    fn test_workers_share_a_pool() {
        unsafe {
            let (session, _sample, _, ds) = sample_source();
            let scan = plan(ds, ptr::null());
            let mut error = ptr::null_mut();
            let pool = vx_scan_open(scan, &raw mut error);
            assert_no_error(error);

            let first = vx_scan_pool_worker(pool, &raw mut error);
            assert_no_error(error);
            let second = vx_scan_pool_worker(pool, &raw mut error);
            assert_no_error(error);
            // Workers outlive the pool handle.
            vx_scan_pool_free(pool);

            let mut rows = ages(&drain(first));
            rows.extend(ages(&drain(second)));
            rows.sort_unstable();
            assert_eq!(rows, (0..SAMPLE_ROWS as u64).collect::<Vec<_>>());

            vx_data_source_free(ds);
            vx_session_free(session);
        }
    }

    #[test]
    #[cfg_attr(miri, ignore)]
    fn test_limit_is_shared_by_workers() {
        unsafe {
            let (session, _sample, _, ds) = sample_source();
            let opts = vx_scan_options {
                limit: 50,
                ..Default::default()
            };
            let scan = plan(ds, &raw const opts);
            let mut error = ptr::null_mut();
            let pool = vx_scan_open(scan, &raw mut error);
            assert_no_error(error);
            let first = vx_scan_pool_worker(pool, &raw mut error);
            assert_no_error(error);
            let second = vx_scan_pool_worker(pool, &raw mut error);
            assert_no_error(error);
            vx_scan_pool_free(pool);

            let rows = ages(&drain(first)).len() + ages(&drain(second)).len();
            assert_eq!(rows, 50);

            vx_data_source_free(ds);
            vx_session_free(session);
        }
    }

    #[test]
    #[cfg_attr(miri, ignore)]
    fn test_split_pieces_cover_the_scan_in_order() {
        unsafe {
            let (session, _sample, _, ds) = sample_source();
            let scan = plan(ds, ptr::null());
            let mut error = ptr::null_mut();
            let mut pieces = [ptr::null_mut::<vx_scan>(); 4];
            let count = vx_scan_split(scan, pieces.len(), pieces.as_mut_ptr(), &raw mut error);
            assert_no_error(error);
            assert!((1..=pieces.len()).contains(&count));

            let mut rows = Vec::new();
            for &piece in &pieces[..count] {
                let pool = vx_scan_open(piece, &raw mut error);
                assert_no_error(error);
                let worker = vx_scan_pool_worker(pool, &raw mut error);
                assert_no_error(error);
                vx_scan_pool_free(pool);
                let mut piece_rows = ages(&drain(worker));
                piece_rows.sort_unstable();
                rows.extend(piece_rows);
            }
            assert_eq!(rows, (0..SAMPLE_ROWS as u64).collect::<Vec<_>>());

            vx_data_source_free(ds);
            vx_session_free(session);
        }
    }

    #[test]
    #[cfg_attr(miri, ignore)]
    fn test_split_into_zero_pieces_fails() {
        unsafe {
            let (session, _sample, _, ds) = sample_source();
            let scan = plan(ds, ptr::null());
            let mut error = ptr::null_mut();
            let mut piece = ptr::null_mut::<vx_scan>();
            assert_eq!(vx_scan_split(scan, 0, &raw mut piece, &raw mut error), 0);
            assert_error(error);
            assert!(piece.is_null());

            vx_data_source_free(ds);
            vx_session_free(session);
        }
    }

    #[test]
    #[cfg_attr(miri, ignore)]
    fn test_execute_ordered_keeps_row_order() {
        unsafe {
            let (session, _sample, _, ds) = sample_source();
            let opts = vx_scan_options {
                ordered: true,
                ..Default::default()
            };
            let scan = plan(ds, &raw const opts);
            let mut error = ptr::null_mut();
            let stream = vx_scan_execute(scan, 4, &raw mut error);
            assert_no_error(error);
            assert_eq!(
                ages(&drain(stream)),
                (0..SAMPLE_ROWS as u64).collect::<Vec<_>>()
            );

            vx_data_source_free(ds);
            vx_session_free(session);
        }
    }
}
