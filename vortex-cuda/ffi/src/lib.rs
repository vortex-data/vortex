// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

#![deny(missing_docs)]

//! Native CUDA FFI helpers for cuDF interop.
//!
//! This crate keeps CUDA out of `vortex-ffi` and exports borrowed `vx_array` handles as the
//! `ArrowSchema + ArrowDeviceArray` pair that callers pass to cuDF's Arrow Device import APIs.

use std::os::raw::c_int;
use std::ptr;

use arrow_schema::ffi::FFI_ArrowSchema;
use vortex::array::ArrayRef;
use vortex::array::stream::ArrayStreamExt;
use vortex::compressor::CompressionSession;
use vortex::dtype::FieldName;
use vortex::dtype::FieldNames;
use vortex::error::VortexResult;
use vortex::error::vortex_ensure;
use vortex::error::vortex_err;
use vortex::expr::root;
use vortex::expr::select;
use vortex::file::OpenOptionsSessionExt;
use vortex::file::VortexFile;
use vortex::io::runtime::BlockingRuntime;
use vortex::layout::scan::scan_builder::ScanBuilder;
use vortex::layout::scan::split_by::SplitBy;
use vortex::session::SessionExt;
use vortex::session::VortexSession;
use vortex::utils::aliases::hash_set::HashSet;
use vortex_cuda::CudaExecutionCtx;
use vortex_cuda::CudaOpenOptionsExt;
use vortex_cuda::CudaSession;
use vortex_cuda::DictionaryExport;
use vortex_cuda::PooledFileReadAtOptions;
use vortex_cuda::arrow::ArrowDeviceArray;
use vortex_cuda::arrow::ArrowDeviceArrayStream;
use vortex_cuda::arrow::DeviceArrayExt;
use vortex_cuda::arrow::DeviceArrayStreamExt;
use vortex_cuda::layout::cuda_write_strategy;
use vortex_cuda::layout::register_cuda_layout;
use vortex_ffi::ffi_runtime;
use vortex_ffi::try_or;
use vortex_ffi::vx_array;
use vortex_ffi::vx_array_ref;
use vortex_ffi::vx_array_sink;
use vortex_ffi::vx_array_sink_open_file_with_strategy;
use vortex_ffi::vx_dtype;
use vortex_ffi::vx_error;
use vortex_ffi::vx_partition;
use vortex_ffi::vx_partition_into_array_stream;
use vortex_ffi::vx_session;
use vortex_ffi::vx_session_new_with;
use vortex_ffi::vx_session_ref;
use vortex_ffi::vx_view;

const VX_CUDA_OK: c_int = 0;
const VX_CUDA_ERR: c_int = 1;

/// Bypass the operating system page cache for pooled data-plane reads.
/// Footer and zone-map reads remain buffered. Supported only on Linux.
pub const VX_CUDA_SCAN_FLAG_DIRECT_IO: u32 = 1u32 << 0;

const VX_CUDA_SCAN_KNOWN_FLAGS: u32 = VX_CUDA_SCAN_FLAG_DIRECT_IO;

/// Options for scanning a CUDA-compatible Vortex file.
///
/// Zero-initialize this struct to use buffered file I/O and layout-derived batch splitting.
#[repr(C)]
#[derive(Default)]
pub struct vx_cuda_scan_options {
    /// A bitwise combination of `VX_CUDA_SCAN_FLAG_*` values. Unknown bits are rejected.
    pub flags: u32,
    /// Rows per batch, except for a possibly smaller final batch. Zero preserves layout boundaries.
    /// Nonzero counts ignore layout boundaries and may require unsupported CUDA `Chunked`
    /// concatenation.
    pub batch_rows: usize,
}

fn session_with_cuda(session: &VortexSession) -> &VortexSession {
    session.get::<CudaSession>();
    register_cuda_layout(session);
    session
}

/// Create a CUDA Vortex session.
///
/// Returns an owned handle with reusable CUDA state, or null and an optional `vx_error` on failure.
///
/// # Safety
///
/// If `error_out` is non-null, it must be valid for writing one error pointer.
#[unsafe(no_mangle)]
pub unsafe extern "C-unwind" fn vx_cuda_session_new(
    error_out: *mut *mut vx_error,
) -> *mut vx_session {
    try_or(error_out, ptr::null_mut(), || {
        let cuda_session = CudaSession::try_default()?;
        Ok(vx_session_new_with(|session| {
            let session = session.with_some(cuda_session);
            register_cuda_layout(&session);
            session.register(CompressionSession::cuda());
            session
        }))
    })
}

/// Open a Vortex file sink configured to produce CUDA-readable files.
///
/// Push host arrays and close/abort with `vx_array_sink_*`. Only on-disk encodings and layouts
/// change; writing does not move arrays to the GPU. The sink compresses with the session's
/// schemes, so pass a session from `vx_cuda_session_new` to use only those the GPU decodes.
///
/// # Safety
///
/// `session`, `path`, and `dtype` follow `vx_array_sink_open_file`'s requirements.
/// Non-null `error_out` must be writable for one error pointer.
#[unsafe(no_mangle)]
pub unsafe extern "C-unwind" fn vx_cuda_array_sink_open_file(
    session: *const vx_session,
    path: vx_view,
    dtype: *const vx_dtype,
    error_out: *mut *mut vx_error,
) -> *mut vx_array_sink {
    // SAFETY: The forwarded pointers satisfy the same requirements as this wrapper.
    unsafe { vx_cuda_array_sink_open_file_block_rows(session, path, dtype, 0, error_out) }
}

/// Open a CUDA-readable Vortex file sink with a fixed row block size.
///
/// Zero `block_rows` uses default writer sizing. Nonzero values disable byte-size coalescing and
/// outer layout dictionaries, but retain per-block dictionary compression.
/// Write sizing is independent of scan `batch_rows`; see `vx_cuda_scan_options`.
///
/// # Safety
///
/// Same requirements as `vx_cuda_array_sink_open_file`.
#[unsafe(no_mangle)]
pub unsafe extern "C-unwind" fn vx_cuda_array_sink_open_file_block_rows(
    session: *const vx_session,
    path: vx_view,
    dtype: *const vx_dtype,
    block_rows: usize,
    error_out: *mut *mut vx_error,
) -> *mut vx_array_sink {
    try_or(error_out, ptr::null_mut(), || {
        // SAFETY: The caller supplies a live borrowed session handle.
        let vortex_session = session_with_cuda(unsafe { vx_session_ref(session) }?);
        // SAFETY: All borrowed inputs satisfy the underlying sink's requirements.
        unsafe {
            vx_array_sink_open_file_with_strategy(
                session,
                path,
                dtype,
                cuda_write_strategy(vortex_session, block_rows),
            )
        }
    })
}

/// Scan a local CUDA-readable file with buffered I/O into an Arrow C Device stream.
/// Dictionaries export as plain values. Returns `0` on success, `1` on error.
/// Release the stream and batches via Arrow callbacks; free errors with `vx_error_free`.
///
/// # Safety
///
/// `session` must be a live borrowed `vortex-ffi` handle; `path` must contain readable UTF-8
/// for this call. `out_stream` and non-null `error_out` must point to writable output storage.
#[unsafe(no_mangle)]
pub unsafe extern "C-unwind" fn vx_cuda_scan_path_arrow_device_stream(
    session: *const vx_session,
    path: vx_view,
    out_stream: *mut ArrowDeviceArrayStream,
    error_out: *mut *mut vx_error,
) -> c_int {
    // SAFETY: The forwarded pointers satisfy this wrapper's requirements; null options is valid.
    unsafe {
        vx_cuda_scan_path_arrow_device_stream_with_options(
            session,
            path,
            ptr::null(),
            out_stream,
            error_out,
        )
    }
}

/// Like `vx_cuda_scan_path_arrow_device_stream`, with `batch_rows` as in `vx_cuda_scan_options`.
///
/// # Safety
///
/// Same requirements as `vx_cuda_scan_path_arrow_device_stream`.
#[unsafe(no_mangle)]
pub unsafe extern "C-unwind" fn vx_cuda_scan_path_arrow_device_stream_batch_rows(
    session: *const vx_session,
    path: vx_view,
    batch_rows: usize,
    out_stream: *mut ArrowDeviceArrayStream,
    error_out: *mut *mut vx_error,
) -> c_int {
    let options = vx_cuda_scan_options {
        batch_rows,
        ..Default::default()
    };
    // SAFETY: The caller supplies valid pointers; the local options remain live during the call.
    unsafe {
        vx_cuda_scan_path_arrow_device_stream_with_options(
            session,
            path,
            &raw const options,
            out_stream,
            error_out,
        )
    }
}

/// Like `vx_cuda_scan_path_arrow_device_stream`, with explicit scan options.
///
/// Null or zero-initialized `options` selects buffered I/O and layout-derived batch splitting.
///
/// # Safety
///
/// Same requirements as `vx_cuda_scan_path_arrow_device_stream`; non-null `options` must
/// point to a valid `vx_cuda_scan_options`.
#[unsafe(no_mangle)]
pub unsafe extern "C-unwind" fn vx_cuda_scan_path_arrow_device_stream_with_options(
    session: *const vx_session,
    path: vx_view,
    options: *const vx_cuda_scan_options,
    out_stream: *mut ArrowDeviceArrayStream,
    error_out: *mut *mut vx_error,
) -> c_int {
    // SAFETY: The caller supplies valid borrowed inputs and writable outputs.
    unsafe {
        vx_cuda_scan_path_arrow_device_stream_projected(
            session,
            path,
            options,
            ptr::null(),
            0,
            out_stream,
            error_out,
        )
    }
}

/// Scan a local Vortex file with ordered top-level column projection.
///
/// Otherwise follows `vx_cuda_scan_path_arrow_device_stream_with_options`.
/// Names are literal, case-sensitive, and unique; nonempty projections require a struct file.
/// Zero `ncolumns` ignores `columns` and selects all. Unknown names fail.
/// Skips unselected column I/O where the layout allows. Errors leave `out_stream` unchanged.
///
/// # Safety
///
/// Same requirements as `vx_cuda_scan_path_arrow_device_stream_with_options`.
/// For nonzero `ncolumns`, `columns` must reference that many valid, aligned `vx_view` values.
/// Name bytes must be readable UTF-8 for this call; null is allowed only for zero length.
#[unsafe(no_mangle)]
pub unsafe extern "C-unwind" fn vx_cuda_scan_path_arrow_device_stream_projected(
    session: *const vx_session,
    path: vx_view,
    options: *const vx_cuda_scan_options,
    columns: *const vx_view,
    ncolumns: usize,
    out_stream: *mut ArrowDeviceArrayStream,
    error_out: *mut *mut vx_error,
) -> c_int {
    try_or(error_out, VX_CUDA_ERR, || {
        vortex_ensure!(!out_stream.is_null(), "null ArrowDeviceArrayStream output");

        // SAFETY: The caller keeps options, column views and their bytes, path bytes, and the
        // borrowed session handle valid for this call.
        let (options, columns, path, session) = unsafe {
            (
                scan_options(options)?,
                scan_columns(columns, ncolumns)?,
                path.as_str()?,
                vx_session_ref(session)?,
            )
        };
        let session = session_with_cuda(session);
        let file = ffi_runtime().block_on(
            session
                .open_options()
                .with_cuda()
                .with_read_at_options(options.read_at_options)
                .open_path(path),
        )?;
        let scan = projected_scan(&file, columns, options.batch_rows)?;
        let array_stream = scan.into_array_stream()?.boxed();
        let ctx = scan_export_ctx(session)?;
        let device_stream = ArrowDeviceArrayStream::new(array_stream, ctx, ffi_runtime());

        // SAFETY: The output is non-null and the caller guarantees writable storage.
        unsafe { ptr::write(out_stream, device_stream) };
        Ok(VX_CUDA_OK)
    })
}

/// Copy ordered, literal field names, rejecting invalid views and duplicates. Zero selects all.
///
/// # Safety
///
/// Inputs passing null/alignment/size checks must reference `ncolumns` live views at `columns`,
/// with `len` readable bytes at each non-null name pointer.
unsafe fn scan_columns(columns: *const vx_view, ncolumns: usize) -> VortexResult<FieldNames> {
    if ncolumns == 0 {
        return Ok(FieldNames::default());
    }
    vortex_ensure!(
        !columns.is_null(),
        "null CUDA scan columns with nonzero count"
    );
    vortex_ensure!(columns.is_aligned(), "unaligned CUDA scan columns pointer");
    vortex_ensure!(
        ncolumns <= isize::MAX as usize / size_of::<vx_view>(),
        "CUDA scan column count is too large"
    );
    // SAFETY: Null, alignment, and size were checked; the caller guarantees readable views.
    let columns = unsafe { std::slice::from_raw_parts(columns, ncolumns) };
    let mut names = Vec::<FieldName>::with_capacity(ncolumns);
    let mut seen = HashSet::<&str>::with_capacity(ncolumns);
    for (index, column) in columns.iter().enumerate() {
        vortex_ensure!(
            column.len <= isize::MAX as usize,
            "CUDA scan column {index} name is too long"
        );
        // SAFETY: The caller guarantees readable name bytes. as_str checks null and UTF-8.
        let name = unsafe { column.as_str() }
            .map_err(|error| vortex_err!("invalid CUDA scan column {index}: {error}"))?;
        vortex_ensure!(seen.insert(name), "duplicate CUDA scan column: {name:?}");
        names.push(FieldName::from(name));
    }
    Ok(names.into())
}

fn projected_scan(
    file: &VortexFile,
    columns: FieldNames,
    batch_rows: usize,
) -> VortexResult<ScanBuilder<ArrayRef>> {
    let mut scan = file.scan()?;
    if !columns.is_empty() {
        let projection = select(columns, root()).optimize(file.dtype())?;
        scan = scan.with_projection(projection.bind(file.dtype())?);
    }
    let split_by = if batch_rows == 0 {
        SplitBy::Layout
    } else {
        SplitBy::RowCount(batch_rows)
    };
    Ok(scan.with_split_by(split_by))
}

struct CudaScanOptions {
    read_at_options: PooledFileReadAtOptions,
    batch_rows: usize,
}

/// Select plain Arrow export for one scan without mutating the shared session.
fn scan_export_ctx(session: &VortexSession) -> VortexResult<CudaExecutionCtx> {
    Ok(
        CudaSession::create_execution_ctx(session)?
            .with_dictionary_export(DictionaryExport::Decode),
    )
}

/// Parse scan settings; null selects defaults and unknown flags are rejected.
///
/// # Safety
///
/// Non-null `options` must point to an initialized, aligned `vx_cuda_scan_options`.
unsafe fn scan_options(options: *const vx_cuda_scan_options) -> VortexResult<CudaScanOptions> {
    let defaults = vx_cuda_scan_options::default();
    // SAFETY: The caller guarantees that a non-null options pointer is valid for this call.
    let options = unsafe { options.as_ref() }.unwrap_or(&defaults);
    vortex_ensure!(
        options.flags & !VX_CUDA_SCAN_KNOWN_FLAGS == 0,
        "unsupported CUDA scan option flags: {:#x}",
        options.flags & !VX_CUDA_SCAN_KNOWN_FLAGS
    );
    let read_at_options = PooledFileReadAtOptions::default();
    let read_at_options = if options.flags & VX_CUDA_SCAN_FLAG_DIRECT_IO == 0 {
        read_at_options
    } else {
        #[cfg(target_os = "linux")]
        {
            read_at_options.with_direct_io()
        }
        #[cfg(not(target_os = "linux"))]
        {
            return Err(vortex::error::vortex_err!(
                "direct CUDA file I/O is only supported on Linux"
            ));
        }
    };

    Ok(CudaScanOptions {
        read_at_options,
        batch_rows: options.batch_rows,
    })
}

/// Export a borrowed Vortex array for cuDF's Arrow Device import path.
///
/// Returns `0` with independently owned `out_schema` and `out_array`. Pass them to cuDF, then
/// release both via their Arrow callbacks after import. Returns `1` on error, writing a `vx_error`
/// if `error_out` is non-null; free it with `vx_error_free`.
///
/// `out_array` is exported on `ARROW_DEVICE_CUDA`; struct arrays become table-shaped schemas,
/// non-struct arrays a single column field.
///
/// Export is stream-ordered; `out_array->sync_event` is valid until `out_array` is released.
///
/// # Safety
///
/// `session` and `array` must be valid borrowed handles created by `vortex-ffi`. `out_schema`
/// and `out_array` must be valid writable pointers. If `error_out` is non-null, it must be valid
/// for writing one error pointer.
#[unsafe(no_mangle)]
pub unsafe extern "C-unwind" fn vx_cuda_array_export_arrow_device(
    session: *const vx_session,
    array: *const vx_array,
    out_schema: *mut FFI_ArrowSchema,
    out_array: *mut ArrowDeviceArray,
    error_out: *mut *mut vx_error,
) -> c_int {
    try_or(error_out, VX_CUDA_ERR, || {
        vortex_ensure!(!out_schema.is_null(), "null ArrowSchema output");
        vortex_ensure!(!out_array.is_null(), "null ArrowDeviceArray output");

        // SAFETY: The caller supplies a live borrowed session handle.
        let session = session_with_cuda(unsafe { vx_session_ref(session) }?);
        // SAFETY: The caller supplies a live borrowed array handle.
        let array = unsafe { vx_array_ref(array) }?.clone();
        let mut ctx = CudaSession::create_execution_ctx(session)?;
        let exported =
            futures::executor::block_on(array.export_device_array_with_schema(&mut ctx))?;

        // SAFETY: Both outputs are non-null and the caller guarantees writable storage.
        unsafe {
            ptr::write(out_schema, exported.schema);
            ptr::write(out_array, exported.array);
        }
        Ok(VX_CUDA_OK)
    })
}

/// Scan a Vortex partition as an Arrow C Device stream.
///
/// Consumes `partition`, even on error. Return codes and output ownership follow
/// `vx_cuda_scan_path_arrow_device_stream`.
///
/// # Safety
///
/// `session` must be a valid borrowed handle created by `vortex-ffi`. `partition` must be an owned
/// partition handle created by `vortex-ffi`. `out_stream` must be a valid writable pointer. If
/// `error_out` is non-null, it must be valid for writing one error pointer.
#[unsafe(no_mangle)]
pub unsafe extern "C-unwind" fn vx_cuda_partition_scan_arrow_device_stream(
    session: *const vx_session,
    partition: *mut vx_partition,
    out_stream: *mut ArrowDeviceArrayStream,
    error_out: *mut *mut vx_error,
) -> c_int {
    try_or(error_out, VX_CUDA_ERR, || {
        vortex_ensure!(!partition.is_null(), "null vx_partition");

        // SAFETY: The caller transfers ownership of this non-null partition handle.
        let array_stream = unsafe { vx_partition_into_array_stream(partition) }?;
        vortex_ensure!(!out_stream.is_null(), "null ArrowDeviceArrayStream output");

        // SAFETY: The caller supplies a live borrowed session handle.
        let session = session_with_cuda(unsafe { vx_session_ref(session) }?);
        // Drive the stream on the same runtime the partition's scan spawned its work onto.
        let device_stream = array_stream.export_device_array_stream(session, ffi_runtime())?;

        // SAFETY: The output is non-null and the caller guarantees writable storage.
        unsafe { ptr::write(out_stream, device_stream) };
        Ok(VX_CUDA_OK)
    })
}

#[cfg(test)]
mod tests {
    use std::ffi::CStr;
    use std::io::Write;
    use std::mem::MaybeUninit;
    use std::ptr;
    use std::sync::Arc;
    use std::sync::atomic::AtomicUsize;
    use std::sync::atomic::Ordering;

    use arrow_schema::DataType;
    use arrow_schema::Field;
    use arrow_schema::Schema;
    use cudarc::driver::CudaContext;
    use cudarc::driver::result;
    use cudarc::driver::sys::CUevent;
    use futures::TryStreamExt;
    use tempfile::NamedTempFile;
    use vortex::VortexSessionDefault;
    use vortex::array::ArrayRef;
    use vortex::array::IntoArray;
    use vortex::array::VortexSessionExecute;
    use vortex::array::arrays::ChunkedArray;
    use vortex::array::arrays::DictArray;
    use vortex::array::arrays::PrimitiveArray;
    use vortex::array::arrays::StructArray;
    use vortex::array::assert_arrays_eq;
    use vortex::array::memory::BufferAllocatorRef;
    use vortex::array::memory::MemorySessionExt;
    use vortex::array::memory::StaticBufferAllocator;
    use vortex::array::validity::Validity;
    use vortex::buffer::BitBuffer;
    use vortex::buffer::Buffer;
    use vortex::buffer::ByteBuffer;
    use vortex::buffer::ByteBufferMut;
    use vortex::dtype::NativePType;
    use vortex::dtype::Nullability;
    use vortex::error::VortexResult;
    use vortex::file::WriteOptionsSessionExt;
    use vortex::io::session::RuntimeSessionExt;
    use vortex::layout::LayoutStrategy;
    use vortex::layout::layouts::flat::writer::FlatLayoutStrategy;
    use vortex::layout::layouts::table::TableStrategy;
    use vortex::layout::segments::SegmentFuture;
    use vortex::layout::segments::SegmentId;
    use vortex::layout::segments::SegmentSource;
    use vortex_cuda::arrow::ARROW_DEVICE_CUDA;
    use vortex_cuda::arrow::ArrowArray;
    use vortex_cuda::arrow::release_device_array;
    use vortex_cuda::arrow::release_schema;
    use vortex_cuda_macros::cuda_not_available;
    use vortex_cuda_macros::test as cuda_test;
    use vortex_ffi::vx_array_free as free_test_array;
    use vortex_ffi::vx_error_free;
    use vortex_ffi::vx_error_message;
    use vortex_ffi::vx_session_free as free_test_session;

    use super::*;

    fn test_session(session: VortexSession) -> *mut vx_session {
        Box::into_raw(Box::new(session)).cast::<vx_session>()
    }

    fn test_array(array: impl IntoArray) -> *const vx_array {
        Box::into_raw(Box::new(array.into_array())).cast::<vx_array>()
    }

    fn stream_error(stream: &mut ArrowDeviceArrayStream) -> String {
        // SAFETY: The callback and returned C string belong to this live stream.
        unsafe {
            stream
                .get_last_error
                .and_then(|callback| callback(stream).as_ref())
                .map(|message| CStr::from_ptr(message).to_string_lossy().into_owned())
                .unwrap_or_default()
        }
    }

    fn stream_schema(stream: &mut ArrowDeviceArrayStream) -> FFI_ArrowSchema {
        let mut schema = FFI_ArrowSchema::empty();
        let get_schema = stream.get_schema.expect("missing get_schema");
        // SAFETY: This live stream owns the callback; schema is writable.
        assert_eq!(
            unsafe { get_schema(stream, (&raw mut schema).cast()) },
            0,
            "{}",
            stream_error(stream)
        );
        schema
    }

    /// # Safety
    /// `session` and `array` must be valid borrowed FFI handles for the duration of the call.
    unsafe fn export_array(
        session: *const vx_session,
        array: *const vx_array,
    ) -> (FFI_ArrowSchema, ArrowDeviceArray) {
        let mut error = ptr::null_mut();
        let mut schema = FFI_ArrowSchema::empty();
        let mut device_array = ArrowDeviceArray::empty();
        // SAFETY: The caller guarantees valid handles; all outputs are live and writable.
        let status = unsafe {
            vx_cuda_array_export_arrow_device(
                session,
                array,
                &raw mut schema,
                &raw mut device_array,
                &raw mut error,
            )
        };
        assert_eq!(status, VX_CUDA_OK);
        assert!(error.is_null());
        (schema, device_array)
    }

    fn view(value: &str) -> vx_view {
        vx_view {
            ptr: value.as_ptr().cast(),
            len: value.len(),
        }
    }

    fn names(values: &[&str]) -> VortexResult<FieldNames> {
        let views: Vec<_> = values.iter().map(|name| view(name)).collect();
        // SAFETY: All views and their string bytes remain live throughout parsing.
        unsafe { scan_columns(views.as_ptr(), views.len()) }
    }

    fn assert_error<T>(result: VortexResult<T>, message: &str) {
        let error = result.err().expect("expected an error");
        assert!(error.to_string().contains(message), "{error}");
    }

    fn session() -> VortexSession {
        VortexSession::default().with_handle(ffi_runtime().handle())
    }

    fn table() -> VortexResult<StructArray> {
        StructArray::try_new(
            ["ids", "unused", "値.x"].into(),
            vec![
                PrimitiveArray::from_iter(0u32..5).into_array(),
                PrimitiveArray::from_iter([1.0f64, 2.0, 3.0, 4.0, 5.0]).into_array(),
                PrimitiveArray::from_option_iter([Some(10i64), None, Some(30), None, Some(50)])
                    .into_array(),
            ],
            5,
            Validity::NonNullable,
        )
    }

    fn file_bytes(
        session: &VortexSession,
        array: ArrayRef,
        strategy: Arc<dyn LayoutStrategy>,
    ) -> VortexResult<ByteBuffer> {
        let mut bytes = ByteBufferMut::empty();
        ffi_runtime().block_on(
            session
                .write_options()
                .with_strategy(strategy)
                .write(&mut bytes, array.to_array_stream()),
        )?;
        Ok(bytes.freeze())
    }

    fn open_file(
        session: &VortexSession,
        array: ArrayRef,
        strategy: Arc<dyn LayoutStrategy>,
    ) -> VortexResult<VortexFile> {
        session
            .open_options()
            .open_buffer(file_bytes(session, array, strategy)?)
    }

    fn flat_ids_file(session: &VortexSession, rows: u32) -> VortexResult<VortexFile> {
        let ids = PrimitiveArray::from_iter(0..rows).into_array();
        let rows = ids.len();
        let input = StructArray::try_new(["ids"].into(), vec![ids], rows, Validity::NonNullable)?
            .into_array();
        open_file(session, input, Arc::new(FlatLayoutStrategy::default()))
    }

    struct RejectSegments {
        inner: Arc<dyn SegmentSource>,
        forbidden: Vec<SegmentId>,
        rejected: AtomicUsize,
    }

    impl SegmentSource for RejectSegments {
        fn request(&self, id: SegmentId) -> SegmentFuture {
            if self.forbidden.contains(&id) {
                self.rejected.fetch_add(1, Ordering::Relaxed);
                return Box::pin(async move {
                    Err(vortex_err!("unselected column segment requested: {id}"))
                });
            }
            self.inner.request(id)
        }
    }

    /// # Safety
    /// `error` must be null or an owned FFI error, consumed by this call.
    unsafe fn take_error_message(error: *mut vx_error) -> Option<String> {
        if error.is_null() {
            return None;
        }
        // SAFETY: The error remains live while its message is copied, then is freed exactly once.
        let message = unsafe { vx_error_message(error).as_str() }
            .map(str::to_owned)
            .unwrap_or_else(|error| error.to_string());
        unsafe { vx_error_free(error) };
        Some(message)
    }

    struct OwnedDeviceStream(ArrowDeviceArrayStream);

    impl Drop for OwnedDeviceStream {
        fn drop(&mut self) {
            if let Some(release) = self.0.release {
                // SAFETY: This owner holds the live stream and releases it exactly once.
                unsafe { release(&raw mut self.0) };
            }
        }
    }

    fn open_stream(
        session: &VortexSession,
        path: &str,
        options: &vx_cuda_scan_options,
        columns: &[&str],
    ) -> OwnedDeviceStream {
        let mut output = MaybeUninit::<ArrowDeviceArrayStream>::uninit();
        let mut error = ptr::null_mut();
        let handle = test_session(session.clone());
        let columns: Vec<_> = columns.iter().map(|name| view(name)).collect();
        // SAFETY: All borrowed inputs and writable outputs are live for this call.
        let status = unsafe {
            vx_cuda_scan_path_arrow_device_stream_projected(
                handle,
                view(path),
                options,
                columns.as_ptr(),
                columns.len(),
                output.as_mut_ptr(),
                &raw mut error,
            )
        };
        // SAFETY: This call owns both the session handle and any returned error.
        let message = unsafe {
            free_test_session(handle);
            take_error_message(error)
        };
        assert_eq!(
            status,
            VX_CUDA_OK,
            "{}",
            message.as_deref().unwrap_or("no FFI error")
        );
        // SAFETY: A successful call initialized the stream, which owns its session state.
        let stream = OwnedDeviceStream(unsafe { output.assume_init() });
        assert!(message.is_none(), "unexpected FFI error: {message:?}");
        assert!(stream.0.release.is_some(), "missing release");
        stream
    }

    /// # Safety
    /// `array` must be a live Arrow primitive array of `T` on the current CUDA context.
    /// Its producer's sync event must have completed before this call.
    unsafe fn read_primitive<T: NativePType>(
        array: &ArrowArray,
        nullability: Nullability,
    ) -> VortexResult<ArrayRef> {
        assert!(array.release.is_some());
        assert!(array.dictionary.is_null());
        assert_eq!(array.n_buffers, 2);
        assert_eq!(array.n_children, 0);
        assert!(!array.buffers.is_null());
        let len = usize::try_from(array.length)?;
        let offset = usize::try_from(array.offset)?;
        // SAFETY: The live primitive array owns two buffer pointers.
        let buffers = unsafe { std::slice::from_raw_parts(array.buffers, 2) };
        let mut values = vec![T::default(); offset + len];
        // SAFETY: The synchronized data buffer contains offset + len values of T and remains live.
        unsafe { result::memcpy_dtoh_sync(&mut values, buffers[1] as u64) }
            .map_err(|error| vortex_err!("copying Arrow values: {error}"))?;

        let validity = if buffers[0].is_null() {
            assert_eq!(array.null_count, 0);
            match nullability {
                Nullability::NonNullable => Validity::NonNullable,
                Nullability::Nullable => Validity::AllValid,
            }
        } else {
            let mut bytes = vec![0u8; (offset + len).div_ceil(8)];
            // SAFETY: The synchronized bitmap covers offset + len bits and remains live.
            unsafe { result::memcpy_dtoh_sync(&mut bytes, buffers[0] as u64) }
                .map_err(|error| vortex_err!("copying Arrow validity: {error}"))?;
            let bits = BitBuffer::new_with_offset(ByteBuffer::from(bytes), len, offset);
            assert_eq!(array.null_count, i64::try_from(len - bits.true_count())?);
            match nullability {
                Nullability::NonNullable => {
                    assert_eq!(array.null_count, 0);
                    Validity::NonNullable
                }
                Nullability::Nullable => Validity::from(bits),
            }
        };
        Ok(PrimitiveArray::new(Buffer::from(values).slice(offset..), validity).into_array())
    }

    /// Read the fixture's projected Int64/UInt32 columns through the public Arrow Device ABI.
    ///
    /// # Safety
    /// `array` must be a live batch from the fixture's projected stream, with its schema checked.
    unsafe fn read_projected_batch(array: &ArrowDeviceArray) -> VortexResult<ArrayRef> {
        assert_eq!(array.device_type, ARROW_DEVICE_CUDA);
        let context = CudaContext::new(usize::try_from(array.device_id)?)
            .map_err(|error| vortex_err!("opening Arrow device context: {error}"))?;
        context
            .bind_to_thread()
            .map_err(|error| vortex_err!("binding Arrow device context: {error}"))?;
        if !array.sync_event.is_null() {
            // SAFETY: Arrow's CUDA sync_event points to a live event handle owned by this batch.
            unsafe { result::event::synchronize(*array.sync_event.cast::<CUevent>()) }
                .map_err(|error| vortex_err!("waiting for Arrow device batch: {error}"))?;
        }
        let array = &array.array;
        assert!(array.dictionary.is_null());
        assert_eq!(array.offset, 0);
        assert_eq!(array.null_count, 0);
        assert_eq!(array.n_children, 2);
        assert!(!array.children.is_null());
        // SAFETY: The live struct owns two children matching the previously checked schema.
        let fields = unsafe {
            let values = (*array.children).as_ref().expect("missing values child");
            let ids = (*array.children.add(1))
                .as_ref()
                .expect("missing ids child");
            assert_eq!(values.length, array.length);
            assert_eq!(ids.length, array.length);
            vec![
                read_primitive::<i64>(values, Nullability::Nullable)?,
                read_primitive::<u32>(ids, Nullability::NonNullable)?,
            ]
        };
        Ok(StructArray::try_new(
            ["値.x", "ids"].into(),
            fields,
            usize::try_from(array.length)?,
            Validity::NonNullable,
        )?
        .into_array())
    }

    fn read_projected_batches(stream: &mut ArrowDeviceArrayStream) -> VortexResult<Vec<ArrayRef>> {
        let get_next = stream.get_next.expect("missing get_next");
        let mut batches = Vec::new();
        loop {
            let mut array = ArrowDeviceArray::empty();
            // SAFETY: This live stream owns the callback; array is writable.
            let status = unsafe { get_next(stream, &raw mut array) };
            vortex_ensure!(status == 0, "get_next failed: {}", stream_error(stream));
            if array.array.release.is_none() {
                break;
            }
            // SAFETY: The fixture's schema was checked, and this batch remains live during readback.
            let batch = unsafe { read_projected_batch(&array) };
            release_device_array(&mut array);
            batches.push(batch?);
        }
        Ok(batches)
    }

    #[test]
    fn test_projection_names_are_owned_and_zero_count_means_all() -> VortexResult<()> {
        let parsed = {
            let name = String::from("値.x");
            names(&[&name, "ids", ""])?
        };
        assert_eq!(parsed, ["値.x", "ids", ""]);
        // SAFETY: Zero count ignores the pointer, including a null pointer.
        assert!(unsafe { scan_columns(ptr::null(), 0) }?.is_empty());
        let empty = vx_view {
            ptr: ptr::null(),
            len: 0,
        };
        // SAFETY: A null, zero-length view is a valid empty name.
        assert_eq!(unsafe { scan_columns(&raw const empty, 1) }?, [""]);
        Ok(())
    }

    #[test]
    fn test_projection_rejects_invalid_names_and_counts() {
        let invalid_utf8 = vx_view {
            ptr: [0xffu8].as_ptr().cast(),
            len: 1,
        };
        let null_name = vx_view {
            ptr: ptr::null(),
            len: 1,
        };
        let long_name = vx_view {
            ptr: "x".as_ptr().cast(),
            len: usize::MAX,
        };
        let duplicate = String::from("x");
        let aligned = [view("x"), view(&duplicate)];
        let misaligned = aligned.as_ptr().cast::<u8>().wrapping_add(1).cast();
        for (columns, count, message) in [
            (ptr::null(), 1, "null CUDA scan columns"),
            (aligned.as_ptr(), usize::MAX, "column count is too large"),
            (misaligned, 1, "unaligned CUDA scan columns"),
            (&raw const invalid_utf8, 1, "invalid utf-8"),
            (&raw const null_name, 1, "null vx_view pointer"),
            (&raw const long_name, 1, "name is too long"),
            (aligned.as_ptr(), 2, "duplicate CUDA scan column: \"x\""),
        ] {
            // SAFETY: Invalid pointer/length combinations must be rejected before dereferencing;
            // all remaining views and bytes are live.
            assert_error(unsafe { scan_columns(columns, count) }, message);
        }
    }

    #[test]
    fn test_projected_scan_zero_batch_rows_preserves_large_layout_span() -> VortexResult<()> {
        let session = session();
        // Exceed the default scan split cap to catch accidentally leaving it enabled.
        let file = flat_ids_file(&session, 1_000_000)?;
        for columns in [names(&[])?, names(&["ids"])?] {
            let splits = projected_scan(&file, columns, 0)?.full_file_splits()?;
            assert_eq!(splits, [0, 1_000_000]);
        }
        Ok(())
    }

    #[test]
    fn test_projected_scan_exact_batch_rows_with_final_tail() -> VortexResult<()> {
        let session = session();
        let file = flat_ids_file(&session, 10)?;
        for columns in [names(&[])?, names(&["ids"])?] {
            let batches: Vec<ArrayRef> = ffi_runtime().block_on(
                projected_scan(&file, columns, 3)?
                    .into_array_stream()?
                    .try_collect(),
            )?;
            let lengths: Vec<_> = batches.iter().map(|batch| batch.len()).collect();
            assert_eq!(lengths, [3, 3, 3, 1]);
        }
        Ok(())
    }

    #[test]
    fn test_projection_cpu_never_requests_unselected_column_segments() -> VortexResult<()> {
        let session = session();
        let input = table()?;
        let columns = names(&["値.x", "ids"])?;
        let expected = input.project(columns.as_ref())?.into_array();
        let flat: Arc<dyn LayoutStrategy> = Arc::new(FlatLayoutStrategy::default());
        let strategy = Arc::new(TableStrategy::new(Arc::clone(&flat), flat));
        let file = open_file(&session, input.into_array(), strategy)?;
        // TableStrategy writes one flat child per column, so child 1 is exactly the unused column.
        let children = file.footer().layout().children()?;
        let forbidden = children[1].segment_ids();
        assert!(!forbidden.is_empty());
        let source = Arc::new(RejectSegments {
            inner: file.segment_source(),
            forbidden,
            rejected: AtomicUsize::new(0),
        });
        let file = file.with_segment_source(Arc::<RejectSegments>::clone(&source));
        let actual = ffi_runtime().block_on(
            projected_scan(&file, columns, 2)?
                .into_array_stream()?
                .read_all(),
        )?;
        assert_arrays_eq!(actual, expected, &mut session.create_execution_ctx());
        assert_eq!(source.rejected.load(Ordering::Relaxed), 0);
        // The same reader must fail without projection, proving the guard actually observes reads.
        assert_error(
            ffi_runtime().block_on(
                projected_scan(&file, names(&[])?, 0)?
                    .into_array_stream()?
                    .read_all(),
            ),
            "unselected column segment requested",
        );
        assert!(source.rejected.load(Ordering::Relaxed) > 0);
        Ok(())
    }

    #[test]
    fn test_projection_ffi_validation_without_cuda() {
        let mut stream = ArrowDeviceArrayStream {
            device_type: -1,
            get_schema: None,
            get_next: None,
            get_last_error: None,
            release: None,
            private_data: ptr::null_mut(),
        };
        let mut error = ptr::null_mut();
        // SAFETY: Output pointers are writable; invalid columns must fail before session/path use.
        let status = unsafe {
            vx_cuda_scan_path_arrow_device_stream_projected(
                ptr::null(),
                view(""),
                ptr::null(),
                ptr::null(),
                1,
                &raw mut stream,
                &raw mut error,
            )
        };
        // SAFETY: This call owns the returned error and frees it exactly once.
        let message = unsafe { take_error_message(error) }.expect("missing FFI error");
        assert_eq!(status, VX_CUDA_ERR, "{message}");
        assert!(
            message.contains("null CUDA scan columns with nonzero count"),
            "{message}"
        );
        assert_eq!(stream.device_type, -1);
        assert!(stream.release.is_none());
        error = ptr::null_mut();
        // SAFETY: Null output is rejected before any other input is used; error is writable.
        assert_eq!(
            unsafe {
                vx_cuda_scan_path_arrow_device_stream_projected(
                    ptr::null(),
                    view(""),
                    ptr::null(),
                    ptr::null(),
                    0,
                    ptr::null_mut(),
                    &raw mut error,
                )
            },
            VX_CUDA_ERR
        );
        // SAFETY: This call owns the returned error and frees it exactly once.
        let message = unsafe { take_error_message(error) }.expect("missing FFI error");
        assert!(
            message.contains("null ArrowDeviceArrayStream output"),
            "{message}"
        );
    }

    #[cuda_test]
    fn test_projection_gpu_values_and_validity() -> VortexResult<()> {
        let session = session().with_some(CudaSession::try_default()?);
        register_cuda_layout(&session);
        session.register(CompressionSession::cuda());
        let input = table()?;
        let columns = ["値.x", "ids"];
        let expected = input.project(names(&columns)?.as_ref())?.into_array();
        let mut file = NamedTempFile::new()?;
        file.write_all(&file_bytes(
            &session,
            input.into_array(),
            cuda_write_strategy(&session, 5),
        )?)?;
        let path = file
            .path()
            .to_str()
            .ok_or_else(|| vortex_err!("non-UTF-8 test path"))?;
        let options = vx_cuda_scan_options {
            batch_rows: 2,
            ..Default::default()
        };
        let mut stream = open_stream(&session, path, &options, &columns);
        // The stream must retain its session state after the caller releases its session.
        drop(session);
        let schema = stream_schema(&mut stream.0);
        let expected_fields = vec![
            Field::new("値.x", DataType::Int64, true),
            Field::new("ids", DataType::UInt32, false),
        ];
        assert_eq!(Schema::try_from(&schema)?, Schema::new(expected_fields));
        let batches = read_projected_batches(&mut stream.0)?;
        assert_eq!(
            batches.iter().map(|batch| batch.len()).collect::<Vec<_>>(),
            [2, 2, 1]
        );
        let actual = ChunkedArray::try_new(batches, expected.dtype().clone())?.into_array();
        assert_arrays_eq!(
            actual,
            expected,
            &mut VortexSession::default().create_execution_ctx()
        );
        Ok(())
    }

    #[test]
    fn test_scan_options_default_to_buffered_io() -> VortexResult<()> {
        let options = vx_cuda_scan_options::default();

        for pointer in [ptr::null(), &raw const options] {
            // SAFETY: Each pointer is either null or points to the live options above.
            let parsed = unsafe { scan_options(pointer) }?;
            assert_eq!(parsed.read_at_options, PooledFileReadAtOptions::default());
            assert_eq!(parsed.batch_rows, 0);
        }
        Ok(())
    }

    #[test]
    fn test_maps_scan_options() -> VortexResult<()> {
        let buffered = PooledFileReadAtOptions::default();
        for (flags, batch_rows, read_at_options) in [
            (0, 8192, buffered),
            #[cfg(target_os = "linux")]
            (VX_CUDA_SCAN_FLAG_DIRECT_IO, 0, buffered.with_direct_io()),
        ] {
            let options = vx_cuda_scan_options { flags, batch_rows };
            // SAFETY: options lives for the duration of parsing.
            let parsed = unsafe { scan_options(&raw const options) }?;
            assert_eq!(parsed.read_at_options, read_at_options, "flags={flags}");
            assert_eq!(parsed.batch_rows, batch_rows, "flags={flags}");
        }
        Ok(())
    }

    #[test]
    fn test_scan_options_reject_unknown_flags() {
        for flags in [1 << 1, VX_CUDA_SCAN_FLAG_DIRECT_IO | (1 << 1)] {
            let options = vx_cuda_scan_options {
                flags,
                batch_rows: 0,
            };
            // SAFETY: options remains live throughout parsing.
            let error = unsafe { scan_options(&raw const options) }
                .err()
                .expect("unknown flags must be rejected");
            assert!(
                error
                    .to_string()
                    .contains("unsupported CUDA scan option flags"),
                "{error}"
            );
        }
    }

    #[cuda_test]
    fn test_scan_context_preserves_session_resources_and_policy() -> VortexResult<()> {
        // A distinct allocator detects accidental reconstruction of a default session.
        let allocator = BufferAllocatorRef::new(StaticBufferAllocator);
        let session = VortexSession::default()
            .with_some(CudaSession::try_default()?)
            .with_allocator(allocator.clone());
        let mut ctx = scan_export_ctx(session_with_cuda(&session))?;

        assert_eq!(
            session.get::<CudaSession>().dictionary_export(),
            DictionaryExport::Preserve
        );
        assert!(ctx.execution_ctx().allocator().ptr_eq(&allocator));
        let export_session = ctx.execution_ctx().session();
        assert!(Arc::ptr_eq(
            session.get::<CudaSession>().pinned_buffer_pool(),
            export_session.get::<CudaSession>().pinned_buffer_pool(),
        ));
        let array = DictArray::try_new(
            PrimitiveArray::from_iter([1u8, 0, 1]).into_array(),
            PrimitiveArray::from_iter([10i32, 20]).into_array(),
        )?
        .into_array();
        let mut exported =
            ffi_runtime().block_on(array.export_device_array_with_schema(&mut ctx))?;
        release_device_array(&mut exported.array);
        assert_eq!(
            Field::try_from(&exported.schema)?.data_type(),
            &DataType::Int32
        );
        Ok(())
    }

    #[cuda_test]
    fn test_export_primitive_arrow_device() {
        let session = test_session(VortexSession::default());
        let array = test_array(PrimitiveArray::from_iter(0u32..5));
        // SAFETY: Both handles remain live until cleanup below.
        let (mut schema, mut device_array) = unsafe { export_array(session, array) };

        let field = Field::try_from(&schema).expect("schema should be a field");
        assert_eq!(field.name(), "");
        assert_eq!(device_array.array.length, 5);
        assert_eq!(device_array.array.n_buffers, 2);
        assert_eq!(device_array.device_type, ARROW_DEVICE_CUDA);
        assert_eq!(device_array.reserved, [0; 3]);
        assert!(device_array.array.release.is_some());

        unsafe {
            release_device_array(&mut device_array);
            release_schema(&mut schema);
            free_test_array(array);
            free_test_session(session);
        }
    }

    #[cuda_test]
    fn test_export_struct_arrow_device_table() -> VortexResult<()> {
        let session = test_session(VortexSession::default());
        let array = test_array(StructArray::try_new(
            ["ids", "values"].into(),
            vec![
                PrimitiveArray::from_iter(0u32..3).into_array(),
                PrimitiveArray::from_iter([10i64, 20, 30]).into_array(),
            ],
            3,
            Validity::NonNullable,
        )?);
        // SAFETY: Both handles remain live until cleanup below.
        let (mut schema, mut device_array) = unsafe { export_array(session, array) };

        let arrow_schema = Schema::try_from(&schema)?;
        assert_eq!(arrow_schema.fields().len(), 2);
        assert_eq!(arrow_schema.field(0).name(), "ids");
        assert_eq!(arrow_schema.field(1).name(), "values");

        assert_eq!(device_array.device_type, ARROW_DEVICE_CUDA);
        assert_eq!(device_array.reserved, [0; 3]);
        assert_eq!(device_array.array.length, 3);
        assert_eq!(device_array.array.n_buffers, 1);
        assert_eq!(device_array.array.n_children, 2);
        assert!(device_array.array.release.is_some());

        let children = unsafe { std::slice::from_raw_parts(device_array.array.children, 2) };
        for child in children {
            let child = unsafe { &**child };
            assert_eq!(child.length, 3);
            assert_eq!(child.n_buffers, 2);
            assert!(child.release.is_some());
        }

        unsafe {
            release_device_array(&mut device_array);
            assert!(device_array.array.release.is_none());
            release_schema(&mut schema);
            free_test_array(array);
            free_test_session(session);
        }
        Ok(())
    }

    #[cuda_test]
    fn test_cuda_session_new_export() {
        let mut error = ptr::null_mut();
        let session = unsafe { vx_cuda_session_new(&raw mut error) };
        assert!(error.is_null());
        assert!(!session.is_null());

        let array = test_array(PrimitiveArray::from_iter(0u32..5));
        // SAFETY: Both handles remain live until cleanup below.
        let (mut schema, mut device_array) = unsafe { export_array(session, array) };
        assert_eq!(device_array.array.length, 5);
        assert_eq!(device_array.device_type, ARROW_DEVICE_CUDA);

        unsafe {
            release_device_array(&mut device_array);
            release_schema(&mut schema);
            free_test_array(array);
            vortex_ffi::vx_session_free(session);
        }
    }

    #[cuda_not_available]
    #[test]
    fn test_export_reports_cuda_initialization_error() {
        let session = test_session(VortexSession::default());
        let array = test_array(PrimitiveArray::from_iter(0u32..5));
        let mut schema = FFI_ArrowSchema::empty();
        let mut device_array = ArrowDeviceArray::empty();
        let mut error = ptr::null_mut();

        let status = unsafe {
            vx_cuda_array_export_arrow_device(
                session,
                array,
                &raw mut schema,
                &raw mut device_array,
                &raw mut error,
            )
        };
        assert_eq!(status, VX_CUDA_ERR);
        assert!(!error.is_null());
        unsafe {
            vx_error_free(error);
            free_test_array(array);
            free_test_session(session);
        }
    }
}
