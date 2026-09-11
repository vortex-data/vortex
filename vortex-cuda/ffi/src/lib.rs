// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

#![deny(missing_docs)]

//! Native CUDA FFI helpers for cuDF interop.
//!
//! This crate keeps CUDA out of `vortex-ffi` and exports borrowed `vx_array` handles as the
//! `ArrowSchema + ArrowDeviceArray` pair that callers pass to cuDF's Arrow Device import APIs.

use std::os::raw::c_int;
use std::ptr;
use std::sync::Arc;

use arrow_schema::ffi::FFI_ArrowSchema;
use vortex::VortexSessionDefault;
use vortex::array::ArrayRef;
use vortex::array::stream::ArrayStreamExt;
use vortex::compressor::BtrBlocksCompressorBuilder;
use vortex::dtype::FieldName;
use vortex::dtype::FieldNames;
use vortex::editions::ComponentKind;
use vortex::editions::EditionSessionExt;
use vortex::error::VortexResult;
use vortex::error::vortex_ensure;
use vortex::error::vortex_err;
use vortex::expr::root;
use vortex::expr::select;
use vortex::file::OpenOptionsSessionExt;
use vortex::file::VortexFile;
use vortex::file::WriteStrategyBuilder;
use vortex::io::runtime::BlockingRuntime;
use vortex::layout::LayoutStrategy;
use vortex::layout::scan::scan_builder::ScanBuilder;
use vortex::layout::scan::split_by::SplitBy;
use vortex::session::SessionExt;
use vortex::session::VortexSession;
use vortex_cuda::CudaOpenOptionsExt;
use vortex_cuda::CudaSession;
use vortex_cuda::DictionaryExport;
use vortex_cuda::PooledFileReadAtOptions;
use vortex_cuda::arrow::ArrowDeviceArray;
use vortex_cuda::arrow::ArrowDeviceArrayStream;
use vortex_cuda::arrow::DeviceArrayExt;
use vortex_cuda::arrow::DeviceArrayStreamExt;
use vortex_cuda::layout::CudaFlatLayoutStrategy;
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

/// Enable direct I/O for pooled CUDA file reads.
pub const VX_CUDA_SCAN_FLAG_DIRECT_IO: u32 = 1 << 0;
/// Decode dictionaries on CUDA and export their logical plain Arrow types across scan batches.
pub const VX_CUDA_SCAN_FLAG_DECODE_DICTIONARIES: u32 = 1 << 1;
const VX_CUDA_SCAN_KNOWN_FLAGS: u32 =
    VX_CUDA_SCAN_FLAG_DIRECT_IO | VX_CUDA_SCAN_FLAG_DECODE_DICTIONARIES;

/// Options for scanning a CUDA-compatible Vortex file.
///
/// Zero-initialize this struct to use buffered file I/O, layout-derived batch splitting, and
/// dictionary-preserving Arrow exports.
#[repr(C)]
#[derive(Default)]
pub struct vx_cuda_scan_options {
    /// A bitwise combination of `VX_CUDA_SCAN_FLAG_*` values.
    pub flags: u32,
    /// Maximum rows in each output batch. Zero uses layout-derived splitting.
    /// Physical layout boundaries may produce shorter batches.
    pub batch_rows: usize,
}

/// Return a Vortex session with a [`CudaSession`] session variable.
///
/// If `session` already has CUDA support, this returns a clone of it. Otherwise it
/// returns a new session cloned from `session` with a default [`CudaSession`] attached.
fn session_with_cuda(session: &VortexSession) -> VortexResult<VortexSession> {
    session.get::<CudaSession>();
    register_cuda_layout(session);
    Ok(session.clone())
}

fn cuda_write_strategy(session: &VortexSession, block_rows: usize) -> Arc<dyn LayoutStrategy> {
    let allowed_encodings = session
        .enabled_component_ids(ComponentKind::Array)
        .into_iter()
        .collect();
    let mut strategy = WriteStrategyBuilder::default()
        .with_btrblocks_builder(
            BtrBlocksCompressorBuilder::default()
                .only_cuda_compatible()
                .retain_allowed_encodings(&allowed_encodings),
        )
        .with_flat_strategy(Arc::new(CudaFlatLayoutStrategy::default()));
    if block_rows > 0 {
        // Preserve explicit row blocks: layout dictionaries can otherwise split a high-cardinality
        // block into u16-sized dictionary runs, while a byte target can coalesce adjacent blocks.
        strategy = strategy
            .with_probe_compressor(BtrBlocksCompressorBuilder::empty().build())
            .with_row_block_size(block_rows)
            .with_data_block_target_bytes(None);
    }
    strategy.build()
}

/// Create a CUDA Vortex session.
///
/// Repeated [`vx_cuda_array_export_arrow_device`] calls reuse this CUDA state. Returns an owned
/// session handle, or null and an optional `vx_error` on failure.
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
            session
        }))
    })
}

/// Open a Vortex file sink configured to produce CUDA-readable files.
///
/// Push host-resident arrays and close or abort the returned sink with the standard
/// `vx_array_sink_*` functions. This function configures the on-disk encodings and layout; it does
/// not move arrays to the GPU during the write.
///
/// # Safety
///
/// `session`, `path`, and `dtype` must satisfy the same requirements as
/// `vx_array_sink_open_file`. If `error_out` is non-null, it must be valid for writing one error
/// pointer.
#[unsafe(no_mangle)]
pub unsafe extern "C-unwind" fn vx_cuda_array_sink_open_file(
    session: *const vx_session,
    path: vx_view,
    dtype: *const vx_dtype,
    error_out: *mut *mut vx_error,
) -> *mut vx_array_sink {
    unsafe { vx_cuda_array_sink_open_file_block_rows(session, path, dtype, 0, error_out) }
}

/// Open a CUDA-readable Vortex file sink with a fixed row block size.
///
/// `block_rows` controls the row granularity of CUDA-flat data blocks. Passing zero preserves the
/// default writer strategy used by [`vx_cuda_array_sink_open_file`]. Any nonzero value disables
/// byte-size coalescing and layout dictionaries so data blocks retain the requested row granularity.
///
/// Write and scan sizing are independent. To align on-disk row blocks with scan batches, pass the
/// same nonzero value to this function and [`vx_cuda_scan_path_arrow_device_stream_batch_rows`].
///
/// # Safety
///
/// `session`, `path`, and `dtype` must satisfy the same requirements as
/// `vx_array_sink_open_file`. If `error_out` is non-null, it must be valid for writing one error
/// pointer.
#[unsafe(no_mangle)]
pub unsafe extern "C-unwind" fn vx_cuda_array_sink_open_file_block_rows(
    session: *const vx_session,
    path: vx_view,
    dtype: *const vx_dtype,
    block_rows: usize,
    error_out: *mut *mut vx_error,
) -> *mut vx_array_sink {
    try_or(error_out, ptr::null_mut(), || {
        let vortex_session = unsafe { vx_session_ref(session) }?;
        session_with_cuda(vortex_session)?;
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

/// Scan a local Vortex file with buffered I/O and export an Arrow C Device stream.
///
/// Footer and zone-map reads remain on the host. Data segments are staged through pinned host
/// buffers and transferred directly to the GPU.
///
/// The file must use encodings and layouts supported by the CUDA execution path, such as files
/// written by [`vx_cuda_array_sink_open_file`]. Pinned staging buffers are reused across scans made
/// with the same CUDA session.
///
/// On success returns `0` and writes an owned [`ArrowDeviceArrayStream`] to `out_stream`. The
/// caller must release the stream and each array produced by it through their embedded Arrow
/// release callbacks.
///
/// On error returns `1` and, when `error_out` is non-null, writes a `vx_error` (free with
/// `vx_error_free`).
///
/// # Safety
///
/// `session` must be a valid borrowed handle created by `vortex-ffi`. `path` must be valid for the
/// duration of this call and contain UTF-8. `out_stream` must be a valid writable pointer. If
/// `error_out` is non-null, it must be valid for writing one error pointer.
#[unsafe(no_mangle)]
pub unsafe extern "C-unwind" fn vx_cuda_scan_path_arrow_device_stream(
    session: *const vx_session,
    path: vx_view,
    out_stream: *mut ArrowDeviceArrayStream,
    error_out: *mut *mut vx_error,
) -> c_int {
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

/// Scan a local Vortex file and export an Arrow C Device stream with bounded row batches.
///
/// `batch_rows` sets the maximum number of rows in each output batch. Physical layout boundaries
/// may produce shorter batches. Passing zero preserves the layout-derived splitting used by
/// [`vx_cuda_scan_path_arrow_device_stream`].
///
/// Scan and write sizing are independent; scan batches preserve on-disk layout boundaries.
///
/// # Safety
///
/// `session` must be a valid borrowed handle created by `vortex-ffi`. `path` must be valid for the
/// duration of this call and contain UTF-8. `out_stream` must be a valid writable pointer. If
/// `error_out` is non-null, it must be valid for writing one error pointer.
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

/// Scan a local Vortex file with explicit options and export an Arrow C Device stream.
///
/// This has the same ownership and file compatibility requirements as
/// [`vx_cuda_scan_path_arrow_device_stream`]. Pass a null `options` pointer or a zero-initialized
/// [`vx_cuda_scan_options`] to use buffered file I/O, layout-derived batch splitting, and
/// dictionary-preserving exports. Set [`VX_CUDA_SCAN_FLAG_DECODE_DICTIONARIES`] to export logical
/// plain types even when batches vary between dictionary/plain encodings or dictionary index widths.
/// The export policy applies only to this scan, not to the caller's session.
///
/// # Safety
///
/// `session` must be a valid borrowed handle created by `vortex-ffi`. `path` must be valid for the
/// duration of this call and contain UTF-8. `options`, when non-null, must point to a valid
/// [`vx_cuda_scan_options`]. `out_stream` must be a valid writable pointer. If `error_out` is
/// non-null, it must be valid for writing one error pointer.
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

/// Scan selected top-level columns of a local Vortex file as an Arrow C Device stream.
///
/// This has the same options, ownership, and file compatibility requirements as
/// [`vx_cuda_scan_path_arrow_device_stream_with_options`]. Zero `ncolumns` selects all columns and
/// ignores `columns`. Otherwise, names are case-sensitive literal top-level field names (not field
/// paths), returned in the requested order. Unknown or duplicate names and non-struct file dtypes
/// are rejected. Projection is applied by the scan builder before reading or decoding column data.
///
/// Names are copied during this call; the stream does not borrow them. The projected schema is
/// available even for a zero-row file. On error, `out_stream` is left unchanged.
///
/// # Safety
///
/// `session`, `path`, `options`, `out_stream`, and `error_out` must satisfy the requirements of
/// [`vx_cuda_scan_path_arrow_device_stream_with_options`]. For nonzero `ncolumns`, `columns` must
/// point to that many initialized, aligned [`vx_view`] values. Each name must point to `len`
/// readable bytes for this call, or be null with zero length. Names must contain UTF-8.
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

        // SAFETY: The caller keeps the borrowed options and column views alive for this call.
        let options = unsafe { scan_options(options) }?;
        let columns = unsafe { scan_columns(columns, ncolumns) }?;
        let path = unsafe { path.as_str() }?.to_owned();
        let session = session_with_cuda(unsafe { vx_session_ref(session) }?)?;
        let array_stream = ffi_runtime().block_on(async {
            let file = session
                .open_options()
                .with_cuda()
                .with_read_at_options(options.read_at_options)
                .open_path(path)
                .await?;
            let scan = projected_scan(&file, columns, options.batch_rows)?;
            Ok::<_, vortex::error::VortexError>(scan.into_array_stream()?.boxed())
        })?;
        let export_session = scan_export_session(&session, &options);
        let device_stream =
            array_stream.export_device_array_stream(&export_session, ffi_runtime())?;

        unsafe { ptr::write(out_stream, device_stream) };
        Ok(VX_CUDA_OK)
    })
}

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
    for (index, column) in columns.iter().enumerate() {
        vortex_ensure!(
            column.len <= isize::MAX as usize,
            "CUDA scan column {index} name is too long"
        );
        // SAFETY: The caller guarantees readable name bytes. as_str checks null and UTF-8.
        let name = unsafe { column.as_str() }
            .map_err(|error| vortex_err!("invalid CUDA scan column {index}: {error}"))?;
        vortex_ensure!(
            !names.iter().any(|existing| existing.as_ref() == name),
            "duplicate CUDA scan column: {name:?}"
        );
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
        let fields = file.dtype().as_struct_fields_opt().ok_or_else(|| {
            vortex_err!("CUDA scan column projection requires a struct file dtype")
        })?;
        for name in columns.iter() {
            vortex_ensure!(
                fields.find(name).is_some(),
                "unknown CUDA scan column: {name:?}"
            );
        }
        let projection = select(columns, root()).optimize_recursive(file.dtype())?;
        scan = scan.with_projection(projection.bind(file.dtype())?);
    }
    if batch_rows != 0 {
        let max_rows = u64::try_from(batch_rows)
            .map_err(|_| vortex_err!("CUDA scan batch row count is too large"))?;
        scan = scan.with_split_by(SplitBy::LayoutSubSplitting { max_rows });
    }
    Ok(scan)
}

struct CudaScanOptions {
    read_at_options: PooledFileReadAtOptions,
    batch_rows: usize,
    dictionary_export: DictionaryExport,
}

fn scan_export_session(session: &VortexSession, options: &CudaScanOptions) -> VortexSession {
    if options.dictionary_export == DictionaryExport::Preserve {
        return session.clone();
    }

    // VortexSession clones share their variable store. Use an independent export session so
    // opting in cannot change subsequent scans/exports through the caller's session. Cloning
    // CudaSession still shares the context, kernels, streams, and pinned buffer pool.
    let cuda_session = (*session.get::<CudaSession>())
        .clone()
        .with_dictionary_export(options.dictionary_export);
    VortexSession::default().with_some(cuda_session)
}

unsafe fn scan_options(options: *const vx_cuda_scan_options) -> VortexResult<CudaScanOptions> {
    let (flags, batch_rows) = if options.is_null() {
        (0, 0)
    } else {
        let options = unsafe { &*options };
        (options.flags, options.batch_rows)
    };
    vortex_ensure!(
        flags & !VX_CUDA_SCAN_KNOWN_FLAGS == 0,
        "unsupported CUDA scan option flags: {:#x}",
        flags & !VX_CUDA_SCAN_KNOWN_FLAGS
    );

    let read_at_options = PooledFileReadAtOptions::default();
    let read_at_options = if flags & VX_CUDA_SCAN_FLAG_DIRECT_IO == 0 {
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
        batch_rows,
        dictionary_export: if flags & VX_CUDA_SCAN_FLAG_DECODE_DICTIONARIES == 0 {
            DictionaryExport::Preserve
        } else {
            DictionaryExport::Decode
        },
    })
}

/// Export a borrowed Vortex array for cuDF's Arrow Device import path.
///
/// On success returns `0` and writes independently releasable `out_schema` and `out_array`; the
/// caller passes them to cuDF and releases both via their embedded Arrow callbacks after import. On
/// error returns `1` and, when `error_out` is non-null, writes a `vx_error` (free with
/// `vx_error_free`).
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

        let session = session_with_cuda(unsafe { vx_session_ref(session) }?)?;
        let array = unsafe { vx_array_ref(array) }?.clone();
        let mut ctx = CudaSession::create_execution_ctx(&session)?;
        let exported =
            futures::executor::block_on(array.export_device_array_with_schema(&mut ctx))?;

        unsafe {
            ptr::write(out_schema, exported.schema);
            ptr::write(out_array, exported.array);
        }
        Ok(VX_CUDA_OK)
    })
}

/// Consume a Vortex partition and scan it as an Arrow C Device stream.
///
/// This function takes ownership of `partition`. Callers must not free or reuse it after calling
/// this function, regardless of success or failure.
///
/// On success returns `0` and writes an owned `ArrowDeviceArrayStream` to `out_stream`. The stream
/// owns the resulting scan iterator. The caller must release the stream through its embedded Arrow
/// `release` callback, and must release each produced `ArrowDeviceArray` through its embedded
/// `ArrowArray.release` callback.
///
/// On error returns `1` and, when `error_out` is non-null, writes a `vx_error` (free with
/// `vx_error_free`).
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

        let array_stream = unsafe { vx_partition_into_array_stream(partition) }?;
        vortex_ensure!(!out_stream.is_null(), "null ArrowDeviceArrayStream output");

        let session = session_with_cuda(unsafe { vx_session_ref(session) }?)?;
        // Drive the stream on the same runtime the partition's scan spawned its work onto.
        let device_stream = array_stream.export_device_array_stream(&session, ffi_runtime())?;

        unsafe { ptr::write(out_stream, device_stream) };
        Ok(VX_CUDA_OK)
    })
}

#[cfg(test)]
mod tests {
    mod projection;

    use std::ptr;
    use std::sync::Arc;

    use arrow_schema::DataType;
    use arrow_schema::Field;
    use arrow_schema::Schema;
    use vortex::VortexSessionDefault;
    use vortex::array::ArrayRef;
    use vortex::array::IntoArray;
    use vortex::array::arrays::DictArray;
    use vortex::array::arrays::PrimitiveArray;
    use vortex::array::arrays::StructArray;
    use vortex::array::validity::Validity;
    use vortex::error::VortexResult;
    use vortex_cuda::arrow::ARROW_DEVICE_CUDA;
    use vortex_cuda_macros::cuda_not_available;
    use vortex_cuda_macros::test as cuda_test;

    use super::*;

    #[test]
    fn scan_options_default_to_buffered_io() -> VortexResult<()> {
        let options = vx_cuda_scan_options::default();
        assert_eq!(options.flags, 0);
        assert_eq!(options.batch_rows, 0);
        for pointer in [ptr::null(), &raw const options] {
            // SAFETY: Each pointer is either null or points to the live options above.
            let parsed = unsafe { scan_options(pointer) }?;
            assert_eq!(parsed.read_at_options, PooledFileReadAtOptions::default());
            assert_eq!(parsed.batch_rows, 0);
            assert_eq!(parsed.dictionary_export, DictionaryExport::Preserve);
        }
        Ok(())
    }

    #[test]
    fn maps_decode_dictionaries_scan_option() -> VortexResult<()> {
        let options = vx_cuda_scan_options {
            flags: VX_CUDA_SCAN_FLAG_DECODE_DICTIONARIES,
            batch_rows: 8192,
        };
        // SAFETY: options lives for the duration of parsing.
        let parsed = unsafe { scan_options(&raw const options) }?;
        assert_eq!(parsed.dictionary_export, DictionaryExport::Decode);
        assert_eq!(parsed.read_at_options, PooledFileReadAtOptions::default());
        assert_eq!(parsed.batch_rows, 8192);
        Ok(())
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn combines_direct_io_and_decode_dictionaries_scan_options() -> VortexResult<()> {
        let options = vx_cuda_scan_options {
            flags: VX_CUDA_SCAN_FLAG_DIRECT_IO | VX_CUDA_SCAN_FLAG_DECODE_DICTIONARIES,
            batch_rows: 8192,
        };
        // SAFETY: options lives for the duration of parsing.
        let parsed = unsafe { scan_options(&raw const options) }?;
        assert_eq!(parsed.dictionary_export, DictionaryExport::Decode);
        assert_eq!(
            parsed.read_at_options,
            PooledFileReadAtOptions::default().with_direct_io()
        );
        assert_eq!(parsed.batch_rows, 8192);
        Ok(())
    }

    #[test]
    fn rejects_unknown_scan_option_flags() {
        let options = vx_cuda_scan_options {
            flags: VX_CUDA_SCAN_FLAG_DECODE_DICTIONARIES | (1 << 2),
            ..Default::default()
        };
        assert!(unsafe { scan_options(&raw const options) }.is_err());
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn maps_direct_io_scan_option_to_pooled_reader() -> VortexResult<()> {
        let options = vx_cuda_scan_options {
            flags: VX_CUDA_SCAN_FLAG_DIRECT_IO,
            ..Default::default()
        };
        assert_eq!(
            unsafe { scan_options(&raw const options) }?.read_at_options,
            PooledFileReadAtOptions::default().with_direct_io()
        );
        Ok(())
    }

    #[test]
    fn maps_batch_rows_scan_option() -> VortexResult<()> {
        let options = vx_cuda_scan_options {
            batch_rows: 8192,
            ..Default::default()
        };
        assert_eq!(
            unsafe { scan_options(&raw const options) }?.batch_rows,
            8192
        );
        Ok(())
    }

    #[cuda_test]
    fn scan_dictionary_policy_is_isolated_and_reuses_cuda_resources() -> VortexResult<()> {
        let session = VortexSession::default().with_some(CudaSession::try_default()?);
        let options = vx_cuda_scan_options {
            flags: VX_CUDA_SCAN_FLAG_DECODE_DICTIONARIES,
            ..Default::default()
        };
        // SAFETY: options lives for the duration of parsing.
        let options = unsafe { scan_options(&raw const options) }?;
        let export_session = scan_export_session(&session, &options);
        assert_eq!(
            session.get::<CudaSession>().dictionary_export(),
            DictionaryExport::Preserve
        );
        assert_eq!(
            export_session.get::<CudaSession>().dictionary_export(),
            DictionaryExport::Decode
        );
        assert!(Arc::ptr_eq(
            session.get::<CudaSession>().pinned_buffer_pool(),
            export_session.get::<CudaSession>().pinned_buffer_pool(),
        ));
        let array = DictArray::try_new(
            PrimitiveArray::from_iter([1u8, 0, 1]).into_array(),
            PrimitiveArray::from_iter([10i32, 20]).into_array(),
        )?
        .into_array();
        for (session, expected_type, preserved) in [
            (&export_session, DataType::Int32, false),
            (
                &session,
                DataType::Dictionary(Box::new(DataType::Int16), Box::new(DataType::Int32)),
                true,
            ),
        ] {
            let mut ctx = CudaSession::create_execution_ctx(session)?;
            let mut exported =
                ffi_runtime().block_on(array.clone().export_device_array_with_schema(&mut ctx))?;
            assert_eq!(
                Field::try_from(&exported.schema)?.data_type(),
                &expected_type
            );
            assert_eq!(!exported.array.array.dictionary.is_null(), preserved);
            vortex_cuda::arrow::release_device_array(&mut exported.array);
        }
        Ok(())
    }

    fn test_session(session: VortexSession) -> *mut vx_session {
        Box::into_raw(Box::new(session)).cast::<vx_session>()
    }

    unsafe fn free_test_session(session: *mut vx_session) {
        unsafe { drop(Box::from_raw(session.cast::<VortexSession>())) };
    }

    fn test_array(array: impl IntoArray) -> *const vx_array {
        Arc::into_raw(Arc::new(array.into_array())).cast::<vx_array>()
    }

    unsafe fn free_test_array(array: *const vx_array) {
        unsafe { Arc::decrement_strong_count(array.cast::<ArrayRef>()) };
    }

    unsafe fn release_schema(schema: &mut FFI_ArrowSchema) {
        unsafe {
            if let Some(release) = schema.release {
                release(schema);
            }
        }
    }

    unsafe fn release_device_array(array: &mut ArrowDeviceArray) {
        unsafe {
            if let Some(release) = array.array.release {
                release(&raw mut array.array);
            }
        }
    }

    fn empty_device_array() -> ArrowDeviceArray {
        ArrowDeviceArray {
            array: vortex_cuda::arrow::ArrowArray::empty(),
            device_id: 0,
            device_type: 0,
            sync_event: ptr::null_mut(),
            reserved: [0; 3],
        }
    }

    #[cuda_test]
    fn test_export_primitive_arrow_device() {
        let mut error = ptr::null_mut();
        let session = test_session(VortexSession::default());
        let array = test_array(PrimitiveArray::from_iter(0u32..5));
        let mut schema = FFI_ArrowSchema::empty();
        let mut device_array = empty_device_array();

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
        let mut error = ptr::null_mut();
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

        let mut schema = FFI_ArrowSchema::empty();
        let mut device_array = empty_device_array();

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
        let mut schema = FFI_ArrowSchema::empty();
        let mut device_array = empty_device_array();

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
        let mut device_array = empty_device_array();
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
            vortex_ffi::vx_error_free(error);
            free_test_array(array);
            free_test_session(session);
        }
    }
}
