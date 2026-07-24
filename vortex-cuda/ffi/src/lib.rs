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
use vortex::array::stream::ArrayStreamExt;
use vortex::compressor::BtrBlocksCompressorBuilder;
use vortex::error::VortexResult;
use vortex::error::vortex_ensure;
use vortex::file::OpenOptionsSessionExt;
use vortex::file::WriteStrategyBuilder;
use vortex::io::runtime::BlockingRuntime;
use vortex::layout::scan::split_by::SplitBy;
use vortex::session::SessionExt;
use vortex::session::VortexSession;
use vortex_cuda::CudaOpenOptionsExt;
use vortex_cuda::CudaSession;
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
const VX_CUDA_SCAN_KNOWN_FLAGS: u32 = VX_CUDA_SCAN_FLAG_DIRECT_IO;

/// Options for scanning a CUDA-compatible Vortex file.
///
/// Zero-initialize this struct to use buffered file I/O and layout-derived batch splitting.
#[repr(C)]
#[derive(Default)]
pub struct vx_cuda_scan_options {
    /// A bitwise combination of `VX_CUDA_SCAN_FLAG_*` values.
    pub flags: u32,
    /// Number of rows in each output batch. Zero uses layout-derived splitting.
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
/// byte-size coalescing so data blocks retain the requested row granularity.
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
        session_with_cuda(unsafe { vx_session_ref(session) }?)?;
        let mut strategy = WriteStrategyBuilder::default()
            .with_btrblocks_builder(BtrBlocksCompressorBuilder::default().only_cuda_compatible())
            .with_flat_strategy(Arc::new(CudaFlatLayoutStrategy::default()));
        if block_rows > 0 {
            // The default byte-size target can coalesce several row blocks into one data block.
            // A scan using the same row count would then split inside that data block, defeating
            // the requested alignment. The explicit block row count already defines the desired
            // granularity for this opt-in path, so a separate byte-size target is unnecessary.
            strategy = strategy
                .with_row_block_size(block_rows)
                .with_data_block_target_bytes(None);
        }
        unsafe { vx_array_sink_open_file_with_strategy(session, path, dtype, strategy.build()) }
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

/// Scan a local Vortex file and export an Arrow C Device stream with fixed-size row batches.
///
/// `batch_rows` controls the number of rows in each output batch. Passing zero preserves the
/// layout-derived splitting used by [`vx_cuda_scan_path_arrow_device_stream`].
///
/// Scan and write sizing are independent. To align scan batches with on-disk row blocks, pass the
/// same nonzero value to this function and [`vx_cuda_array_sink_open_file_block_rows`].
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
/// [`vx_cuda_scan_options`] to use buffered file I/O and layout-derived batch splitting.
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
    try_or(error_out, VX_CUDA_ERR, || {
        vortex_ensure!(!out_stream.is_null(), "null ArrowDeviceArrayStream output");

        let path = unsafe { path.as_str() }?.to_owned();
        let session = session_with_cuda(unsafe { vx_session_ref(session) }?)?;
        let options = unsafe { scan_options(options) }?;
        let array_stream = ffi_runtime().block_on(async {
            let file = session
                .open_options()
                .with_cuda()
                .with_read_at_options(options.read_at_options)
                .open_path(path)
                .await?;
            let scan = file.scan()?;
            let scan = if options.batch_rows == 0 {
                scan
            } else {
                scan.with_split_by(SplitBy::RowCount(options.batch_rows))
            };
            Ok::<_, vortex::error::VortexError>(scan.into_array_stream()?.boxed())
        })?;
        let device_stream = array_stream.export_device_array_stream(&session, ffi_runtime())?;

        unsafe { ptr::write(out_stream, device_stream) };
        Ok(VX_CUDA_OK)
    })
}

struct CudaScanOptions {
    read_at_options: PooledFileReadAtOptions,
    batch_rows: usize,
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
    use std::ptr;
    use std::sync::Arc;

    use arrow_schema::Field;
    use arrow_schema::Schema;
    use vortex::VortexSessionDefault;
    use vortex::array::ArrayRef;
    use vortex::array::IntoArray;
    use vortex::array::arrays::PrimitiveArray;
    use vortex::array::arrays::StructArray;
    use vortex::array::validity::Validity;
    use vortex::error::VortexResult;
    use vortex_cuda::arrow::ARROW_DEVICE_CUDA;
    use vortex_cuda_macros::cuda_not_available;
    use vortex_cuda_macros::test as cuda_test;

    use super::*;

    #[test]
    fn scan_options_default_to_buffered_io() {
        let options = vx_cuda_scan_options::default();
        assert_eq!(options.flags, 0);
        assert_eq!(options.batch_rows, 0);
    }

    #[test]
    fn rejects_unknown_scan_option_flags() {
        let options = vx_cuda_scan_options {
            flags: 1 << 31,
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
