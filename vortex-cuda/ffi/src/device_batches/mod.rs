// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::os::raw::c_int;
use std::ptr;

use arrow_schema::ffi::FFI_ArrowSchema;
use vortex::array::ArrayRef;
use vortex::array::stream::SendableArrayStream;
use vortex::dtype::DType;
use vortex::error::VortexResult;
use vortex::error::vortex_ensure;
use vortex::error::vortex_ensure_eq;
use vortex::io::runtime::BlockingRuntime;
use vortex::io::runtime::current::CurrentThreadRuntime;
use vortex::session::VortexSession;
use vortex_cuda::CudaExecutionCtx;
use vortex_cuda::CudaSession;
use vortex_cuda::DictionaryExport;
use vortex_cuda::arrow::ArrowDeviceArray;
use vortex_cuda::arrow::ArrowDeviceArrayWithSchema;
use vortex_cuda::arrow::DeviceArrayExt;
use vortex_ffi::ffi_runtime;
use vortex_ffi::try_or;
use vortex_ffi::vx_error;
use vortex_ffi::vx_session;
use vortex_ffi::vx_session_ref;
use vortex_ffi::vx_view;

use crate::VX_CUDA_ERR;
use crate::VX_CUDA_OK;
use crate::open_cuda_scan;
use crate::scan_columns;
use crate::scan_options;
use crate::session_with_cuda;
use crate::vx_cuda_scan_options;

/// Opaque scan handle returning an independently owned schema for every CUDA batch.
///
/// The logical dtype is fixed, but physical Arrow schemas, dictionary values, and index widths
/// may vary between batches. Dictionaries are exported using `DictionaryExport::Preserve`.
/// Plain batches stay plain; dictionaries are not synthesized.
/// The handle retains its runtime and session resources. Use and free it on the creating thread,
/// without concurrent or reentrant calls. Free with `vx_cuda_device_batches_free`.
#[allow(non_camel_case_types)]
pub struct vx_cuda_device_batches {
    array_iter: Box<dyn Iterator<Item = VortexResult<ArrayRef>>>,
    ctx: CudaExecutionCtx,
    runtime: CurrentThreadRuntime,
    dtype: DType,
    state: State,
}

enum State {
    Ready,
    End,
    Failed,
}

impl vx_cuda_device_batches {
    fn new(
        stream: SendableArrayStream,
        session: &VortexSession,
        runtime: &CurrentThreadRuntime,
    ) -> VortexResult<Self> {
        let dtype = stream.dtype().clone();
        let ctx = CudaSession::create_execution_ctx(session)?
            .with_dictionary_export(DictionaryExport::Preserve);
        Ok(Self {
            array_iter: Box::new(runtime.block_on_stream(stream)),
            ctx,
            runtime: runtime.clone(),
            dtype,
            state: State::Ready,
        })
    }

    fn next(&mut self) -> VortexResult<ArrowDeviceArrayWithSchema> {
        vortex_ensure!(
            !matches!(self.state, State::Failed),
            "CUDA device batches are terminal after an earlier scan or export error"
        );
        if matches!(self.state, State::End) {
            return Ok(ArrowDeviceArrayWithSchema {
                schema: FFI_ArrowSchema::empty(),
                array: ArrowDeviceArray::empty(),
            });
        }

        // Poison before polling/exporting so errors and panics caught at the FFI boundary cannot
        // silently skip a consumed batch on retry. Pointer validation happens before this method.
        self.state = State::Failed;
        let Some(array) = self.array_iter.next() else {
            self.state = State::End;
            return Ok(ArrowDeviceArrayWithSchema {
                schema: FFI_ArrowSchema::empty(),
                array: ArrowDeviceArray::empty(),
            });
        };
        let array = array?;
        vortex_ensure_eq!(
            array.dtype(),
            &self.dtype,
            "scan batch logical dtype changed"
        );
        let exported = self
            .runtime
            .block_on(array.export_device_array_with_schema(&mut self.ctx))?;
        self.state = State::Ready;
        Ok(exported)
    }
}

/// Scan a local CUDA-readable file with a native per-batch-schema iterator.
///
/// Returns an owned handle, or null on error. Uses the same options, ordered top-level projection,
/// direct I/O, and batch splitting as `vx_cuda_scan_path_arrow_device_stream_projected`, but
/// preserves dictionaries. Null options selects defaults;
/// zero `ncolumns` ignores `columns` and selects all columns. Names are literal, case-sensitive,
/// unique UTF-8; unknown names fail and nonempty projections require a struct file.
///
/// The session handle, path, options, and column names are borrowed only for this call. The result
/// retains the runtime and session resources, and must be used and freed on the creating thread.
/// Non-null `error_out` is cleared on success or receives an owned error on failure; free errors
/// with `vx_error_free`.
///
/// # Safety
///
/// `session` must be a live borrowed Vortex FFI handle; `path` must contain readable UTF-8.
/// Non-null `options` must point to an initialized, aligned `vx_cuda_scan_options`.
/// For nonzero `ncolumns`, `columns` must reference that many valid, aligned `vx_view` values.
/// Name bytes must be readable for this call; null is allowed only for zero length.
/// Non-null `error_out` must be writable for one error pointer.
#[unsafe(no_mangle)]
pub unsafe extern "C-unwind" fn vx_cuda_scan_path_device_batches(
    session: *const vx_session,
    path: vx_view,
    options: *const vx_cuda_scan_options,
    columns: *const vx_view,
    ncolumns: usize,
    error_out: *mut *mut vx_error,
) -> *mut vx_cuda_device_batches {
    try_or(error_out, ptr::null_mut(), || {
        // SAFETY: The caller keeps all borrowed inputs valid for this call.
        let (options, columns, path, session) = unsafe {
            (
                scan_options(options)?,
                scan_columns(columns, ncolumns)?,
                path.as_str()?,
                vx_session_ref(session)?,
            )
        };
        let session = session_with_cuda(session);
        let stream = open_cuda_scan(session, path, options, columns)?;
        Ok(Box::into_raw(Box::new(vx_cuda_device_batches::new(
            stream,
            session,
            ffi_runtime(),
        )?)))
    })
}

/// Return the next independently owned Arrow schema and CUDA array. Returns `0` on success,
/// `1` on error. Each batch has its own physical schema; never reuse an earlier batch's schema.
///
/// End of stream returns both outputs empty/released (both release callbacks null), including on
/// repeated calls. A zero-row batch has live release callbacks and is not end of stream.
/// Errors leave both outputs unchanged. Null or unaligned handle/output pointers are rejected
/// before consuming input. A scan, logical dtype, export error, or panic makes the handle terminal:
/// later calls fail rather than skipping the failed batch. Free the handle after such an error.
/// Non-null `error_out` is cleared on success or receives an owned error; free with `vx_error_free`.
///
/// Successful outputs outlive the handle and the original session handle. Release the schema via
/// `out_schema->release` and the array via `out_array->array.release`, independently, exactly once.
/// Export is stream-ordered: wait on `out_array->sync_event` before consuming device buffers.
/// The event and device buffers remain valid until the array is released. Releasing the schema
/// or handle does not release the array. The caller owns the output storage itself.
///
/// # Safety
///
/// Use a live handle on its creating thread, without concurrent or reentrant calls. Non-null,
/// aligned output pointers must reference writable, non-overlapping storage, not aliasing the
/// handle or `error_out`. Output storage may be uninitialized; release any previous contents before
/// a successful call overwrites them. Non-null `error_out` must be writable for one error pointer.
#[unsafe(no_mangle)]
pub unsafe extern "C-unwind" fn vx_cuda_device_batches_next(
    handle: *mut vx_cuda_device_batches,
    out_schema: *mut FFI_ArrowSchema,
    out_array: *mut ArrowDeviceArray,
    error_out: *mut *mut vx_error,
) -> c_int {
    try_or(error_out, VX_CUDA_ERR, || {
        vortex_ensure!(!handle.is_null(), "null vx_cuda_device_batches");
        vortex_ensure!(handle.is_aligned(), "unaligned vx_cuda_device_batches");
        vortex_ensure!(!out_schema.is_null(), "null ArrowSchema output");
        vortex_ensure!(out_schema.is_aligned(), "unaligned ArrowSchema output");
        vortex_ensure!(!out_array.is_null(), "null ArrowDeviceArray output");
        vortex_ensure!(out_array.is_aligned(), "unaligned ArrowDeviceArray output");
        // SAFETY: The pointer checks passed and the caller guarantees exclusive access to a live
        // handle on its creating thread.
        let exported = unsafe { &mut *handle }.next()?;
        // SAFETY: Both outputs passed pointer checks and the caller guarantees separate writable
        // storage. Publish only after the entire export succeeds.
        unsafe {
            ptr::write(out_schema, exported.schema);
            ptr::write(out_array, exported.array);
        }
        Ok(VX_CUDA_OK)
    })
}

/// Free a device-batches handle. Null is a no-op. Previously returned schemas and arrays remain
/// owned by the caller and must still be released independently via their Arrow callbacks.
///
/// # Safety
///
/// A non-null handle must be live, exclusively owned, and freed exactly once on its creating
/// thread, without a concurrent or reentrant call.
#[unsafe(no_mangle)]
pub unsafe extern "C-unwind" fn vx_cuda_device_batches_free(handle: *mut vx_cuda_device_batches) {
    if !handle.is_null() {
        // SAFETY: The caller transfers sole ownership of a live handle on its creating thread.
        drop(unsafe { Box::from_raw(handle) });
    }
}

#[cfg(test)]
mod tests;
