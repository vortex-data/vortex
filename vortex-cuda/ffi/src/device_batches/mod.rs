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
use vortex_cuda::arrow::ARROW_DEVICE_CUDA;
use vortex_cuda::arrow::ArrowDeviceArray;
use vortex_cuda::arrow::ArrowDeviceArrayWithSchema;
use vortex_cuda::arrow::DeviceArrayExt;
use vortex_cuda::arrow::release_device_array;
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

/// Dictionary-preserving CUDA scan with an owned Arrow schema per batch.
///
/// Logical dtype and CUDA device are fixed; physical schemas, dictionary values, and index widths
/// may vary. Uses `DictionaryExport::Preserve`; plain batches remain plain.
/// Retains runtime/session resources. Use and free only on the creating thread, without concurrent
/// or reentrant calls. Free with `vx_cuda_device_batches_free`.
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

    /// Return released outputs carrying this scan's CUDA device metadata.
    ///
    /// Matches the decoded stream; release callbacks still determine end-of-stream.
    fn eos(&self) -> ArrowDeviceArrayWithSchema {
        ArrowDeviceArrayWithSchema {
            schema: FFI_ArrowSchema::empty(),
            array: ArrowDeviceArray {
                device_id: self.ctx.stream().context().ordinal() as i64,
                device_type: ARROW_DEVICE_CUDA,
                ..ArrowDeviceArray::empty()
            },
        }
    }

    fn next(&mut self) -> VortexResult<ArrowDeviceArrayWithSchema> {
        vortex_ensure!(
            !matches!(self.state, State::Failed),
            "CUDA device batches are terminal after an earlier scan or export error"
        );
        if matches!(self.state, State::End) {
            return Ok(self.eos());
        }

        // Poison before polling/exporting so a failure or caught panic cannot skip a batch on retry.
        self.state = State::Failed;
        let Some(array) = self.array_iter.next() else {
            self.state = State::End;
            return Ok(self.eos());
        };
        let array = array?;
        vortex_ensure_eq!(
            array.dtype(),
            &self.dtype,
            "scan batch logical dtype changed"
        );
        let mut exported = self
            .runtime
            .block_on(array.export_device_array_with_schema(&mut self.ctx))?;
        if let Err(error) = self.check_device(&exported.array) {
            // The schema releases itself on drop; the array needs explicit release.
            release_device_array(&mut exported.array);
            return Err(error);
        }
        self.state = State::Ready;
        Ok(exported)
    }

    /// Require exports to use the handle's CUDA device.
    ///
    /// Physical schemas may vary between batches, but the execution device must not.
    fn check_device(&self, array: &ArrowDeviceArray) -> VortexResult<()> {
        vortex_ensure_eq!(
            array.device_type,
            ARROW_DEVICE_CUDA,
            "scan batch exported on non-CUDA device type"
        );
        vortex_ensure_eq!(
            array.device_id,
            self.ctx.stream().context().ordinal() as i64,
            "scan batch moved from CUDA device"
        );
        Ok(())
    }
}

/// Open a local CUDA scan that preserves per-batch dictionaries.
///
/// Returns an owned handle, or null on error. Options, projection, and splitting match
/// `vx_cuda_scan_path_arrow_device_stream_projected`: null options use defaults; zero `ncolumns`
/// ignores `columns` and selects all. Names must be unique, literal, case-sensitive UTF-8;
/// unknown names fail, and nonempty projections require a struct file.
///
/// Inputs are borrowed for this call; the handle retains runtime/session resources.
/// Non-null `error_out` is cleared on success or receives an error to free with `vx_error_free`.
///
/// # Safety
///
/// Borrow a live `session` and readable UTF-8 `path`. Non-null `options` must be aligned and
/// initialized. For nonzero `ncolumns`, `columns` must reference that many valid, aligned views
/// with readable name bytes (null only for zero length). Non-null `error_out` must be writable.
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

/// Return the next owned Arrow schema and CUDA array (`0` on success, `1` on error).
///
/// Use each batch's own schema. EOS (including repeated calls) sets both release callbacks to null;
/// live zero-row batches are not EOS. Errors leave outputs unchanged. Null/unaligned handle or
/// output pointers are rejected before polling; scan/export failures, dtype/device mismatches,
/// and panics make the handle terminal (later calls fail). Free it after a terminal error.
/// Non-null `error_out` is cleared on success or receives an error to free with `vx_error_free`.
///
/// Outputs outlive the handle/session. Release each independently, exactly once, via its Arrow
/// callback. Wait on `out_array->sync_event` before consuming buffers; retain the array until GPU
/// consumers finish. The event/buffers live until array release; output storage is caller-owned.
///
/// # Safety
///
/// Use a live handle exclusively on its creating thread, without reentrancy. Non-null, aligned
/// output pointers must reference writable storage, disjoint from each other, the handle, and
/// `error_out`. Outputs may be uninitialized; release previous contents before overwriting them.
/// Non-null `error_out` must be writable for one error pointer.
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

/// Free the handle; null is a no-op.
///
/// Returned outputs remain caller-owned and must be released via their Arrow callbacks.
///
/// # Safety
///
/// Free a live, exclusively owned handle exactly once on its creating thread, without reentrancy.
#[unsafe(no_mangle)]
pub unsafe extern "C-unwind" fn vx_cuda_device_batches_free(handle: *mut vx_cuda_device_batches) {
    if !handle.is_null() {
        // SAFETY: The caller transfers sole ownership of a live handle on its creating thread.
        drop(unsafe { Box::from_raw(handle) });
    }
}

#[cfg(test)]
mod tests;
