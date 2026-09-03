// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Private support for the Velox tests and benchmarks.

use std::ptr;

use arrow_array::array::make_array;
use arrow_array::ffi::FFI_ArrowArray;
use arrow_array::ffi::FFI_ArrowSchema;
use arrow_array::ffi::from_ffi;
use arrow_schema::Field;
use futures::SinkExt;
use futures::TryStreamExt;
use futures::channel::mpsc;
use futures::channel::mpsc::Sender;
use vortex::array::ArrayRef;
use vortex::array::stream::ArrayStreamAdapter;
use vortex::dtype::DType;
use vortex::error::VortexResult;
use vortex::error::vortex_ensure;
use vortex::error::vortex_err;
use vortex::file::WriteOptionsSessionExt;
use vortex::file::WriteStrategyBuilder;
use vortex::file::WriteSummary;
use vortex::io::runtime::BlockingRuntime;
use vortex::io::runtime::Task;
use vortex::io::session::RuntimeSessionExt;
use vortex::session::VortexSession;
use vortex_arrow::ArrowSessionExt;

use crate::ffi::ffi_runtime;
use crate::ffi::try_or;
use crate::ffi::vx_array_new_with;
use crate::ffi::vx_session_new;
use crate::ffi::vx_session_ref;
use crate::ffi::vx_velox_array;
use crate::ffi::vx_velox_error;
use crate::ffi::vx_velox_expression;
use crate::ffi::vx_velox_view;

struct TestSink {
    input: Sender<VortexResult<ArrayRef>>,
    writer: Task<VortexResult<WriteSummary>>,
    dtype: DType,
}

impl TestSink {
    fn try_new(session: &VortexSession, path: String, dtype: DType) -> VortexResult<Self> {
        let (input, output) = mpsc::channel(32);
        let stream = ArrayStreamAdapter::new(dtype.clone(), output.into_stream());
        let writer_session = session.clone();
        let writer = session.handle().spawn(async move {
            let mut file = async_fs::File::create(path).await?;
            writer_session
                .write_options()
                .with_strategy(WriteStrategyBuilder::default().build())
                .write(&mut file, stream)
                .await
        });
        Ok(Self {
            input,
            writer,
            dtype,
        })
    }

    fn push(&mut self, array: ArrayRef) -> VortexResult<()> {
        vortex_ensure!(
            array.dtype() == &self.dtype,
            "array dtype {} does not match writer dtype {}",
            array.dtype(),
            self.dtype
        );
        ffi_runtime()
            .block_on(self.input.send(Ok(array)))
            .map_err(|error| vortex_err!("Vortex test writer send failed: {error}"))
    }

    fn close(self) -> VortexResult<()> {
        drop(self.input);
        ffi_runtime().block_on(async {
            self.writer.await?;
            Ok(())
        })
    }
}

/// An opaque file writer for Velox tests and benchmarks.
pub struct vx_velox_test_writer {
    session: *mut crate::ffi::vx_velox_session,
    path: String,
    sink: Option<TestSink>,
}

impl Drop for vx_velox_test_writer {
    fn drop(&mut self) {
        // SAFETY: The writer owns this session handle.
        unsafe { crate::ffi::vx_session_free(self.session) };
    }
}

unsafe fn import_arrow(
    session: &VortexSession,
    array: *mut FFI_ArrowArray,
    schema: *mut FFI_ArrowSchema,
) -> VortexResult<ArrayRef> {
    vortex_ensure!(!array.is_null(), "Arrow array must not be null");
    vortex_ensure!(!schema.is_null(), "Arrow schema must not be null");
    // SAFETY: The caller transfers both initialized Arrow C Data structures.
    let array = unsafe { ptr::replace(array, FFI_ArrowArray::empty()) };
    // SAFETY: The caller transfers both initialized Arrow C Data structures.
    let schema = unsafe { ptr::replace(schema, FFI_ArrowSchema::empty()) };
    let array_data = unsafe { from_ffi(array, &schema) }?;
    let field = Field::try_from(&schema)?.with_nullable(false);
    let arrow_array = make_array(array_data);
    session.arrow().from_arrow_array(arrow_array, &field)
}

/// Import one Arrow C Data batch and apply an expression for adapter tests.
///
/// # Safety
///
/// Every pointer must satisfy the adapter header contract.
#[unsafe(no_mangle)]
pub unsafe extern "C-unwind" fn vx_velox_test_array_from_arrow_apply(
    session: *const crate::ffi::vx_velox_session,
    array: *mut FFI_ArrowArray,
    schema: *mut FFI_ArrowSchema,
    expression: *const vx_velox_expression,
    error_out: *mut *mut vx_velox_error,
) -> *const vx_velox_array {
    try_or(error_out, ptr::null(), || {
        let session = unsafe { vx_session_ref(session) }?;
        vortex_ensure!(
            !expression.is_null(),
            "Vortex test expression must not be null"
        );
        let array = unsafe { import_arrow(session, array, schema) }?;
        let expression = unsafe { vx_velox_expression::as_ref(expression) };
        Ok(vx_array_new_with(array.apply(expression)?))
    })
}

/// Create a private Vortex file writer for Velox tests.
///
/// # Safety
///
/// `path` and `error_out` must satisfy the adapter header contract.
#[unsafe(no_mangle)]
pub unsafe extern "C-unwind" fn vx_velox_test_writer_new(
    path: vx_velox_view,
    error_out: *mut *mut vx_velox_error,
) -> *mut vx_velox_test_writer {
    try_or(error_out, ptr::null_mut(), || {
        let path = unsafe { path.as_str() }?.to_owned();
        vortex_ensure!(
            !path.is_empty(),
            "Vortex test writer path must not be empty"
        );
        Ok(Box::into_raw(Box::new(vx_velox_test_writer {
            session: vx_session_new(),
            path,
            sink: None,
        })))
    })
}

/// Push one Arrow C Data batch into a private Vortex test writer.
///
/// # Safety
///
/// Every pointer must satisfy the adapter header contract.
#[unsafe(no_mangle)]
pub unsafe extern "C-unwind" fn vx_velox_test_writer_push(
    writer: *mut vx_velox_test_writer,
    array: *mut FFI_ArrowArray,
    schema: *mut FFI_ArrowSchema,
    error_out: *mut *mut vx_velox_error,
) -> i32 {
    try_or(error_out, 1, || {
        let writer = unsafe {
            writer
                .as_mut()
                .ok_or_else(|| vortex_err!("Vortex test writer must not be null"))?
        };
        let session = unsafe { vx_session_ref(writer.session) }?;
        let array = unsafe { import_arrow(session, array, schema) }?;
        if writer.sink.is_none() {
            writer.sink = Some(TestSink::try_new(
                session,
                writer.path.clone(),
                array.dtype().clone(),
            )?);
        }
        writer
            .sink
            .as_mut()
            .ok_or_else(|| vortex_err!("Vortex test writer did not create a sink"))?
            .push(array)?;
        Ok(0)
    })
}

/// Close a private Vortex test writer.
///
/// # Safety
///
/// `writer` must transfer one live writer. `error_out` must be null or writable.
#[unsafe(no_mangle)]
pub unsafe extern "C-unwind" fn vx_velox_test_writer_close(
    writer: *mut vx_velox_test_writer,
    error_out: *mut *mut vx_velox_error,
) -> i32 {
    try_or(error_out, 1, || {
        vortex_ensure!(!writer.is_null(), "Vortex test writer must not be null");
        // SAFETY: The caller transfers one writer from `vx_velox_test_writer_new`.
        let mut writer = unsafe { Box::from_raw(writer) };
        writer
            .sink
            .take()
            .ok_or_else(|| vortex_err!("Vortex test writer received no batches"))?
            .close()?;
        Ok(0)
    })
}

/// Abort and free a private Vortex test writer.
///
/// # Safety
///
/// `writer` must be null or transfer one live writer.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn vx_velox_test_writer_abort(writer: *mut vx_velox_test_writer) {
    if !writer.is_null() {
        // SAFETY: The caller transfers one writer from `vx_velox_test_writer_new`.
        drop(unsafe { Box::from_raw(writer) });
    }
}
