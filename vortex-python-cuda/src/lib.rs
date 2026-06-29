// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Optional CUDA extension for PyVortex.
//!
//! Builds the separate `vortex-data-cuda` wheel (imported as `vortex_cuda`), installed alongside
//! the CPU-only `vortex-data` wheel. Keeping CUDA in its own extension keeps the base wheel free of
//! CUDA build/runtime dependencies; `vortex.cuda_extension_installed()` reports whether it is present.

use std::ffi::CStr;
use std::ffi::c_void;
use std::ptr::NonNull;
use std::sync::LazyLock;

use arrow_schema::Field;
use arrow_schema::Schema;
use arrow_schema::ffi::FFI_ArrowSchema;
use pyo3::exceptions::PyNotImplementedError;
use pyo3::exceptions::PyRuntimeError;
use pyo3::exceptions::PyValueError;
use pyo3::ffi;
use pyo3::ffi::c_str;
use pyo3::prelude::*;
use pyo3::types::PyCapsule;
use pyo3::types::PyDict;
use pyo3::types::PyList;
use pyo3::types::PyTuple;
use vortex::VortexSessionDefault;
use vortex::array::ArrayId;
use vortex::array::ArrayRef;
use vortex::array::buffer::BufferHandle;
use vortex::array::serde::ArrayChildren;
use vortex::array::session::ArraySessionExt;
use vortex::buffer::ByteBuffer;
use vortex::dtype::DType;
use vortex::error::VortexError;
use vortex::error::VortexResult;
use vortex::error::vortex_ensure;
use vortex::error::vortex_err;
use vortex::flatbuffers::FlatBuffer;
use vortex::io::runtime::BlockingRuntime;
use vortex::io::runtime::current::CurrentThreadRuntime;
use vortex::session::VortexSession;
use vortex_cuda::CudaSession;
use vortex_cuda::arrow::ARROW_DEVICE_CUDA;
use vortex_cuda::arrow::ArrowDeviceArray;
use vortex_cuda::arrow::ArrowDeviceArrayWithSchema;
use vortex_cuda::arrow::DeviceArrayExt;
use vortex_cuda::arrow::release_device_array;
use vortex_cuda::arrow::release_schema;
use vortex_python_abi::BUFFER_EXPORT_CAPSULE_NAME;
use vortex_python_abi::VORTEX_BUFFER_EXPORT_VERSION;
use vortex_python_abi::VORTEX_BUFFER_HOST;
use vortex_python_abi::VortexBufferExport;

const ARROW_SCHEMA_CAPSULE_NAME: &CStr = c_str!("arrow_schema");
const USED_ARROW_SCHEMA_CAPSULE_NAME: &CStr = c_str!("used_arrow_schema");
const ARROW_DEVICE_ARRAY_CAPSULE_NAME: &CStr = c_str!("arrow_device_array");
const USED_ARROW_DEVICE_ARRAY_CAPSULE_NAME: &CStr = c_str!("used_arrow_device_array");

struct BufferExportGuard {
    export: NonNull<VortexBufferExport>,
}

impl BufferExportGuard {
    fn export(&self) -> &VortexBufferExport {
        unsafe { self.export.as_ref() }
    }
}

impl AsRef<[u8]> for BufferExportGuard {
    fn as_ref(&self) -> &[u8] {
        let export = self.export();
        if export.len == 0 {
            &[]
        } else {
            unsafe { std::slice::from_raw_parts(export.ptr, export.len) }
        }
    }
}

impl Drop for BufferExportGuard {
    fn drop(&mut self) {
        // The producer's release callback owns cleanup of both private data and the descriptor.
        let export = unsafe { self.export.as_ref() };
        if let Some(release) = export.release {
            unsafe { release(self.export.as_ptr()) };
        }
    }
}

// The guard is moved into `Bytes::from_owner`, which requires `Send + Sync`. After import we disable
// the source capsule destructor and own the C export until this guard is dropped.
unsafe impl Send for BufferExportGuard {}
unsafe impl Sync for BufferExportGuard {}

fn import_buffer_from_capsule(capsule: &Bound<'_, PyCapsule>) -> PyResult<BufferHandle> {
    let export_ptr = capsule
        .pointer_checked(Some(BUFFER_EXPORT_CAPSULE_NAME))?
        .cast::<VortexBufferExport>();
    let export = unsafe { export_ptr.as_ref() };

    if export.version != VORTEX_BUFFER_EXPORT_VERSION {
        return Err(PyValueError::new_err(format!(
            "unsupported VortexBufferExport version {}",
            export.version
        )));
    }
    if export.kind != VORTEX_BUFFER_HOST {
        return Err(PyValueError::new_err(format!(
            "unsupported buffer kind {} (only host buffers are supported in metadata bridge)",
            export.kind
        )));
    }

    if export.len != 0 && export.ptr.is_null() {
        return Err(PyValueError::new_err(
            "non-empty VortexBufferExport has null data pointer",
        ));
    }
    if export.release.is_none() {
        return Err(PyValueError::new_err(
            "VortexBufferExport is missing a release callback",
        ));
    }

    let len = export.len;
    let alignment = vortex::buffer::Alignment::try_from(
        u32::try_from(export.alignment)
            .map_err(|_| PyValueError::new_err("buffer alignment exceeds u32"))?,
    )
    .map_err(|e| PyValueError::new_err(e.to_string()))?;

    if len != 0 && !alignment.is_ptr_aligned(export.ptr) {
        return Err(PyValueError::new_err(format!(
            "buffer pointer is not aligned to requested alignment {alignment}"
        )));
    }

    // Transfer ownership of the boxed VortexBufferExport from the producer capsule into the Bytes
    // owner below. Otherwise the producer capsule could be dropped before the reconstructed
    // BufferHandle, leaving the Bytes owner with a dangling export pointer.
    unsafe { ffi::PyCapsule_SetDestructor(capsule.as_ptr(), None) };
    if PyErr::occurred(capsule.py()) {
        return Err(PyErr::fetch(capsule.py()));
    }

    let guard = BufferExportGuard { export: export_ptr };

    let byte_buffer = if len == 0 {
        drop(guard);
        ByteBuffer::empty_aligned(alignment)
    } else {
        ByteBuffer::from(bytes::Bytes::from_owner(guard)).aligned(alignment)
    };

    Ok(BufferHandle::new_host(byte_buffer))
}

struct ExportedDeviceArray(ArrowDeviceArrayWithSchema);

// The exported Arrow C Device structs own CPU-side metadata plus CUDA device pointers through their
// Arrow release callbacks. `Python::detach` requires a `Send` return value even though it executes
// the closure synchronously with the GIL released; this wrapper lets us move the owned export result
// back across that boundary without changing the ABI structs themselves.
unsafe impl Send for ExportedDeviceArray {}

static RUNTIME: LazyLock<CurrentThreadRuntime> = LazyLock::new(CurrentThreadRuntime::new);
static METADATA_SESSION: LazyLock<VortexSession> =
    LazyLock::new(<VortexSession as VortexSessionDefault>::default);
static CUDA_SESSION: LazyLock<Result<VortexSession, String>> = LazyLock::new(|| {
    if !vortex_cuda::cuda_available() {
        return Err("CUDA is not available: no usable CUDA driver/device was found".to_string());
    }

    let cuda_session = CudaSession::try_default().map_err(|err| err.to_string())?;
    Ok(<VortexSession as VortexSessionDefault>::default().with_some(cuda_session))
});

fn cuda_session() -> PyResult<&'static VortexSession> {
    match &*CUDA_SESSION {
        Ok(session) => Ok(session),
        Err(err) => Err(PyRuntimeError::new_err(err.clone())),
    }
}

fn to_py_err(err: VortexError) -> PyErr {
    PyRuntimeError::new_err(err.to_string())
}

/// Return whether a usable CUDA device is available in the current process.
///
/// This performs a runtime probe of the CUDA driver and device. It differs from
/// `vortex.cuda_extension_installed()`, which only reports whether this extension package is
/// installed.
#[pyfunction]
fn cuda_available() -> bool {
    vortex_cuda::cuda_available()
}

struct ArrayMetadata {
    encoding_id: String,
    dtype: Vec<u8>,
    len: usize,
    metadata: Vec<u8>,
    buffers: Vec<BufferHandle>,
    children: Vec<ArrayMetadata>,
}

struct MetadataChildren(Vec<ArrayRef>);

impl ArrayChildren for MetadataChildren {
    fn get(&self, index: usize, dtype: &DType, len: usize) -> VortexResult<ArrayRef> {
        let child = self
            .0
            .as_slice()
            .get(index)
            .ok_or_else(|| vortex_err!("array metadata child index {index} out of bounds"))?
            .clone();
        vortex_ensure!(
            child.dtype() == dtype,
            "array metadata child {index} has dtype {}, expected {dtype}",
            child.dtype()
        );
        vortex_ensure!(
            child.len() == len,
            "array metadata child {index} has length {}, expected {len}",
            child.len()
        );
        Ok(child)
    }

    fn len(&self) -> usize {
        self.0.len()
    }
}

fn extract_array_metadata(array: &Bound<'_, PyAny>) -> PyResult<ArrayMetadata> {
    let metadata = array.call_method0("__vortex_array_metadata__")?;
    parse_array_metadata(&metadata)
}

fn parse_array_metadata(value: &Bound<'_, PyAny>) -> PyResult<ArrayMetadata> {
    let tuple = value.cast::<PyTuple>()?;
    if tuple.len() != 6 {
        return Err(PyValueError::new_err(format!(
            "expected Vortex array metadata tuple of length 6, got {}",
            tuple.len()
        )));
    }

    let buffers = tuple
        .get_item(4)?
        .cast::<PyList>()?
        .iter()
        .map(|item| {
            let capsule: Bound<'_, PyCapsule> = item.extract()?;
            import_buffer_from_capsule(&capsule)
        })
        .collect::<PyResult<Vec<_>>>()?;

    let children = tuple
        .get_item(5)?
        .cast::<PyList>()?
        .iter()
        .map(|child| parse_array_metadata(&child))
        .collect::<PyResult<Vec<_>>>()?;

    Ok(ArrayMetadata {
        encoding_id: tuple.get_item(0)?.extract()?,
        dtype: tuple.get_item(1)?.extract()?,
        len: tuple.get_item(2)?.extract()?,
        metadata: tuple.get_item(3)?.extract()?,
        buffers,
        children,
    })
}

fn dtype_from_metadata(metadata: &ArrayMetadata, session: &VortexSession) -> VortexResult<DType> {
    let flatbuffer = FlatBuffer::align_from(ByteBuffer::from(metadata.dtype.clone()));
    DType::from_flatbuffer(flatbuffer, session)
}

fn deserialize_metadata_tree(
    metadata: &ArrayMetadata,
    session: &VortexSession,
) -> VortexResult<ArrayRef> {
    let dtype = dtype_from_metadata(metadata, session)?;
    let children = metadata
        .children
        .iter()
        .map(|child| deserialize_metadata_tree(child, session))
        .collect::<VortexResult<Vec<_>>>()?;
    let children = MetadataChildren(children);
    let encoding_id = ArrayId::new(&metadata.encoding_id);
    let plugin = session
        .arrays()
        .registry()
        .find(&encoding_id)
        .ok_or_else(|| vortex_err!("Unknown array encoding: {}", metadata.encoding_id))?;
    let decoded = plugin.deserialize(
        &dtype,
        metadata.len,
        &metadata.metadata,
        &metadata.buffers,
        &children,
        session,
    )?;
    vortex_ensure!(
        decoded.len() == metadata.len,
        "Array decoded from {} has incorrect length {}, expected {}",
        metadata.encoding_id,
        decoded.len(),
        metadata.len
    );
    vortex_ensure!(
        decoded.dtype() == &dtype,
        "Array decoded from {} has incorrect dtype {}, expected {}",
        metadata.encoding_id,
        decoded.dtype(),
        dtype
    );
    vortex_ensure!(
        plugin.is_supported_encoding(&decoded.encoding_id()),
        "Array decoded from {} has incorrect encoding {}",
        metadata.encoding_id,
        decoded.encoding_id()
    );
    Ok(decoded)
}

// PyO3 exposes a synchronous Python API, while the CUDA Arrow Device export is async.
// Keep this adapter private to the Python extension so this PR does not add a public
// blocking convenience API to `vortex-cuda`.
fn export_device_array_with_schema_blocking(
    array: ArrayRef,
    session: &VortexSession,
    runtime: &CurrentThreadRuntime,
) -> VortexResult<ArrowDeviceArrayWithSchema> {
    let mut ctx = CudaSession::create_execution_ctx(session)?;
    runtime.block_on(array.export_device_array_with_schema(&mut ctx))
}

/// Return the dtype string after crossing the private vtable-metadata bridge.
#[pyfunction]
fn _debug_array_metadata_dtype(array: Bound<'_, PyAny>) -> PyResult<String> {
    let metadata = extract_array_metadata(&array)?;
    let array = deserialize_metadata_tree(&metadata, &METADATA_SESSION).map_err(to_py_err)?;
    Ok(array.dtype().to_string())
}

/// Return array values after crossing the private vtable-metadata bridge.
#[pyfunction]
fn _debug_array_metadata_display_values(array: Bound<'_, PyAny>) -> PyResult<String> {
    let metadata = extract_array_metadata(&array)?;
    let array = deserialize_metadata_tree(&metadata, &METADATA_SESSION).map_err(to_py_err)?;
    Ok(array.display_values().to_string())
}

/// Export a PyVortex array as Arrow C Device schema and array PyCapsules.
#[pyfunction]
#[pyo3(signature = (array, requested_schema = None, **kwargs))]
fn export_device_array<'py>(
    py: Python<'py>,
    array: Bound<'py, PyAny>,
    requested_schema: Option<Bound<'py, PyAny>>,
    kwargs: Option<&Bound<'py, PyDict>>,
) -> PyResult<(Bound<'py, PyCapsule>, Bound<'py, PyCapsule>)> {
    reject_unsupported_kwargs(kwargs)?;

    let metadata = extract_array_metadata(&array)?;
    let session = cuda_session()?;
    let array = deserialize_metadata_tree(&metadata, session).map_err(to_py_err)?;
    let dtype = array.dtype().clone();

    let exported = py
        .detach(move || {
            export_device_array_with_schema_blocking(array, session, &RUNTIME)
                .map(ExportedDeviceArray)
        })
        .map_err(to_py_err)?;
    let mut exported = exported.0;

    if let Err(err) = check_requested_schema(requested_schema.as_ref(), &exported.schema, &dtype) {
        release_exported(&mut exported);
        return Err(err);
    }

    let ArrowDeviceArrayWithSchema { schema, mut array } = exported;
    let schema = match schema_capsule(py, schema) {
        Ok(schema) => schema,
        Err(err) => {
            release_device_array(&mut array);
            return Err(err);
        }
    };
    let array = device_array_capsule(py, array)?;
    Ok((schema, array))
}

fn reject_unsupported_kwargs(kwargs: Option<&Bound<'_, PyDict>>) -> PyResult<()> {
    let Some(kwargs) = kwargs else {
        return Ok(());
    };

    for (name, value) in kwargs.iter() {
        if !value.is_none() {
            return Err(PyNotImplementedError::new_err(format!(
                "unsupported __arrow_c_device_array__ keyword argument {name}={value:?}"
            )));
        }
    }
    Ok(())
}

fn check_requested_schema(
    requested_schema: Option<&Bound<'_, PyAny>>,
    exported_schema: &FFI_ArrowSchema,
    dtype: &DType,
) -> PyResult<()> {
    let Some(requested_schema) = requested_schema else {
        return Ok(());
    };
    if requested_schema.is_none() {
        return Ok(());
    }

    let requested_schema = requested_schema.cast::<PyCapsule>()?;
    let requested_schema = unsafe {
        requested_schema
            .pointer_checked(Some(ARROW_SCHEMA_CAPSULE_NAME))?
            .cast::<FFI_ArrowSchema>()
            .as_ref()
    };

    if matches!(dtype, DType::Struct(..)) {
        let requested = Schema::try_from(requested_schema)
            .map_err(|err| PyValueError::new_err(err.to_string()))?;
        let exported = Schema::try_from(exported_schema)
            .map_err(|err| PyValueError::new_err(err.to_string()))?;
        if requested == exported {
            return Ok(());
        }
    } else {
        let requested = Field::try_from(requested_schema)
            .map_err(|err| PyValueError::new_err(err.to_string()))?;
        let exported = Field::try_from(exported_schema)
            .map_err(|err| PyValueError::new_err(err.to_string()))?;
        if requested == exported {
            return Ok(());
        }
    }

    Err(PyNotImplementedError::new_err(
        "requested_schema coercion is not supported by vortex_cuda.export_device_array",
    ))
}

fn release_exported(exported: &mut ArrowDeviceArrayWithSchema) {
    release_schema(&mut exported.schema);
    release_device_array(&mut exported.array);
}

fn schema_capsule<'py>(
    py: Python<'py>,
    schema: FFI_ArrowSchema,
) -> PyResult<Bound<'py, PyCapsule>> {
    let ptr = Box::into_raw(Box::new(schema)).cast::<c_void>();
    let ptr = NonNull::new(ptr)
        .ok_or_else(|| PyRuntimeError::new_err("failed to allocate ArrowSchema capsule"))?;
    let capsule = unsafe {
        PyCapsule::new_with_pointer_and_destructor(
            py,
            ptr,
            ARROW_SCHEMA_CAPSULE_NAME,
            Some(release_schema_capsule),
        )
    };
    match capsule {
        Ok(capsule) => Ok(capsule),
        Err(err) => {
            let mut schema = unsafe { Box::from_raw(ptr.as_ptr().cast::<FFI_ArrowSchema>()) };
            release_schema(&mut schema);
            Err(err)
        }
    }
}

fn device_array_capsule<'py>(
    py: Python<'py>,
    array: ArrowDeviceArray,
) -> PyResult<Bound<'py, PyCapsule>> {
    debug_assert_eq!(array.device_type, ARROW_DEVICE_CUDA);
    let ptr = Box::into_raw(Box::new(array)).cast::<c_void>();
    let ptr = NonNull::new(ptr)
        .ok_or_else(|| PyRuntimeError::new_err("failed to allocate ArrowDeviceArray capsule"))?;
    let capsule = unsafe {
        PyCapsule::new_with_pointer_and_destructor(
            py,
            ptr,
            ARROW_DEVICE_ARRAY_CAPSULE_NAME,
            Some(release_device_array_capsule),
        )
    };
    match capsule {
        Ok(capsule) => Ok(capsule),
        Err(err) => {
            let mut array = unsafe { Box::from_raw(ptr.as_ptr().cast::<ArrowDeviceArray>()) };
            release_device_array(&mut array);
            Err(err)
        }
    }
}

// The `used_*` names are only seen after a consumer imports and renames the capsule. CI cannot
// exercise that path without a CUDA Arrow Device consumer, but the destructor must still reclaim
// the outer boxed C struct after the consumer move-nulls the embedded release callback.
unsafe fn capsule_pointer_with_name_or_used(
    capsule: *mut ffi::PyObject,
    name: &CStr,
    used_name: &CStr,
) -> *mut c_void {
    let ptr = unsafe { ffi::PyCapsule_GetPointer(capsule, name.as_ptr()) };
    if !ptr.is_null() {
        return ptr;
    }
    unsafe { ffi::PyErr_Clear() };

    let ptr = unsafe { ffi::PyCapsule_GetPointer(capsule, used_name.as_ptr()) };
    if !ptr.is_null() {
        return ptr;
    }
    unsafe { ffi::PyErr_Clear() };
    std::ptr::null_mut()
}

unsafe extern "C" fn release_schema_capsule(capsule: *mut ffi::PyObject) {
    let ptr = unsafe {
        capsule_pointer_with_name_or_used(
            capsule,
            ARROW_SCHEMA_CAPSULE_NAME,
            USED_ARROW_SCHEMA_CAPSULE_NAME,
        )
    };
    if ptr.is_null() {
        return;
    }

    let mut schema = unsafe { Box::from_raw(ptr.cast::<FFI_ArrowSchema>()) };
    release_schema(&mut schema);
}

unsafe extern "C" fn release_device_array_capsule(capsule: *mut ffi::PyObject) {
    let ptr = unsafe {
        capsule_pointer_with_name_or_used(
            capsule,
            ARROW_DEVICE_ARRAY_CAPSULE_NAME,
            USED_ARROW_DEVICE_ARRAY_CAPSULE_NAME,
        )
    };
    if ptr.is_null() {
        return;
    }

    let mut array = unsafe { Box::from_raw(ptr.cast::<ArrowDeviceArray>()) };
    release_device_array(&mut array);
}

/// The `vortex_cuda._lib` extension module.
#[cfg(feature = "extension-module")]
#[pymodule]
fn _lib(m: &Bound<PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(cuda_available, m)?)?;
    m.add_function(wrap_pyfunction!(_debug_array_metadata_dtype, m)?)?;
    m.add_function(wrap_pyfunction!(_debug_array_metadata_display_values, m)?)?;
    m.add_function(wrap_pyfunction!(export_device_array, m)?)?;
    Ok(())
}
