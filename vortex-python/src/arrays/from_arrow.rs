// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use arrow_array::RecordBatchReader;
use arrow_array::ffi_stream::ArrowArrayStreamReader;
use arrow_array::make_array;
use arrow_data::ArrayData as ArrowArrayData;
use arrow_schema::DataType;
use arrow_schema::Field;
use pyo3::exceptions::PyValueError;
use pyo3::intern;
use pyo3::prelude::*;
use vortex::array::ArrayRef;
use vortex::array::IntoArray;
use vortex::array::LEGACY_SESSION;
use vortex::array::arrays::ChunkedArray;
use vortex::array::arrays::ExtensionArray;
use vortex::array::arrow::FromArrowArray;
use vortex::dtype::DType;
use vortex::dtype::arrow::ARROW_EXT_NAME_VARIANT;
use vortex::dtype::arrow::FromArrowType;
use vortex::dtype::extension::ExtId;
use vortex::dtype::session::DTypeSessionExt;
use vortex::error::VortexError;
use vortex::error::VortexResult;
use vortex::session::VortexSession;

use crate::SESSION;
use crate::arrays::PyArrayRef;
use crate::arrow::FromPyArrow;
use crate::classes::array_class;
use crate::classes::chunked_array_class;
use crate::classes::extension_type_class;
use crate::classes::table_class;
use crate::error::PyVortexError;
use crate::error::PyVortexResult;

/// Convert a Python `pyarrow` array (including `pa.ExtensionArray`) into a Vortex array.
///
/// The Arrow C ABI strips extension identity from leaf arrays; we recover it from the
/// Python object via `extension_name` and `__arrow_ext_serialize__`.
pub trait FromPyArrowArray: Sized {
    /// Convert a Python `pyarrow` array to a Vortex array.
    fn from_pyarrow(py_array: &Bound<'_, PyAny>, nullable: bool) -> PyResult<Self>;

    /// Like [`Self::from_pyarrow`], but consults `session` to resolve `pa.ExtensionType`
    /// arrays back into `DType::Extension`.
    fn from_pyarrow_with_session(
        py_array: &Bound<'_, PyAny>,
        nullable: bool,
        session: &VortexSession,
    ) -> PyResult<Self> {
        let _ = session;
        Self::from_pyarrow(py_array, nullable)
    }
}

impl FromPyArrowArray for ArrayRef {
    fn from_pyarrow(py_array: &Bound<'_, PyAny>, nullable: bool) -> PyResult<Self> {
        Self::from_pyarrow_with_session(py_array, nullable, &LEGACY_SESSION)
    }

    fn from_pyarrow_with_session(
        py_array: &Bound<'_, PyAny>,
        nullable: bool,
        session: &VortexSession,
    ) -> PyResult<Self> {
        let ext_info = extract_extension_info(py_array)?;
        let array_data = ArrowArrayData::from_pyarrow(&py_array.as_borrowed())?;
        let storage =
            ArrayRef::from_arrow_with_session(make_array(array_data).as_ref(), nullable, session)
                .map_err(PyVortexError::from)?;
        let Some((name, meta)) = ext_info else {
            return Ok(storage);
        };
        Ok(wrap_with_extension(storage, &name, &meta, session).map_err(PyVortexError::from)?)
    }
}

/// Raw bytes from `__arrow_ext_serialize__` — no base64 (that's only for the
/// Arrow Field-metadata string channel). Variant short-circuits to `None` so it surfaces
/// as `DType::Variant` via the storage path, mirroring `dtype/arrow.rs::dtype_from_field`.
fn extract_extension_info(py_array: &Bound<'_, PyAny>) -> PyResult<Option<(String, Vec<u8>)>> {
    let py = py_array.py();
    let py_type = py_array.getattr(intern!(py, "type"))?;
    if !py_type.is_instance(extension_type_class(py)?)? {
        return Ok(None);
    }
    let ext_name: String = py_type.getattr(intern!(py, "extension_name"))?.extract()?;
    if ext_name == ARROW_EXT_NAME_VARIANT {
        return Ok(None);
    }
    let ext_meta_bytes: Vec<u8> = py_type
        .call_method0(intern!(py, "__arrow_ext_serialize__"))?
        .extract()?;
    Ok(Some((ext_name, ext_meta_bytes)))
}

/// Soft fallback to storage on registry miss or malformed metadata, mirroring
/// `dtype/arrow.rs::resolve_extension_dtype`.
fn wrap_with_extension(
    storage: ArrayRef,
    ext_name: &str,
    ext_meta_bytes: &[u8],
    session: &VortexSession,
) -> VortexResult<ArrayRef> {
    let ext_id = ExtId::new(ext_name);
    let dtypes = session.dtypes();
    let Some(plugin) = dtypes.registry().find(&ext_id) else {
        log::warn!("pyarrow extension {ext_name:?} not registered on session; using storage dtype");
        return Ok(storage);
    };
    let ext_dtype = match plugin.deserialize(ext_meta_bytes, storage.dtype().clone()) {
        Ok(dt) => dt,
        Err(e) => {
            log::warn!(
                "pyarrow extension {ext_name:?} failed to deserialize metadata ({e}); \
                 using storage dtype",
            );
            return Ok(storage);
        }
    };
    Ok(ExtensionArray::try_new(ext_dtype, storage)?.into_array())
}

/// Convert an Arrow object to a Vortex array.
pub(super) fn from_arrow(obj: &Borrowed<'_, '_, PyAny>) -> PyVortexResult<PyArrayRef> {
    let py = obj.py();
    let pa_array = array_class(py)?;
    let chunked_array = chunked_array_class(py)?;
    let table = table_class(py)?;

    if obj.is_instance(pa_array)? {
        let bound = obj.to_owned();
        let ext_info = extract_extension_info(&bound)?;
        let arrow_array = ArrowArrayData::from_pyarrow(&obj.as_borrowed()).map(make_array)?;
        let storage = ArrayRef::from_arrow_with_session(
            arrow_array.as_ref(),
            arrow_array.is_nullable(),
            &SESSION,
        )?;
        let enc_array = match ext_info {
            None => storage,
            Some((name, meta)) => {
                wrap_with_extension(storage, &name, &meta, &SESSION).map_err(PyVortexError::from)?
            }
        };
        Ok(PyArrayRef::from(enc_array))
    } else if obj.is_instance(chunked_array)? {
        let chunks: Vec<Bound<PyAny>> = obj.getattr(intern!(py, "chunks"))?.extract()?;
        // ChunkedArray has a uniform type — peek extension identity once and reuse.
        let bound = obj.to_owned();
        let ext_info = extract_extension_info(&bound)?;
        let encoded_chunks = chunks
            .iter()
            .map(|chunk| {
                let arrow_array =
                    ArrowArrayData::from_pyarrow(&chunk.as_borrowed()).map(make_array)?;
                let storage = ArrayRef::from_arrow_with_session(
                    arrow_array.as_ref(),
                    arrow_array.is_nullable(),
                    &SESSION,
                )
                .map_err(PyVortexError::from)?;
                match &ext_info {
                    None => Ok(storage),
                    Some((name, meta)) => wrap_with_extension(storage, name, meta, &SESSION)
                        .map_err(|e| PyVortexError::from(e).into()),
                }
            })
            .collect::<PyResult<Vec<_>>>()?;
        let dtype: DType = if let Some(first) = encoded_chunks.first() {
            first.dtype().clone()
        } else {
            // Empty array: `obj.type` over the C ABI loses extension metadata, so we
            // recover only the storage dtype.
            obj.getattr(intern!(py, "type"))
                .and_then(|v| DataType::from_pyarrow(&v.as_borrowed()))
                .map(|dt| DType::from_arrow_with_session(&Field::new("_", dt, false), &SESSION))?
        };
        Ok(PyArrayRef::from(
            ChunkedArray::try_new(encoded_chunks, dtype)?.into_array(),
        ))
    } else if obj.is_instance(table)? {
        // The C ABI Stream carries Field metadata on the schema — session-aware
        // conversion recovers extensions directly, no Python peek needed.
        let array_stream = ArrowArrayStreamReader::from_pyarrow(&obj.as_borrowed())?;
        let dtype = DType::from_arrow_with_session(array_stream.schema(), &SESSION);
        let chunks = array_stream
            .into_iter()
            .map(|b| {
                b.map_err(VortexError::from)
                    .and_then(|b| ArrayRef::from_arrow_with_session(b, false, &SESSION))
            })
            .collect::<VortexResult<Vec<_>>>()?;
        Ok(PyArrayRef::from(
            ChunkedArray::try_new(chunks, dtype)?.into_array(),
        ))
    } else {
        Err(PyValueError::new_err("Cannot convert object to Vortex array").into())
    }
}
