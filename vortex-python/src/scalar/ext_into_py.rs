// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Python values for the extension scalars that have a canonical Python type.
//!
//! Temporal scalars convert through the equivalent pyarrow scalar, so that `as_py` returns the
//! same value as `pyarrow.Scalar.as_py` and `Array.to_pylist`. For nanosecond timestamps and
//! durations, pyarrow returns `pandas.Timestamp` and `pandas.Timedelta` when pandas is installed,
//! and otherwise raises if the value has a sub-microsecond part.

use pyo3::Bound;
use pyo3::PyAny;
use pyo3::PyResult;
use pyo3::Python;
use pyo3::exceptions::PyValueError;
use pyo3::intern;
use pyo3::prelude::PyAnyMethods;
use pyo3::prelude::PyDictMethods;
use pyo3::types::PyBytes;
use pyo3::types::PyDict;
use vortex::dtype::DType;
use vortex::encodings::uuid::Uuid;
use vortex::extension::datetime::AnyTemporal;
use vortex::scalar::ExtScalar;
use vortex::scalar::Scalar;
use vortex_arrow::ArrowSessionExt;

use crate::arrow::ToPyArrow;
use crate::error::PyVortexError;
use crate::session::session;

/// Convert `scalar` into its canonical Python value.
///
/// Returns `None` when the extension has no canonical Python type, so that the caller converts
/// its storage value instead.
pub(super) fn ext_scalar_into_py<'py>(
    py: Python<'py>,
    scalar: &ExtScalar<'_>,
) -> PyResult<Option<Bound<'py, PyAny>>> {
    let ext_dtype = scalar.ext_dtype();
    let is_temporal = ext_dtype.is::<AnyTemporal>();
    let is_uuid = ext_dtype.is::<Uuid>();

    if !is_temporal && !is_uuid {
        return Ok(None);
    }

    if scalar.value().is_none() {
        return Ok(Some(py.None().into_bound(py)));
    }

    let storage = scalar.to_storage_scalar();

    if is_uuid {
        return uuid_into_py(py, &storage).map(Some);
    }

    temporal_into_py(py, scalar.dtype(), &storage).map(Some)
}

/// Convert a non-null temporal scalar through the pyarrow scalar of the same Arrow type.
fn temporal_into_py<'py>(
    py: Python<'py>,
    dtype: &DType,
    storage: &Scalar,
) -> PyResult<Bound<'py, PyAny>> {
    let arrow_type = session()
        .arrow()
        .to_arrow_datatype(dtype)
        .map_err(PyVortexError::from)?
        .to_pyarrow(py)?;

    let kwargs = PyDict::new(py);
    kwargs.set_item(intern!(py, "type"), arrow_type)?;

    let value = storage.as_primitive().as_::<i64>();

    py.import(intern!(py, "pyarrow"))?
        .call_method(intern!(py, "scalar"), (value,), Some(&kwargs))?
        .call_method0(intern!(py, "as_py"))
}

/// Convert non-null UUID storage, a fixed-size list of 16 big-endian bytes, into a `uuid.UUID`.
fn uuid_into_py<'py>(py: Python<'py>, storage: &Scalar) -> PyResult<Bound<'py, PyAny>> {
    let elements = storage
        .as_list()
        .elements()
        .ok_or_else(|| PyValueError::new_err("UUID storage is null"))?;

    let bytes = elements
        .iter()
        .map(|element| {
            element
                .as_primitive()
                .typed_value::<u8>()
                .ok_or_else(|| PyValueError::new_err("UUID storage has a null byte"))
        })
        .collect::<PyResult<Vec<u8>>>()?;

    let kwargs = PyDict::new(py);
    kwargs.set_item(intern!(py, "bytes"), PyBytes::new(py, &bytes))?;

    py.import(intern!(py, "uuid"))?
        .getattr(intern!(py, "UUID"))?
        .call((), Some(&kwargs))
}
