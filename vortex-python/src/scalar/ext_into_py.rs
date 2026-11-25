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
use vortex::encodings::uuid::Uuid;
use vortex::extension::datetime::Date;
use vortex::extension::datetime::Duration;
use vortex::extension::datetime::Time;
use vortex::extension::datetime::TimeUnit;
use vortex::extension::datetime::Timestamp;
use vortex::scalar::ExtScalar;
use vortex::scalar::Scalar;

/// Convert `scalar` into its canonical Python value.
///
/// Returns `None` when the extension has no canonical Python type, so that the caller converts
/// its storage value instead.
pub(super) fn ext_scalar_into_py<'py>(
    py: Python<'py>,
    scalar: &ExtScalar<'_>,
) -> PyResult<Option<Bound<'py, PyAny>>> {
    let ext_dtype = scalar.ext_dtype();

    if ext_dtype.is::<Uuid>() {
        return uuid_into_py(py, &scalar.to_storage_scalar()).map(Some);
    }

    let pyarrow = py.import(intern!(py, "pyarrow"))?;

    let arrow_type = if let Some(unit) = ext_dtype.metadata_opt::<Date>() {
        match unit {
            TimeUnit::Days => pyarrow.call_method0(intern!(py, "date32"))?,
            TimeUnit::Milliseconds => pyarrow.call_method0(intern!(py, "date64"))?,
            _ => return Err(unsupported_unit("date", *unit)),
        }
    } else if let Some(unit) = ext_dtype.metadata_opt::<Time>() {
        match unit {
            TimeUnit::Seconds | TimeUnit::Milliseconds => {
                pyarrow.call_method1(intern!(py, "time32"), (unit_name("time", *unit)?,))?
            }
            _ => pyarrow.call_method1(intern!(py, "time64"), (unit_name("time", *unit)?,))?,
        }
    } else if let Some(options) = ext_dtype.metadata_opt::<Timestamp>() {
        let kwargs = PyDict::new(py);
        kwargs.set_item(intern!(py, "tz"), options.tz.as_deref())?;
        pyarrow.call_method(
            intern!(py, "timestamp"),
            (unit_name("timestamp", options.unit)?,),
            Some(&kwargs),
        )?
    } else if let Some(unit) = ext_dtype.metadata_opt::<Duration>() {
        pyarrow.call_method1(intern!(py, "duration"), (unit_name("duration", *unit)?,))?
    } else {
        return Ok(None);
    };

    let storage = scalar.to_storage_scalar();
    let value = storage.as_primitive().as_::<i64>();

    let kwargs = PyDict::new(py);
    kwargs.set_item(intern!(py, "type"), arrow_type)?;

    pyarrow
        .call_method(intern!(py, "scalar"), (value,), Some(&kwargs))?
        .call_method0(intern!(py, "as_py"))
        .map(Some)
}

/// Convert UUID storage, a fixed-size list of 16 big-endian bytes, into a `uuid.UUID`.
fn uuid_into_py<'py>(py: Python<'py>, storage: &Scalar) -> PyResult<Bound<'py, PyAny>> {
    let Some(elements) = storage.as_list().elements() else {
        return Ok(py.None().into_bound(py));
    };

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

/// The pyarrow name of `unit`.
fn unit_name(kind: &str, unit: TimeUnit) -> PyResult<&'static str> {
    match unit {
        TimeUnit::Nanoseconds => Ok("ns"),
        TimeUnit::Microseconds => Ok("us"),
        TimeUnit::Milliseconds => Ok("ms"),
        TimeUnit::Seconds => Ok("s"),
        TimeUnit::Days => Err(unsupported_unit(kind, unit)),
    }
}

fn unsupported_unit(kind: &str, unit: TimeUnit) -> pyo3::PyErr {
    PyValueError::new_err(format!("Unsupported {kind} unit {unit}"))
}
