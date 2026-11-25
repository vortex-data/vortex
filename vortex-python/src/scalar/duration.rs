// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use pyo3::Bound;
use pyo3::IntoPyObject;
use pyo3::Py;
use pyo3::PyAny;
use pyo3::PyResult;
use pyo3::PyTypeInfo;
use pyo3::Python;
use pyo3::exceptions::PyModuleNotFoundError;
use pyo3::exceptions::PyOverflowError;
use pyo3::exceptions::PyValueError;
use pyo3::intern;
use pyo3::prelude::PyAnyMethods;
use pyo3::sync::PyOnceLock;
use pyo3::types::IntoPyDict;
use pyo3::types::PyDelta;
use pyo3::types::PyType;
use vortex::dtype::DType;
use vortex::dtype::Nullability;
use vortex::error::vortex_err;
use vortex::extension::datetime::Duration;
use vortex::extension::datetime::TimeUnit;
use vortex::scalar::Scalar;

use crate::error::PyVortexError;
use crate::error::PyVortexResult;

const NANOS_PER_MICROSECOND: i128 = 1_000;
const NANOS_PER_SECOND: i128 = 1_000_000_000;

fn duration_unit_name(unit: TimeUnit) -> &'static str {
    match unit {
        TimeUnit::Nanoseconds => "ns",
        TimeUnit::Microseconds => "us",
        TimeUnit::Milliseconds => "ms",
        TimeUnit::Seconds => "s",
        TimeUnit::Days => unreachable!("days are not a valid Duration unit"),
    }
}

fn nanos_per_unit(unit: TimeUnit) -> i128 {
    match unit {
        TimeUnit::Nanoseconds => 1,
        TimeUnit::Microseconds => NANOS_PER_MICROSECOND,
        TimeUnit::Milliseconds => 1_000_000,
        TimeUnit::Seconds => NANOS_PER_SECOND,
        TimeUnit::Days => unreachable!("days are not a valid Duration unit"),
    }
}

pub(super) fn to_py<'py>(
    py: Python<'py>,
    scalar: &Scalar,
    unit: TimeUnit,
) -> PyResult<Bound<'py, PyAny>> {
    let Some(value) = scalar.value() else {
        return Ok(py.None().into_pyobject(py)?);
    };
    let value = value
        .as_primitive()
        .cast::<i64>()
        .map_err(|error| PyValueError::new_err(error.to_string()))?;
    let keyword = match unit {
        TimeUnit::Seconds => intern!(py, "seconds"),
        TimeUnit::Milliseconds => intern!(py, "milliseconds"),
        TimeUnit::Microseconds => intern!(py, "microseconds"),
        TimeUnit::Nanoseconds => {
            if value == i64::MIN {
                return Err(PyOverflowError::new_err(
                    "pandas.Timedelta reserves i64::MIN nanoseconds for NaT",
                ));
            }
            return get_pandas_timedelta(py)?.call1((value, "ns"));
        }
        TimeUnit::Days => unreachable!("days are not a valid Duration unit"),
    };
    let kwargs = [(keyword, value)].into_py_dict(py)?;
    PyDelta::type_object(py).call((), Some(&kwargs))
}

fn get_pandas_timedelta(py: Python<'_>) -> PyVortexResult<&Bound<'_, PyType>> {
    static PANDAS_TIMEDELTA: PyOnceLock<Py<PyType>> = PyOnceLock::new();
    PANDAS_TIMEDELTA.import(py, "pandas", "Timedelta").map_err(|error| {
        if error.is_instance_of::<PyModuleNotFoundError>(py)
            && error
                .value(py)
                .getattr(intern!(py, "name"))
                .and_then(|name| name.extract::<String>())
                .is_ok_and(|name| name == "pandas")
        {
            vortex_err!(
                "Exporting duration[ns] to Python requires pandas. Install it with `pip install pandas`."
            )
            .into()
        } else {
            PyVortexError::from(error)
        }
    })
}

pub(super) fn from_py(value: &Bound<'_, PyDelta>, dtype: Option<&DType>) -> PyVortexResult<Scalar> {
    let (value, source_unit) = timedelta_value(value)?;
    let unit = dtype
        .and_then(DType::as_extension_opt)
        .and_then(|ext| ext.metadata_opt::<Duration>())
        .copied()
        .unwrap_or(source_unit);
    let value = if unit == source_unit {
        value
    } else {
        // The intermediate nanoseconds can exceed i64 even when the target unit fits.
        let nanos = value * nanos_per_unit(source_unit);
        let per_unit = nanos_per_unit(unit);
        if nanos % per_unit != 0 {
            return Err(PyValueError::new_err(format!(
                "Timedelta cannot be represented in {} without losing precision",
                duration_unit_name(unit)
            ))
            .into());
        }
        nanos / per_unit
    };
    let storage = i64::try_from(value).map_err(|_| {
        PyOverflowError::new_err(format!(
            "Timedelta overflows duration[{}]",
            duration_unit_name(unit)
        ))
    })?;
    Ok(Scalar::try_new(
        DType::Extension(Duration::try_new(unit, Nullability::NonNullable)?.erased()),
        Some(storage.into()),
    )?)
}

fn timedelta_value(value: &Bound<'_, PyDelta>) -> PyResult<(i128, TimeUnit)> {
    let py = value.py();
    // pandas exposes a NumPy scalar view in its own unit through `asm8`.
    if let Some(asm8) = value.getattr_opt(intern!(py, "asm8"))? {
        let unit = value.getattr(intern!(py, "unit"))?.extract::<String>()?;
        let unit = match unit.as_str() {
            "s" => TimeUnit::Seconds,
            "ms" => TimeUnit::Milliseconds,
            "us" => TimeUnit::Microseconds,
            "ns" => TimeUnit::Nanoseconds,
            _ => {
                return Err(PyValueError::new_err(format!(
                    "Unsupported timedelta unit {unit}"
                )));
            }
        };
        let value = asm8
            .call_method1(intern!(py, "view"), ("int64",))?
            .extract::<i64>()?;
        return Ok((i128::from(value), unit));
    }

    // Every native timedelta is an exact multiple of its microsecond resolution.
    let timedelta = PyDelta::type_object(py);
    let microsecond = timedelta.getattr(intern!(py, "resolution"))?;
    let value = timedelta
        .call_method1(intern!(py, "__floordiv__"), (value, microsecond))?
        .extract::<i128>()?;
    Ok((value, TimeUnit::Microseconds))
}
