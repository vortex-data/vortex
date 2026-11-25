// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use pyo3::Bound;
use pyo3::IntoPyObject;
use pyo3::Py;
use pyo3::PyAny;
use pyo3::PyResult;
use pyo3::PyTypeInfo;
use pyo3::Python;
use pyo3::exceptions::PyOverflowError;
use pyo3::exceptions::PyValueError;
use pyo3::intern;
use pyo3::prelude::PyAnyMethods;
use pyo3::sync::PyOnceLock;
use pyo3::types::PyDelta;
use pyo3::types::PyType;
use vortex::dtype::DType;
use vortex::dtype::DurationUnit;
use vortex::dtype::Nullability;
use vortex::scalar::Scalar;

use crate::error::PyVortexResult;

const NANOS_PER_MICROSECOND: i128 = 1_000;
const NANOS_PER_SECOND: i128 = 1_000_000_000;
const NANOS_PER_DAY: i128 = 86_400 * NANOS_PER_SECOND;

fn nanos_per_unit(unit: DurationUnit) -> i128 {
    match unit {
        DurationUnit::Nanoseconds => 1,
        DurationUnit::Microseconds => NANOS_PER_MICROSECOND,
        DurationUnit::Milliseconds => 1_000_000,
        DurationUnit::Seconds => NANOS_PER_SECOND,
    }
}

pub(super) fn to_py<'py>(
    py: Python<'py>,
    scalar: &Scalar,
    unit: DurationUnit,
) -> PyResult<Bound<'py, PyAny>> {
    let Some(value) = scalar.value() else {
        return Ok(py.None().into_pyobject(py)?);
    };
    let value = value
        .as_primitive()
        .cast::<i64>()
        .map_err(|error| PyValueError::new_err(error.to_string()))?;
    if unit == DurationUnit::Nanoseconds {
        if value == i64::MIN {
            return Err(PyOverflowError::new_err(
                "pandas.Timedelta reserves i64::MIN nanoseconds for NaT",
            ));
        }
        static TIMEDELTA: PyOnceLock<Py<PyType>> = PyOnceLock::new();
        return TIMEDELTA
            .import(py, "pandas", "Timedelta")?
            .call1((value, "ns"));
    }

    let nanos = i128::from(value) * nanos_per_unit(unit);
    // Python normalizes the parts of a negative timedelta so only the days can be negative.
    let days = nanos.div_euclid(NANOS_PER_DAY);
    let nanos_within_day = nanos.rem_euclid(NANOS_PER_DAY);
    let seconds = nanos_within_day / NANOS_PER_SECOND;
    let microseconds = (nanos_within_day % NANOS_PER_SECOND) / NANOS_PER_MICROSECOND;
    PyDelta::type_object(py).call1((days, seconds, microseconds))
}

pub(super) fn from_py(value: &Bound<'_, PyDelta>, dtype: Option<&DType>) -> PyVortexResult<Scalar> {
    let py = value.py();
    let days: i64 = value.getattr(intern!(py, "days"))?.extract()?;
    let seconds: i64 = value.getattr(intern!(py, "seconds"))?.extract()?;
    let microseconds: i64 = value.getattr(intern!(py, "microseconds"))?.extract()?;
    // Timedelta subclasses such as pandas.Timedelta can carry a sub-microsecond remainder.
    let nanoseconds: i64 = value
        .getattr_opt(intern!(py, "nanoseconds"))?
        .map(|part| part.extract())
        .transpose()?
        .unwrap_or(0);
    let nanos = i128::from(days) * NANOS_PER_DAY
        + i128::from(seconds) * NANOS_PER_SECOND
        + i128::from(microseconds) * NANOS_PER_MICROSECOND
        + i128::from(nanoseconds);

    let unit = match dtype {
        Some(DType::Duration(unit, _)) => *unit,
        _ => match value
            .getattr_opt(intern!(py, "unit"))?
            .map(|unit| unit.extract::<String>())
            .transpose()?
            .as_deref()
        {
            Some("s") => DurationUnit::Seconds,
            Some("ms") => DurationUnit::Milliseconds,
            Some("us") => DurationUnit::Microseconds,
            Some("ns") => DurationUnit::Nanoseconds,
            _ if nanoseconds != 0 => DurationUnit::Nanoseconds,
            _ => DurationUnit::Microseconds,
        },
    };
    let per_unit = nanos_per_unit(unit);
    if nanos % per_unit != 0 {
        return Err(PyValueError::new_err(format!(
            "Timedelta cannot be represented in {unit} without losing precision"
        ))
        .into());
    }
    let storage = i64::try_from(nanos / per_unit)
        .map_err(|_| PyOverflowError::new_err(format!("Timedelta overflows duration[{unit}]")))?;
    Ok(Scalar::try_new(
        DType::Duration(unit, Nullability::NonNullable),
        Some(storage.into()),
    )?)
}
