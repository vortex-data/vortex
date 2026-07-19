use std::time::Duration;

use humantime::format_duration;
use pyo3::prelude::*;

/// A wrapper around `String` used to store values for config values.
///
/// Supported Python input:
///
/// - `True` and `False` (becomes `"true"` and `"false"`)
/// - `timedelta`
/// - `str`
#[derive(Clone, Debug, PartialEq, Eq, Hash, IntoPyObject, IntoPyObjectRef)]
pub struct PyConfigValue(pub String);

impl PyConfigValue {
    pub(crate) fn new(val: impl Into<String>) -> Self {
        Self(val.into())
    }
}

impl AsRef<str> for PyConfigValue {
    fn as_ref(&self) -> &str {
        &self.0
    }
}

impl<'py> FromPyObject<'_, 'py> for PyConfigValue {
    type Error = PyErr;

    fn extract(obj: Borrowed<'_, 'py, pyo3::PyAny>) -> PyResult<Self> {
        if let Ok(val) = obj.extract::<bool>() {
            Ok(val.into())
        } else if let Ok(duration) = obj.extract::<Duration>() {
            Ok(duration.into())
        } else {
            Ok(Self(obj.extract()?))
        }
    }
}

impl From<PyConfigValue> for String {
    fn from(value: PyConfigValue) -> Self {
        value.0
    }
}

impl From<bool> for PyConfigValue {
    fn from(value: bool) -> Self {
        Self(value.to_string())
    }
}

impl From<Duration> for PyConfigValue {
    fn from(value: Duration) -> Self {
        Self(format_duration(value).to_string())
    }
}
