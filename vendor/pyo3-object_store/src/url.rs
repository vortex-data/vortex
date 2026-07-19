use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::pybacked::PyBackedStr;
use pyo3::types::PyString;
use pyo3::FromPyObject;
use url::Url;

/// A wrapper around [`url::Url`] that implements [`FromPyObject`].
#[derive(Debug, Clone, PartialEq)]
pub struct PyUrl(Url);

impl PyUrl {
    /// Create a new PyUrl from a [Url]
    pub fn new(url: Url) -> Self {
        Self(url)
    }

    /// Consume self and return the underlying [Url]
    pub fn into_inner(self) -> Url {
        self.0
    }
}

impl<'py> FromPyObject<'_, 'py> for PyUrl {
    type Error = PyErr;

    fn extract(obj: Borrowed<'_, 'py, PyAny>) -> Result<Self, Self::Error> {
        let s = obj.extract::<PyBackedStr>()?;
        let url = Url::parse(&s).map_err(|err| PyValueError::new_err(err.to_string()))?;
        Ok(Self(url))
    }
}

impl<'py> IntoPyObject<'py> for PyUrl {
    type Target = PyString;
    type Output = Bound<'py, PyString>;
    type Error = std::convert::Infallible;

    fn into_pyobject(self, py: Python<'py>) -> Result<Self::Output, Self::Error> {
        Ok(PyString::new(py, self.0.as_str()))
    }
}

impl<'py> IntoPyObject<'py> for &PyUrl {
    type Target = PyString;
    type Output = Bound<'py, PyString>;
    type Error = std::convert::Infallible;

    fn into_pyobject(self, py: Python<'py>) -> Result<Self::Output, Self::Error> {
        Ok(PyString::new(py, self.0.as_str()))
    }
}

impl AsRef<Url> for PyUrl {
    fn as_ref(&self) -> &Url {
        &self.0
    }
}

impl From<PyUrl> for String {
    fn from(value: PyUrl) -> Self {
        value.0.into()
    }
}
