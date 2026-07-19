use object_store::path::Path;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::pybacked::PyBackedStr;
use pyo3::types::PyString;

/// A Python-facing wrapper around a [`Path`].
#[derive(Clone, Debug, Default, PartialEq)]
pub struct PyPath(Path);

impl<'py> FromPyObject<'_, 'py> for PyPath {
    type Error = PyErr;

    fn extract(obj: Borrowed<'_, 'py, PyAny>) -> Result<Self, Self::Error> {
        let path = Path::parse(obj.extract::<PyBackedStr>()?)
            .map_err(|err| PyValueError::new_err(format!("Could not parse path: {err}")))?;
        Ok(Self(path))
    }
}

impl PyPath {
    /// Consume self and return the underlying [`Path`].
    pub fn into_inner(self) -> Path {
        self.0
    }
}

impl<'py> IntoPyObject<'py> for PyPath {
    type Target = PyString;
    type Output = Bound<'py, PyString>;
    type Error = PyErr;

    fn into_pyobject(self, py: Python<'py>) -> Result<Self::Output, Self::Error> {
        Ok(PyString::new(py, self.0.as_ref()))
    }
}

impl<'py> IntoPyObject<'py> for &PyPath {
    type Target = PyString;
    type Output = Bound<'py, PyString>;
    type Error = PyErr;

    fn into_pyobject(self, py: Python<'py>) -> Result<Self::Output, Self::Error> {
        Ok(PyString::new(py, self.0.as_ref()))
    }
}

impl AsRef<Path> for PyPath {
    fn as_ref(&self) -> &Path {
        &self.0
    }
}

impl From<PyPath> for Path {
    fn from(value: PyPath) -> Self {
        value.0
    }
}

impl From<Path> for PyPath {
    fn from(value: Path) -> Self {
        Self(value)
    }
}
