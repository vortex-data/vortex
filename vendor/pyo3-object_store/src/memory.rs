use std::sync::Arc;

use object_store::memory::InMemory;
use pyo3::intern;
use pyo3::prelude::*;
use pyo3::types::PyString;

/// A Python-facing wrapper around an [`InMemory`].
#[derive(Debug, Clone)]
#[pyclass(name = "MemoryStore", frozen, subclass, from_py_object)]
pub struct PyMemoryStore(Arc<InMemory>);

impl AsRef<Arc<InMemory>> for PyMemoryStore {
    fn as_ref(&self) -> &Arc<InMemory> {
        &self.0
    }
}

impl From<Arc<InMemory>> for PyMemoryStore {
    fn from(value: Arc<InMemory>) -> Self {
        Self(value)
    }
}

impl<'py> PyMemoryStore {
    /// Consume self and return the underlying [`InMemory`].
    pub fn into_inner(self) -> Arc<InMemory> {
        self.0
    }

    fn __repr__(&'py self, py: Python<'py>) -> &'py Bound<'py, PyString> {
        intern!(py, "MemoryStore")
    }
}

#[pymethods]
impl PyMemoryStore {
    #[new]
    fn py_new() -> Self {
        Self(Arc::new(InMemory::new()))
    }

    fn __eq__(slf: Py<Self>, other: &Bound<PyAny>) -> bool {
        // Two memory stores are equal only if they are the same object
        slf.is(other)
    }
}
