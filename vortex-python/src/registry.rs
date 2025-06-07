use std::sync::Arc;

use itertools::Itertools;
use parking_lot::RwLock;
use pyo3::prelude::*;
use pyo3::{Bound, PyResult, Python};
use vortex::ArrayRegistryBuilder;
use vortex::file::ArrayRegistryExt;

use crate::arrays::py::PythonEncoding;
use crate::install_module;
use crate::serde::context::PyArrayContext;

/// Register serde functions and classes.
pub(crate) fn init(py: Python, parent: &Bound<PyModule>) -> PyResult<()> {
    let m = PyModule::new(py, "registry")?;
    parent.add_submodule(&m)?;
    install_module("vortex._lib.registry", &m)?;

    m.add_class::<PyRegistry>()?;

    Ok(())
}

/// Return the default Python registry.
#[allow(dead_code)]
pub(crate) fn default_registry(py: Python) -> PyResult<Bound<PyRegistry>> {
    Ok(py
        .import("vortex")?
        .getattr("registry")?
        .downcast_into::<PyRegistry>()?)
}

/// A register of known array and layout encodings.
#[pyclass(name = "Registry", module = "vortex", frozen)]
pub(crate) struct PyRegistry {
    array_registry: Arc<RwLock<ArrayRegistryBuilder>>,
}

#[pymethods]
impl PyRegistry {
    #[new]
    fn new() -> Self {
        Self {
            array_registry: Arc::new(RwLock::new(ArrayRegistryBuilder::full())),
        }
    }

    /// Register an array encoding implemented by subclassing `PyArray`.
    ///
    /// It's not currently possible to register a layout encoding from Python.
    pub(crate) fn register(&self, cls: PythonEncoding) -> PyResult<()> {
        let encoding = cls.to_encoding();
        let mut handle = self.array_registry.write();
        let x = &mut *handle;
        let mut builder = std::mem::take(&mut *handle);
        let builder = builder.register(encoding);

        Ok(())
    }

    /// Create an :class:`~vortex.ArrayContext` containing the given encodings.
    fn array_ctx(&self, encodings: Vec<Bound<PyAny>>) -> PyResult<PyArrayContext> {
        let registry = self.array_registry.read();
        Ok(PyArrayContext::from(registry.new_context(
            encoding_ids(&encodings)?.iter().map(|s| s.as_str()),
        )?))
    }
}

fn encoding_ids(objects: &[Bound<PyAny>]) -> PyResult<Vec<String>> {
    objects
        .iter()
        .map(|e| {
            // Try to extract the "id" attribute from the encoding class.
            if e.hasattr("id")? {
                e.getattr("id")?.extract::<String>()
            } else {
                // Otherwise, we assume it's a string
                e.extract::<String>()
            }
        })
        .try_collect()
}
