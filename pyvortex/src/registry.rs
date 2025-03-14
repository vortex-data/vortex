use std::sync::{Arc, RwLock};

use itertools::Itertools;
use pyo3::prelude::*;
use pyo3::{Bound, PyResult, Python};
use vortex::ArrayRegistry;
use vortex::arcref::ArcRef;
use vortex::error::VortexExpect;
use vortex::file::DEFAULT_REGISTRY;
use vortex::layout::{LayoutRegistry, LayoutRegistryExt};

use crate::arrays::py::PyEncodingClass;
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
    array_registry: Arc<RwLock<ArrayRegistry>>,
    #[allow(dead_code)]
    layout_registry: Arc<RwLock<LayoutRegistry>>,
}

#[pymethods]
impl PyRegistry {
    #[new]
    fn new() -> Self {
        let mut array = ArrayRegistry::canonical_only();
        array.register_many(DEFAULT_REGISTRY.vtables().cloned());
        let layout = LayoutRegistry::default();
        Self {
            array_registry: Arc::new(RwLock::new(array)),
            layout_registry: Arc::new(RwLock::new(layout)),
        }
    }

    /// Register an array encoding implemented by subclassing `PyArray`.
    ///
    /// It's not currently possible to register a layout encoding from Python.
    pub(crate) fn register(&self, cls: PyEncodingClass) -> PyResult<()> {
        let encoding = ArcRef::new_arc(Arc::new(cls) as _);
        self.array_registry
            .write()
            .vortex_expect("poisoned lock")
            .register(encoding);
        Ok(())
    }

    /// Create an :class:`~vortex.ArrayContext` containing the given encodings.
    fn array_ctx(&self, encodings: Vec<Bound<PyAny>>) -> PyResult<PyArrayContext> {
        let registry = self.array_registry.read().vortex_expect("poisoned lock");
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
