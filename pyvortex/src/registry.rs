use std::sync::{Arc, LazyLock, RwLock};

use pyo3::{PyResult, pyfunction};
use vortex::ArrayRegistry;
use vortex::arcref::ArcRef;
use vortex::error::VortexExpect;
use vortex::file::DEFAULT_REGISTRY;

use crate::arrays::py::PyEncodingClass;

static ARRAY_REGISTRY: LazyLock<Arc<RwLock<ArrayRegistry>>> = LazyLock::new(|| {
    // Set up a registry using the default encodings from vortex-file.
    let mut registry = ArrayRegistry::default();
    registry.register_many(DEFAULT_REGISTRY.vtables().cloned());
    Arc::new(RwLock::new(registry))
});

/// Register an array encoding implemented by subclassing `PyArray`.
#[pyfunction(name = "register")]
pub fn register(cls: PyEncodingClass) -> PyResult<()> {
    let encoding = ArcRef::new_arc(Arc::new(cls) as _);
    ARRAY_REGISTRY
        .write()
        .vortex_expect("poisoned lock")
        .register(encoding);
    Ok(())
}
