// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! PyO3 bindings for object store registry functions.

use pyo3::prelude::*;

/// Add the `_store` module to the path for downstream consumers to use.
pub(crate) fn init(py: Python, parent: &Bound<PyModule>) -> PyResult<()> {
    pyo3_object_store::register_store_module(py, parent, "vortex._lib", "store")?;
    pyo3_object_store::register_exceptions_module(py, parent, "vortex._lib", "exceptions")?;

    Ok(())
}
