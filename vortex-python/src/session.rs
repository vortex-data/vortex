// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::ops::Deref;

use pyo3::prelude::*;
use vortex::VortexSessionDefault;
use vortex::array::ExecutionCtx;
use vortex::array::VortexSessionExecute;
use vortex::io::runtime::BlockingRuntime;
use vortex::io::session::RuntimeSessionExt;
use vortex::session::VortexSession;

use crate::RUNTIME;
use crate::install_module;

pub(crate) fn init(py: Python, parent: &Bound<PyModule>) -> PyResult<()> {
    let m = PyModule::new(py, "session")?;
    parent.add_submodule(&m)?;
    install_module("vortex._lib.session", &m)?;

    m.add_class::<PyVortexSession>()?;

    Ok(())
}

#[pyclass(name = "Session", module = "vortex", frozen)]
pub struct PyVortexSession {
    inner: VortexSession,
}

impl Default for PyVortexSession {
    fn default() -> Self {
        Self::new()
    }
}

impl From<VortexSession> for PyVortexSession {
    fn from(inner: VortexSession) -> Self {
        Self { inner }
    }
}

impl PyVortexSession {
    pub fn new() -> Self {
        Self {
            inner: VortexSession::default().with_handle(RUNTIME.handle()),
        }
    }

    pub fn inner(&self) -> &VortexSession {
        &self.inner
    }

    pub fn create_execution_ctx(&self) -> ExecutionCtx {
        self.inner.create_execution_ctx()
    }
}

impl Deref for PyVortexSession {
    type Target = VortexSession;

    fn deref(&self) -> &Self::Target {
        &self.inner
    }
}

#[pymethods]
impl PyVortexSession {
    #[new]
    fn py_new() -> Self {
        Self::new()
    }

    fn __repr__(&self) -> &'static str {
        "vortex.Session()"
    }
}
