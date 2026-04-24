// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;

use crate::SESSION;
use crate::TOKIO_RUNTIME;
use crate::install_module;

pub(crate) fn init(py: Python, parent: &Bound<PyModule>) -> PyResult<()> {
    let m = PyModule::new(py, "cli")?;
    parent.add_submodule(&m)?;
    install_module("vortex._lib.cli", &m)?;

    m.add_function(wrap_pyfunction!(launch, &m)?)?;

    Ok(())
}

/// Launch the `vx` CLI with the given arguments.
///
/// Parameters
/// ----------
/// args : list[str]
///     Command-line arguments, typically ``sys.argv``.
#[pyfunction]
fn launch(py: Python, args: Vec<String>) -> PyResult<()> {
    py.detach(|| TOKIO_RUNTIME.block_on(vortex_tui::launch_from(&SESSION, args)))
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))
}
