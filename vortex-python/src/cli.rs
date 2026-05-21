// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::sync::LazyLock;

use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;
use tokio::runtime::Runtime;
use vortex::VortexSessionDefault;
use vortex::error::VortexError;
use vortex::error::VortexExpect as _;
use vortex::io::runtime::BlockingRuntime;
use vortex::io::runtime::tokio::TokioRuntime;
use vortex::io::session::RuntimeSessionExt;
use vortex::session::VortexSession;

use crate::install_module;

static TOKIO_RUNTIME: LazyLock<Runtime> = LazyLock::new(|| {
    Runtime::new()
        .map_err(VortexError::from)
        .vortex_expect("tokio runtime must not fail to start")
});
static TUI_RUNTIME: LazyLock<TokioRuntime> =
    LazyLock::new(|| TokioRuntime::new(TOKIO_RUNTIME.handle().clone()));
static TUI_SESSION: LazyLock<VortexSession> =
    LazyLock::new(|| VortexSession::default().with_handle(TUI_RUNTIME.handle()));

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
    py.detach(|| TOKIO_RUNTIME.block_on(vortex_tui::launch_from(&TUI_SESSION, args)))
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))
}
