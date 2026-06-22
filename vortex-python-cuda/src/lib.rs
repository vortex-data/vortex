// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Optional CUDA extension for PyVortex.
//!
//! Builds the separate `vortex-data-cuda` wheel (imported as `vortex_cuda`), installed alongside
//! the CPU-only `vortex-data` wheel. Keeping CUDA in its own extension keeps the base wheel free of
//! CUDA build/runtime dependencies; `vortex.cuda_extension_installed()` reports whether it is present.

use pyo3::prelude::*;

/// Return whether a usable CUDA device is available in the current process.
///
/// This performs a runtime probe of the CUDA driver and device. It differs from
/// `vortex.cuda_extension_installed()`, which only reports whether this extension package is
/// installed.
#[pyfunction]
fn cuda_available() -> bool {
    vortex_cuda::cuda_available()
}

/// The `vortex_cuda._lib` extension module.
#[cfg(feature = "extension-module")]
#[pymodule]
fn _lib(m: &Bound<PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(cuda_available, m)?)?;
    Ok(())
}
