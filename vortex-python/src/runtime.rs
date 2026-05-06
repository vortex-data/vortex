// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use crate::POOL;
use crate::install_module;

pub(crate) fn init(py: Python, parent: &Bound<PyModule>) -> PyResult<()> {
    let m = PyModule::new(py, "runtime")?;
    parent.add_submodule(&m)?;
    install_module("vortex._lib.runtime", &m)?;

    m.add_function(wrap_pyfunction!(set_worker_threads, &m)?)?;
    m.add_function(wrap_pyfunction!(
        set_worker_threads_to_available_parallelism,
        &m
    )?)?;
    m.add_function(wrap_pyfunction!(worker_count, &m)?)?;

    Ok(())
}

/// Set the number of background worker threads driving Vortex futures.
#[pyfunction]
pub fn set_worker_threads(n: isize) -> PyResult<()> {
    if n < 0 {
        return Err(PyValueError::new_err(
            "worker thread count must be non-negative",
        ));
    }
    POOL.set_workers(n as usize);
    Ok(())
}

/// Set the worker pool to `available_parallelism() - 1`.
#[pyfunction]
pub fn set_worker_threads_to_available_parallelism() {
    POOL.set_workers_to_available_parallelism();
}

/// Return the current number of background worker threads.
#[pyfunction]
pub fn worker_count() -> usize {
    POOL.worker_count()
}
