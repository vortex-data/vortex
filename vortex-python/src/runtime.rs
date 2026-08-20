// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use crate::install_module;
use crate::set_requested_workers;
use crate::with_pool;

pub(crate) fn init(py: Python, parent: &Bound<PyModule>) -> PyResult<()> {
    let m = PyModule::new(py, "runtime")?;
    parent.add_submodule(&m)?;
    install_module("vortex._lib.runtime", &m)?;

    m.add_function(wrap_pyfunction!(set_worker_threads, &m)?)?;
    m.add_function(wrap_pyfunction!(worker_threads, &m)?)?;

    Ok(())
}

/// Set the number of background worker threads driving Vortex futures.
///
/// If `n` is `None`, resets the pool to `available_parallelism() - 1`.
#[pyfunction]
#[pyo3(signature = (n=None))]
pub fn set_worker_threads(n: Option<isize>) -> PyResult<()> {
    if let Some(n) = n
        && n < 0
    {
        return Err(PyValueError::new_err(
            "worker thread count must be non-negative",
        ));
    }
    let workers = with_pool(|pool| {
        match n {
            Some(n) => pool.set_workers(n as usize),
            None => pool.set_workers_to_available_parallelism(),
        }
        pool.worker_count()
    });
    // Remember the request so that a forked child rebuilds its pool with the same size.
    set_requested_workers(workers);
    Ok(())
}

/// Return the current number of background worker threads.
#[pyfunction]
pub fn worker_threads() -> usize {
    with_pool(|pool| pool.worker_count())
}
