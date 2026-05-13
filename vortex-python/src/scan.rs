// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::sync::Arc;

use pyo3::exceptions::PyIndexError;
use pyo3::prelude::*;
use vortex::array::ArrayRef;
use vortex::array::VortexSessionExecute;
use vortex::error::VortexResult;
use vortex::layout::scan::repeated_scan::RepeatedScan;
use vortex::scalar::Scalar;

use crate::RUNTIME;
use crate::error::PyVortexResult;
use crate::install_module;
use crate::iter::PyArrayIterator;
use crate::scalar::PyScalar;
use crate::session::session;

pub(crate) fn init(py: Python, parent: &Bound<PyModule>) -> PyResult<()> {
    let m = PyModule::new(py, "scan")?;
    parent.add_submodule(&m)?;
    install_module("vortex._lib.scan", &m)?;

    m.add_class::<PyRepeatedScan>()?;

    Ok(())
}

#[pyclass(name = "RepeatedScan", module = "vortex", frozen)]
pub struct PyRepeatedScan {
    pub scan: Arc<RepeatedScan<ArrayRef>>,
    pub row_count: u64,
}

#[pymethods]
impl PyRepeatedScan {
    #[pyo3(signature = (*, start = None, stop = None))]
    fn execute(
        slf: Bound<Self>,
        start: Option<u64>,
        stop: Option<u64>,
    ) -> PyVortexResult<PyArrayIterator> {
        let row_count = slf.get().row_count;
        let row_range = match (start, stop) {
            (Some(start), Some(stop)) => Some(start..stop),
            (Some(start), None) => Some(start..row_count),
            (None, Some(stop)) => Some(0..stop),
            (None, None) => None,
        };

        let scan = Arc::clone(&slf.get().scan);
        slf.py().detach(move || {
            Ok(PyArrayIterator::new(Box::new(
                scan.execute_array_iter(row_range, &*RUNTIME)?,
            )))
        })
    }

    fn scalar_at(slf: Bound<Self>, index: u64) -> PyVortexResult<Bound<PyScalar>> {
        let row_count = slf.get().row_count;
        if index >= row_count {
            return Err(PyIndexError::new_err(format!(
                "Index out of bounds: {} >= {}",
                index, row_count
            ))
            .into());
        }

        let scan = Arc::clone(&slf.get().scan);
        let scalar = slf.py().detach(move || -> VortexResult<Option<Scalar>> {
            let session = session();
            for batch in scan.execute_array_iter(Some(index..index + 1), &*RUNTIME)? {
                let array = batch?;
                if array.is_empty() {
                    continue;
                }
                let scalar = array.execute_scalar(0, &mut session.create_execution_ctx())?;
                return Ok(Some(scalar));
            }

            Ok(None)
        })?;

        match scalar {
            Some(scalar) => Ok(PyScalar::init(slf.py(), scalar)?),
            None => {
                Err(PyIndexError::new_err(format!("Index {} not found in the scan", index)).into())
            }
        }
    }
}
