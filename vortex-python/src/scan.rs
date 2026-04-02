// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use pyo3::exceptions::PyIndexError;
use pyo3::prelude::*;
use vortex::array::ArrayRef;
use vortex::layout::scan::repeated_scan::RepeatedScan;

use crate::RUNTIME;
use crate::error::PyVortexResult;
use crate::install_module;
use crate::iter::PyArrayIterator;
use crate::scalar::PyScalar;

pub(crate) fn init(py: Python, parent: &Bound<PyModule>) -> PyResult<()> {
    let m = PyModule::new(py, "scan")?;
    parent.add_submodule(&m)?;
    install_module("vortex._lib.scan", &m)?;

    m.add_class::<PyRepeatedScan>()?;

    Ok(())
}

#[pyclass(name = "RepeatedScan", module = "vortex", frozen)]
pub struct PyRepeatedScan {
    pub scan: RepeatedScan<ArrayRef>,
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

        Ok(PyArrayIterator::new(Box::new(
            slf.get().scan.execute_array_iter(row_range, &*RUNTIME)?,
        )))
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

        for batch in slf
            .get()
            .scan
            .execute_array_iter(Some(index..index + 1), &*RUNTIME)?
        {
            let array = batch?;
            if array.is_empty() {
                continue;
            }
            let scalar = array.scalar_at(0)?;
            return Ok(PyScalar::init(slf.py(), scalar)?);
        }

        Err(PyIndexError::new_err(format!("Index {} not found in the scan", index)).into())
    }
}
