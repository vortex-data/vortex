// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::sync::Arc;

use pyo3::prelude::*;
use vortex::expr::root;
use vortex::file::{VortexFile, VortexOpenOptions};
use vortex::layout::LayoutReader;
use vortex::scan::ScanBuilder;
use vortex::{Array};

use crate::file::PyIntoProjection;
use crate::install_module;
use crate::scalar::PyScalar;

pub(crate) fn init(py: Python, parent: &Bound<PyModule>) -> PyResult<()> {
    let m = PyModule::new(py, "scan")?;
    parent.add_submodule(&m)?;
    install_module("vortex._lib.scan", &m)?;

    m.add_function(wrap_pyfunction!(open, &m)?)?;
    m.add_class::<PyVortexScan>()?;

    Ok(())
}

#[pyfunction]
pub fn open(path: &str) -> PyResult<PyVortexScan> {
    let vxf = VortexOpenOptions::file()
        .without_segment_cache()
        .open_blocking(path)?;
    let layout_reader = vxf.layout_reader()?;
    Ok(PyVortexScan { vxf, layout_reader })
}

#[pyclass(name = "VortexScan", module = "vortex", frozen)]
pub struct PyVortexScan {
    vxf: VortexFile,
    layout_reader: Arc<dyn LayoutReader>,
}

#[pymethods]
impl PyVortexScan {
    fn __len__(slf: PyRef<Self>) -> PyResult<usize> {
        Ok(usize::try_from(slf.vxf.row_count())?)
    }

    #[pyo3(signature = (idx, projection = None))]
    fn scalar_at(slf: Bound<Self>, idx: usize, projection: Option<PyIntoProjection>,) -> PyResult<Bound<PyScalar>> {
        let mut scan = ScanBuilder::new(slf.get().layout_reader.clone())
            .with_projection(projection.map(|p| p.0).unwrap_or_else(root))
            .with_row_range((idx as u64)..(idx as u64 + 1))
            .into_array_iter_multithread()?;
        let next_batch = scan.next().expect("Index out of bounds")?;
        let scalar = next_batch.scalar_at(0);

        PyScalar::init(slf.py(), scalar)
    }
}
