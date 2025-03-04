use std::ops::Deref;

use pyo3::{Bound, PyRef, PyResult, pyclass, pymethods};
use vortex::file::{GenericScanDriver, GenericVortexFile, Scan, VortexFile};
use vortex::io::TokioFile;

use crate::dtype::PyDType;

#[pyclass(name = "VortexFile", module = "vortex")]
pub struct PyVortexFile {
    vxf: VortexFile<GenericVortexFile<TokioFile>>,
}

#[pymethods]
impl PyVortexFile {
    /// The dtype of the file.
    fn dtype(slf: Bound<Self>) -> PyResult<Bound<PyDType>> {
        PyDType::init(slf.py(), slf.get().vxf.dtype().clone())
    }

    fn to_polars(slf: PyRef<Self>) -> PyResult<PyVortexPolarsSource> {
        let scan = slf.vxf.scan();
    }
}

#[pyclass(name = "VortexPolarsSource", module = "vortex")]
pub struct PyVortexPolarsSource {
    scan: Scan<GenericScanDriver<TokioFile>>,
}
