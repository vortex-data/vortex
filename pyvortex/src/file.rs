use arrow::array::RecordBatchReader;
use arrow::pyarrow::IntoPyArrow;
use pyo3::prelude::*;
use vortex::arrow::infer_schema;
use vortex::file::{GenericVortexFile, VortexFile, VortexOpenOptions};
use vortex::io::TokioFile;

use crate::dtype::PyDType;
use crate::record_batch_reader::VortexRecordBatchReader;
use crate::{TOKIO_RUNTIME, install_module};

pub(crate) fn init(py: Python, parent: &Bound<PyModule>) -> PyResult<()> {
    let m = PyModule::new(py, "file")?;
    parent.add_submodule(&m)?;
    install_module("vortex._lib.file", &m)?;

    m.add_function(wrap_pyfunction!(open, &m)?)?;
    m.add_class::<PyVortexFile>()?;

    Ok(())
}

#[pyfunction]
pub fn open(path: &str) -> PyResult<PyVortexFile> {
    let vxf = TOKIO_RUNTIME.block_on(VortexOpenOptions::file(TokioFile::open(path)?).open())?;
    Ok(PyVortexFile { vxf })
}

#[pyclass(name = "VortexFile", module = "vortex", frozen)]
pub struct PyVortexFile {
    vxf: VortexFile<GenericVortexFile<TokioFile>>,
}

#[pymethods]
impl PyVortexFile {
    /// The dtype of the file.
    #[getter]
    fn dtype(slf: Bound<Self>) -> PyResult<Bound<PyDType>> {
        PyDType::init(slf.py(), slf.get().vxf.dtype().clone())
    }

    /// Scan the vortex file as a :class:`pyarrow.RecordBatchReader`.
    fn to_arrow(slf: Bound<Self>) -> PyResult<PyObject> {
        let stream = slf
            .get()
            .vxf
            .scan()
            .with_canonicalize(true)
            .build()?
            .into_array_stream()?;

        let rbr: Box<dyn RecordBatchReader + Send> =
            Box::new(VortexRecordBatchReader::try_new(stream, &*TOKIO_RUNTIME)?);
        rbr.into_pyarrow(slf.py())
    }

    /// Returns a :class:`polars.LazyFrame` that reads the file.
    fn to_polars(slf: Bound<Self>) -> PyResult<Bound<PyAny>> {
        let _scan = slf.get().vxf.scan();
        let _schema = infer_schema(slf.get().vxf.dtype())?;
        todo!()
        //
        // // An IO source is a callable that returns an Iterator of pl.DataFrame.
        // let io_source = Bound::new(slf.py(), PyVortexPolarsSource { scan_builder })?;
        // let schema = pyo3_polars::PySchema(Arc::new(schema));
        //
        // let plugins = slf.py().import("polars.io.plugins")?;
        // plugins.call_method("register_io_source", (io_source, schema), None)
    }
}
