use std::sync::Arc;

use arrow::array::RecordBatchReader;
use arrow::pyarrow::IntoPyArrow;
use pyo3::prelude::*;
use vortex::expr::{ident, select};
use vortex::file::{GenericVortexFile, SplitBy, VortexFile, VortexOpenOptions};
use vortex::io::TokioFile;
use vortex::stream::ArrayStreamExt;

use crate::dtype::PyDType;
use crate::expr::PyExpr;
use crate::iter::{ArrayStreamToIterator, PyArrayIterator};
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
    fn __len__(slf: PyRef<Self>) -> PyResult<usize> {
        Ok(usize::try_from(slf.vxf.row_count())?)
    }

    /// The dtype of the file.
    #[getter]
    fn dtype(slf: Bound<Self>) -> PyResult<Bound<PyDType>> {
        PyDType::init(slf.py(), slf.get().vxf.dtype().clone())
    }

    /// Scan the Vortex file returning a :class:`vortex.ArrayIterator`.
    #[pyo3(signature = (projection = None, *, expr = None, batch_size = None))]
    fn scan(
        slf: Bound<Self>,
        projection: Option<PyExpr>,
        expr: Option<PyExpr>,
        batch_size: Option<usize>,
    ) -> PyResult<PyArrayIterator> {
        let mut builder = slf
            .get()
            .vxf
            .scan()
            .with_some_filter(expr.map(|e| e.into_inner()))
            .with_projection(
                projection
                    .map(|e| e.into_inner())
                    .unwrap_or_else(|| ident()),
            );

        if let Some(batch_size) = batch_size {
            builder = builder.with_split_by(SplitBy::RowCount(batch_size));
        }

        let iter = ArrayStreamToIterator::new(ArrayStreamExt::boxed(
            builder.build()?.into_array_stream()?,
        ));
        Ok(PyArrayIterator::new(Box::new(iter)))
    }

    /// Scan the Vortex file as a :class:`pyarrow.RecordBatchReader`.
    // TODO(ngates): columns should instead be a projection expression
    #[pyo3(signature = (columns = None, *, expr = None, batch_size = None))]
    fn to_arrow(
        slf: Bound<Self>,
        columns: Option<Vec<String>>,
        expr: Option<PyExpr>,
        batch_size: Option<usize>,
    ) -> PyResult<PyObject> {
        let mut builder = slf
            .get()
            .vxf
            .scan()
            .with_canonicalize(true)
            .with_some_filter(expr.map(|e| e.into_inner()))
            .with_projection(
                columns
                    .map(|cols| {
                        select(
                            cols.into_iter()
                                .map(|s| s.into())
                                .collect::<Vec<Arc<str>>>(),
                            ident(),
                        )
                    })
                    .unwrap_or_else(|| ident()),
            );

        if let Some(batch_size) = batch_size {
            builder = builder.with_split_by(SplitBy::RowCount(batch_size));
        }

        let iter = ArrayStreamToIterator::new(ArrayStreamExt::boxed(
            builder.build()?.into_array_stream()?,
        ));
        let rbr: Box<dyn RecordBatchReader + Send> =
            Box::new(VortexRecordBatchReader::try_new(iter)?);
        rbr.into_pyarrow(slf.py())
    }
}
