// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::sync::Arc;

use arrow_array::RecordBatchReader;
use pyo3::exceptions::PyTypeError;
use pyo3::prelude::*;
use pyo3::types::PyList;
use vortex::compute::cast;
use vortex::dtype::Nullability::NonNullable;
use vortex::dtype::{DType, PType};
use vortex::error::VortexResult;
use vortex::expr::{ExprRef, root, select};
use vortex::file::segments::MokaSegmentCache;
use vortex::file::{VortexFile, VortexOpenOptions};
use vortex::scan::{ScanBuilder, SplitBy};
use vortex::{ArrayRef, ToCanonical};

use crate::arrays::PyArrayRef;
use crate::arrow::IntoPyArrow;
use crate::dataset::PyVortexDataset;
use crate::dtype::PyDType;
use crate::expr::PyExpr;
use crate::install_module;
use crate::iter::PyArrayIterator;
use crate::scan::PyRepeatedScan;

pub(crate) fn init(py: Python, parent: &Bound<PyModule>) -> PyResult<()> {
    let m = PyModule::new(py, "file")?;
    parent.add_submodule(&m)?;
    install_module("vortex._lib.file", &m)?;

    m.add_function(wrap_pyfunction!(open, &m)?)?;
    m.add_class::<PyVortexFile>()?;

    Ok(())
}

#[pyfunction]
#[pyo3(signature = (path, *, without_segment_cache = false))]
pub fn open(path: &str, without_segment_cache: bool) -> PyResult<PyVortexFile> {
    let mut options = VortexOpenOptions::file();
    if without_segment_cache {
        options = options.without_segment_cache();
    } else {
        // TODO(ngates): use a globally shared segment cache for all files
        options = options.with_segment_cache(Arc::new(MokaSegmentCache::new(256 << 20)));
    }

    let vxf = options.open_blocking(path)?;
    Ok(PyVortexFile { vxf })
}

#[pyclass(name = "VortexFile", module = "vortex", frozen)]
pub struct PyVortexFile {
    vxf: VortexFile,
}

#[pymethods]
impl PyVortexFile {
    fn __len__(slf: PyRef<Self>) -> PyResult<usize> {
        Ok(usize::try_from(slf.vxf.row_count())?)
    }

    #[getter]
    fn dtype(slf: Bound<Self>) -> PyResult<Bound<PyDType>> {
        PyDType::init(slf.py(), slf.get().vxf.dtype().clone())
    }

    #[pyo3(signature = (projection = None, *, expr = None, indices = None, batch_size = None))]
    fn scan(
        slf: Bound<Self>,
        projection: Option<PyIntoProjection>,
        expr: Option<PyExpr>,
        indices: Option<PyArrayRef>,
        batch_size: Option<usize>,
    ) -> PyResult<PyArrayIterator> {
        let builder = slf.get().scan_builder(
            projection.map(|p| p.0),
            expr.map(|e| e.into_inner()),
            indices.map(|i| i.into_inner()),
            batch_size,
        )?;

        Ok(PyArrayIterator::new(Box::new(
            builder.into_array_iter_multithread()?,
        )))
    }

    #[pyo3(signature = (projection = None, *, expr = None, indices = None, batch_size = None))]
    fn prepare(
        slf: Bound<Self>,
        projection: Option<PyIntoProjection>,
        expr: Option<PyExpr>,
        indices: Option<PyArrayRef>,
        batch_size: Option<usize>,
    ) -> PyResult<PyRepeatedScan> {
        let builder = slf.get().scan_builder(
            projection.map(|p| p.0),
            expr.map(|e| e.into_inner()),
            indices.map(|i| i.into_inner()),
            batch_size,
        )?;

        let scan = builder.prepare()?;

        Ok(PyRepeatedScan {
            scan,
            row_count: slf.get().vxf.row_count(),
        })
    }

    #[pyo3(signature = (projection = None, *, expr = None, batch_size = None))]
    fn to_arrow(
        slf: Bound<Self>,
        projection: Option<PyIntoProjection>,
        expr: Option<PyExpr>,
        batch_size: Option<usize>,
    ) -> PyResult<PyObject> {
        let vxf = slf.get().vxf.clone();

        let reader = slf.py().allow_threads(|| {
            let mut builder = vxf
                .scan()?
                .with_some_filter(expr.map(|e| e.into_inner()))
                .with_projection(projection.map(|p| p.0).unwrap_or_else(root));

            if let Some(batch_size) = batch_size {
                builder = builder.with_split_by(SplitBy::RowCount(batch_size));
            }

            let schema = Arc::new(builder.dtype()?.to_arrow_schema()?);
            builder.into_record_batch_reader_multithread(schema)
        })?;

        let rbr: Box<dyn RecordBatchReader + Send> = Box::new(reader);
        rbr.into_pyarrow(slf.py())
    }

    fn to_dataset(slf: Bound<Self>) -> PyResult<PyVortexDataset> {
        Ok(PyVortexDataset::try_new(slf.get().vxf.clone())?)
    }
}

impl PyVortexFile {
    fn scan_builder(
        &self,
        projection: Option<ExprRef>,
        expr: Option<ExprRef>,
        indices: Option<ArrayRef>,
        batch_size: Option<usize>,
    ) -> VortexResult<ScanBuilder<ArrayRef>> {
        let mut builder = self
            .vxf
            .scan()?
            .with_some_filter(expr)
            .with_projection(projection.unwrap_or_else(root));

        if let Some(indices) = indices {
            let indices = cast(indices.as_ref(), &DType::Primitive(PType::U64, NonNullable))?
                .to_primitive()
                .into_buffer::<u64>();
            builder = builder.with_row_indices(indices);
        }

        if let Some(batch_size) = batch_size {
            builder = builder.with_split_by(SplitBy::RowCount(batch_size));
        }

        Ok(builder)
    }
}

pub struct PyIntoProjection(ExprRef);

impl<'py> FromPyObject<'py> for PyIntoProjection {
    fn extract_bound(ob: &Bound<'py, PyAny>) -> PyResult<Self> {
        // If it's a list of strings, convert to a column selection.
        if let Ok(py_list) = ob.downcast::<PyList>() {
            let cols = py_list
                .iter()
                .map(|item| item.extract::<String>())
                .collect::<PyResult<Vec<String>>>()?;
            return Ok(PyIntoProjection(select(
                cols.into_iter().map(Arc::<str>::from).collect::<Vec<_>>(),
                root(),
            )));
        }

        // If it's an expression, just return it.
        if let Ok(py_expr) = ob.downcast::<PyExpr>() {
            return Ok(PyIntoProjection(py_expr.get().inner().clone()));
        }

        Err(PyTypeError::new_err(
            "projection must be a list of strings or a vortex.Expr",
        ))
    }
}
