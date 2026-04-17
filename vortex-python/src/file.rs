// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::sync::Arc;

use arrow_array::RecordBatchReader;
use pyo3::exceptions::PyTypeError;
use pyo3::prelude::*;
use pyo3::types::PyList;
use pyo3_object_store::PyObjectStore;
use vortex::array::ArrayRef;
#[expect(deprecated)]
use vortex::array::ToCanonical;
use vortex::array::builtins::ArrayBuiltins;
use vortex::dtype::DType;
use vortex::dtype::FieldNames;
use vortex::dtype::Nullability::NonNullable;
use vortex::dtype::PType;
use vortex::error::VortexResult;
use vortex::expr::Expression;
use vortex::expr::root;
use vortex::expr::select;
use vortex::file::OpenOptionsSessionExt;
use vortex::file::VortexFile;
use vortex::layout::scan::scan_builder::ScanBuilder;
use vortex::layout::scan::split_by::SplitBy;
use vortex::layout::segments::MokaSegmentCache;

use crate::RUNTIME;
use crate::SESSION;
use crate::TOKIO_RUNTIME;
use crate::arrays::PyArrayRef;
use crate::arrow::IntoPyArrow;
use crate::dataset::PyVortexDataset;
use crate::dtype::PyDType;
use crate::error::PyVortexResult;
use crate::expr::PyExpr;
use crate::install_module;
use crate::iter::PyArrayIterator;
use crate::object_store::resolve::ResolvedStore;
use crate::object_store::resolve::resolve_store;
use crate::scan::PyRepeatedScan;

pub(crate) fn init(py: Python, parent: &Bound<PyModule>) -> PyResult<()> {
    let m = PyModule::new(py, "file")?;
    parent.add_submodule(&m)?;
    install_module("vortex._lib.file", &m)?;

    m.add_function(wrap_pyfunction!(open, &m)?)?;
    m.add_class::<PyVortexFile>()?;

    Ok(())
}

/// Open a Vortex file for reading.
///
/// Callers can optionally configure an object store to build from using one of the definitions
/// in the `vortex.store` crate.
#[pyfunction]
#[pyo3(signature = (path, *, store = None, without_segment_cache = false))]
pub fn open(
    py: Python,
    path: &str,
    store: Option<PyObjectStore>,
    without_segment_cache: bool,
) -> PyVortexResult<PyVortexFile> {
    let vxf = py.detach(|| {
        TOKIO_RUNTIME.block_on(async move {
            let mut options = SESSION.open_options();
            if !without_segment_cache {
                // TODO(ngates): use a globally shared segment cache for all files
                options = options.with_segment_cache(Arc::new(MokaSegmentCache::new(256 << 20)));
            }

            match resolve_store(path, store.map(|x| x.into_inner()))? {
                ResolvedStore::ObjectStore(store, path) => {
                    options.open_object_store(&store, path.as_ref()).await
                }
                ResolvedStore::Path(path) => options.open_path(path).await,
            }
        })
    })?;

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

    #[pyo3(signature = (projection = None, *, expr = None, limit = None, indices = None, batch_size = None))]
    fn scan(
        slf: Bound<Self>,
        projection: Option<PyIntoProjection>,
        expr: Option<PyExpr>,
        limit: Option<u64>,
        indices: Option<PyArrayRef>,
        batch_size: Option<usize>,
    ) -> PyVortexResult<PyArrayIterator> {
        let builder = slf.get().scan_builder(
            projection.map(|p| p.0),
            expr.map(|e| e.into_inner()),
            limit,
            indices.map(|i| i.into_inner()),
            batch_size,
        )?;

        Ok(PyArrayIterator::new(Box::new(
            builder.into_array_iter(&*RUNTIME)?,
        )))
    }

    #[pyo3(signature = (projection = None, *, expr = None, limit = None, indices = None, batch_size = None))]
    fn prepare(
        slf: Bound<Self>,
        projection: Option<PyIntoProjection>,
        expr: Option<PyExpr>,
        limit: Option<u64>,
        indices: Option<PyArrayRef>,
        batch_size: Option<usize>,
    ) -> PyVortexResult<PyRepeatedScan> {
        let builder = slf.get().scan_builder(
            projection.map(|p| p.0),
            expr.map(|e| e.into_inner()),
            limit,
            indices.map(|i| i.into_inner()),
            batch_size,
        )?;

        let scan = builder.prepare()?;

        Ok(PyRepeatedScan {
            scan,
            row_count: slf.get().vxf.row_count(),
        })
    }

    #[pyo3(signature = (projection = None, *, expr = None, limit = None, batch_size = None))]
    fn to_arrow(
        slf: Bound<Self>,
        projection: Option<PyIntoProjection>,
        expr: Option<PyExpr>,
        limit: Option<u64>,
        batch_size: Option<usize>,
    ) -> PyVortexResult<Py<PyAny>> {
        let vxf = slf.get().vxf.clone();

        let reader = slf.py().detach(|| {
            let mut builder = vxf
                .scan()?
                .with_some_filter(expr.map(|e| e.into_inner()))
                .with_projection(projection.map(|p| p.0).unwrap_or_else(root));

            if let Some(limit) = limit {
                builder = builder.with_limit(limit);
            }

            if let Some(batch_size) = batch_size {
                builder = builder.with_split_by(SplitBy::RowCount(batch_size));
            }

            let schema = Arc::new(builder.dtype()?.to_arrow_schema()?);
            builder.into_record_batch_reader(schema, &*RUNTIME)
        })?;

        let rbr: Box<dyn RecordBatchReader + Send> = Box::new(reader);
        Ok(rbr.into_pyarrow(slf.py())?)
    }

    fn to_dataset(slf: Bound<Self>) -> PyVortexResult<PyVortexDataset> {
        Ok(PyVortexDataset::try_new(slf.get().vxf.clone())?)
    }

    #[pyo3(signature = (*))]
    pub fn splits(&self) -> PyVortexResult<Vec<(u64, u64)>> {
        Ok(self
            .vxf
            .splits()?
            .into_iter()
            .map(|x| (x.start, x.end))
            .collect())
    }
}

impl PyVortexFile {
    fn scan_builder(
        &self,
        projection: Option<Expression>,
        expr: Option<Expression>,
        limit: Option<u64>,
        indices: Option<ArrayRef>,
        batch_size: Option<usize>,
    ) -> VortexResult<ScanBuilder<ArrayRef>> {
        let mut builder = self
            .vxf
            .scan()?
            .with_some_filter(expr)
            .with_projection(projection.unwrap_or_else(root));

        if let Some(limit) = limit {
            builder = builder.with_limit(limit);
        }

        if let Some(indices) = indices {
            let casted = indices.cast(DType::Primitive(PType::U64, NonNullable))?;
            #[expect(deprecated)]
            let indices = casted.to_primitive().into_buffer::<u64>();
            builder = builder.with_row_indices(indices);
        }

        if let Some(batch_size) = batch_size {
            builder = builder.with_split_by(SplitBy::RowCount(batch_size));
        }

        Ok(builder)
    }
}

pub struct PyIntoProjection(Expression);

impl<'py> FromPyObject<'_, 'py> for PyIntoProjection {
    type Error = PyErr;

    fn extract(ob: Borrowed<'_, 'py, PyAny>) -> Result<Self, Self::Error> {
        // If it's a list of strings, convert to a column selection.
        if let Ok(py_list) = ob.cast::<PyList>() {
            let cols = py_list
                .iter()
                .map(|item| item.extract::<String>())
                .collect::<PyResult<Vec<String>>>()?;
            return Ok(PyIntoProjection(select(
                cols.into_iter().collect::<FieldNames>(),
                root(),
            )));
        }

        // If it's an expression, just return it.
        if let Ok(py_expr) = ob.cast::<PyExpr>() {
            return Ok(PyIntoProjection(py_expr.get().inner().clone()));
        }

        Err(PyTypeError::new_err(
            "projection must be a list of strings or a vortex.Expr",
        ))
    }
}
