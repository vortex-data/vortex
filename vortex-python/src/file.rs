// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::sync::Arc;

use arrow_array::RecordBatchReader;
use arrow_schema::Schema;
use pyo3::exceptions::PyTypeError;
use pyo3::intern;
use pyo3::prelude::*;
use pyo3::types::PyList;
use pyo3::types::PyString;
use vortex::array::ArrayRef;
use vortex::array::ExecutionCtx;
use vortex::array::VortexSessionExecute;
use vortex::array::arrays::PrimitiveArray;
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
use vortex::io::VortexReadAt;
use vortex::io::runtime::BlockingRuntime;
use vortex::io::session::RuntimeSessionExt;
use vortex::layout::scan::scan_builder::ScanBuilder;
use vortex::layout::scan::split_by::SplitBy;
use vortex::layout::segments::MokaSegmentCache;
use vortex::scan::strict_sorted_buffer::StrictSortedBuffer;
use vortex_arrow::ArrowSessionExt;

use crate::arrays::PyArrayRef;
use crate::arrow::FromPyArrow;
use crate::arrow::IntoPyArrow;
use crate::current_runtime;
use crate::dataset::PyVortexDataset;
use crate::dtype::PyDType;
use crate::error::PyVortexResult;
use crate::expr::PyExpr;
use crate::install_module;
use crate::io::AnyVortexStore;
use crate::iter::PyArrayIterator;
use crate::object_store::resolve::ResolvedStore;
use crate::object_store::resolve::resolve_store;
use crate::readable::PyReadable;
use crate::scan::PyRepeatedScan;
use crate::session::session;

pub(crate) fn init(py: Python, parent: &Bound<PyModule>) -> PyResult<()> {
    let m = PyModule::new(py, "file")?;
    parent.add_submodule(&m)?;
    install_module("vortex._lib.file", &m)?;

    m.add_function(wrap_pyfunction!(open, &m)?)?;
    m.add_function(wrap_pyfunction!(_reopen, &m)?)?;
    m.add_class::<PyVortexFile>()?;

    Ok(())
}

/// Reopen a Vortex file by path. The unpickling half of [`PyVortexFile::__reduce__`].
#[pyfunction]
fn _reopen(py: Python, path: &str, without_segment_cache: bool) -> PyVortexResult<PyVortexFile> {
    let path = PyString::new(py, path);
    open(py, path.as_any(), None, without_segment_cache, None)
}

/// Open a Vortex file for reading.
///
/// `source` is a path or URL, an `os.PathLike`, or a Python object that performs the IO itself
/// (see [`PyReadable`]). Callers can optionally configure an object store for a path using one of
/// the definitions in the `vortex.store` module.
#[pyfunction]
#[pyo3(signature = (source, *, store = None, without_segment_cache = false, concurrency = None))]
pub fn open(
    py: Python,
    source: &Bound<PyAny>,
    store: Option<AnyVortexStore>,
    without_segment_cache: bool,
    concurrency: Option<usize>,
) -> PyVortexResult<PyVortexFile> {
    let path = if let Ok(path) = source.cast::<PyString>() {
        Some(path.to_str()?.to_string())
    } else if source.hasattr(intern!(py, "__fspath__"))? {
        Some(
            PyModule::import(py, intern!(py, "os"))?
                .call_method1(intern!(py, "fspath"), (source,))?
                .extract::<String>()?,
        )
    } else {
        None
    };

    let origin = match (&path, &store) {
        (Some(_), None) => Origin::Path,
        (Some(_), Some(_)) => Origin::Store,
        (None, None) => Origin::Readable,
        (None, Some(_)) => {
            return Err(PyTypeError::new_err(
                "`store` can only be combined with a path or URL, not a Python readable",
            )
            .into());
        }
    };

    let (readable, owned_path) = match &path {
        Some(_) if concurrency.is_some() => {
            return Err(PyTypeError::new_err(
                "`concurrency` applies to a vortex.io.ReadAt reader, not a path or URL",
            )
            .into());
        }
        Some(path) => (None, path.clone()),
        None => {
            let readable = Arc::new(PyReadable::try_new(
                source,
                session().handle(),
                concurrency,
            )?);
            let name = readable
                .uri()
                .map(|uri| uri.to_string())
                .unwrap_or_default();
            (Some(readable), name)
        }
    };

    let vxf = py.detach(move || {
        current_runtime().block_on(async move {
            let mut options = session().open_options();
            if !without_segment_cache {
                // TODO(ngates): use a globally shared segment cache for all files
                options = options.with_segment_cache(Arc::new(MokaSegmentCache::new(256 << 20)));
            }

            if let Some(readable) = readable {
                return options.open(readable).await;
            }
            let path = path.as_deref().unwrap_or_default();
            match resolve_store(path, store.map(|x| x.into_inner()))? {
                ResolvedStore::ObjectStore(store, path) => {
                    options.open_object_store(&store, path).await
                }
                ResolvedStore::Path(path) => options.open_path(path).await,
            }
        })
    })?;

    Ok(PyVortexFile {
        vxf,
        path: owned_path,
        origin,
        without_segment_cache,
    })
}

/// Where a [`PyVortexFile`] was opened from, which decides whether it can be pickled.
#[derive(Clone, Copy)]
enum Origin {
    /// A path or URL resolved through the default registry: reopenable by path alone.
    Path,
    /// A path within an explicit object store, which is not picklable.
    Store,
    /// A Python readable, whose IO state cannot be transferred.
    Readable,
}

#[pyclass(name = "VortexFile", module = "vortex", frozen)]
pub struct PyVortexFile {
    vxf: VortexFile,
    /// The path this file was opened from, retained so that it can be reopened in another process.
    /// For a Python readable this is its `name`, if it has one, and empty otherwise.
    path: String,
    origin: Origin,
    without_segment_cache: bool,
}

#[pymethods]
impl PyVortexFile {
    fn __len__(slf: PyRef<Self>) -> PyResult<usize> {
        Ok(usize::try_from(slf.vxf.row_count())?)
    }

    /// The path or URL this file was opened from.
    ///
    /// Returns
    /// -------
    /// :class:`.str`
    #[getter]
    fn path(slf: PyRef<Self>) -> String {
        slf.path.clone()
    }

    /// Support for Python's pickle protocol: the file is reopened by path in the receiving process.
    ///
    /// Only the path is transferred, never any read state or segment cache, so this is cheap enough
    /// to send a file to every worker of a multiprocessing pool or Ray job.
    ///
    /// Raises
    /// ------
    /// :class:`TypeError`
    ///     If the file was opened with an explicit ``store``, since object stores cannot be
    ///     pickled, or from a Python file object or readable. Pass the URL to
    ///     :func:`vortex.open` in the worker instead.
    fn __reduce__<'py>(slf: PyRef<'py, Self>) -> PyResult<(Bound<'py, PyAny>, (String, bool))> {
        match slf.origin {
            Origin::Path => {}
            Origin::Store => {
                return Err(PyTypeError::new_err(
                    "cannot pickle a VortexFile opened with an explicit store, because object \
                     stores are not picklable; open it from its URL in the receiving process \
                     instead",
                ));
            }
            Origin::Readable => {
                return Err(PyTypeError::new_err(
                    "cannot pickle a VortexFile opened from a Python readable; open it again in \
                     the receiving process instead",
                ));
            }
        }
        let py = slf.py();
        let module = PyModule::import(py, "vortex._lib.file")?;
        let reopen = module.getattr(intern!(py, "_reopen"))?;
        Ok((reopen, (slf.path.clone(), slf.without_segment_cache)))
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
        let vxf = slf.get().vxf.clone();
        let projection = projection.map(|p| p.0);
        let expr = expr.map(|e| e.into_inner());
        let indices = indices.map(|i| i.into_inner());

        slf.py().detach(move || {
            let session = session();
            let mut ctx = session.create_execution_ctx();
            let builder =
                scan_builder(&vxf, projection, expr, limit, indices, batch_size, &mut ctx)?;
            let runtime = current_runtime();
            Ok(PyArrayIterator::new(Box::new(
                builder.into_array_iter(&runtime)?,
            )))
        })
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
        let vxf = slf.get().vxf.clone();
        let projection = projection.map(|p| p.0);
        let expr = expr.map(|e| e.into_inner());
        let indices = indices.map(|i| i.into_inner());

        let scan = slf.py().detach(move || {
            let session = session();
            let mut ctx = session.create_execution_ctx();
            scan_builder(&vxf, projection, expr, limit, indices, batch_size, &mut ctx)?.prepare()
        })?;

        Ok(PyRepeatedScan {
            scan: Arc::new(scan),
            row_count: slf.get().vxf.row_count(),
        })
    }

    #[pyo3(signature = (projection = None, *, expr = None, limit = None, batch_size = None, schema = None))]
    fn to_arrow(
        slf: Bound<Self>,
        projection: Option<PyIntoProjection>,
        expr: Option<PyExpr>,
        limit: Option<u64>,
        batch_size: Option<usize>,
        schema: Option<&Bound<PyAny>>,
    ) -> PyVortexResult<Py<PyAny>> {
        let vxf = slf.get().vxf.clone();
        let schema = schema
            .map(|schema| Schema::from_pyarrow(&schema.as_borrowed()))
            .transpose()?
            .map(Arc::new);

        let runtime = current_runtime();
        let reader = slf.py().detach(|| {
            let filter = expr
                .map(|e| e.into_inner().bind(vxf.dtype())?.optimize_recursive())
                .transpose()?;
            let projection = projection
                .map(|p| p.0)
                .unwrap_or_else(root)
                .bind(vxf.dtype())?
                .optimize_recursive()?;
            let mut builder = vxf
                .scan()?
                .with_some_filter(filter)
                .with_projection(projection);

            if let Some(limit) = limit {
                builder = builder.with_limit(limit);
            }

            if let Some(batch_size) = batch_size {
                builder = builder.with_split_by(SplitBy::RowCount(batch_size));
            }

            let schema = match schema {
                Some(schema) => schema,
                None => Arc::new(session().arrow().to_arrow_schema(&builder.dtype()?)?),
            };
            builder.into_record_batch_reader(schema, &runtime)
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

fn scan_builder(
    vxf: &VortexFile,
    projection: Option<Expression>,
    expr: Option<Expression>,
    limit: Option<u64>,
    indices: Option<ArrayRef>,
    batch_size: Option<usize>,
    ctx: &mut ExecutionCtx,
) -> VortexResult<ScanBuilder<ArrayRef>> {
    let projection = projection
        .unwrap_or_else(root)
        .bind(vxf.dtype())?
        .optimize_recursive()?;
    let expr = expr
        .map(|expr| expr.bind(vxf.dtype())?.optimize_recursive())
        .transpose()?;
    let mut builder = vxf
        .scan()?
        .with_some_filter(expr)
        .with_projection(projection);

    if let Some(limit) = limit {
        builder = builder.with_limit(limit);
    }

    if let Some(indices) = indices {
        let casted = indices.cast(DType::Primitive(PType::U64, NonNullable))?;
        let indices = casted.execute::<PrimitiveArray>(ctx)?.into_buffer::<u64>();
        builder = builder.with_row_indices(StrictSortedBuffer::try_new(indices)?);
    }

    if let Some(batch_size) = batch_size {
        builder = builder.with_split_by(SplitBy::RowCount(batch_size));
    }

    Ok(builder)
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
