// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::sync::Arc;

use arrow_array::RecordBatchReader;
use arrow_schema::Schema;
use pyo3::exceptions::PyTypeError;
use pyo3::exceptions::PyValueError;
use pyo3::intern;
use pyo3::prelude::*;
use pyo3::types::PyList;
use vortex::array::ArrayRef;
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
use vortex::file::Footer;
use vortex::file::OpenOptionsSessionExt;
use vortex::file::VortexFile;
use vortex::file::VortexOpenOptions;
use vortex::io::VortexReadAt;
use vortex::io::runtime::BlockingRuntime;
use vortex::io::session::RuntimeSessionExt;
use vortex::layout::scan::scan_builder::ScanBuilder;
use vortex::layout::scan::split_by::SplitBy;
use vortex::layout::segments::MokaSegmentCache;
use vortex::layout::segments::SegmentEviction;
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
    m.add_function(wrap_pyfunction!(open_readable, &m)?)?;
    m.add_function(wrap_pyfunction!(_reopen, &m)?)?;
    m.add_class::<PyVortexFile>()?;
    m.add_class::<PyFooter>()?;
    m.add_class::<PySegmentCache>()?;

    Ok(())
}

/// Reopen a Vortex file by path. The unpickling half of [`PyVortexFile::__reduce__`].
#[pyfunction]
fn _reopen(py: Python, path: &str, without_segment_cache: bool) -> PyVortexResult<PyVortexFile> {
    open(py, path, None, None, without_segment_cache, None, None)
}

/// Open a Vortex file for reading.
///
/// Callers can optionally configure an object store to build from using one of the definitions
/// in the `vortex.store` crate.
#[pyfunction]
#[pyo3(signature = (
    path,
    *,
    store = None,
    footer = None,
    without_segment_cache = false,
    segment_cache = None,
    cache_key = None,
))]
pub fn open(
    py: Python,
    path: &str,
    store: Option<AnyVortexStore>,
    footer: Option<PyRef<PyFooter>>,
    without_segment_cache: bool,
    segment_cache: Option<PyRef<PySegmentCache>>,
    cache_key: Option<String>,
) -> PyVortexResult<PyVortexFile> {
    let origin = if store.is_some() {
        Origin::Store
    } else {
        Origin::Path
    };
    let mut options = open_options(
        without_segment_cache,
        segment_cache.as_deref(),
        cache_key.as_deref(),
    )?;
    if let Some(footer) = footer {
        // The file size is not known without IO here, so a footer from a different file is not
        // detected. The caller must pass the footer of this same, unchanged file.
        options = options.with_footer(footer.footer.clone());
    }

    let vxf = py.detach(|| {
        current_runtime().block_on(async {
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
        path: path.to_string(),
        origin,
        without_segment_cache,
    })
}

/// Open a Vortex file through a Python object that performs the IO itself (see [`PyReadable`]).
///
/// A `footer` taken from an earlier open of the same file skips the footer read entirely.
#[pyfunction]
#[pyo3(signature = (
    reader,
    *,
    footer = None,
    concurrency = None,
    without_segment_cache = false,
    segment_cache = None,
    cache_key = None,
))]
pub fn open_readable(
    py: Python,
    reader: &Bound<PyAny>,
    footer: Option<PyRef<PyFooter>>,
    concurrency: Option<usize>,
    without_segment_cache: bool,
    segment_cache: Option<PyRef<PySegmentCache>>,
    cache_key: Option<String>,
) -> PyVortexResult<PyVortexFile> {
    let options = open_options(
        without_segment_cache,
        segment_cache.as_deref(),
        cache_key.as_deref(),
    )?;
    let readable = Arc::new(PyReadable::try_new(
        reader,
        session().handle(),
        concurrency,
    )?);
    let name = readable
        .uri()
        .map(|uri| uri.to_string())
        .unwrap_or_default();

    let footer = footer.map(|footer| footer.footer.clone());

    let vxf = py.detach(|| {
        current_runtime().block_on(async {
            let mut options = options;
            if let Some(footer) = footer {
                // The size was read when the readable was created, so this does no IO. With it,
                // the open rejects a footer whose segments lie past the end of this source.
                let file_size = readable.size().await?;
                options = options.with_footer(footer).with_file_size(file_size);
            }
            options.open(readable).await
        })
    })?;

    Ok(PyVortexFile {
        vxf,
        path: name,
        origin: Origin::Readable,
        without_segment_cache,
    })
}

/// Open options with the segment cache that the caller asked for.
///
/// A shared `segment_cache` needs a `cache_key`. Without one, each file gets a private cache,
/// unless `without_segment_cache` is set.
fn open_options(
    without_segment_cache: bool,
    segment_cache: Option<&PySegmentCache>,
    cache_key: Option<&str>,
) -> PyResult<VortexOpenOptions> {
    let options = session().open_options();

    match (segment_cache, cache_key) {
        (Some(_), _) if without_segment_cache => Err(PyValueError::new_err(
            "segment_cache cannot be combined with without_segment_cache=True",
        )),
        (Some(cache), Some(key)) => {
            Ok(options.with_segment_cache(Arc::new(cache.cache.for_file(key))))
        }
        (Some(_), None) => Err(PyValueError::new_err(
            "segment_cache requires a cache_key that identifies the file's contents",
        )),
        (None, Some(_)) => Err(PyValueError::new_err("cache_key requires a segment_cache")),
        (None, None) if without_segment_cache => Ok(options),
        // A private cache holds only this file, so any key will do. It serves re-reads of this
        // one file, where TinyLFU keeps a stable hot set that LRU would evict on every pass.
        (None, None) => Ok(options.with_segment_cache(Arc::new(
            MokaSegmentCache::new(256 << 20, SegmentEviction::TinyLfu).for_file(""),
        ))),
    }
}

/// A segment cache that many Vortex files share, capped by total bytes.
///
/// Pass it to :func:`vortex.open` or :func:`vortex.open_readable` with a ``cache_key``. Files
/// opened with the same key share cached segments, so a file opened again, for example once per
/// batch, does not read them again. The key must identify the file's contents, not only its
/// location: a file that has changed must get a new key.
///
/// It is safe to share between threads, but it is not picklable: create one in each process.
///
/// Parameters
/// ----------
/// max_bytes : :class:`int`
///     The most bytes of segments to hold. The least recently used segments are evicted first.
#[pyclass(name = "SegmentCache", module = "vortex", frozen)]
pub struct PySegmentCache {
    cache: MokaSegmentCache,
}

#[pymethods]
impl PySegmentCache {
    #[new]
    fn new(max_bytes: u64) -> Self {
        Self {
            cache: MokaSegmentCache::new(max_bytes, SegmentEviction::Lru),
        }
    }

    /// The total bytes of the cached segments.
    ///
    /// Returns
    /// -------
    /// :class:`int`
    #[getter]
    fn size_bytes(&self, py: Python) -> u64 {
        self.sync(py);
        self.cache.weighted_size()
    }

    /// The number of cached segments.
    ///
    /// Returns
    /// -------
    /// :class:`int`
    #[getter]
    fn entry_count(&self, py: Python) -> u64 {
        self.sync(py);
        self.cache.entry_count()
    }

    /// Remove every cached segment.
    fn clear(&self) {
        self.cache.invalidate_all();
    }
}

impl PySegmentCache {
    /// Apply pending inserts and evictions, so that the counts are exact.
    fn sync(&self, py: Python) {
        py.detach(|| current_runtime().block_on(self.cache.run_pending_tasks()));
    }
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

/// The parsed footer of a Vortex file: its layout, segment map and dtype.
///
/// Pass it to :func:`vortex.open` or :func:`vortex.open_readable` to open the same file again
/// without reading the footer. It holds no IO state, but it is not picklable.
#[pyclass(name = "Footer", module = "vortex", frozen)]
pub struct PyFooter {
    footer: Footer,
}

#[pymethods]
impl PyFooter {
    /// The number of rows in the file.
    ///
    /// Returns
    /// -------
    /// :class:`int`
    #[getter]
    fn row_count(&self) -> u64 {
        self.footer.row_count()
    }
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

    /// The parsed footer, for opening the same file again without reading it.
    #[getter]
    fn footer(&self) -> PyFooter {
        PyFooter {
            footer: self.vxf.footer().clone(),
        }
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
        let vxf = &slf.get().vxf;
        let projection = projection.map(|p| p.0);
        let expr = expr.map(|e| e.into_inner());
        let indices = row_indices(slf.py(), indices)?;

        // Building the scan is lazy and cheap, so it runs without releasing the GIL.
        let builder = scan_builder(vxf, projection, expr, limit, indices, batch_size)?;
        Ok(PyArrayIterator::new(Box::new(
            builder.into_array_iter(&current_runtime())?,
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
        let vxf = &slf.get().vxf;
        let projection = projection.map(|p| p.0);
        let expr = expr.map(|e| e.into_inner());
        let indices = row_indices(slf.py(), indices)?;

        let scan = scan_builder(vxf, projection, expr, limit, indices, batch_size)?.prepare()?;

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
        let vxf = &slf.get().vxf;
        let schema = schema
            .map(|schema| Schema::from_pyarrow(&schema.as_borrowed()))
            .transpose()?
            .map(Arc::new);

        // Building the reader is lazy and cheap, so it runs without releasing the GIL. The scan
        // runs as pyarrow pulls batches, and pyarrow releases the GIL while it does.
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
        let reader = builder.into_record_batch_reader(schema, &current_runtime())?;

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

/// Decode row indices into a sorted `u64` buffer, releasing the GIL since this is O(n).
fn row_indices(
    py: Python,
    indices: Option<PyArrayRef>,
) -> VortexResult<Option<StrictSortedBuffer<u64>>> {
    let Some(indices) = indices else {
        return Ok(None);
    };
    let indices = indices.into_inner();
    py.detach(move || {
        let casted = indices.cast(DType::Primitive(PType::U64, NonNullable))?;
        let indices = casted
            .execute::<PrimitiveArray>(&mut session().create_execution_ctx())?
            .into_buffer::<u64>();
        Ok(Some(StrictSortedBuffer::try_new(indices)?))
    })
}

fn scan_builder(
    vxf: &VortexFile,
    projection: Option<Expression>,
    expr: Option<Expression>,
    limit: Option<u64>,
    indices: Option<StrictSortedBuffer<u64>>,
    batch_size: Option<usize>,
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
        builder = builder.with_row_indices(indices);
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
