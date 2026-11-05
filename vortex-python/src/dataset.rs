// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::ops::Range;
use std::path::PathBuf;
use std::sync::Arc;

use arrow_array::RecordBatch;
use arrow_array::RecordBatchReader;
use arrow_array::cast::AsArray;
use arrow_schema::ArrowError;
use arrow_schema::Field;
use arrow_schema::SchemaRef;
use futures::StreamExt;
use futures::TryStreamExt;
use futures::future;
use futures::stream;
use futures::stream::BoxStream;
use itertools::Itertools;
use object_store::ObjectStore;
use pyo3::exceptions::PyIndexError;
use pyo3::exceptions::PyTypeError;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use vortex::array::ArrayRef;
use vortex::array::VortexSessionExecute;
use vortex::array::iter::ArrayIteratorAdapter;
use vortex::array::iter::ArrayIteratorExt;
use vortex::buffer::Buffer;
use vortex::dtype::DType;
use vortex::dtype::FieldName;
use vortex::dtype::FieldNames;
use vortex::error::VortexError;
use vortex::error::VortexResult;
use vortex::error::vortex_bail;
use vortex::error::vortex_err;
use vortex::expr::Expression;
use vortex::expr::root;
use vortex::expr::select;
use vortex::file::OpenOptionsSessionExt;
use vortex::file::VORTEX_FILE_EXTENSION;
use vortex::file::VortexFile;
use vortex::io::filesystem::FileListing;
use vortex::io::filesystem::FileSystem;
use vortex::io::filesystem::FileSystemRef;
use vortex::io::object_store::ObjectStoreFileSystem;
use vortex::io::object_store::object_path_from_literal;
use vortex::io::runtime::BlockingRuntime;
use vortex::io::session::RuntimeSessionExt;
use vortex::layout::scan::arrow::RecordBatchIteratorAdapter;
use vortex::layout::scan::scan_builder::ScanBuilder;
use vortex::layout::scan::split_by::SplitBy;
use vortex::scan::strict_sorted_buffer::StrictSortedBuffer;
use vortex::utils::parallelism::get_available_parallelism;
use vortex_arrow::ArrowSessionExt;

use crate::arrays::PyArrayRef;
use crate::arrow::IntoPyArrow;
use crate::arrow::ToPyArrow;
use crate::current_runtime;
use crate::error::PyVortexResult;
use crate::expr::PyExpr;
use crate::file::check_concurrency;
use crate::file::row_indices;
use crate::install_module;
use crate::io::AnyVortexStore;
use crate::object_store::resolve::ResolvedStore;
use crate::object_store::resolve::resolve_store;
use crate::session::session;

/// Files opened at once while opening a dataset.
const OPEN_CONCURRENCY: usize = 16;

pub(crate) fn init(py: Python, parent: &Bound<PyModule>) -> PyResult<()> {
    let m = PyModule::new(py, "dataset")?;
    parent.add_submodule(&m)?;
    install_module("vortex._lib.dataset", &m)?;

    m.add_class::<PyVortexDataset>()?;

    m.add_function(wrap_pyfunction!(dataset_from_url, &m)?)?;
    m.add_function(wrap_pyfunction!(dataset_from_sources, &m)?)?;

    Ok(())
}

/// A projected column, selected either by name or by positional index.
#[derive(FromPyObject)]
pub enum ProjectionColumn {
    Name(String),
    Index(usize),
}

fn projection_from_python(
    columns: Option<Vec<ProjectionColumn>>,
    dtype: &DType,
) -> PyResult<Expression> {
    Ok(match columns {
        None => root(),
        Some(columns) => {
            let fields = columns
                .into_iter()
                .map(|column| match column {
                    ProjectionColumn::Name(name) => Ok(FieldName::from(name.as_str())),
                    ProjectionColumn::Index(index) => {
                        // Positional projection: map the index onto the top-level field name.
                        let DType::Struct(struct_dtype, _) = dtype else {
                            return Err(PyTypeError::new_err(
                                "projection: integer indices are only valid for a struct-typed file",
                            ));
                        };
                        struct_dtype.field_name(index).cloned().ok_or_else(|| {
                            PyIndexError::new_err(format!(
                                "projection: column index {index} is out of range for {} columns",
                                struct_dtype.nfields()
                            ))
                        })
                    }
                })
                .collect::<PyResult<Vec<_>>>()?;
            select(FieldNames::from(fields), root())
        }
    })
}

fn filter_from_python(row_filter: Option<&Bound<PyExpr>>) -> Option<Expression> {
    row_filter.map(|x| x.borrow().inner().clone())
}

/// Vortex files that read as one table: the rows of each file, in file order.
///
/// Every file has the same dtype. Row ranges, row indices and splits count rows across the whole
/// dataset, so the first row of the second file follows the last row of the first.
#[pyclass(name = "VortexDataset", module = "dataset")]
pub struct PyVortexDataset {
    files: Arc<[VortexFile]>,
    /// The dataset row of the first row of each file, followed by the dataset's row count.
    offsets: Arc<[u64]>,
    schema: SchemaRef,
    /// The most splits a read fetches and decodes ahead, for reads that do not set their own.
    concurrency: Option<usize>,
}

impl PyVortexDataset {
    pub fn try_new(files: Vec<VortexFile>, concurrency: Option<usize>) -> VortexResult<Self> {
        let Some(first) = files.first() else {
            vortex_bail!("a dataset needs at least one file");
        };
        let dtype = first.dtype();
        if let Some((idx, file)) = files.iter().find_position(|file| file.dtype() != dtype) {
            vortex_bail!(
                "file {idx} of the dataset has dtype {}, but file 0 has dtype {dtype}",
                file.dtype()
            );
        }
        let schema = Arc::new(session().arrow().to_arrow_schema(dtype)?);
        let offsets = std::iter::once(0)
            .chain(files.iter().scan(0, |end, file| {
                *end += file.row_count();
                Some(*end)
            }))
            .collect();
        Ok(Self {
            files: files.into(),
            offsets,
            schema,
            concurrency,
        })
    }

    pub async fn from_url(
        url: &str,
        store: Option<Arc<dyn object_store::ObjectStore>>,
    ) -> VortexResult<Self> {
        let session = session();
        let vxf = match resolve_store(url, store)? {
            ResolvedStore::ObjectStore(store, path) => {
                session
                    .open_options()
                    .open_object_store(&store, path)
                    .await?
            }
            ResolvedStore::Path(path) => session.open_options().open_path(path).await?,
        };
        PyVortexDataset::try_new(vec![vxf], None)
    }

    fn dtype(&self) -> &DType {
        self.files[0].dtype()
    }

    /// The rows of each file that the dataset rows `row_range` cover, as the file's index and a
    /// range of its own rows. Files with no rows in the range are left out.
    fn file_ranges(&self, row_range: Option<(u64, u64)>) -> Vec<(usize, Range<u64>)> {
        let (start, end) = row_range.unwrap_or((0, u64::MAX));
        self.offsets
            .iter()
            .tuple_windows()
            .enumerate()
            .filter_map(|(idx, (&file_start, &file_end))| {
                let first = start.max(file_start);
                let last = end.min(file_end);
                // `last` is below `file_start` when the range ends before the file starts.
                if first < last {
                    Some((idx, first - file_start..last - file_start))
                } else {
                    None
                }
            })
            .collect()
    }

    /// One scan for each file that a read covers, in file order, and the dtype they return.
    ///
    /// `row_range` and `indices` count dataset rows. Each scan reads the rows of its file that
    /// they select, and files they select no rows of are not scanned.
    fn scans(
        &self,
        projection: Expression,
        filter: Option<Expression>,
        indices: Option<StrictSortedBuffer<u64>>,
        row_range: Option<(u64, u64)>,
        split_by: Option<usize>,
    ) -> VortexResult<(DType, Vec<ScanBuilder<ArrayRef>>)> {
        let projection = projection.bind(self.dtype())?.optimize_recursive()?;
        let filter = filter
            .map(|filter| filter.bind(self.dtype())?.optimize_recursive())
            .transpose()?;
        let dtype = projection.dtype().clone();

        let mut scans = Vec::new();
        for (idx, rows) in self.file_ranges(row_range) {
            let file = &self.files[idx];
            let file_start = self.offsets[idx];
            let file_indices = match &indices {
                None => None,
                Some(indices) => {
                    let first = indices.partition_point(|&row| row < file_start + rows.start);
                    let last = indices.partition_point(|&row| row < file_start + rows.end);
                    if first == last {
                        continue;
                    }
                    let local: Buffer<u64> = indices[first..last]
                        .iter()
                        .map(|&row| row - file_start)
                        .collect();
                    Some(StrictSortedBuffer::try_new(local)?)
                }
            };

            let mut scan = file
                .scan()?
                .with_projection(projection.clone())
                .with_some_filter(filter.clone())
                .with_split_by(split_by.map(SplitBy::RowCount).unwrap_or_default());
            if rows != (0..file.row_count()) {
                scan = scan.with_row_range(rows);
            }
            if let Some(file_indices) = file_indices {
                scan = scan.with_row_indices(file_indices);
            }
            scans.push(scan);
        }
        Ok((dtype, scans))
    }

    /// The most splits a read keeps in flight: `concurrency` if the read sets it, else the
    /// dataset's, else what a scan of one file keeps in flight by default.
    fn read_concurrency<A: 'static + Send>(
        &self,
        scans: &[ScanBuilder<A>],
        concurrency: Option<usize>,
    ) -> usize {
        concurrency.or(self.concurrency).unwrap_or_else(|| {
            scans.first().map_or(1, ScanBuilder::concurrency)
                * get_available_parallelism().unwrap_or(1)
        })
    }

    pub(crate) fn to_array_inner<'py>(
        &self,
        py: Python<'py>,
        columns: Option<Vec<ProjectionColumn>>,
        row_filter: Option<&Bound<'py, PyExpr>>,
        indices: Option<PyArrayRef>,
        row_range: Option<(u64, u64)>,
        concurrency: Option<usize>,
    ) -> PyVortexResult<PyArrayRef> {
        let projection = projection_from_python(columns, self.dtype())?;
        let filter = filter_from_python(row_filter);
        let indices = row_indices(py, indices)?;
        let concurrency = check_concurrency(concurrency)?;

        let (dtype, scans) = self.scans(projection, filter, indices, row_range, None)?;
        let concurrency = self.read_concurrency(&scans, concurrency);
        let array = py.detach(move || {
            let splits = current_runtime().block_on_stream(split_stream(scans, concurrency));
            ArrayIteratorAdapter::new(dtype, splits).read_all()
        })?;
        Ok(PyArrayRef::from(array))
    }
}

/// The splits of `scans` in order, with at most `concurrency` of them fetched and decoded ahead
/// of the consumer across all the scans.
///
/// A file's splits start as soon as there is room for them, so a read keeps `concurrency` splits
/// in flight across file boundaries instead of draining each file before starting the next.
fn split_stream<A: Send + 'static>(
    scans: Vec<ScanBuilder<A>>,
    concurrency: usize,
) -> BoxStream<'static, VortexResult<A>> {
    let handle = session().handle();
    let prepare = handle.clone();
    stream::iter(scans)
        // Preparing a scan walks its file's layout, so it runs on a CPU worker. A file is only
        // prepared once the buffer below has room for its first split.
        .then(move |scan| prepare.spawn_cpu(move || scan.build()))
        .map_ok(|splits| stream::iter(splits).map(Ok::<_, VortexError>))
        .try_flatten()
        .map_ok(move |split| handle.spawn(split))
        .try_buffered(concurrency)
        .try_filter_map(|array| future::ready(Ok(array)))
        .boxed()
}

fn to_record_batch(chunk: ArrayRef, target: &Field) -> VortexResult<RecordBatch> {
    let arrow = session().arrow().execute_arrow(
        chunk,
        Some(target),
        &mut session().create_execution_ctx(),
    )?;
    Ok(RecordBatch::from(arrow.as_struct().clone()))
}

#[pymethods]
impl PyVortexDataset {
    fn schema(self_: PyRef<Self>) -> PyResult<Py<PyAny>> {
        Arc::clone(&self_.schema).to_pyarrow(self_.py())
    }

    /// The most splits a read fetches and decodes ahead, for reads that do not set their own.
    #[getter]
    fn concurrency(&self) -> Option<usize> {
        self.concurrency
    }

    /// The number of files in the dataset.
    #[getter]
    fn file_count(&self) -> usize {
        self.files.len()
    }

    #[pyo3(signature = (*, columns = None, row_filter = None, indices = None, row_range = None, concurrency = None))]
    pub fn to_array<'py>(
        self_: PyRef<'py, Self>,
        columns: Option<Vec<ProjectionColumn>>,
        row_filter: Option<&Bound<'py, PyExpr>>,
        indices: Option<PyArrayRef>,
        row_range: Option<(u64, u64)>,
        concurrency: Option<usize>,
    ) -> PyVortexResult<PyArrayRef> {
        self_.to_array_inner(
            self_.py(),
            columns,
            row_filter,
            indices,
            row_range,
            concurrency,
        )
    }

    #[pyo3(signature = (*, columns = None, row_filter = None, split_by = None, row_range = None, concurrency = None))]
    pub fn to_record_batch_reader(
        self_: PyRef<Self>,
        columns: Option<Vec<ProjectionColumn>>,
        row_filter: Option<&Bound<'_, PyExpr>>,
        split_by: Option<usize>,
        row_range: Option<(u64, u64)>,
        concurrency: Option<usize>,
    ) -> PyVortexResult<Py<PyAny>> {
        let projection = projection_from_python(columns, self_.dtype())?;
        let filter = filter_from_python(row_filter);
        let concurrency = check_concurrency(concurrency)?;

        // Building the reader is lazy and cheap, so it runs without releasing the GIL. The scan
        // runs as pyarrow pulls batches, and pyarrow releases the GIL while it does.
        let (dtype, scans) = self_.scans(projection, filter, None, row_range, split_by)?;
        let concurrency = self_.read_concurrency(&scans, concurrency);
        let schema = Arc::new(session().arrow().to_arrow_schema(&dtype)?);
        let target = Field::new_struct("", schema.fields().clone(), false);
        let scans = scans
            .into_iter()
            .map(|scan| {
                let target = target.clone();
                scan.map(move |chunk| to_record_batch(chunk, &target))
            })
            .collect();
        let batches = current_runtime()
            .block_on_stream(split_stream(scans, concurrency))
            .map(|batch| batch.map_err(|e| ArrowError::ExternalError(Box::new(e))));

        let reader: Box<dyn RecordBatchReader + Send> =
            Box::new(RecordBatchIteratorAdapter::new(batches, schema));
        Ok(reader.into_pyarrow(self_.py())?)
    }

    /// The number of rows matching the filter.
    #[pyo3(signature = (*, row_filter = None, split_by = None, row_range = None, concurrency = None))]
    pub fn count_rows(
        self_: PyRef<Self>,
        row_filter: Option<&Bound<'_, PyExpr>>,
        split_by: Option<usize>,
        row_range: Option<(u64, u64)>,
        concurrency: Option<usize>,
    ) -> PyVortexResult<usize> {
        let concurrency = check_concurrency(concurrency)?;
        if row_filter.is_none() {
            let row_count: u64 = self_
                .file_ranges(row_range)
                .iter()
                .map(|(_, rows)| rows.end - rows.start)
                .sum();
            return row_count
                .try_into()
                .map_err(|e| PyValueError::new_err(e).into());
        }

        let filter = filter_from_python(row_filter);
        let projection = select(FieldNames::empty(), root());
        let (_, scans) = self_.scans(projection, filter, None, row_range, split_by)?;
        let concurrency = self_.read_concurrency(&scans, concurrency);
        let scans = scans
            .into_iter()
            .map(|scan| scan.map(|array| Ok(array.len())))
            .collect();
        let n_rows = self_.py().detach(move || {
            current_runtime()
                .block_on_stream(split_stream(scans, concurrency))
                .sum::<VortexResult<usize>>()
        })?;

        Ok(n_rows)
    }

    /// The natural splits of this Dataset, as ranges of dataset rows.
    #[pyo3(signature = (*))]
    pub fn splits(&self) -> PyVortexResult<Vec<(u64, u64)>> {
        let mut splits = Vec::new();
        for (file, &file_start) in self.files.iter().zip(self.offsets.iter()) {
            splits.extend(
                file.splits()?
                    .into_iter()
                    .map(|rows| (file_start + rows.start, file_start + rows.end)),
            );
        }
        Ok(splits)
    }
}

#[pyfunction]
#[pyo3(signature = (url, *, store = None))]
pub fn dataset_from_url(
    py: Python,
    url: &str,
    store: Option<Bound<PyAny>>,
) -> PyVortexResult<PyVortexDataset> {
    let store_arc = if let Some(store_obj) = store {
        let py_store: pyo3_object_store::PyObjectStore = store_obj.extract()?;
        Some(py_store.into_inner())
    } else {
        None
    };

    Ok(py.detach(move || current_runtime().block_on(PyVortexDataset::from_url(url, store_arc)))?)
}

/// Open the Vortex files that `sources` name as one dataset, in the order the sources name them.
#[pyfunction]
#[pyo3(signature = (sources, *, store = None, concurrency = None))]
pub fn dataset_from_sources(
    py: Python,
    sources: Vec<String>,
    store: Option<AnyVortexStore>,
    concurrency: Option<usize>,
) -> PyVortexResult<PyVortexDataset> {
    let concurrency = check_concurrency(concurrency)?;
    let store = store.map(AnyVortexStore::into_inner);
    let files = py.detach(move || current_runtime().block_on(open_sources(sources, store)))?;
    Ok(PyVortexDataset::try_new(files, concurrency)?)
}

/// A Vortex file that a dataset source names.
enum SourceFile {
    Local(PathBuf),
    Object(Arc<dyn ObjectStore>, object_store::path::Path),
}

impl SourceFile {
    fn name(&self) -> String {
        match self {
            SourceFile::Local(path) => path.display().to_string(),
            SourceFile::Object(_, path) => path.to_string(),
        }
    }

    async fn open(self) -> VortexResult<VortexFile> {
        let options = session().open_options();
        match self {
            SourceFile::Local(path) => options.open_path(path).await,
            SourceFile::Object(store, path) => options.open_object_store(&store, path).await,
        }
    }
}

async fn open_sources(
    sources: Vec<String>,
    store: Option<Arc<dyn ObjectStore>>,
) -> VortexResult<Vec<VortexFile>> {
    if sources.is_empty() {
        vortex_bail!("a dataset needs at least one source");
    }

    let mut files = Vec::new();
    for source in &sources {
        let found = expand_source(source, store.clone()).await?;
        if found.is_empty() {
            vortex_bail!("no Vortex files found for {source:?}");
        }
        files.extend(found);
    }

    let names = files.iter().map(SourceFile::name).collect::<Vec<_>>();
    let opened: Vec<VortexFile> = stream::iter(files.into_iter().map(SourceFile::open))
        .buffered(OPEN_CONCURRENCY)
        .try_collect()
        .await?;

    let dtype = opened[0].dtype();
    if let Some((name, file)) = names.iter().zip(&opened).find(|(_, file)| file.dtype() != dtype) {
        vortex_bail!(
            "{name} has dtype {}, but {} has dtype {dtype}; every file of a dataset needs the same dtype",
            file.dtype(),
            names[0],
        );
    }
    Ok(opened)
}

fn has_glob(text: &str) -> bool {
    text.contains(['*', '?', '['])
}

/// The files that `source` names, in sorted order.
///
/// A source is a file, a directory or an object store prefix ending in `/` (meaning the Vortex
/// files below it, at any depth), or a glob pattern (meaning the files it matches).
async fn expand_source(
    source: &str,
    store: Option<Arc<dyn ObjectStore>>,
) -> VortexResult<Vec<SourceFile>> {
    match resolve_store(source, store)? {
        ResolvedStore::Path(path) => Ok(expand_local(path)?
            .into_iter()
            .map(SourceFile::Local)
            .collect()),
        ResolvedStore::ObjectStore(store, path) => {
            // Object store paths drop a trailing `/`, so the source says whether it is a prefix.
            let is_prefix = source.ends_with('/') || path.as_ref().is_empty();
            Ok(expand_object_store(&store, path, is_prefix)
                .await?
                .into_iter()
                .map(|path| SourceFile::Object(Arc::clone(&store), path))
                .collect())
        }
    }
}

fn expand_local(path: PathBuf) -> VortexResult<Vec<PathBuf>> {
    if path.is_file() {
        return Ok(vec![path]);
    }
    let pattern = if path.is_dir() {
        let dir = path
            .to_str()
            .ok_or_else(|| vortex_err!("{} is not a UTF-8 path", path.display()))?;
        format!(
            "{}/**/*.{VORTEX_FILE_EXTENSION}",
            glob::Pattern::escape(dir.trim_end_matches(['/', '\\']))
        )
    } else if let Some(pattern) = path.to_str().filter(|text| has_glob(text)) {
        pattern.to_string()
    } else {
        // Not a file, a directory or a pattern: opening it reports that it does not exist.
        return Ok(vec![path]);
    };

    let mut files = Vec::new();
    for entry in glob::glob(&pattern)
        .map_err(|e| vortex_err!("invalid glob pattern {pattern:?}: {e}"))?
    {
        let file = entry
            .map_err(|e| vortex_err!("cannot read {}: {}", e.path().display(), e.error()))?;
        if file.is_file() {
            files.push(file);
        }
    }
    files.sort();
    Ok(files)
}

async fn expand_object_store(
    store: &Arc<dyn ObjectStore>,
    path: object_store::path::Path,
    is_prefix: bool,
) -> VortexResult<Vec<object_store::path::Path>> {
    let pattern = path.as_ref();
    if !is_prefix && !has_glob(pattern) {
        return Ok(vec![path]);
    }

    let fs: FileSystemRef = Arc::new(ObjectStoreFileSystem::new(
        Arc::clone(store),
        session().handle(),
    ));
    let listings: Vec<FileListing> = if has_glob(pattern) {
        // Match like a local glob, where `*` and `?` stop at `/` and only `**` crosses it.
        let glob = glob::Pattern::new(pattern)
            .map_err(|e| vortex_err!("invalid glob pattern {pattern:?}: {e}"))?;
        let options = glob::MatchOptions {
            require_literal_separator: true,
            ..Default::default()
        };
        let literal = &pattern[..pattern.find(['*', '?', '[']).unwrap_or(pattern.len())];
        let prefix = literal.rfind('/').map_or("", |slash| &literal[..slash]);
        fs.list(prefix)
            .try_filter(|listing| future::ready(glob.matches_with(&listing.path, options)))
            .try_collect()
            .await?
    } else {
        let suffix = format!(".{VORTEX_FILE_EXTENSION}");
        fs.list(pattern)
            .try_filter(|listing| future::ready(listing.path.ends_with(&suffix)))
            .try_collect()
            .await?
    };

    let mut paths = listings
        .into_iter()
        .map(|listing| listing.path)
        .collect::<Vec<_>>();
    paths.sort();
    Ok(paths
        .into_iter()
        .map(|path| object_path_from_literal(&path))
        .collect())
}
