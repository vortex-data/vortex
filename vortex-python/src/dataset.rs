// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::ops::Range;
use std::path::Path;
use std::sync::Arc;
use std::task::Poll;

use arrow_array::RecordBatch;
use arrow_array::RecordBatchReader;
use arrow_array::cast::AsArray;
use arrow_schema::ArrowError;
use arrow_schema::Field;
use arrow_schema::SchemaRef;
use futures::StreamExt;
use futures::TryStreamExt;
use futures::future;
use futures::ready;
use futures::stream;
use futures::stream::BoxStream;
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
use vortex::expr::BoundExpression;
use vortex::expr::Expression;
use vortex::expr::root;
use vortex::expr::select;
use vortex::file::OpenOptionsSessionExt;
use vortex::file::VORTEX_FILE_EXTENSION;
use vortex::file::VortexFile;
use vortex::io::filesystem::FileListing;
use vortex::io::filesystem::FileSystemRef;
use vortex::io::object_store::ObjectStoreFileSystem;
use vortex::io::runtime::BlockingRuntime;
use vortex::io::session::RuntimeSessionExt;
use vortex::io::std_file::StdFileSystem;
use vortex::layout::scan::arrow::RecordBatchIteratorAdapter;
use vortex::layout::scan::scan_builder::ScanBuilder;
use vortex::layout::scan::split_by::SplitBy;
use vortex::scan::strict_sorted_buffer::StrictSortedBuffer;
use vortex::utils::parallelism::get_available_parallelism;
use vortex_arrow::ArrowSessionExt;
use vortex_arrow::ExecuteArrowExt;

use crate::arrays::PyArrayRef;
use crate::arrow::IntoPyArrow;
use crate::arrow::ToPyArrow;
use crate::current_runtime;
use crate::error::PyVortexResult;
use crate::expr::PyExpr;
use crate::file::row_indices;
use crate::install_module;
use crate::io::AnyVortexStore;
use crate::object_store::resolve::ResolvedStore;
use crate::object_store::resolve::resolve_store;
use crate::session::session;

/// Files opened at once when scanning metadata across a dataset.
const OPEN_CONCURRENCY: usize = 16;

/// Default splits fetched and decoded ahead per worker thread.
const DEFAULT_CONCURRENCY_PER_THREAD: usize = 2;

pub(crate) fn init(py: Python, parent: &Bound<PyModule>) -> PyResult<()> {
    let m = PyModule::new(py, "dataset")?;
    parent.add_submodule(&m)?;
    install_module("vortex._lib.dataset", &m)?;

    m.add_class::<PyVortexDataset>()?;

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

#[derive(Clone)]
enum DatasetSource {
    Opened(Box<VortexFile>),
    Lazy {
        files: Arc<[(FileSystemRef, FileListing)]>,
        dtype: DType,
    },
}

#[derive(Clone)]
struct ScanParams {
    projection: BoundExpression,
    filter: Option<BoundExpression>,
    indices: Option<StrictSortedBuffer<u64>>,
    row_range: Option<(u64, u64)>,
    split_by: SplitBy,
}

/// Vortex files that read as one table: the rows of each file, in file order.
///
/// Every file has the same dtype. Row ranges, row indices and splits count rows across the whole
/// dataset, so the first row of the second file follows the last row of the first.
///
/// Multi-file datasets open only the first file eagerly to validate the source and read the schema;
/// each file is opened lazily during scans and dropped once its splits finish, so a dataset with
/// thousands of files does not hold every file open at once.
#[pyclass(name = "VortexDataset", module = "dataset")]
pub struct PyVortexDataset {
    source: DatasetSource,
    schema: SchemaRef,
    concurrency: Option<usize>,
}

impl PyVortexDataset {
    pub fn try_new(vxf: VortexFile, concurrency: Option<usize>) -> VortexResult<Self> {
        let schema = Arc::new(session().arrow().to_arrow_schema(vxf.dtype())?);
        Ok(Self {
            source: DatasetSource::Opened(Box::new(vxf)),
            schema,
            concurrency,
        })
    }

    fn from_lazy(
        files: Vec<(FileSystemRef, FileListing)>,
        dtype: DType,
        concurrency: Option<usize>,
    ) -> VortexResult<Self> {
        let schema = Arc::new(session().arrow().to_arrow_schema(&dtype)?);
        Ok(Self {
            source: DatasetSource::Lazy {
                files: files.into(),
                dtype,
            },
            schema,
            concurrency,
        })
    }

    fn read_concurrency(&self, concurrency: Option<usize>) -> usize {
        concurrency
            .or(self.concurrency)
            .unwrap_or_else(default_read_concurrency)
    }

    fn dtype(&self) -> &DType {
        match &self.source {
            DatasetSource::Opened(file) => file.dtype(),
            DatasetSource::Lazy { dtype, .. } => dtype,
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn scan_stream<A, F>(
        &self,
        projection: Expression,
        filter: Option<Expression>,
        indices: Option<StrictSortedBuffer<u64>>,
        row_range: Option<(u64, u64)>,
        split_by: Option<usize>,
        concurrency: Option<usize>,
        map_fn: F,
    ) -> VortexResult<(DType, BoxStream<'static, VortexResult<A>>)>
    where
        A: Send + 'static,
        F: Fn(ArrayRef) -> VortexResult<A> + Send + Sync + Clone + 'static,
    {
        let projection = projection.bind(self.dtype())?.optimize_recursive()?;
        let filter = filter
            .map(|filter| filter.bind(self.dtype())?.optimize_recursive())
            .transpose()?;
        let dtype = projection.dtype().clone();
        let split_by = split_by.map(SplitBy::RowCount).unwrap_or_default();
        let concurrency = self.read_concurrency(concurrency);

        let params = ScanParams {
            projection,
            filter,
            indices,
            row_range,
            split_by,
        };
        let scans = lazy_scan_stream(self.source.clone(), params, map_fn);
        Ok((dtype, split_stream(scans, concurrency)))
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

        let (dtype, splits) = self.scan_stream(
            projection,
            filter,
            indices,
            row_range,
            None,
            concurrency,
            Ok,
        )?;
        let array = py.detach(move || {
            let splits = current_runtime().block_on_stream(splits);
            ArrayIteratorAdapter::new(dtype, splits).read_all()
        })?;
        Ok(PyArrayRef::from(array))
    }
}

fn check_concurrency(concurrency: Option<usize>) -> PyResult<Option<usize>> {
    if concurrency == Some(0) {
        return Err(PyValueError::new_err("concurrency must be at least 1"));
    }
    Ok(concurrency)
}

fn default_read_concurrency() -> usize {
    DEFAULT_CONCURRENCY_PER_THREAD * get_available_parallelism().unwrap_or(1)
}

async fn open_lazy_file(
    fs: &FileSystemRef,
    listing: &FileListing,
    expected_dtype: Option<&DType>,
    first_name: &str,
) -> VortexResult<VortexFile> {
    let source = fs.open_read(&listing.path).await?;
    let mut options = session().open_options();
    if let Some(size) = listing.size {
        options = options.with_file_size(size);
    }
    let file = options.open(source).await?;
    if let Some(expected) = expected_dtype
        && file.dtype() != expected
    {
        vortex_bail!(
            "{} has dtype {}, but {first_name} has dtype {expected}; every file of a dataset needs the same dtype",
            listing.path,
            file.dtype(),
        );
    }
    Ok(file)
}

fn build_file_scan<A, F>(
    file: &VortexFile,
    file_start: u64,
    params: &ScanParams,
    map_fn: &F,
) -> VortexResult<Option<ScanBuilder<A>>>
where
    A: Send + 'static,
    F: Fn(ArrayRef) -> VortexResult<A> + Send + Sync + Clone + 'static,
{
    let file_rows = file.row_count();
    let file_end = file_start + file_rows;

    let rows = match params.row_range {
        None => 0..file_rows,
        Some((start, end)) => {
            let first = start.max(file_start);
            let last = end.min(file_end);
            if first >= last {
                return Ok(None);
            }
            (first - file_start)..(last - file_start)
        }
    };

    let file_indices = match params.indices.as_ref() {
        None => None,
        Some(indices) => {
            let first = indices.partition_point(|&row| row < file_start + rows.start);
            let last = indices.partition_point(|&row| row < file_start + rows.end);
            if first == last {
                return Ok(None);
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
        .with_projection(params.projection.clone())
        .with_some_filter(params.filter.clone())
        .with_split_by(params.split_by);
    if rows != (0..file_rows) {
        scan = scan.with_row_range(rows);
    }
    if let Some(file_indices) = file_indices {
        scan = scan.with_row_indices(file_indices);
    }
    Ok(Some(scan.map(map_fn.clone())))
}

/// Produce one [`ScanBuilder`] per covered file lazily, opening each file only when its turn
/// arrives and stopping early once `row_range` or `indices` can no longer match later files.
fn lazy_scan_stream<A, F>(
    source: DatasetSource,
    params: ScanParams,
    map_fn: F,
) -> BoxStream<'static, VortexResult<ScanBuilder<A>>>
where
    A: Send + 'static,
    F: Fn(ArrayRef) -> VortexResult<A> + Send + Sync + Clone + 'static,
{
    match source {
        DatasetSource::Opened(file) => match build_file_scan(&file, 0, &params, &map_fn) {
            Ok(Some(scan)) => stream::once(future::ready(Ok(scan))).boxed(),
            Ok(None) => stream::empty().boxed(),
            Err(e) => stream::once(future::ready(Err(e))).boxed(),
        },
        DatasetSource::Lazy { files, dtype } => {
            let needs_offsets = params.row_range.is_some() || params.indices.is_some();
            if !needs_offsets {
                let first_name = files[0].1.path.clone();
                return stream::iter(0..files.len())
                    .then(move |idx| {
                        let files = Arc::clone(&files);
                        let dtype = dtype.clone();
                        let first_name = first_name.clone();
                        let params = params.clone();
                        let map_fn = map_fn.clone();
                        async move {
                            let (fs, listing) = &files[idx];
                            let file =
                                open_lazy_file(fs, listing, Some(&dtype), &first_name).await?;
                            let scan = file
                                .scan()?
                                .with_projection(params.projection)
                                .with_some_filter(params.filter)
                                .with_split_by(params.split_by)
                                .map(map_fn);
                            Ok::<_, VortexError>(scan)
                        }
                    })
                    .boxed();
            }

            let max_index = params.indices.as_ref().and_then(|idx| idx.last().copied());
            let range_end = params.row_range.map(|(_, end)| end);
            let first_name = files[0].1.path.clone();

            stream::unfold(
                (0usize, 0u64, files, dtype, first_name, params, map_fn),
                move |(mut idx, mut file_start, files, dtype, first_name, params, map_fn)| async move {
                    while idx < files.len() {
                        if let Some(end) = range_end
                            && file_start >= end
                        {
                            return None;
                        }
                        if let Some(last_idx) = max_index
                            && file_start > last_idx
                        {
                            return None;
                        }

                        let (fs, listing) = &files[idx];
                        idx += 1;
                        let file =
                            match open_lazy_file(fs, listing, Some(&dtype), &first_name).await {
                                Ok(file) => file,
                                Err(e) => {
                                    return Some((
                                        Err(e),
                                        (files.len(), file_start, files, dtype, first_name, params, map_fn),
                                    ));
                                }
                            };

                        let file_rows = file.row_count();
                        let scan = build_file_scan(&file, file_start, &params, &map_fn);
                        file_start += file_rows;

                        match scan {
                            Ok(Some(scan)) => {
                                return Some((
                                    Ok(scan),
                                    (idx, file_start, files, dtype, first_name, params, map_fn),
                                ));
                            }
                            Ok(None) => continue,
                            Err(e) => {
                                return Some((
                                    Err(e),
                                    (files.len(), file_start, files, dtype, first_name, params, map_fn),
                                ));
                            }
                        }
                    }
                    None
                },
            )
            .boxed()
        }
    }
}

/// The splits of `scans` in order, with at most `concurrency` of them fetched and decoded ahead
/// of the consumer across all the scans.
fn split_stream<A: Send + 'static>(
    scans: BoxStream<'static, VortexResult<ScanBuilder<A>>>,
    concurrency: usize,
) -> BoxStream<'static, VortexResult<A>> {
    let handle = session().handle();
    let prepare = handle.clone();
    scans
        // Preparing a scan walks its file's layout, so it runs on a CPU worker. A file is only
        // prepared once the buffer below has room for its first split.
        .and_then(move |scan| prepare.spawn_cpu(move || scan.build()))
        .map_ok(|splits| stream::iter(splits).map(Ok::<_, VortexError>))
        .try_flatten()
        .map_ok(move |split| handle.spawn(split))
        .try_buffered(concurrency)
        .try_filter_map(|array| future::ready(Ok(array)))
        .boxed()
}

/// The first `limit` rows of `batches`. Drops the upstream stream immediately once the row
/// budget is reached (including `limit == 0`) so no extra batch is polled.
fn take_rows(
    batches: BoxStream<'static, VortexResult<RecordBatch>>,
    limit: u64,
) -> BoxStream<'static, VortexResult<RecordBatch>> {
    let mut remaining = usize::try_from(limit).unwrap_or(usize::MAX);
    let mut batches = (remaining > 0).then_some(batches);
    stream::poll_fn(move |cx| {
        let Some(stream) = batches.as_mut() else {
            return Poll::Ready(None);
        };
        let batch = match ready!(stream.poll_next_unpin(cx)) {
            Some(Ok(batch)) => batch,
            end => {
                batches = None;
                return Poll::Ready(end);
            }
        };
        let rows = batch.num_rows().min(remaining);
        remaining -= rows;
        if remaining == 0 {
            batches = None;
        }
        Poll::Ready(Some(Ok(if rows < batch.num_rows() {
            batch.slice(0, rows)
        } else {
            batch
        })))
    })
    .boxed()
}

fn to_record_batch(chunk: ArrayRef, target: &Field) -> VortexResult<RecordBatch> {
    let arrow = chunk.execute_arrow(Some(target), &mut session().create_execution_ctx())?;
    Ok(RecordBatch::from(arrow.as_struct().clone()))
}

#[pymethods]
impl PyVortexDataset {
    fn schema(self_: PyRef<Self>) -> PyResult<Py<PyAny>> {
        Arc::clone(&self_.schema).to_pyarrow(self_.py())
    }

    /// The default read concurrency of the dataset, or `None` for the automatic default.
    #[getter]
    fn concurrency(&self) -> Option<usize> {
        self.concurrency
    }

    /// The number of files in the dataset.
    #[getter]
    fn file_count(&self) -> usize {
        match &self.source {
            DatasetSource::Opened(_) => 1,
            DatasetSource::Lazy { files, .. } => files.len(),
        }
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

    #[pyo3(signature = (*, columns = None, row_filter = None, split_by = None, row_range = None, limit = None, concurrency = None))]
    pub fn to_record_batch_reader(
        self_: PyRef<Self>,
        columns: Option<Vec<ProjectionColumn>>,
        row_filter: Option<&Bound<'_, PyExpr>>,
        split_by: Option<usize>,
        row_range: Option<(u64, u64)>,
        limit: Option<u64>,
        concurrency: Option<usize>,
    ) -> PyVortexResult<Py<PyAny>> {
        let projection = projection_from_python(columns, self_.dtype())?;
        let filter = filter_from_python(row_filter);
        let concurrency = check_concurrency(concurrency)?;

        // Building the reader is lazy and cheap, so it runs without releasing the GIL. The scan
        // runs as pyarrow pulls batches, and pyarrow releases the GIL while it does.
        let bound_projection = projection.bind(self_.dtype())?.optimize_recursive()?;
        let schema = Arc::new(
            session()
                .arrow()
                .to_arrow_schema(bound_projection.dtype())?,
        );
        let target = Field::new_struct("", schema.fields().clone(), false);

        let (_, mut batches) = self_.scan_stream(
            projection,
            filter,
            None,
            row_range,
            split_by,
            concurrency,
            move |chunk| to_record_batch(chunk, &target),
        )?;
        if let Some(limit) = limit {
            batches = take_rows(batches, limit);
        }

        let batches = current_runtime()
            .block_on_stream(batches)
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
            let source = self_.source.clone();
            let row_count: u64 = self_.py().detach(move || {
                current_runtime().block_on(count_rows_metadata(source, row_range))
            })?;
            return row_count
                .try_into()
                .map_err(|e| PyValueError::new_err(e).into());
        }

        let filter = filter_from_python(row_filter);
        let projection = select(FieldNames::empty(), root());
        let (_, counts) = self_.scan_stream(
            projection,
            filter,
            None,
            row_range,
            split_by,
            concurrency,
            |array| Ok(array.len()),
        )?;
        let n_rows = self_.py().detach(move || {
            current_runtime()
                .block_on_stream(counts)
                .sum::<VortexResult<usize>>()
        })?;

        Ok(n_rows)
    }

    /// The natural splits of this Dataset, as ranges of dataset rows.
    #[pyo3(signature = (*))]
    pub fn splits(&self, py: Python<'_>) -> PyVortexResult<Vec<(u64, u64)>> {
        Ok(self
            .fragment_splits(py)?
            .into_iter()
            .map(|(_, range)| range)
            .collect())
    }

    /// The natural splits of this Dataset, each paired with its owning file's starting dataset row.
    #[pyo3(signature = (*))]
    pub fn fragment_splits(&self, py: Python<'_>) -> PyVortexResult<Vec<(u64, (u64, u64))>> {
        let source = self.source.clone();
        Ok(py.detach(move || current_runtime().block_on(collect_fragment_splits(source)))?)
    }
}

async fn count_rows_metadata(
    source: DatasetSource,
    row_range: Option<(u64, u64)>,
) -> VortexResult<u64> {
    match source {
        DatasetSource::Opened(file) => {
            let (start, end) = row_range.unwrap_or((0, u64::MAX));
            let last = end.min(file.row_count());
            Ok(last.saturating_sub(start))
        }
        DatasetSource::Lazy { files, dtype } => {
            let first_name = files[0].1.path.clone();
            let counts: Vec<u64> = stream::iter(0..files.len())
                .map(|idx| {
                    let files = Arc::clone(&files);
                    let dtype = dtype.clone();
                    let first_name = first_name.clone();
                    async move {
                        let (fs, listing) = &files[idx];
                        let file = open_lazy_file(fs, listing, Some(&dtype), &first_name).await?;
                        Ok::<_, VortexError>(file.row_count())
                    }
                })
                .buffered(OPEN_CONCURRENCY)
                .try_collect()
                .await?;

            let (start, end) = row_range.unwrap_or((0, u64::MAX));
            let mut total = 0u64;
            let mut file_start = 0u64;
            for count in counts {
                let file_end = file_start + count;
                let first = start.max(file_start);
                let last = end.min(file_end);
                total += last.saturating_sub(first);
                file_start = file_end;
            }
            Ok(total)
        }
    }
}

async fn collect_fragment_splits(source: DatasetSource) -> VortexResult<Vec<(u64, (u64, u64))>> {
    match source {
        DatasetSource::Opened(file) => Ok(file
            .splits()?
            .into_iter()
            .map(|rows| (0, (rows.start, rows.end)))
            .collect()),
        DatasetSource::Lazy { files, dtype } => {
            let first_name = files[0].1.path.clone();
            let per_file: Vec<(u64, Vec<Range<u64>>)> = stream::iter(0..files.len())
                .map(|idx| {
                    let files = Arc::clone(&files);
                    let dtype = dtype.clone();
                    let first_name = first_name.clone();
                    async move {
                        let (fs, listing) = &files[idx];
                        let file = open_lazy_file(fs, listing, Some(&dtype), &first_name).await?;
                        let row_count = file.row_count();
                        let splits = file.splits()?.into_iter().collect::<Vec<_>>();
                        Ok::<_, VortexError>((row_count, splits))
                    }
                })
                .buffered(OPEN_CONCURRENCY)
                .try_collect()
                .await?;

            let mut out = Vec::new();
            let mut file_start = 0u64;
            for (row_count, splits) in per_file {
                for rows in splits {
                    out.push((file_start, (file_start + rows.start, file_start + rows.end)));
                }
                file_start += row_count;
            }
            Ok(out)
        }
    }
}

/// Open the Vortex files that `sources` name as one dataset, in the order the sources name them.
#[pyfunction]
#[pyo3(signature = (sources, *, store = None, concurrency = None, expand = true))]
pub fn dataset_from_sources(
    py: Python,
    sources: Vec<String>,
    store: Option<AnyVortexStore>,
    concurrency: Option<usize>,
    expand: bool,
) -> PyVortexResult<PyVortexDataset> {
    open_dataset(py, sources, store, concurrency, expand)
}

/// Open the Vortex files that `sources` name as one dataset.
///
/// Every Python entry point that reads from paths or URLs opens its files here, so all of them
/// expand directories, prefixes and glob patterns the same way.
pub(crate) fn open_dataset(
    py: Python,
    sources: Vec<String>,
    store: Option<AnyVortexStore>,
    concurrency: Option<usize>,
    expand: bool,
) -> PyVortexResult<PyVortexDataset> {
    let concurrency = check_concurrency(concurrency)?;
    let store = store.map(AnyVortexStore::into_inner);
    let (files, dtype) =
        py.detach(move || current_runtime().block_on(discover_sources(sources, store, expand)))?;

    Ok(PyVortexDataset::from_lazy(files, dtype, concurrency)?)
}

async fn discover_sources(
    sources: Vec<String>,
    store: Option<Arc<dyn ObjectStore>>,
    expand: bool,
) -> VortexResult<(Vec<(FileSystemRef, FileListing)>, DType)> {
    if sources.is_empty() {
        vortex_bail!("a dataset needs at least one source");
    }

    let mut files = Vec::new();
    for source in &sources {
        let found = discover_source(source, store.clone(), expand).await?;
        if found.is_empty() {
            vortex_bail!("no Vortex files found for {source:?}");
        }
        files.extend(found);
    }

    // Open only the first file eagerly to confirm it is readable and learn the dataset's dtype;
    // the remaining files open lazily when scanned.
    let (first_fs, first_listing) = &files[0];
    let first_file = open_lazy_file(first_fs, first_listing, None, &first_listing.path).await?;
    let dtype = first_file.dtype().clone();
    drop(first_file);

    Ok((files, dtype))
}

fn is_url_source(source: &str) -> bool {
    source.contains("://")
}

/// Normalize a relative local glob pattern whose prefix before the first glob character has no
/// directory separator (e.g. `*.vortex` -> `./*.vortex`) so `StdFileSystem::list` lists `.`.
fn normalize_local_glob(pattern: &str) -> String {
    let glob_pos = pattern.find(['*', '?', '[']).unwrap_or(pattern.len());
    if !Path::new(pattern).is_absolute() && !pattern[..glob_pos].contains('/') {
        format!("./{pattern}")
    } else {
        pattern.to_string()
    }
}

/// Resolve `source` to `(FileSystemRef, FileListing)` pairs in sorted path order.
///
/// When `expand` is true, existing literal files are taken literally before globbing, directories
/// and trailing-`/` prefixes list all `.vortex` files below them, and other patterns resolve
/// through [`vortex::io::filesystem::FileSystem::glob`]. When `expand` is false, each source is treated as an exact
/// file path without glob expansion.
async fn discover_source(
    source: &str,
    store: Option<Arc<dyn ObjectStore>>,
    expand: bool,
) -> VortexResult<Vec<(FileSystemRef, FileListing)>> {
    if store.is_none() && is_url_source(source) && source.contains('?') {
        vortex_bail!(
            "URL source {source:?} contains '?'; '?' is not supported in URL sources because URL parsing treats it as a query string"
        );
    }

    match resolve_store(source, store)? {
        ResolvedStore::Path(path) => {
            let fs: FileSystemRef = Arc::new(StdFileSystem::new(session().handle()));
            let path_str = path
                .to_str()
                .ok_or_else(|| vortex_err!("{} is not a UTF-8 path", path.display()))?;

            if !expand {
                return Ok(vec![(
                    fs,
                    FileListing {
                        path: path_str.to_string(),
                        size: None,
                    },
                )]);
            }

            // Take an existing file literally before treating metacharacters like `[` as a glob.
            if let Some(listing) = fs.head(path_str).await? {
                return Ok(vec![(fs, listing)]);
            }

            let mut listings: Vec<FileListing> = if path.is_dir() {
                let suffix = format!(".{VORTEX_FILE_EXTENSION}");
                fs.list(path_str)
                    .try_filter(|listing| future::ready(listing.path.ends_with(&suffix)))
                    .try_collect()
                    .await?
            } else {
                let pattern = normalize_local_glob(path_str);
                fs.glob(&pattern)?.try_collect().await?
            };

            listings.sort_by(|a, b| a.path.cmp(&b.path));
            Ok(listings
                .into_iter()
                .map(|listing| (Arc::clone(&fs), listing))
                .collect())
        }
        ResolvedStore::ObjectStore(store, path) => {
            let fs: FileSystemRef = Arc::new(ObjectStoreFileSystem::new(
                Arc::clone(&store),
                session().handle(),
            ));
            let key = path.as_ref();

            if !expand {
                return Ok(vec![(
                    fs,
                    FileListing {
                        path: key.to_string(),
                        size: None,
                    },
                )]);
            }

            // Object store paths drop a trailing `/`, so the source says whether it is a prefix.
            let is_prefix = source.ends_with('/') || key.is_empty();
            let mut listings: Vec<FileListing> = if is_prefix {
                let suffix = format!(".{VORTEX_FILE_EXTENSION}");
                fs.list(key)
                    .try_filter(|listing| future::ready(listing.path.ends_with(&suffix)))
                    .try_collect()
                    .await?
            } else if let Some(listing) = fs.head(key).await? {
                // Take an existing object literally before treating metacharacters like `[` as a glob.
                return Ok(vec![(fs, listing)]);
            } else {
                fs.glob(key)?.try_collect().await?
            };

            listings.sort_by(|a, b| a.path.cmp(&b.path));
            Ok(listings
                .into_iter()
                .map(|listing| (Arc::clone(&fs), listing))
                .collect())
        }
    }
}
