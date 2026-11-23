// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::ops::Range;
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
use glob::MatchOptions;
use glob::Pattern;
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
use crate::file::validate_positive_split_size;
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

struct LazyOffsetState<F> {
    idx: usize,
    file_start: u64,
    files: Arc<[(FileSystemRef, FileListing)]>,
    dtype: DType,
    first_name: String,
    params: ScanParams,
    map_fn: F,
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
            let first_name = files[0].1.path.clone();
            if params.row_range.is_none() && params.indices.is_none() {
                return stream::iter(0..files.len())
                    .then(move |idx| {
                        let (files, dtype, first_name, params, map_fn) = (
                            Arc::clone(&files),
                            dtype.clone(),
                            first_name.clone(),
                            params.clone(),
                            map_fn.clone(),
                        );
                        async move {
                            let (fs, listing) = &files[idx];
                            let file =
                                open_lazy_file(fs, listing, Some(&dtype), &first_name).await?;
                            Ok::<_, VortexError>(
                                file.scan()?
                                    .with_projection(params.projection)
                                    .with_some_filter(params.filter)
                                    .with_split_by(params.split_by)
                                    .map(map_fn),
                            )
                        }
                    })
                    .boxed();
            }

            let state = LazyOffsetState {
                idx: 0,
                file_start: 0,
                files,
                dtype,
                first_name,
                params,
                map_fn,
            };
            stream::unfold(state, |mut st| async move {
                let max_index = st.params.indices.as_ref().and_then(|i| i.last().copied());
                let range_end = st.params.row_range.map(|(_, end)| end);
                while st.idx < st.files.len() {
                    if range_end.is_some_and(|end| st.file_start >= end)
                        || max_index.is_some_and(|last| st.file_start > last)
                    {
                        return None;
                    }
                    let (fs, listing) = &st.files[st.idx];
                    st.idx += 1;
                    let file =
                        match open_lazy_file(fs, listing, Some(&st.dtype), &st.first_name).await {
                            Ok(file) => file,
                            Err(e) => {
                                st.idx = st.files.len();
                                return Some((Err(e), st));
                            }
                        };
                    let file_rows = file.row_count();
                    let scan = build_file_scan(&file, st.file_start, &st.params, &st.map_fn);
                    st.file_start += file_rows;
                    match scan {
                        Ok(Some(scan)) => return Some((Ok(scan), st)),
                        Ok(None) => continue,
                        Err(e) => {
                            st.idx = st.files.len();
                            return Some((Err(e), st));
                        }
                    }
                }
                None
            })
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
        let split_by = validate_positive_split_size(split_by)?;
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
        let split_by = validate_positive_split_size(split_by)?;
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
        let source = self.source.clone();
        let per_file =
            py.detach(move || current_runtime().block_on(collect_file_splits(source)))?;
        let mut out = Vec::new();
        let mut file_start = 0u64;
        for (row_count, splits) in per_file {
            for rows in splits {
                out.push((file_start + rows.start, file_start + rows.end));
            }
            file_start += row_count;
        }
        Ok(out)
    }

    /// The natural splits of this Dataset, each paired with its owning file index and file-local row range.
    #[pyo3(signature = (*))]
    pub fn fragment_splits(&self, py: Python<'_>) -> PyVortexResult<Vec<(usize, (u64, u64))>> {
        let source = self.source.clone();
        let per_file =
            py.detach(move || current_runtime().block_on(collect_file_splits(source)))?;
        let mut out = Vec::new();
        for (file_idx, (_, splits)) in per_file.into_iter().enumerate() {
            for rows in splits {
                out.push((file_idx, (rows.start, rows.end)));
            }
        }
        Ok(out)
    }

    /// Return a single-file dataset view over the `file_idx`-th file of this dataset.
    pub fn for_file(&self, file_idx: usize) -> PyVortexResult<Self> {
        let source = match &self.source {
            DatasetSource::Opened(file) if file_idx == 0 => DatasetSource::Opened(file.clone()),
            DatasetSource::Lazy { files, dtype } => {
                let entry = files.get(file_idx).ok_or_else(|| {
                    vortex_err!(
                        "file index {file_idx} out of range for dataset with {} files",
                        files.len()
                    )
                })?;
                DatasetSource::Lazy {
                    files: Arc::from([entry.clone()]),
                    dtype: dtype.clone(),
                }
            }
            DatasetSource::Opened(_) => {
                return Err(vortex_err!(
                    "file index {file_idx} out of range for single-file dataset"
                )
                .into());
            }
        };
        Ok(Self {
            source,
            schema: Arc::clone(&self.schema),
            concurrency: self.concurrency,
        })
    }
}

async fn map_lazy_files<T, F>(
    files: Arc<[(FileSystemRef, FileListing)]>,
    dtype: DType,
    f: F,
) -> VortexResult<Vec<T>>
where
    F: Fn(&VortexFile) -> VortexResult<T> + Send + Sync + Clone,
{
    let first_name = files[0].1.path.clone();
    stream::iter(0..files.len())
        .map(|idx| {
            let (files, dtype, first_name, f) = (
                Arc::clone(&files),
                dtype.clone(),
                first_name.clone(),
                f.clone(),
            );
            async move {
                let (fs, listing) = &files[idx];
                let file = open_lazy_file(fs, listing, Some(&dtype), &first_name).await?;
                f(&file)
            }
        })
        .buffered(OPEN_CONCURRENCY)
        .try_collect()
        .await
}

async fn count_rows_metadata(
    source: DatasetSource,
    row_range: Option<(u64, u64)>,
) -> VortexResult<u64> {
    let counts = match source {
        DatasetSource::Opened(file) => vec![file.row_count()],
        DatasetSource::Lazy { files, dtype } => {
            map_lazy_files(files, dtype, |file| Ok(file.row_count())).await?
        }
    };
    let (start, end) = row_range.unwrap_or((0, u64::MAX));
    let mut total = 0u64;
    let mut file_start = 0u64;
    for count in counts {
        let file_end = file_start + count;
        total += end.min(file_end).saturating_sub(start.max(file_start));
        file_start = file_end;
    }
    Ok(total)
}

async fn collect_file_splits(source: DatasetSource) -> VortexResult<Vec<(u64, Vec<Range<u64>>)>> {
    match source {
        DatasetSource::Opened(file) => Ok(vec![(
            file.row_count(),
            file.splits()?.into_iter().collect(),
        )]),
        DatasetSource::Lazy { files, dtype } => {
            map_lazy_files(files, dtype, |file| {
                Ok((file.row_count(), file.splits()?.into_iter().collect()))
            })
            .await
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

fn has_glob(pattern: &str) -> bool {
    pattern.contains(['*', '?', '['])
}

fn glob_match_options() -> MatchOptions {
    MatchOptions {
        require_literal_separator: true,
        ..MatchOptions::new()
    }
}

/// Resolve `source` to `(FileSystemRef, FileListing)` pairs in sorted path order.
///
/// When `expand` is true, existing literal files are taken literally before globbing, directories
/// and trailing-`/` prefixes list all `.vortex` files below them, and globs require literal path
/// separators (`*` does not cross `/`). Local paths are anchored to the current working directory
/// at discovery time (escaping glob metacharacters in the working directory) so later `os.chdir()`
/// calls cannot redirect lazy opens. When `expand` is false, each source is treated as an exact
/// file path without glob expansion.
async fn discover_source(
    source: &str,
    store: Option<Arc<dyn ObjectStore>>,
    expand: bool,
) -> VortexResult<Vec<(FileSystemRef, FileListing)>> {
    if store.is_none() && source.contains("://") && source.contains('?') {
        vortex_bail!(
            "URL source {source:?} contains '?'; '?' is not supported in URL sources because URL parsing treats it as a query string"
        );
    }

    let (fs, literal_key, glob_key, is_prefix): (FileSystemRef, String, String, bool) =
        match resolve_store(source, store)? {
            ResolvedStore::Path(path) => {
                let (abs, glob_pattern) = if path.is_relative() {
                    let cwd = std::env::current_dir()?;
                    let cwd_str = cwd
                        .to_str()
                        .ok_or_else(|| vortex_err!("{} is not a UTF-8 path", cwd.display()))?;
                    let rel_str = path
                        .to_str()
                        .ok_or_else(|| vortex_err!("{} is not a UTF-8 path", path.display()))?;
                    let sep = if cwd_str.ends_with('/') { "" } else { "/" };
                    let escaped = format!("{}{sep}{rel_str}", Pattern::escape(cwd_str));
                    (cwd.join(&path), escaped)
                } else {
                    let abs_str = path
                        .to_str()
                        .ok_or_else(|| vortex_err!("{} is not a UTF-8 path", path.display()))?
                        .to_string();
                    (path, abs_str)
                };
                let abs_str = abs
                    .to_str()
                    .ok_or_else(|| vortex_err!("{} is not a UTF-8 path", abs.display()))?
                    .to_string();
                let is_dir = abs.is_dir();
                let fs: FileSystemRef = Arc::new(StdFileSystem::new(session().handle()));
                (fs, abs_str, glob_pattern, is_dir)
            }
            ResolvedStore::ObjectStore(store, path) => {
                let fs: FileSystemRef =
                    Arc::new(ObjectStoreFileSystem::new(store, session().handle()));
                let key = path.as_ref().to_string();
                let is_prefix = source.ends_with('/') || key.is_empty();
                (fs, key.clone(), key, is_prefix)
            }
        };

    if !expand {
        return Ok(vec![(
            fs,
            FileListing {
                path: literal_key,
                size: None,
            },
        )]);
    }

    let mut listings: Vec<FileListing> = if is_prefix {
        let suffix = format!(".{VORTEX_FILE_EXTENSION}");
        fs.list(&literal_key)
            .try_filter(|listing| future::ready(listing.path.ends_with(&suffix)))
            .try_collect()
            .await?
    } else if !has_glob(&glob_key) {
        fs.head(&literal_key).await?.into_iter().collect()
    } else if let Ok(Some(listing)) = fs.head(&literal_key).await {
        // Take an existing file/object literally before globbing; ignore HEAD errors (e.g.
        // S3 403 on prefix-scoped roles or Windows invalid-name on `*`) so globbing runs.
        return Ok(vec![(fs, listing)]);
    } else {
        let compiled = Pattern::new(&glob_key)
            .map_err(|e| vortex_err!("invalid glob pattern {source:?}: {e}"))?;
        let options = glob_match_options();
        let raw: Vec<FileListing> = fs.glob(&glob_key)?.try_collect().await?;
        raw.into_iter()
            .filter(|listing| compiled.matches_with(&listing.path, options))
            .collect()
    };

    listings.sort_by(|a, b| a.path.cmp(&b.path));
    Ok(listings
        .into_iter()
        .map(|listing| (Arc::clone(&fs), listing))
        .collect())
}
