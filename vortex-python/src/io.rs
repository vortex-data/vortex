// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::sync::Arc;

use arrow_array::RecordBatchReader;
use arrow_array::ffi_stream::RecordBatchStreamReader;
use arrow_schema::Schema;
use pyo3::exceptions::PyIndexError;
use pyo3::exceptions::PyTypeError;
use pyo3::exceptions::PyValueError;
use pyo3::intern;
use pyo3::prelude::*;
use pyo3::pyfunction;
use pyo3_object_store::PyObjectStore;
use vortex::array::ArrayRef;
use vortex::array::Canonical;
use vortex::array::IntoArray;
use vortex::array::iter::ArrayIterator;
use vortex::array::iter::ArrayIteratorAdapter;
use vortex::array::iter::ArrayIteratorExt;
use vortex::buffer::Buffer;
use vortex::compressor::BtrBlocksCompressorBuilder;
use vortex::error::VortexError;
use vortex::error::VortexResult;
use vortex::file::WriteOptionsSessionExt;
use vortex::file::WriteStrategyBuilder;
use vortex::io::VortexWrite;
use vortex::io::object_store::ObjectStoreWrite;
use vortex::io::runtime::BlockingRuntime;
use vortex::io::std_file::FileWrite;
use vortex::scan::strict_sorted_buffer::StrictSortedBuffer;
use vortex_arrow::ArrowSessionExt;

use crate::arrays::PyArray;
use crate::arrays::PyArrayRef;
use crate::arrow::FromPyArrow;
use crate::classes::record_batch_reader_class;
use crate::classes::table_class;
use crate::current_runtime;
use crate::dataset::ProjectionColumn;
use crate::dataset::open_dataset;
use crate::error::PyVortexResult;
use crate::expr::PyExpr;
use crate::file::row_indices;
use crate::hf_store::HfStore;
use crate::install_module;
use crate::iter::PyArrayIterator;
use crate::object_store::resolve::ResolvedStore;
use crate::object_store::resolve::resolve_store;
use crate::opendal_store::CosStore;
use crate::opendal_store::GoosefsStore;
use crate::session::session;
use crate::session::write_session;

pub(crate) fn init(py: Python, parent: &Bound<PyModule>) -> PyResult<()> {
    let m = PyModule::new(py, "io")?;
    parent.add_submodule(&m)?;
    install_module("vortex._lib.io", &m)?;

    m.add_function(wrap_pyfunction!(read_url, &m)?)?;
    m.add_function(wrap_pyfunction!(write, &m)?)?;

    m.add_class::<PyVortexWriteOptions>()?;

    Ok(())
}

/// Read a vortex struct array from a URL.
///
/// Parameters
/// ----------
/// url : str
///     The URL to read from. Like a source of :func:`vortex.open_dataset`, it can also name a
///     directory, an object store prefix ending in ``/`` or a glob pattern, and then reads the rows
///     of every Vortex file it names, in sorted order.
/// store : vortex.store.AzureStore | vortex.store.CosStore | vortex.store.GCSStore | vortex.store.HfStore | vortex.store.HTTPStore | vortex.store.LocalStore | vortex.store.MemoryStore | vortex.store.S3Store | None
///     Pre-configured object store with credentials and settings.
///     If provided, uses this store's configuration.
///     If None, checks session registry for matching URL pattern.
///     If not found, raises VortexError.
/// projection : list[str | int] | None
///     The columns to read identified either by their index or name.
/// row_filter : Expr | None
///     Keep only the rows for which this expression evaluates to true.
/// indices : Array | None
///     A list of rows to keep identified by the zero-based index within the file. NB: If row_range
///     is specified, these indices are within the row range, not the file!
/// row_range : tuple[int, int] | None
///     A left-inclusive, right-exclusive range of rows to read.
///
/// Examples
/// --------
///
/// Read an array from an HTTPS URL:
///
/// >>> import vortex as vx
/// >>> a = vx.io.read_url("https://example.com/dataset.vortex")  # doctest: +SKIP
///
/// Read an array from an S3 URL:
///
/// >>> a = vx.io.read_url("s3://bucket/path/to/dataset.vortex")  # doctest: +SKIP
///
/// Read an array from an Azure Blob File System URL:
///
/// >>> a = vx.io.read_url("abfss://my_file_system@my_account.dfs.core.windows.net/path/to/dataset.vortex")  # doctest: +SKIP
///
/// Read an array from an Azure Blob Storage URL:
///
/// >>> a = vx.io.read_url("https://my_account.blob.core.windows.net/my_container/path/to/dataset.vortex")  # doctest: +SKIP
///
/// Read an array from a Google Storage URL:
///
/// >>> a = vx.io.read_url("gs://bucket/path/to/dataset.vortex")  # doctest: +SKIP
///
/// Read an array from a local file URL:
///
/// >>> a = vx.io.read_url("file:///path/to/dataset.vortex")  # doctest: +SKIP
///
/// Read from S3 with explicit credentials:
///
/// >>> from vortex import store as S
/// >>> store = S.S3Store(
/// ...     bucket="my-bucket",
/// ...     region="us-east-1",
/// ...     access_key_id="AKIA...",
/// ...     secret_access_key="..."
/// ... )
/// >>> a = vx.io.read_url("s3://my-bucket/data.vortex", store=store)  # doctest: +SKIP

#[pyfunction]
#[pyo3(signature = (url, *, store = None, projection = None, row_filter = None, indices = None, row_range = None))]
pub fn read_url<'py>(
    py: Python<'py>,
    url: &str,
    store: Option<Bound<'py, PyAny>>,
    projection: Option<Vec<ProjectionColumn>>,
    row_filter: Option<&Bound<'py, PyExpr>>,
    indices: Option<PyArrayRef>,
    row_range: Option<(u64, u64)>,
) -> PyVortexResult<PyArrayRef> {
    let store = store
        .map(|store| store.extract::<AnyVortexStore>())
        .transpose()?;
    let dataset = open_dataset(py, vec![url.to_string()], store, None, true)?;
    let indices = row_indices(py, indices)?;
    let (indices, row_range) = match (indices, row_range) {
        (Some(indices), Some((start, end))) => {
            let shifted = shift_row_range_indices(indices, start, end)?;
            // Every shifted index already falls within `start..end`. Drop `row_range` when
            // non-empty so `dataset.to_array_inner` still checks shifted indices against the
            // dataset's actual row count (e.g. when `end` exceeds the file row count).
            let effective_range = shifted.is_empty().then_some((start, end));
            (Some(shifted), effective_range)
        }
        other => other,
    };

    dataset.to_array_inner(py, projection, row_filter, indices, row_range, None)
}

/// Translate indices relative to `start..end` into dataset-global indices, rejecting any index
/// outside the range length.
fn shift_row_range_indices(
    indices: StrictSortedBuffer<u64>,
    start: u64,
    end: u64,
) -> PyVortexResult<StrictSortedBuffer<u64>> {
    let range_len = end.saturating_sub(start);
    if let Some(&last) = indices.last()
        && last >= range_len
    {
        return Err(PyIndexError::new_err(format!(
            "index {last} is out of bounds for row_range ({start}, {end}) of length {range_len}"
        ))
        .into());
    }
    if start == 0 {
        return Ok(indices);
    }
    // Every index is strictly less than `end - start`, so `start + idx < end <= u64::MAX`
    // cannot overflow and adding a constant preserves strict ascending order.
    let shifted: Buffer<u64> = indices.iter().map(|&idx| start + idx).collect();
    Ok(StrictSortedBuffer::try_new(shifted)?)
}

/// A store object accepted by `read_url` / `write`.
///
/// This recognizes the built-in `pyo3-object_store` classes (S3, Azure, GCS, HTTP, Local, Memory)
/// and Vortex's own classes: the Hugging Face Hub store and the OpenDAL-backed `CosStore` and
/// `GoosefsStore`.
pub(crate) enum AnyVortexStore {
    /// A store extracted from one of the built-in `pyo3-object_store` classes.
    Builtin(PyObjectStore),
    /// Vortex's Hugging Face Hub store.
    Hf(HfStore),
    /// Vortex's OpenDAL-backed COS store.
    Cos(CosStore),
    /// Vortex's OpenDAL-backed GooseFS store.
    Goosefs(GoosefsStore),
}

impl AnyVortexStore {
    /// Consume self and return the underlying `Arc<dyn ObjectStore>`.
    pub(crate) fn into_inner(self) -> Arc<dyn object_store::ObjectStore> {
        match self {
            AnyVortexStore::Builtin(s) => s.into_inner(),
            AnyVortexStore::Hf(s) => s.to_arc(),
            AnyVortexStore::Cos(s) => s.to_arc(),
            AnyVortexStore::Goosefs(s) => s.to_arc(),
        }
    }
}

impl<'py> FromPyObject<'_, 'py> for AnyVortexStore {
    type Error = PyErr;

    fn extract(obj: Borrowed<'_, 'py, PyAny>) -> PyResult<Self> {
        if let Ok(builtin) = obj.extract::<PyObjectStore>() {
            return Ok(AnyVortexStore::Builtin(builtin));
        }
        if let Ok(hf) = obj.extract::<HfStore>() {
            return Ok(AnyVortexStore::Hf(hf));
        }
        if let Ok(cos) = obj.extract::<CosStore>() {
            return Ok(AnyVortexStore::Cos(cos));
        }
        if let Ok(goosefs) = obj.extract::<GoosefsStore>() {
            return Ok(AnyVortexStore::Goosefs(goosefs));
        }
        Err(PyTypeError::new_err(
            "Expected an object store instance (S3/Azure/GCS/HTTP/Local/Memory/HF/COS/OSS/GooseFS store)",
        ))
    }
}

/// Write an array to a Vortex file.
///
/// Parameters
/// ----------
/// iter : vortex.Array | vortex.ArrayIterator | pyarrow.Table | pyarrow.RecordBatchReader
///     The data to write. Can be a single array, an array iterator, or a PyArrow object that supports streaming.
///     When using PyArrow objects, data is streamed directly without loading the entire dataset into memory.
///
/// path : str
///     The file path.
///
/// store : vortex.store.AzureStore | vortex.store.CosStore | vortex.store.GCSStore | vortex.store.HfStore | vortex.store.HTTPStore | vortex.store.LocalStore | vortex.store.MemoryStore | vortex.store.S3Store | None
///     An optional object store configuration to use for writing the output.
///
/// edition : str | None
///     An optional Vortex edition, for example ``"core2025.10.0"``. The file then contains only
///     the formats of this edition, so that older Vortex readers can read it. If None, the
///     default edition is used.
///
/// Examples
/// --------
///
/// Write a single Vortex array `a` to the local file `a.vortex`.
///
/// >>> import vortex as vx
/// >>> a = vx.array([
/// ...     {'x': 1},
/// ...     {'x': 2},
/// ...     {'x': 10},
/// ...     {'x': 11},
/// ...     {'x': None},
/// ... ])
/// >>> vx.io.write(a, "a.vortex") # doctest: +SKIP
///
/// Stream a PyArrow Table directly to Vortex without loading into memory:
///
/// >>> import pyarrow as pa
/// >>> import vortex as vx
/// >>> table = pa.table({'x': [1, 2, 3, 4, 5]})
/// >>> vx.io.write(table, "streamed.vortex")  # doctest: +SKIP
///
/// Stream from a PyArrow RecordBatchReader:
///
/// >>> import pyarrow as pa
/// >>> import vortex as vx
/// >>> reader = pa.RecordBatchReader.from_batches(schema, batches) # doctest: +SKIP
/// >>> vx.io.write(reader, "streamed.vortex")  # doctest: +SKIP
///
/// See also
/// --------
///
/// :func:`vortex.io.VortexWriteOptions`
#[pyfunction]
#[pyo3(signature = (iter, path, *, store = None, edition = None))]
pub fn write<'py>(
    py: Python<'py>,
    iter: &Bound<'py, PyAny>,
    path: &str,
    store: Option<AnyVortexStore>,
    edition: Option<&str>,
) -> PyVortexResult<()> {
    write_inner(py, iter, path, store, edition, None)
}

/// Write Vortex files with custom configuration.
///
/// See also
/// --------
///
/// :func:`vortex.io.write`.
#[pyclass(name = "VortexWriteOptions", module = "io", frozen)]
pub(crate) struct PyVortexWriteOptions {
    use_compact_encodings: bool,
    edition: Option<String>,
}

#[pymethods]
impl PyVortexWriteOptions {
    /// Balance size, read-throughput, and read-latency.
    #[staticmethod]
    pub fn default() -> Self {
        Self {
            use_compact_encodings: false,
            edition: None,
        }
    }

    /// Prioritize small size over read-throughput and read-latency.
    ///
    /// Let's model some stock ticker data. As you may know, the stock market always (noisly) goes
    /// up:
    ///
    /// >>> import random
    /// >>> sprl = vx.array([random.randint(i, i + 10) for i in range(100_000)])
    ///
    /// If we naively wrote 8-bytes for each of these integers to a file we'd have 800,000 bytes!
    /// Let's see how small the array buffers are when we write with the default Vortex write
    /// options (which are also used by :func:`vortex.io.write`):
    ///
    /// >>> vx.io.VortexWriteOptions.default().write(sprl, "chonky.vortex")
    /// >>> vx.open("chonky.vortex").scan().read_all().nbytes
    /// 213248
    ///
    /// Wow, Vortex manages to use about two bytes per integer! So advanced. So tiny.
    ///
    /// But can we do better?
    ///
    /// We sure can.
    ///
    /// >>> vx.io.VortexWriteOptions.compact().write(sprl, "tiny.vortex")
    /// >>> vx.open("tiny.vortex").scan().read_all().nbytes
    /// 52564
    ///
    /// Random numbers are not (usually) composed of random bytes!
    #[staticmethod]
    pub fn compact() -> Self {
        Self {
            use_compact_encodings: true,
            edition: None,
        }
    }

    /// Return a copy of these options that writes with the Vortex edition ``edition``.
    ///
    /// The file then contains only the formats of this edition, so that older Vortex readers can
    /// read it. For example, ``"core2025.10.0"`` writes files that Vortex 0.73 can read.
    ///
    /// Parameters
    /// ----------
    /// edition : str
    ///     The name of the edition, for example ``"core2025.10.0"``.
    ///
    /// Raises
    /// ------
    /// ValueError
    ///     If Vortex does not know the edition.
    ///
    /// Examples
    /// --------
    ///
    /// >>> import vortex as vx
    /// >>> a = vx.array([0, 1, 2, 3, None, 4])
    /// >>> vx.io.VortexWriteOptions.default().with_edition("core2025.10.0").write(a, "a.vortex") # doctest: +SKIP
    pub fn with_edition(&self, edition: String) -> PyResult<Self> {
        write_session(Some(&edition)).map_err(|error| PyValueError::new_err(error.to_string()))?;

        Ok(Self {
            use_compact_encodings: self.use_compact_encodings,
            edition: Some(edition),
        })
    }

    /// Write an array or iterator of arrays to a file.
    ///
    ///
    /// Parameters
    /// ----------
    /// iter : vortex.Array | vortex.ArrayIterator | pyarrow.Table | pyarrow.RecordBatchReader
    ///     The data to write. Can be a single array, an array iterator, or a PyArrow object that supports streaming.
    ///     When using PyArrow objects, data is streamed directly without loading the entire dataset into memory.
    ///
    /// path : str
    ///     The file path.
    ///
    /// store : vortex.store.AzureStore | vortex.store.CosStore | vortex.store.GCSStore | vortex.store.HfStore | vortex.store.HTTPStore | vortex.store.LocalStore | vortex.store.MemoryStore | vortex.store.S3Store | None
    ///     An optional object store configuration to use for writing the output.
    ///
    /// Examples
    /// --------
    ///
    /// Write a single Vortex array `a` to the local file `a.vortex` using the default settings:
    ///
    /// >>> import vortex as vx
    /// >>> import random
    /// >>> a = vx.array([0, 1, 2, 3, None, 4])
    /// >>> vx.io.VortexWriteOptions.default().write(a, "a.vortex") # doctest: +SKIP
    ///
    /// Write the same array while preferring small file sizes over read-throughput and
    /// read-latency:
    ///
    /// >>> import vortex as vx
    /// >>> vx.io.VortexWriteOptions.compact().write(a, "a.vortex") # doctest: +SKIP
    ///
    /// See also
    /// --------
    ///
    /// :func:`vortex.io.write`
    #[pyo3(signature = (iter, path, *, store = None))]
    pub fn write<'py>(
        &self,
        py: Python<'py>,
        iter: &Bound<'py, PyAny>,
        path: &str,
        store: Option<AnyVortexStore>,
    ) -> PyVortexResult<()> {
        write_inner(
            py,
            iter,
            path,
            store,
            self.edition.as_deref(),
            Some(self.use_compact_encodings),
        )
    }
}

enum OpenWriter {
    ObjectStore(ObjectStoreWrite),
    File(FileWrite),
}

fn write_inner<'py>(
    py: Python<'py>,
    iter: &Bound<'py, PyAny>,
    path: &str,
    store: Option<AnyVortexStore>,
    edition: Option<&str>,
    compact_override: Option<bool>,
) -> PyVortexResult<()> {
    let session = write_session(edition)?;
    let resolved = resolve_store(path, store.map(|x| x.into_inner()))?;
    let input = WriteInput::validate(iter)?;

    let writer = py.detach(|| {
        current_runtime().block_on(async move {
            match resolved {
                ResolvedStore::ObjectStore(store, path) => VortexResult::Ok(
                    OpenWriter::ObjectStore(ObjectStoreWrite::new(store, &path).await?),
                ),
                ResolvedStore::Path(path) => VortexResult::Ok(OpenWriter::File(
                    FileWrite::create(path, current_runtime().handle()).await?,
                )),
            }
        })
    })?;

    let array_iter = input.into_array_iter(py)?;

    py.detach(|| {
        let write_options = match compact_override {
            Some(use_compact_encodings) => {
                let mut compressor = BtrBlocksCompressorBuilder::from_session(&session);
                if use_compact_encodings {
                    compressor = compressor.with_compact();
                }
                let strategy = WriteStrategyBuilder::from_session(&session)
                    .with_btrblocks_builder(compressor)
                    .build();
                session.write_options().with_strategy(strategy)
            }
            None => session.write_options(),
        };

        current_runtime().block_on(async move {
            match writer {
                OpenWriter::ObjectStore(mut store) => {
                    write_options
                        .write(&mut store, array_iter.into_array_stream())
                        .await?;
                    store.shutdown().await?;
                }
                OpenWriter::File(mut file) => {
                    write_options
                        .write(&mut file, array_iter.into_array_stream())
                        .await?;
                    file.shutdown().await?;
                }
            }
            VortexResult::Ok(())
        })
    })?;

    Ok(())
}

/// Validated write input whose stream ownership is deferred until the output writer is open.
enum WriteInput<'py> {
    PyIter(Bound<'py, PyArrayIterator>),
    Array(ArrayRef),
    Arrow(Bound<'py, PyAny>),
}

impl<'py> WriteInput<'py> {
    fn validate(ob: &Bound<'py, PyAny>) -> PyResult<Self> {
        if let Ok(py_iter) = ob.cast::<PyArrayIterator>() {
            py_iter.get().check_not_reentrant()?;
            return Ok(Self::PyIter(py_iter.clone()));
        }

        if let Ok(py_array) = ob.cast::<PyArray>() {
            return Ok(Self::Array(py_array.extract::<PyArrayRef>()?.into_inner()));
        }

        let py = ob.py();
        let pa_table = table_class(py)?;
        let pa_record_batch_reader = record_batch_reader_class(py)?;
        if ob.is_instance(pa_table)? || ob.is_instance(pa_record_batch_reader)? {
            // Validate the Arrow schema without exporting or consuming the stream yet.
            let schema_obj = ob.getattr(intern!(py, "schema"))?;
            let schema = Schema::from_pyarrow(&schema_obj.as_borrowed())?;
            session()
                .arrow()
                .from_arrow_schema(&schema)
                .map_err(|e| PyValueError::new_err(e.to_string()))?;
            return Ok(Self::Arrow(ob.clone()));
        }

        Err(PyTypeError::new_err(
            "Expected an object that can be converted to an ArrayIterator (PyArrayIterator, PyArray, or PyArrow object with streaming support)",
        ))
    }

    fn into_array_iter(self, py: Python<'py>) -> PyResult<Box<dyn ArrayIterator + Send>> {
        match self {
            Self::PyIter(py_iter) => {
                let py_iter = py_iter.get();
                let iter = py.detach(|| py_iter.take())?;
                Ok(iter.unwrap_or_else(|| {
                    Box::new(
                        Canonical::empty(py_iter.dtype())
                            .into_array()
                            .to_array_iterator(),
                    )
                }))
            }
            Self::Array(array) => Ok(Box::new(array.to_array_iterator())),
            Self::Arrow(ob) => arrow_stream_to_iterator(&ob.as_borrowed()),
        }
    }
}

/// Convert a validated PyArrow table or reader to a Vortex ArrayIterator using Arrow FFI streams.
fn arrow_stream_to_iterator(
    ob: &Borrowed<'_, '_, PyAny>,
) -> PyResult<Box<dyn ArrayIterator + Send>> {
    // Convert to Arrow stream using FFI
    let arrow_stream = RecordBatchStreamReader::from_pyarrow(ob)?;
    let dtype = session()
        .arrow()
        .from_arrow_schema(arrow_stream.schema().as_ref())
        .map_err(|e| PyValueError::new_err(e.to_string()))?;

    // Convert Arrow RecordBatch stream to Vortex ArrayIterator
    let vortex_iter = arrow_stream
        .into_iter()
        .map(|batch_result| -> VortexResult<ArrayRef> {
            let batch = batch_result.map_err(VortexError::from)?;
            let schema = batch.schema();
            session().arrow().from_arrow_record_batch(batch, &schema)
        });

    Ok(Box::new(ArrayIteratorAdapter::new(dtype, vortex_iter)))
}

#[cfg(test)]
mod tests {
    use rstest::rstest;
    use vortex::buffer::Buffer;
    use vortex::error::VortexResult;
    use vortex::scan::strict_sorted_buffer::StrictSortedBuffer;

    use super::shift_row_range_indices;
    use crate::error::PyVortexError;

    #[rstest]
    #[case::nonzero_start(&[0, 2, 9], 10, 20, &[10, 12, 19])]
    #[case::zero_start(&[0, 4], 0, 10, &[0, 4])]
    #[case::empty_indices(&[], 10, 20, &[])]
    #[case::empty_indices_empty_range(&[], 10, 10, &[])]
    #[case::near_u64_max(&[0, 4], u64::MAX - 5, u64::MAX, &[u64::MAX - 5, u64::MAX - 1])]
    fn test_shift_row_range_indices_valid(
        #[case] input: &[u64],
        #[case] start: u64,
        #[case] end: u64,
        #[case] expected: &[u64],
    ) -> VortexResult<()> {
        let sorted = StrictSortedBuffer::try_new(Buffer::copy_from(input))?;
        let shifted = match shift_row_range_indices(sorted, start, end) {
            Ok(shifted) => shifted,
            Err(PyVortexError::Vortex(err)) => return Err(err),
            Err(PyVortexError::Py(err)) => panic!("unexpected Python error: {err}"),
        };
        assert_eq!(shifted.as_slice(), expected);
        Ok(())
    }

    #[rstest]
    #[case::at_end(&[0, 10], 10, 20)]
    #[case::past_end(&[11], 10, 20)]
    #[case::empty_range(&[0], 10, 10)]
    #[case::inverted_range(&[0], 20, 10)]
    fn test_shift_row_range_indices_out_of_bounds(
        #[case] input: &[u64],
        #[case] start: u64,
        #[case] end: u64,
    ) -> VortexResult<()> {
        let sorted = StrictSortedBuffer::try_new(Buffer::copy_from(input))?;
        assert!(matches!(
            shift_row_range_indices(sorted, start, end),
            Err(PyVortexError::Py(_))
        ));
        Ok(())
    }
}
