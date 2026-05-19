// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use arrow_array::RecordBatchReader;
use arrow_array::ffi_stream::ArrowArrayStreamReader;
use async_fs::File;
use pyo3::exceptions::PyTypeError;
use pyo3::prelude::*;
use pyo3::pyfunction;
use pyo3_object_store::PyObjectStore;
use vortex::array::ArrayRef;
use vortex::array::Canonical;
use vortex::array::IntoArray;
use vortex::array::arrow::FromArrowArray;
use vortex::array::iter::ArrayIterator;
use vortex::array::iter::ArrayIteratorAdapter;
use vortex::array::iter::ArrayIteratorExt;
use vortex::compressor::BtrBlocksCompressorBuilder;
use vortex::dtype::DType;
use vortex::dtype::arrow::FromArrowType;
use vortex::error::VortexError;
use vortex::error::VortexResult;
use vortex::file::WriteOptionsSessionExt;
use vortex::file::WriteStrategyBuilder;
use vortex::io::VortexWrite;
use vortex::io::object_store::ObjectStoreWrite;
use vortex::io::runtime::BlockingRuntime;

use crate::PyVortex;
use crate::RUNTIME;
use crate::arrays::PyArray;
use crate::arrays::PyArrayRef;
use crate::arrow::FromPyArrow;
use crate::classes::record_batch_reader_class;
use crate::classes::table_class;
use crate::dataset::PyVortexDataset;
use crate::error::PyVortexResult;
use crate::expr::PyExpr;
use crate::install_module;
use crate::iter::PyArrayIterator;
use crate::object_store::resolve::ResolvedStore;
use crate::object_store::resolve::resolve_store;
use crate::session::session;

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
///     The URL to read from.
/// store : vortex.store.AzureStore | vortex.store.GCSStore | vortex.store.HTTPStore | vortex.store.LocalStore | vortex.store.MemoryStore | vortex.store.S3Store | None
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
///
#[pyfunction]
#[pyo3(signature = (url, *, store = None, projection = None, row_filter = None, indices = None, row_range = None))]
pub fn read_url<'py>(
    py: Python<'py>,
    url: &str,
    store: Option<Bound<'py, PyAny>>,
    projection: Option<Vec<Bound<'py, PyAny>>>,
    row_filter: Option<&Bound<'py, PyExpr>>,
    indices: Option<PyArrayRef>,
    row_range: Option<(u64, u64)>,
) -> PyVortexResult<PyArrayRef> {
    let store_arc = if let Some(store_obj) = store {
        let py_store: PyObjectStore = store_obj.extract()?;
        Some(py_store.into_inner())
    } else {
        None
    };

    let dataset = py.detach(move || RUNTIME.block_on(PyVortexDataset::from_url(url, store_arc)))?;
    dataset.to_array_inner(py, projection, row_filter, indices, row_range)
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
/// store : vortex.store.AzureStore | vortex.store.GCSStore | vortex.store.HTTPStore | vortex.store.LocalStore | vortex.store.MemoryStore | vortex.store.S3Store | None
///     An optional object store configuration to use for writing the output.
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
#[pyo3(signature = (iter, path, *, store = None))]
pub fn write(
    py: Python,
    iter: PyIntoArrayIterator,
    path: &str,
    store: Option<PyObjectStore>,
) -> PyVortexResult<()> {
    let session = session();
    py.detach(|| {
        RUNTIME.block_on(async move {
            match resolve_store(path, store.map(|x| x.into_inner()))? {
                ResolvedStore::ObjectStore(store, path) => {
                    let mut store = ObjectStoreWrite::new(store, &path).await?;
                    session
                        .write_options()
                        .write(&mut store, iter.into_inner().into_array_stream())
                        .await?;
                    store.shutdown().await?;
                    VortexResult::Ok(())
                }
                ResolvedStore::Path(path) => {
                    let mut w = File::create(path).await?;
                    session
                        .write_options()
                        .write(&mut w, iter.into_inner().into_array_stream())
                        .await?;
                    w.shutdown().await?;
                    VortexResult::Ok(())
                }
            }
        })
    })?;

    Ok(())
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
}

#[pymethods]
impl PyVortexWriteOptions {
    /// Balance size, read-throughput, and read-latency.
    #[staticmethod]
    pub fn default() -> Self {
        Self {
            use_compact_encodings: false,
        }
    }

    /// Prioritize small size over read-throughput and read-latency.
    ///
    /// Let's model some stock ticker data. As you may know, the stock market always (noisly) goes
    /// up:
    ///
    /// >>> import os
    /// >>> import random
    /// >>> sprl = vx.array([random.randint(i, i + 10) for i in range(100_000)])
    ///
    /// If we naively wrote 4-bytes for each of these integers to a file we'd have 400,000 bytes!
    /// Let's see how small this is when we write with the default Vortex write options (which are
    /// also used by :func:`vortex.io.write`):
    ///
    /// >>> vx.io.VortexWriteOptions.default().write(sprl, "chonky.vortex")
    /// >>> import os
    /// >>> os.path.getsize('chonky.vortex')
    /// 216004
    ///
    /// Wow, Vortex manages to use about two bytes per integer! So advanced. So tiny.
    ///
    /// But can we do better?
    ///
    /// We sure can.
    ///
    /// >>> vx.io.VortexWriteOptions.compact().write(sprl, "tiny.vortex")
    /// >>> os.path.getsize('tiny.vortex')
    /// 55120
    ///
    /// Random numbers are not (usually) composed of random bytes!
    #[staticmethod]
    pub fn compact() -> Self {
        Self {
            use_compact_encodings: true,
        }
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
    /// store : vortex.store.AzureStore | vortex.store.GCSStore | vortex.store.HTTPStore | vortex.store.LocalStore | vortex.store.MemoryStore | vortex.store.S3Store | None
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
    pub fn write(
        &self,
        py: Python,
        iter: PyIntoArrayIterator,
        path: &str,
        store: Option<PyObjectStore>,
    ) -> PyVortexResult<()> {
        let session = session();
        py.detach(|| {
            let mut strategy = WriteStrategyBuilder::default();
            if self.use_compact_encodings {
                strategy = strategy
                    .with_btrblocks_builder(BtrBlocksCompressorBuilder::default().with_compact());
            }
            let strategy = strategy.build();
            RUNTIME.block_on(async move {
                match resolve_store(path, store.map(|x| x.into_inner()))? {
                    ResolvedStore::ObjectStore(store, path) => {
                        let mut store = ObjectStoreWrite::new(store, &path).await?;
                        session
                            .write_options()
                            .with_strategy(strategy)
                            .write(&mut store, iter.into_inner().into_array_stream())
                            .await?;
                        store.shutdown().await?;
                        VortexResult::Ok(())
                    }
                    ResolvedStore::Path(path) => {
                        let mut w = File::create(path).await?;
                        session
                            .write_options()
                            .with_strategy(strategy)
                            .write(&mut w, iter.into_inner().into_array_stream())
                            .await?;
                        w.shutdown().await?;
                        VortexResult::Ok(())
                    }
                }
            })
        })?;

        Ok(())
    }
}

/// Conversion type for converting Python objects into a [`vortex::ArrayIterator`].
pub type PyIntoArrayIterator = PyVortex<Box<dyn ArrayIterator + Send>>;

impl<'py> FromPyObject<'_, 'py> for PyIntoArrayIterator {
    type Error = PyErr;

    fn extract(ob: Borrowed<'_, 'py, PyAny>) -> Result<Self, Self::Error> {
        if let Ok(py_iter) = ob.cast::<PyArrayIterator>() {
            return Ok(PyVortex(py_iter.get().take().unwrap_or_else(|| {
                Box::new(
                    Canonical::empty(py_iter.get().dtype())
                        .into_array()
                        .to_array_iterator(),
                )
            })));
        }

        if let Ok(py_array) = ob.cast::<PyArray>() {
            return Ok(PyVortex(Box::new(
                py_array
                    .extract::<PyArrayRef>()?
                    .into_inner()
                    .to_array_iterator(),
            )));
        }

        // Try to convert from Arrow objects (Table, RecordBatchReader, etc.)
        if let Ok(arrow_iter) = try_arrow_stream_to_iterator(&ob) {
            return Ok(PyVortex(arrow_iter));
        }

        Err(PyTypeError::new_err(
            "Expected an object that can be converted to an ArrayIterator (PyArrayIterator, PyArray, or PyArrow object with streaming support)",
        ))
    }
}

/// Try to convert a PyArrow object to a Vortex ArrayIterator using Arrow FFI streams.
fn try_arrow_stream_to_iterator(
    ob: &Borrowed<'_, '_, PyAny>,
) -> PyResult<Box<dyn ArrayIterator + Send>> {
    let py = ob.py();

    let pa_table = table_class(py)?;
    let pa_record_batch_reader = record_batch_reader_class(py)?;

    if ob.is_instance(pa_table)? || ob.is_instance(pa_record_batch_reader)? {
        // Convert to Arrow stream using FFI
        let arrow_stream = ArrowArrayStreamReader::from_pyarrow(ob)?;
        let dtype = DType::from_arrow(arrow_stream.schema());

        // Convert Arrow RecordBatch stream to Vortex ArrayIterator
        let vortex_iter = arrow_stream
            .into_iter()
            .map(|batch_result| -> VortexResult<ArrayRef> {
                let batch = batch_result.map_err(VortexError::from)?;
                ArrayRef::from_arrow(batch, false)
            });

        Ok(Box::new(ArrayIteratorAdapter::new(dtype, vortex_iter)))
    } else {
        Err(PyTypeError::new_err(
            "Object is not a supported Arrow streaming type",
        ))
    }
}
