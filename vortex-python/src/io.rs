// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use arrow_array::RecordBatchReader;
use arrow_array::ffi_stream::ArrowArrayStreamReader;
use pyo3::exceptions::PyTypeError;
use pyo3::prelude::*;
use pyo3::pyfunction;
use tokio::fs::File;
use vortex::array::ArrayRef;
use vortex::array::Canonical;
use vortex::array::IntoArray;
use vortex::array::arrow::FromArrowArray;
use vortex::array::iter::ArrayIterator;
use vortex::array::iter::ArrayIteratorAdapter;
use vortex::array::iter::ArrayIteratorExt;
use vortex::compressor::CompactCompressor;
use vortex::dtype::DType;
use vortex::dtype::arrow::FromArrowType;
use vortex::error::VortexError;
use vortex::error::VortexResult;
use vortex::file::WriteOptionsSessionExt;
use vortex::file::WriteStrategyBuilder;

use crate::PyVortex;
use crate::SESSION;
use crate::TOKIO_RUNTIME;
use crate::arrays::PyArray;
use crate::arrays::PyArrayRef;
use crate::arrow::FromPyArrow;
use crate::dataset::PyVortexDataset;
use crate::expr::PyExpr;
use crate::install_module;
use crate::iter::PyArrayIterator;

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
/// ```python
/// >>> import vortex as vx
/// >>> a = vx.io.read_url("https://example.com/dataset.vortex")  # doctest: +SKIP
/// ```
///
/// Read an array from an S3 URL:
///
/// ```python
/// >>> a = vx.io.read_url("s3://bucket/path/to/dataset.vortex")  # doctest: +SKIP
/// ```
///
/// Read an array from an Azure Blob File System URL:
///
/// ```python
/// >>> a = vx.io.read_url("abfss://my_file_system@my_account.dfs.core.windows.net/path/to/dataset.vortex")  # doctest: +SKIP
/// ```
///
/// Read an array from an Azure Blob Storage URL:
///
/// ```python
/// >>> a = vx.io.read_url("https://my_account.blob.core.windows.net/my_container/path/to/dataset.vortex")  # doctest: +SKIP
/// ```
///
/// Read an array from a Google Storage URL:
///
/// ```python
/// >>> a = vx.io.read_url("gs://bucket/path/to/dataset.vortex")  # doctest: +SKIP
/// ```
///
/// Read an array from a local file URL:
///
/// ```python
/// >>> a = vx.io.read_url("file:/path/to/dataset.vortex")  # doctest: +SKIP
/// ```
///
#[pyfunction]
#[pyo3(signature = (url, *, projection = None, row_filter = None, indices = None, row_range = None))]
pub fn read_url<'py>(
    py: Python<'py>,
    url: &str,
    projection: Option<Vec<Bound<'py, PyAny>>>,
    row_filter: Option<&Bound<'py, PyExpr>>,
    indices: Option<PyArrayRef>,
    row_range: Option<(u64, u64)>,
) -> PyResult<PyArrayRef> {
    let dataset = py.detach(|| TOKIO_RUNTIME.block_on(PyVortexDataset::from_url(url)))?;
    dataset.to_array(projection, row_filter, indices, row_range)
}

/// Write a vortex struct array to the local filesystem.
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
/// Examples
/// --------
///
/// Write a single Vortex array `a` to the local file `a.vortex`.
///
/// ```python
/// >>> import vortex as vx
/// >>> a = vx.array([
/// ...     {'x': 1},
/// ...     {'x': 2},
/// ...     {'x': 10},
/// ...     {'x': 11},
/// ...     {'x': None},
/// ... ])
/// >>> vx.io.write(a, "a.vortex") # doctest: +SKIP
/// ```
///
/// Stream a PyArrow Table directly to Vortex without loading into memory:
///
/// ```python
/// >>> import pyarrow as pa
/// >>> import vortex as vx
/// >>> table = pa.table({'x': [1, 2, 3, 4, 5]})
/// >>> vx.io.write(table, "streamed.vortex")  # doctest: +SKIP
/// ```
///
/// Stream from a PyArrow RecordBatchReader:
///
/// ```python
/// >>> import pyarrow as pa
/// >>> import vortex as vx
/// >>> reader = pa.RecordBatchReader.from_batches(schema, batches) # doctest: +SKIP
/// >>> vx.io.write(reader, "streamed.vortex")  # doctest: +SKIP
/// ```
///
/// See also
/// --------
///
/// :func:`vortex.io.VortexWriteOptions`
#[pyfunction]
#[pyo3(signature = (iter, path))]
pub fn write(py: Python, iter: PyIntoArrayIterator, path: &str) -> PyResult<()> {
    py.detach(|| {
        TOKIO_RUNTIME.block_on(async move {
            let file = File::create(path).await?;
            SESSION
                .write_options()
                .write(file, iter.into_inner().into_array_stream())
                .await
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
    // TODO(DK): This might need to be an Arc<dyn Compressor> if we actually have multiple
    // compressors.
    compressor: Option<CompactCompressor>,
}

#[pymethods]
impl PyVortexWriteOptions {
    /// Balance size, read-throughput, and read-latency.
    #[staticmethod]
    pub fn default() -> Self {
        Self { compressor: None }
    }

    /// Prioritize small size over read-throughput and read-latency.
    ///
    /// Let's model some stock ticker data. As you may know, the stock market always (noisly) goes
    /// up:
    ///
    /// ```python
    /// >>> import os
    /// >>> import random
    /// >>> sprl = vx.array([random.randint(i, i + 10) for i in range(100_000)])
    /// ```
    ///
    /// If we naively wrote 4-bytes for each of these integers to a file we'd have 400,000 bytes!
    /// Let's see how small this is when we write with the default Vortex write options (which are
    /// also used by :func:`vortex.io.write`):
    ///
    /// ```python
    /// >>> vx.io.VortexWriteOptions.default().write_path(sprl, "chonky.vortex")
    /// >>> import os
    /// >>> os.path.getsize('chonky.vortex')
    /// 215996
    /// ```
    ///
    /// Wow, Vortex manages to use about two bytes per integer! So advanced. So tiny.
    ///
    /// But can we do better?
    ///
    /// We sure can.
    ///
    /// ```python
    /// >>> vx.io.VortexWriteOptions.compact().write_path(sprl, "tiny.vortex")
    /// >>> os.path.getsize('tiny.vortex')
    /// 55116
    /// ```
    ///
    /// Random numbers are not (usually) composed of random bytes!
    #[staticmethod]
    pub fn compact() -> Self {
        Self {
            compressor: Some(CompactCompressor::default()),
        }
    }

    /// Write an array or iterator of arrays into a local file.
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
    /// Examples
    /// --------
    ///
    /// Write a single Vortex array `a` to the local file `a.vortex` using the default settings:
    ///
    /// ```python
    /// >>> import vortex as vx
    /// >>> import random
    /// >>> a = vx.array([0, 1, 2, 3, None, 4])
    /// >>> vx.io.VortexWriteOptions.default().write_path(a, "a.vortex") # doctest: +SKIP
    /// ```
    ///
    /// Write the same array while preferring small file sizes over read-throughput and
    /// read-latency:
    ///
    /// ```python
    /// >>> import vortex as vx
    /// >>> vx.io.VortexWriteOptions.compact().write_path(a, "a.vortex") # doctest: +SKIP
    /// ```
    ///
    /// See also
    /// --------
    ///
    /// :func:`vortex.io.write`
    #[pyo3(signature = (iter, path))]
    pub fn write_path(&self, py: Python, iter: PyIntoArrayIterator, path: &str) -> PyResult<()> {
        py.detach(|| {
            TOKIO_RUNTIME.block_on(async move {
                let file = File::create(path).await?;

                let mut strategy = WriteStrategyBuilder::new();
                if let Some(compressor) = self.compressor.as_ref() {
                    strategy = strategy.with_compressor(compressor.clone())
                }

                SESSION
                    .write_options()
                    .with_strategy(strategy.build())
                    .write(file, iter.into_inner().into_array_stream())
                    .await
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
    let pa = py.import("pyarrow")?;
    let pa_table = pa.getattr("Table")?;
    let pa_record_batch_reader = pa.getattr("RecordBatchReader")?;

    if ob.is_instance(&pa_table)? || ob.is_instance(&pa_record_batch_reader)? {
        // Convert to Arrow stream using FFI
        let arrow_stream = ArrowArrayStreamReader::from_pyarrow(ob)?;
        let dtype = DType::from_arrow(arrow_stream.schema());

        // Convert Arrow RecordBatch stream to Vortex ArrayIterator
        let vortex_iter = arrow_stream
            .into_iter()
            .map(|batch_result| -> VortexResult<ArrayRef> {
                let batch = batch_result.map_err(VortexError::from)?;
                Ok(ArrayRef::from_arrow(batch, false))
            });

        Ok(Box::new(ArrayIteratorAdapter::new(dtype, vortex_iter)))
    } else {
        Err(PyTypeError::new_err(
            "Object is not a supported Arrow streaming type",
        ))
    }
}
