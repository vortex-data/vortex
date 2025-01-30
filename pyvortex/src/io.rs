use std::path::Path;

use pyo3::prelude::*;
use pyo3::pyfunction;
use pyo3::types::PyString;
use tokio::fs::File;
use vortex::file::VortexWriteOptions;
use vortex::sampling_compressor::SamplingCompressor;
use vortex::Array;

use crate::dataset::{ObjectStoreUrlDataset, TokioFileDataset};
use crate::encoding::PyArray;
use crate::expr::PyExpr;
use crate::{install_module, TOKIO_RUNTIME};

pub(crate) fn init(py: Python, parent: &Bound<PyModule>) -> PyResult<()> {
    let m = PyModule::new_bound(py, "io")?;
    parent.add_submodule(&m)?;
    install_module("vortex._lib.io", &m)?;

    m.add_function(wrap_pyfunction!(read_url, &m)?)?;
    m.add_function(wrap_pyfunction!(read_path, &m)?)?;
    m.add_function(wrap_pyfunction!(write_path, &m)?)?;

    Ok(())
}

#[pyfunction]
#[pyo3(signature = (path, *, projection = None, row_filter = None, indices = None))]
pub fn read_path(
    path: Bound<PyString>,
    projection: Option<Vec<Bound<PyAny>>>,
    row_filter: Option<&Bound<PyExpr>>,
    indices: Option<&PyArray>,
) -> PyResult<PyArray> {
    let dataset = TOKIO_RUNTIME.block_on(TokioFileDataset::try_new(path.extract()?))?;
    dataset.to_array(projection, row_filter, indices)
}

/// Read a vortex struct array from a URL.
///
/// .. seealso::
///     :func:`.read_path`
///
/// Parameters
/// ----------
/// url : :class:`str`
///     The URL to read from.
/// projection : :class:`list` [ :class:`str` ``|`` :class:`int` ]
///     The columns to read identified either by their index or name.
/// row_filter : :class:`.Expr`
///     Keep only the rows for which this expression evaluates to true.
///
/// Examples
/// --------
///
/// Read an array from an HTTPS URL:
///
/// >>> a = vortex.io.read_url("https://example.com/dataset.vortex")  # doctest: +SKIP
///
/// Read an array from an S3 URL:
///
/// >>> a = vortex.io.read_url("s3://bucket/path/to/dataset.vortex")  # doctest: +SKIP
///
/// Read an array from an Azure Blob File System URL:
///
/// >>> a = vortex.io.read_url("abfss://my_file_system@my_account.dfs.core.windows.net/path/to/dataset.vortex")  # doctest: +SKIP
///
/// Read an array from an Azure Blob Stroage URL:
///
/// >>> a = vortex.io.read_url("https://my_account.blob.core.windows.net/my_container/path/to/dataset.vortex")  # doctest: +SKIP
///
/// Read an array from a Google Stroage URL:
///
/// >>> a = vortex.io.read_url("gs://bucket/path/to/dataset.vortex")  # doctest: +SKIP
///
/// Read an array from a local file URL:
///
/// >>> a = vortex.io.read_url("file:/path/to/dataset.vortex")  # doctest: +SKIP
///
#[pyfunction]
#[pyo3(signature = (url, *, projection = None, row_filter = None, indices = None))]
pub fn read_url(
    url: Bound<PyString>,
    projection: Option<Vec<Bound<PyAny>>>,
    row_filter: Option<&Bound<PyExpr>>,
    indices: Option<&PyArray>,
) -> PyResult<PyArray> {
    let dataset = TOKIO_RUNTIME.block_on(ObjectStoreUrlDataset::try_new(url.extract()?))?;
    dataset.to_array(projection, row_filter, indices)
}

/// Write a vortex struct array to the local filesystem.
///
/// Parameters
/// ----------
/// array : :class:`~vortex.encoding.Array`
///     The array. Must be an array of structures.
///
/// f : :class:`str`
///     The file path.
///
/// compress : :class:`bool`
///     Compress the array before writing, defaults to ``True``.
///
/// Examples
/// --------
///
/// Write the array `a` to the local file `a.vortex`.
///
/// >>> a = vortex.array([
/// ...     {'x': 1},
/// ...     {'x': 2},
/// ...     {'x': 10},
/// ...     {'x': 11},
/// ...     {'x': None},
/// ... ])
/// >>> vortex.io.write_path(a, "a.vortex")
///
#[pyfunction]
#[pyo3(signature = (array, f, *, compress=true))]
pub fn write_path(
    array: &Bound<'_, PyArray>,
    f: &Bound<'_, PyString>,
    compress: bool,
) -> PyResult<()> {
    async fn run(array: &Array, fname: &str) -> PyResult<()> {
        let file = File::create(Path::new(fname)).await?;
        let _file = VortexWriteOptions::default()
            .write(file, array.clone().into_array_stream())
            .await?;

        Ok(())
    }

    let fname = f.to_str()?; // TODO(dk): support file objects
    let mut array = array.borrow().unwrap().clone();

    if compress {
        let compressor = SamplingCompressor::default();
        array = compressor.compress(&array, None)?.into_array();
    }

    TOKIO_RUNTIME.block_on(run(&array, fname))
}
