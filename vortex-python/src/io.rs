use pyo3::exceptions::PyTypeError;
use pyo3::prelude::*;
use pyo3::pyfunction;
use tokio::fs::File;
use vortex::file::VortexWriteOptions;
use vortex::iter::{ArrayIterator, ArrayIteratorExt};
use vortex::{Canonical, IntoArray};

use crate::arrays::{PyArray, PyArrayRef};
use crate::dataset::PyVortexDataset;
use crate::expr::PyExpr;
use crate::iter::PyArrayIterator;
use crate::{PyVortex, TOKIO_RUNTIME, install_module};

pub(crate) fn init(py: Python, parent: &Bound<PyModule>) -> PyResult<()> {
    let m = PyModule::new(py, "io")?;
    parent.add_submodule(&m)?;
    install_module("vortex._lib.io", &m)?;

    m.add_function(wrap_pyfunction!(read_url, &m)?)?;
    m.add_function(wrap_pyfunction!(write, &m)?)?;

    Ok(())
}

/// Read a vortex struct array from a URL.
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
///     >>> import vortex as vx
///     >>> a = vx.io.read_url("https://example.com/dataset.vortex")  # doctest: +SKIP
///
/// Read an array from an S3 URL:
///
///     >>> a = vx.io.read_url("s3://bucket/path/to/dataset.vortex")  # doctest: +SKIP
///
/// Read an array from an Azure Blob File System URL:
///
///     >>> a = vx.io.read_url("abfss://my_file_system@my_account.dfs.core.windows.net/path/to/dataset.vortex")  # doctest: +SKIP
///
/// Read an array from an Azure Blob Stroage URL:
///
///     >>> a = vx.io.read_url("https://my_account.blob.core.windows.net/my_container/path/to/dataset.vortex")  # doctest: +SKIP
///
/// Read an array from a Google Stroage URL:
///
///     >>> a = vx.io.read_url("gs://bucket/path/to/dataset.vortex")  # doctest: +SKIP
///
/// Read an array from a local file URL:
///
///     >>> a = vx.io.read_url("file:/path/to/dataset.vortex")  # doctest: +SKIP
///
#[pyfunction]
#[pyo3(signature = (url, *, projection = None, row_filter = None, indices = None))]
pub fn read_url<'py>(
    url: &str,
    projection: Option<Vec<Bound<'py, PyAny>>>,
    row_filter: Option<&Bound<'py, PyExpr>>,
    indices: Option<PyArrayRef>,
) -> PyResult<PyArrayRef> {
    let dataset = TOKIO_RUNTIME.block_on(PyVortexDataset::from_url(url))?;
    dataset.to_array(projection, row_filter, indices)
}

/// Write a vortex struct array to the local filesystem.
///
/// Parameters
/// ----------
/// array : :class:`~vortex.Array`
///     The array. Must be an array of structures.
///
/// f : :class:`str`
///     The file path.
///
/// Examples
/// --------
///
/// Write the array `a` to the local file `a.vortex`.
///
///     >>> import vortex as vx
///     >>> a = vx.array([
///     ...     {'x': 1},
///     ...     {'x': 2},
///     ...     {'x': 10},
///     ...     {'x': 11},
///     ...     {'x': None},
///     ... ])
///     >>> vx.io.write(a, "a.vortex")
///
#[pyfunction]
#[pyo3(signature = (iter, path))]
pub fn write(iter: PyIntoArrayIterator, path: &str) -> PyResult<()> {
    TOKIO_RUNTIME.block_on(async move {
        VortexWriteOptions::default()
            .write(
                File::create(path).await?,
                iter.into_inner().into_array_stream(),
            )
            .await
    })?;

    Ok(())
}

/// Conversion type for converting Python objects into a [`vortex::ArrayIterator`].
pub type PyIntoArrayIterator = PyVortex<Box<dyn ArrayIterator + Send>>;

impl<'py> FromPyObject<'py> for PyIntoArrayIterator {
    fn extract_bound(ob: &Bound<'py, PyAny>) -> PyResult<Self> {
        if let Ok(py_iter) = ob.downcast::<PyArrayIterator>() {
            return Ok(PyVortex(py_iter.get().take().unwrap_or_else(|| {
                Box::new(
                    Canonical::empty(py_iter.get().dtype())
                        .into_array()
                        .to_array_iterator(),
                )
            })));
        }

        if let Ok(py_array) = ob.downcast::<PyArray>() {
            return Ok(PyVortex(Box::new(
                py_array
                    .extract::<PyArrayRef>()?
                    .into_inner()
                    .to_array_iterator(),
            )));
        }

        Err(PyTypeError::new_err(
            "Expected an object that can be converted to an ArrayIterator",
        ))
    }
}
