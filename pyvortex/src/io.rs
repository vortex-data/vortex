use std::path::Path;

use pyo3::prelude::*;
use pyo3::pyfunction;
use pyo3::types::PyString;
use tokio::fs::File;
use vortex::file::VortexFileWriter;
use vortex::sampling_compressor::SamplingCompressor;
use vortex::ArrayData;

use crate::dataset::{ObjectStoreUrlDataset, TokioFileDataset};
use crate::expr::PyExpr;
use crate::{PyArray, TOKIO_RUNTIME};

/// Read a vortex struct array from the local filesystem.
///
/// Parameters
/// ----------
/// path : :class:`str`
///     The file path to read from.
/// projection : :class:`list` [ :class:`str` ``|`` :class:`int` ]
///     The columns to read identified either by their index or name.
/// row_filter : :class:`.Expr`
///     Keep only the rows for which this expression evaluates to true.
///
/// Examples
/// --------
///
/// Read an array with a structured column and nulls at multiple levels and in multiple columns.
///
/// >>> a = vortex.array([
/// ...     {'name': 'Joseph', 'age': 25},
/// ...     {'name': None, 'age': 31},
/// ...     {'name': 'Angela', 'age': None},
/// ...     {'name': 'Mikhail', 'age': 57},
/// ...     {'name': None, 'age': None},
/// ... ])
/// >>> vortex.io.write_path(a, "a.vortex")
/// >>> b = vortex.io.read_path("a.vortex")
/// >>> b.to_arrow_array()
/// <pyarrow.lib.StructArray object at ...>
/// -- is_valid: all not null
/// -- child 0 type: int64
///   [
///     25,
///     31,
///     null,
///     57,
///     null
///   ]
/// -- child 1 type: string_view
///   [
///     "Joseph",
///     null,
///     "Angela",
///     "Mikhail",
///     null
///   ]
///
/// Read just the age column:
///
/// >>> c = vortex.io.read_path("a.vortex", projection = ["age"])
/// >>> c.to_arrow_array()
/// <pyarrow.lib.StructArray object at ...>
/// -- is_valid: all not null
/// -- child 0 type: int64
///   [
///     25,
///     31,
///     null,
///     57,
///     null
///   ]
///
/// Read just the name column, by its index:
///
/// >>> d = vortex.io.read_path("a.vortex", projection = [1])
/// >>> d.to_arrow_array()
/// <pyarrow.lib.StructArray object at ...>
/// -- is_valid: all not null
/// -- child 0 type: string_view
///   [
///     "Joseph",
///     null,
///     "Angela",
///     "Mikhail",
///     null
///   ]
///
///
/// Keep rows with an age above 35. This will read O(N_KEPT) rows, when the file format allows.
///
/// >>> e = vortex.io.read_path("a.vortex", row_filter = vortex.expr.column("age") > 35)
/// >>> e.to_arrow_array()
/// <pyarrow.lib.StructArray object at ...>
/// -- is_valid: all not null
/// -- child 0 type: int64
///   [
///     57
///   ]
/// -- child 1 type: string_view
///   [
///     "Mikhail"
///   ]
///
/// TODO(DK): Repeating a column in a projection does not work
///
/// Read the age column by name, twice, and the name column by index, once:
///
/// >>> # e = vortex.io.read_path("a.vortex", projection = ["age", 1, "age"])
/// >>> # e.to_arrow_array()
///
/// TODO(DK): Top-level nullness does not work.
///
/// >>> a = vortex.array([
/// ...     {'name': 'Joseph', 'age': 25},
/// ...     {'name': None, 'age': 31},
/// ...     {'name': 'Angela', 'age': None},
/// ...     None,
/// ...     {'name': 'Mikhail', 'age': 57},
/// ...     {'name': None, 'age': None},
/// ... ])
/// >>> vortex.io.write_path(a, "a.vortex")
/// >>> # b = vortex.io.read_path("a.vortex")
/// >>> # b.to_arrow_array()
///
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
    async fn run(array: &ArrayData, fname: &str) -> PyResult<()> {
        let file = File::create(Path::new(fname)).await?;
        let mut writer = VortexFileWriter::new(file);

        writer = writer.write_array_columns(array.clone()).await?;
        writer.finalize().await?;
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
