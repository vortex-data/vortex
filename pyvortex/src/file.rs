use std::sync::Arc;

use arrow::array::RecordBatchReader;
use arrow::pyarrow::IntoPyArrow;
use futures::{SinkExt, StreamExt};
use pyo3::exceptions::PyTypeError;
use pyo3::prelude::*;
use pyo3::types::PyList;
use vortex::ToCanonical;
use vortex::compute::try_cast;
use vortex::dtype::Nullability::NonNullable;
use vortex::dtype::{DType, PType};
use vortex::error::{VortexExpect, vortex_err};
use vortex::expr::{ExprRef, ident, select};
use vortex::file::scan::SplitBy;
use vortex::file::scan::executor::{TaskExecutor, TokioExecutor};
use vortex::file::{VortexFile, VortexOpenOptions};
use vortex::io::TokioFile;
use vortex::stream::{ArrayStream, ArrayStreamAdapter, ArrayStreamExt};

use crate::arrays::PyArrayRef;
use crate::dataset::PyVortexDataset;
use crate::dtype::PyDType;
use crate::expr::PyExpr;
use crate::iter::{ArrayStreamToIterator, PyArrayIterator};
use crate::record_batch_reader::VortexRecordBatchReader;
use crate::{TOKIO_RUNTIME, install_module};

pub(crate) fn init(py: Python, parent: &Bound<PyModule>) -> PyResult<()> {
    let m = PyModule::new(py, "file")?;
    parent.add_submodule(&m)?;
    install_module("vortex._lib.file", &m)?;

    m.add_function(wrap_pyfunction!(open, &m)?)?;
    m.add_class::<PyVortexFile>()?;

    Ok(())
}

#[pyfunction]
pub fn open(path: &str) -> PyResult<PyVortexFile> {
    let vxf = TOKIO_RUNTIME.block_on(VortexOpenOptions::file().open(TokioFile::open(path)?))?;
    Ok(PyVortexFile { vxf })
}

#[pyclass(name = "VortexFile", module = "vortex", frozen)]
pub struct PyVortexFile {
    vxf: VortexFile,
}

#[pymethods]
impl PyVortexFile {
    fn __len__(slf: PyRef<Self>) -> PyResult<usize> {
        Ok(usize::try_from(slf.vxf.row_count())?)
    }

    /// The dtype of the file.
    #[getter]
    fn dtype(slf: Bound<Self>) -> PyResult<Bound<PyDType>> {
        PyDType::init(slf.py(), slf.get().vxf.dtype().clone())
    }

    /// Scan the Vortex file returning a :class:`vortex.ArrayIterator`.
    ///
    /// Parameters
    /// ----------
    /// projection : :class:`vortex.Expr` | None
    ///     The projection expression to read, or else read all columns.
    /// expr : :class:`vortex.Expr` | None
    ///     The predicate used to filter rows. The filter columns do not need to be in the projection.
    /// indices : :class:`vortex.Array` | None
    ///     The indices of the rows to read. Must be sorted and non-null.
    /// batch_size : :class:`int` | None
    ///     The number of rows to read per chunk.
    ///
    /// Examples
    /// --------
    ///
    /// Scan a file with a structured column and nulls at multiple levels and in multiple columns.
    ///
    ///     >>> import vortex as vx
    ///     >>> import vortex.expr as ve
    ///     >>> a = vx.array([
    ///     ...     {'name': 'Joseph', 'age': 25},
    ///     ...     {'name': None, 'age': 31},
    ///     ...     {'name': 'Angela', 'age': None},
    ///     ...     {'name': 'Mikhail', 'age': 57},
    ///     ...     {'name': None, 'age': None},
    ///     ... ])
    ///     >>> vx.io.write(a, "a.vortex")
    ///     >>> vxf = vx.open("a.vortex")
    ///     >>> vxf.scan().read_all().to_arrow_array()
    ///     <pyarrow.lib.StructArray object at ...>
    ///     -- is_valid: all not null
    ///     -- child 0 type: int64
    ///       [
    ///         25,
    ///         31,
    ///         null,
    ///         57,
    ///         null
    ///       ]
    ///     -- child 1 type: string_view
    ///       [
    ///         "Joseph",
    ///         null,
    ///         "Angela",
    ///         "Mikhail",
    ///         null
    ///       ]
    ///
    /// Read just the age column:
    ///
    ///     >>> vxf.scan(['age']).read_all().to_arrow_array()
    ///     <pyarrow.lib.StructArray object at ...>
    ///     -- is_valid: all not null
    ///     -- child 0 type: int64
    ///       [
    ///         25,
    ///         31,
    ///         null,
    ///         57,
    ///         null
    ///       ]
    ///
    ///
    /// Keep rows with an age above 35. This will read O(N_KEPT) rows, when the file format allows.
    ///
    ///     >>> vxf.scan(expr=ve.column("age") > 35).read_all().to_arrow_array()
    ///     <pyarrow.lib.StructArray object at ...>
    ///     -- is_valid: all not null
    ///     -- child 0 type: int64
    ///       [
    ///         57
    ///       ]
    ///     -- child 1 type: string_view
    ///       [
    ///         "Mikhail"
    ///       ]
    ///
    #[pyo3(signature = (projection = None, *, expr = None, indices = None, batch_size = None))]
    fn scan(
        slf: Bound<Self>,
        projection: Option<PyIntoProjection>,
        expr: Option<PyExpr>,
        indices: Option<PyArrayRef>,
        batch_size: Option<usize>,
    ) -> PyResult<PyArrayIterator> {
        let mut builder = slf
            .get()
            .vxf
            .scan()?
            .with_some_filter(expr.map(|e| e.into_inner()))
            .with_projection(projection.map(|p| p.0).unwrap_or_else(ident));

        if let Some(indices) = indices {
            let indices = try_cast(indices.inner(), &DType::Primitive(PType::U64, NonNullable))?
                .to_primitive()?
                .into_buffer::<u64>();
            builder = builder.with_row_indices(indices);
        }

        if let Some(batch_size) = batch_size {
            builder = builder.with_split_by(SplitBy::RowCount(batch_size));
        }

        let iter = ArrayStreamToIterator::new(ArrayStreamExt::boxed(builder.build()?));
        Ok(PyArrayIterator::new(Box::new(iter)))
    }

    /// Scan the Vortex file as a :class:`pyarrow.RecordBatchReader`.
    // TODO(ngates): columns should instead be a projection expression
    #[pyo3(signature = (projection = None, *, expr = None, batch_size = None))]
    fn to_arrow(
        slf: Bound<Self>,
        projection: Option<PyIntoProjection>,
        expr: Option<PyExpr>,
        batch_size: Option<usize>,
    ) -> PyResult<PyObject> {
        let mut builder = slf
            .get()
            .vxf
            .scan()?
            .with_task_executor(TaskExecutor::Tokio(TokioExecutor::new(
                TOKIO_RUNTIME.handle().clone(),
            )))
            .with_canonicalize(true)
            .with_some_filter(expr.map(|e| e.into_inner()))
            .with_projection(projection.map(|p| p.0).unwrap_or_else(ident));

        if let Some(batch_size) = batch_size {
            builder = builder.with_split_by(SplitBy::RowCount(batch_size));
        }

        let stream = ArrayStreamExt::boxed(builder.build()?);
        let dtype = stream.dtype().clone();

        // The I/O of the array stream won't make progress unless it's polled. So we need to spawn it.
        let (mut send, recv) = futures::channel::mpsc::unbounded();

        TOKIO_RUNTIME
            .block_on(TOKIO_RUNTIME.spawn(async move {
                let mut stream = stream;
                while let Some(batch) = stream.next().await {
                    send.send(batch)
                        .await
                        .map_err(|e| vortex_err!("Send failed {}", e))
                        .vortex_expect("send failed");
                }
            }))
            .vortex_expect("failed to spawn stream");

        let stream = ArrayStreamAdapter::new(dtype, recv);

        let iter = ArrayStreamToIterator::new(stream);
        let rbr: Box<dyn RecordBatchReader + Send> =
            Box::new(VortexRecordBatchReader::try_new(iter)?);
        rbr.into_pyarrow(slf.py())
    }

    /// Scan the Vortex file using the :class:`pyarrow.dataset.Dataset` API.
    fn to_dataset(slf: Bound<Self>) -> PyResult<Bound<PyAny>> {
        let dataset_cls = slf
            .py()
            .import("vortex.dataset")?
            .getattr("VortexDataset")?;
        let dataset = PyVortexDataset::try_new(slf.get().vxf.clone())?;
        dataset_cls.call1((dataset,))
    }
}

pub struct PyIntoProjection(ExprRef);

impl<'py> FromPyObject<'py> for PyIntoProjection {
    fn extract_bound(ob: &Bound<'py, PyAny>) -> PyResult<Self> {
        // If it's a list of strings, convert to a column selection.
        if let Ok(py_list) = ob.downcast::<PyList>() {
            let cols = py_list
                .iter()
                .map(|item| item.extract::<String>())
                .collect::<PyResult<Vec<String>>>()?;
            return Ok(PyIntoProjection(select(
                cols.into_iter().map(Arc::<str>::from).collect::<Vec<_>>(),
                ident(),
            )));
        }

        // If it's an expression, just return it.
        if let Ok(py_expr) = ob.downcast::<PyExpr>() {
            return Ok(PyIntoProjection(py_expr.get().inner().clone()));
        }

        Err(PyTypeError::new_err(
            "projection must be a list of strings or a vortex.Expr",
        ))
    }
}
