// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

mod python;

use std::iter;
use std::sync::Arc;

use arrow_array::RecordBatch;
use arrow_array::RecordBatchIterator;
use arrow_array::RecordBatchReader;
use arrow_array::cast::AsArray;
use arrow_schema::ArrowError;
use arrow_schema::DataType;
use parking_lot::Mutex;
use pyo3::Bound;
use pyo3::PyResult;
use pyo3::Python;
use pyo3::prelude::*;
use pyo3::types::PyIterator;
use vortex::array::Canonical;
use vortex::array::IntoArray;
use vortex::array::VortexSessionExecute;
use vortex::array::arrow::ArrowArrayExecutor;
use vortex::array::iter::ArrayIterator;
use vortex::array::iter::ArrayIteratorAdapter;
use vortex::array::iter::ArrayIteratorExt;
use vortex::dtype::DType;

use crate::arrays::PyArrayRef;
use crate::arrow::IntoPyArrow;
use crate::dtype::PyDType;
use crate::error::PyVortexResult;
use crate::install_module;
use crate::iter::python::PythonArrayIterator;
use crate::session::session;

pub(crate) fn init(py: Python, parent: &Bound<PyModule>) -> PyResult<()> {
    let m = PyModule::new(py, "iter")?;
    parent.add_submodule(&m)?;
    install_module("vortex._lib.iter", &m)?;

    m.add_class::<PyArrayIterator>()?;

    Ok(())
}

#[pyclass(name = "ArrayIterator", module = "vortex", frozen)]
pub struct PyArrayIterator {
    iter: Mutex<Option<Box<dyn ArrayIterator + Send>>>,
    dtype: DType,
}

impl PyArrayIterator {
    pub fn new(iter: Box<dyn ArrayIterator + Send>) -> Self {
        let dtype = iter.dtype().clone();
        Self {
            iter: Mutex::new(Some(iter)),
            dtype,
        }
    }

    pub fn dtype(&self) -> &DType {
        &self.dtype
    }

    pub fn take(&self) -> Option<Box<dyn ArrayIterator + Send>> {
        self.iter.lock().take()
    }
}

#[pymethods]
impl PyArrayIterator {
    /// Return the :class:`vortex.DType` for all chunks of this iterator.
    #[getter]
    #[pyo3(name = "dtype")]
    fn dtype_(slf: PyRef<Self>) -> PyResult<Bound<PyDType>> {
        PyDType::init(slf.py(), slf.dtype.clone())
    }

    /// Supports iteration.
    fn __iter__(slf: PyRef<Self>) -> PyRef<Self> {
        slf
    }

    /// Returns the next chunk from the iterator.
    fn __next__(&self, py: Python) -> PyVortexResult<Option<PyArrayRef>> {
        py.detach(|| {
            Ok(self
                .iter
                .lock()
                .as_mut()
                .and_then(|iter| iter.next())
                .transpose()?
                .map(PyArrayRef::from))
        })
    }

    /// Read all chunks into a single :class:`vortex.Array`. If there are multiple chunks,
    /// this will be a :class:`vortex.ChunkedArray`, otherwise it will be a single array.
    fn read_all(&self, py: Python) -> PyVortexResult<PyArrayRef> {
        let array = py.detach(|| {
            if let Some(iter) = self.iter.lock().take() {
                iter.read_all()
            } else {
                // Otherwise, we continue to return an empty array.
                Ok(Canonical::empty(&self.dtype).into_array())
            }
        })?;
        Ok(PyArrayRef::from(array))
    }

    /// Convert the :class:`vortex.ArrayIterator` into a :class:`pyarrow.RecordBatchReader`.
    ///
    /// Note that this performs the conversion on the current thread.
    fn to_arrow(slf: Bound<Self>) -> PyVortexResult<Py<PyAny>> {
        let schema = Arc::new(slf.get().dtype().to_arrow_schema()?);
        let data_type = DataType::Struct(schema.fields().clone());

        let iter = slf.get().take().unwrap_or_else(|| {
            Box::new(ArrayIteratorAdapter::new(
                slf.get().dtype().clone(),
                iter::empty(),
            ))
        });

        let record_batch_reader: Box<dyn RecordBatchReader + Send> =
            Box::new(RecordBatchIterator::new(
                iter.map(move |chunk| {
                    let data_type = data_type.clone();
                    let session = session();
                    chunk?.execute_arrow(Some(&data_type), &mut session.create_execution_ctx())
                })
                .map(|chunk| chunk.map_err(|e| ArrowError::ExternalError(Box::new(e))))
                .map(|array| array.map(|a| RecordBatch::from(a.as_struct().clone()))),
                schema,
            ));

        Ok(record_batch_reader.into_pyarrow(slf.py())?)
    }

    /// Create a :class:`vortex.ArrayIterator` from an iterator of :class:`vortex.Array`.
    #[staticmethod]
    fn from_iter(dtype: PyDType, iter: Py<PyIterator>) -> PyResult<PyArrayIterator> {
        Ok(PyArrayIterator::new(Box::new(
            PythonArrayIterator::try_new(dtype.into_inner(), iter)?,
        )))
    }
}
