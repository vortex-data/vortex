// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

mod python;

use std::collections::VecDeque;
use std::iter;
use std::sync::Arc;

use arrow_array::RecordBatch;
use arrow_array::RecordBatchIterator;
use arrow_array::RecordBatchReader;
use arrow_array::cast::AsArray;
use arrow_schema::ArrowError;
use arrow_schema::Field;
use parking_lot::Mutex;
use pyo3::Bound;
use pyo3::PyResult;
use pyo3::Python;
use pyo3::prelude::*;
use pyo3::types::PyIterator;
use vortex::array::Canonical;
use vortex::array::IntoArray;
use vortex::array::VortexSessionExecute;
use vortex::array::iter::ArrayIterator;
use vortex::array::iter::ArrayIteratorAdapter;
use vortex::array::iter::ArrayIteratorExt;
use vortex::dtype::DType;
use vortex::error::VortexResult;
use vortex::io::runtime::BlockingRuntime;
use vortex::io::runtime::Task;
use vortex::utils::parallelism::get_available_parallelism;
use vortex_arrow::ArrowSessionExt;

use crate::arrays::PyArrayRef;
use crate::arrow::IntoPyArrow;
use crate::current_runtime;
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
    /// Chunks are pulled on the current thread, and each chunk's conversion to Arrow runs on the
    /// Vortex runtime's worker pool, with up to one conversion in flight per core. Batches are
    /// returned in iterator order.
    fn to_arrow(slf: Bound<Self>) -> PyVortexResult<Py<PyAny>> {
        let schema = Arc::new(session().arrow().to_arrow_schema(slf.get().dtype())?);
        let target = Field::new_struct("", schema.fields().clone(), false);

        let iter = slf.get().take().unwrap_or_else(|| {
            Box::new(ArrayIteratorAdapter::new(
                slf.get().dtype().clone(),
                iter::empty(),
            ))
        });

        let record_batch_reader: Box<dyn RecordBatchReader + Send> = Box::new(
            RecordBatchIterator::new(ParallelArrowIterator::new(iter, target), schema),
        );

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

/// Converts the chunks of an [`ArrayIterator`] to Arrow record batches on the runtime's CPU pool.
///
/// Scans yield lazy arrays whose decoding happens during the Arrow conversion, so converting on
/// the consuming thread serializes most of the scan's CPU work.
struct ParallelArrowIterator {
    iter: Box<dyn ArrayIterator + Send>,
    target: Field,
    pending: VecDeque<Task<VortexResult<RecordBatch>>>,
    max_pending: usize,
    exhausted: bool,
}

impl ParallelArrowIterator {
    fn new(iter: Box<dyn ArrayIterator + Send>, target: Field) -> Self {
        let max_pending = get_available_parallelism().unwrap_or(1).max(1);
        Self {
            iter,
            target,
            pending: VecDeque::with_capacity(max_pending),
            max_pending,
            exhausted: false,
        }
    }
}

impl Iterator for ParallelArrowIterator {
    type Item = Result<RecordBatch, ArrowError>;

    fn next(&mut self) -> Option<Self::Item> {
        let runtime = current_runtime();
        let handle = runtime.handle();
        while !self.exhausted && self.pending.len() < self.max_pending {
            let Some(chunk) = self.iter.next() else {
                self.exhausted = true;
                break;
            };
            // Stop pulling after an error, which is returned after the batches before it.
            self.exhausted = chunk.is_err();
            let target = self.target.clone();
            self.pending.push_back(handle.spawn_cpu(move || {
                let mut ctx = session().create_execution_ctx();
                let array = session()
                    .arrow()
                    .execute_arrow(chunk?, Some(&target), &mut ctx)?;
                Ok(RecordBatch::from(array.as_struct().clone()))
            }));
        }

        let task = self.pending.pop_front()?;
        Some(
            runtime
                .block_on(task)
                .map_err(|e| ArrowError::ExternalError(Box::new(e))),
        )
    }
}
