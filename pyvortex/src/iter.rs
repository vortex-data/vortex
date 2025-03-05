use std::sync::Mutex;

use pyo3::prelude::*;
use pyo3::{Bound, PyResult, Python};
use vortex::dtype::DType;
use vortex::error::VortexExpect;
use vortex::iter::{ArrayIterator, ArrayIteratorExt};
use vortex::{Canonical, IntoArray};

use crate::arrays::PyArrayRef;
use crate::dtype::PyDType;
use crate::install_module;

pub(crate) fn init(py: Python, parent: &Bound<PyModule>) -> PyResult<()> {
    let m = PyModule::new(py, "iter")?;
    parent.add_submodule(&m)?;
    install_module("vortex._lib.iter", &m)?;

    m.add_class::<PyArrayIterator>()?;

    Ok(())
}

#[pyclass(name = "ArrayIterator", module = "vortex")]
pub struct PyArrayIterator {
    iter: Mutex<Option<Box<dyn ArrayIterator + Send>>>,
    dtype: DType,
}

#[pymethods]
impl PyArrayIterator {
    /// Return the :class:`vortex.DType` for all chunks of this iterator.
    #[getter]
    fn dtype(slf: PyRef<Self>) -> PyResult<Bound<PyDType>> {
        PyDType::init(slf.py(), slf.dtype.clone())
    }

    /// Supports iteration.
    fn __iter__(slf: PyRef<Self>) -> PyRef<Self> {
        slf
    }

    /// Returns the next chunk from the iterator.
    fn __next__(&mut self, py: Python) -> PyResult<Option<PyArrayRef>> {
        py.allow_threads(|| {
            Ok(self
                .iter
                .lock()
                .vortex_expect("poisoned lock")
                .as_mut()
                .and_then(|iter| iter.next())
                .transpose()?
                .map(PyArrayRef::from))
        })
    }

    /// Read all chunks into a single :class:`vortex.Array`. If there are multiple chunks,
    /// this will be a :class:`vortex.ChunkedArray`, otherwise it will be a single array.
    fn read_all(&mut self, py: Python) -> PyResult<PyArrayRef> {
        let array = py.allow_threads(|| {
            if let Some(iter) = self.iter.lock().vortex_expect("poisoned lock").take() {
                iter.read_all()
            } else {
                // Otherwise, we continue to return an empty array.
                Ok(Canonical::empty(&self.dtype).into_array())
            }
        })?;
        Ok(PyArrayRef::from(array))
    }
}
