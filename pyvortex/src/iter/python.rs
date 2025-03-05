use pyo3::exceptions::PyTypeError;
use pyo3::prelude::*;
use pyo3::types::PyIterator;
use vortex::dtype::DType;
use vortex::error::VortexResult;
use vortex::iter::ArrayIterator;
use vortex::{Array, ArrayRef};

use crate::arrays::PyArrayRef;

/// Wrap a Python iterator over arrays as an [`ArrayIterator`].
pub struct PythonArrayIterator {
    dtype: DType,
    iter: Py<PyIterator>,
}

impl PythonArrayIterator {
    pub fn try_new(dtype: DType, iter: Py<PyIterator>) -> PyResult<Self> {
        Ok(PythonArrayIterator { dtype, iter })
    }
}

impl ArrayIterator for PythonArrayIterator {
    fn dtype(&self) -> &DType {
        &self.dtype
    }
}

impl Iterator for PythonArrayIterator {
    type Item = VortexResult<ArrayRef>;

    fn next(&mut self) -> Option<Self::Item> {
        Python::with_gil(|py| {
            let mut iter = self.iter.clone_ref(py).into_bound(py);
            iter.next().map(|array| {
                array
                    .and_then(|array| array.extract::<PyArrayRef>())
                    .map(|pyarray| pyarray.into_inner())
                    .and_then(|array| {
                        if array.dtype() != &self.dtype {
                            Err(PyTypeError::new_err(format!(
                                "ArrayIterator dtype mismatch. Expected {:?}, got {:?}",
                                &self.dtype,
                                array.dtype()
                            )))
                        } else {
                            Ok(array)
                        }
                    })
                    .map_err(|pyerr| pyerr.into())
            })
        })
    }
}
