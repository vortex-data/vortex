use pyo3::{Bound, PyRef, PyResult, pyclass, pymethods};
use vortex::arrays::ConstantVTable;

use crate::arrays::native::{AsArrayRef, EncodingSubclass, PyNativeArray};
use crate::scalar::PyScalar;

/// Concrete class for arrays with `vortex.constant` encoding.
#[pyclass(name = "ConstantArray", module = "vortex", extends=PyNativeArray, frozen)]
pub(crate) struct PyConstantArray;

impl EncodingSubclass for PyConstantArray {
    type VTable = ConstantVTable;
}

#[pymethods]
impl PyConstantArray {
    /// Return the scalar value of the constant array.
    pub fn scalar(self_: PyRef<'_, Self>) -> PyResult<Bound<PyScalar>> {
        PyScalar::init(self_.py(), self_.as_array_ref().scalar().clone())
    }
}
