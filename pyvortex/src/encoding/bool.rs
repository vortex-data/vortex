use pyo3::prelude::PyAnyMethods;
use pyo3::{pyclass, pymethods, Bound, PyResult};

use crate::arrays::PyArray;

#[pyclass(name = "BoolArray", module = "vortex.encoding", extends=PyArray)]
pub struct PyBoolArray;

#[pymethods]
impl PyBoolArray {
    #[new]
    pub fn new(array: &Bound<'_, PyArray>) -> PyResult<(Self, PyArray)> {
        Ok((PyBoolArray, array.extract()?))
    }
}
