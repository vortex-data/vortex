use std::ops::Deref;

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::{pyclass, pymethods, Bound, PyResult};
use vortex::array::{BoolArray, BoolEncoding};
use vortex::{Array, Encoding};

use crate::arrays::PyArray;

#[pyclass(name = "BoolArray", module = "vortex.encoding", extends=PyArray)]
pub struct PyBoolArray(BoolArray);

impl Deref for PyBoolArray {
    type Target = BoolArray;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

#[pymethods]
impl PyBoolArray {
    #[new]
    pub fn new(array: &Bound<'_, PyArray>) -> PyResult<(Self, PyArray)> {
        let array: Array = array.extract::<PyArray>()?.0;

        if array.encoding() != BoolEncoding::ID {
            return Err(PyValueError::new_err(format!(
                "Expected array with {} encoding, but found {}",
                BoolEncoding::ID,
                array.encoding(),
            )));
        }

        let bool_array = BoolArray::try_from(array.clone())?;

        Ok((PyBoolArray(bool_array), PyArray(array)))
    }
}

#[pymethods]
impl PyBoolArray {
    /// Returns the number of true values in the array.
    pub fn true_count(&self) -> usize {
        // FIXME(ngates): this ignores nulls
        self.boolean_buffer().count_set_bits()
    }
}
