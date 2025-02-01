use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::{pyclass, pymethods, Bound, PyClass, PyResult};
use vortex::array::BoolEncoding;
use vortex::error::{VortexError, VortexExpect};
use vortex::{Array, Encoding};

use crate::arrays::PyArray;

#[pyclass(name = "BoolArray", module = "vortex.encoding", extends=PyArray)]
pub struct PyBoolArray;

impl ArraySubclass for PyBoolArray {
    type Encoding = BoolEncoding;
}

#[pymethods]
impl PyBoolArray {
    /// Downcasts a :class:`vortex.Array` into a :class:`vortex.encoding.BoolArray`.
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

        Ok((PyBoolArray, PyArray(array)))
    }

    /// Compute the number of true values in the array.
    pub fn true_count(self_: PyRef<'_, Self>) -> PyResult<usize> {
        self_
            .as_array_ref()
            .statistics()
            .compute_true_count()
            .ok_or_else(|| PyValueError::new_err("Failed to compute true count"))
    }
}

/// A marker trait indicating a PyO3 class is a subclass of Vortex `Array`.
pub trait ArraySubclass: PyClass<BaseType = PyArray> {
    type Encoding: Encoding;
}

/// Unwrap a downcasted Vortex array from a `PyRef<ArraySubclass>`.
pub trait AsArrayRef<T> {
    fn as_array_ref(&self) -> &T;
}

impl<A: ArraySubclass> AsArrayRef<<<A as ArraySubclass>::Encoding as Encoding>::Array>
    for PyRef<'_, A>
where
    for<'a> &'a <<A as ArraySubclass>::Encoding as Encoding>::Array:
        TryFrom<&'a Array, Error = VortexError>,
{
    fn as_array_ref(&self) -> &<<A as ArraySubclass>::Encoding as Encoding>::Array {
        <&<<A as ArraySubclass>::Encoding as Encoding>::Array>::try_from(self.as_super().inner())
            .vortex_expect("Failed to downcast array")
    }
}

// TODO(ngates): requires newer PyO3 version
// /// Convert a `Bound<ArraySubclass>` into a Vortex array.
// pub trait IntoArray<T> {
//     fn into_array(self) -> T;
// }
//
// impl<'py, A: ArraySubclass> IntoArray<<<A as ArraySubclass>::Encoding as Encoding>::Array>
//     for Bound<'py, A>
// where
//     <<A as ArraySubclass>::Encoding as Encoding>::Array: TryFrom<Array, Error = VortexError>,
// {
//     fn into_array(self) -> <<A as ArraySubclass>::Encoding as Encoding>::Array {
//         let array = self.into_super().unwrap();
//         <&<<A as ArraySubclass>::Encoding as Encoding>::Array>::try_from(&array.0)
//             .vortex_expect("Failed to downcast array")
//     }
// }
