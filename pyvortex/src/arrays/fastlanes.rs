use pyo3::prelude::*;
use vortex::encodings::fastlanes::{BitPackedEncoding, DeltaEncoding, FoREncoding};

use crate::arrays::{AsArrayRef, EncodingSubclass, PyArray};

/// Concrete class for arrays with `fastlanes.bitpacked` encoding.
#[pyclass(name = "FastLanesBitPackedArray", module = "vortex", extends=PyArray, frozen)]
pub(crate) struct PyFastLanesBitPackedArray;

impl EncodingSubclass for PyFastLanesBitPackedArray {
    type Encoding = BitPackedEncoding;
}

#[pymethods]
impl PyFastLanesBitPackedArray {
    #[new]
    fn new(array: Bound<PyArray>) -> PyResult<Bound<Self>> {
        PyArray::init_encoding(array, &BitPackedEncoding, PyFastLanesBitPackedArray)
    }

    /// Returns the bit width of the packed values.
    #[getter]
    fn bit_width(self_: PyRef<'_, Self>) -> u8 {
        self_.as_array_ref().bit_width()
    }
}

/// Concrete class for arrays with `fastlanes.delta` encoding.
#[pyclass(name = "FastLanesDeltaArray", module = "vortex", extends=PyArray, frozen)]
pub(crate) struct PyFastLanesDeltaArray;

impl EncodingSubclass for PyFastLanesDeltaArray {
    type Encoding = DeltaEncoding;
}

#[pymethods]
impl PyFastLanesDeltaArray {
    #[new]
    fn new(array: Bound<PyArray>) -> PyResult<Bound<Self>> {
        PyArray::init_encoding(array, &DeltaEncoding, PyFastLanesDeltaArray)
    }
}

/// Concrete class for arrays with `fastlanes.for` encoding.
#[pyclass(name = "FastLanesFoRArray", module = "vortex", extends=PyArray, frozen)]
pub(crate) struct PyFastLanesFoRArray;

impl EncodingSubclass for PyFastLanesFoRArray {
    type Encoding = FoREncoding;
}

#[pymethods]
impl PyFastLanesFoRArray {
    #[new]
    fn new(array: Bound<PyArray>) -> PyResult<Bound<Self>> {
        PyArray::init_encoding(array, &FoREncoding, PyFastLanesFoRArray)
    }
}
