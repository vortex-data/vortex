use pyo3::prelude::*;
use vortex::encodings::fastlanes::{BitPackedEncoding, DeltaEncoding, FoREncoding};

use crate::arrays::{AsArrayRef, EncodingSubclass, PyArray};

/// Concrete class for arrays with `fastlanes.bitpacked` encoding.
#[pyclass(name = "FastLanesBitPackedEncoding", module = "vortex", extends=PyArray, frozen)]
pub(crate) struct PyFastLanesBitPackedEncoding;

impl EncodingSubclass for PyFastLanesBitPackedEncoding {
    type Encoding = BitPackedEncoding;
}

#[pymethods]
impl PyFastLanesBitPackedEncoding {
    #[new]
    fn new(array: Bound<PyArray>) -> PyResult<Bound<Self>> {
        PyArray::init_encoding(array, &BitPackedEncoding, PyFastLanesBitPackedEncoding)
    }

    /// Returns the bit width of the packed values.
    #[getter]
    fn bit_width(self_: PyRef<'_, Self>) -> u8 {
        self_.as_array_ref().bit_width()
    }
}

/// Concrete class for arrays with `fastlanes.delta` encoding.
#[pyclass(name = "FastLanesDeltaEncoding", module = "vortex", extends=PyArray, frozen)]
pub(crate) struct PyFastLanesDeltaEncoding;

impl EncodingSubclass for PyFastLanesDeltaEncoding {
    type Encoding = DeltaEncoding;
}

#[pymethods]
impl PyFastLanesDeltaEncoding {
    #[new]
    fn new(array: Bound<PyArray>) -> PyResult<Bound<Self>> {
        PyArray::init_encoding(array, &DeltaEncoding, PyFastLanesDeltaEncoding)
    }
}

/// Concrete class for arrays with `fastlanes.for` encoding.
#[pyclass(name = "FastLanesFoREncoding", module = "vortex", extends=PyArray, frozen)]
pub(crate) struct PyFastLanesFoREncoding;

impl EncodingSubclass for PyFastLanesFoREncoding {
    type Encoding = FoREncoding;
}

#[pymethods]
impl PyFastLanesFoREncoding {
    #[new]
    fn new(array: Bound<PyArray>) -> PyResult<Bound<Self>> {
        PyArray::init_encoding(array, &FoREncoding, PyFastLanesFoREncoding)
    }
}
