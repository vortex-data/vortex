use pyo3::prelude::*;
use vortex::encodings::alp::{ALPEncoding, ALPRDEncoding};
use vortex::encodings::datetime_parts::DateTimePartsEncoding;
use vortex::encodings::dict::DictEncoding;
use vortex::encodings::fsst::FSSTEncoding;
use vortex::encodings::runend::RunEndEncoding;
use vortex::encodings::sparse::SparseEncoding;
use vortex::encodings::zigzag::ZigZagEncoding;

use crate::arrays::{EncodingSubclass, PyArray};

/// Concrete class for arrays with `vortex.alp` encoding.
#[pyclass(name = "AlpArray", module = "vortex", extends=PyArray, frozen)]
pub(crate) struct PyAlpArray;

impl EncodingSubclass for PyAlpArray {
    type Encoding = ALPEncoding;
}

#[pymethods]
impl PyAlpArray {
    #[new]
    fn new(array: Bound<PyArray>) -> PyResult<Bound<Self>> {
        PyArray::init_encoding(array, &ALPEncoding, PyAlpArray)
    }
}

/// Concrete class for arrays with `vortex.alprd` encoding.
#[pyclass(name = "AlpRdArray", module = "vortex", extends=PyArray, frozen)]
pub(crate) struct PyAlpRdArray;

impl EncodingSubclass for PyAlpRdArray {
    type Encoding = ALPRDEncoding;
}

#[pymethods]
impl PyAlpRdArray {
    #[new]
    fn new(array: Bound<PyArray>) -> PyResult<Bound<Self>> {
        PyArray::init_encoding(array, &ALPRDEncoding, PyAlpRdArray)
    }
}

/// Concrete class for arrays with `vortex.datetimeparts` encoding.
#[pyclass(name = "DateTimePartsArray", module = "vortex", extends=PyArray, frozen)]
pub(crate) struct PyDateTimePartsArray;

impl EncodingSubclass for PyDateTimePartsArray {
    type Encoding = DateTimePartsEncoding;
}

#[pymethods]
impl PyDateTimePartsArray {
    #[new]
    fn new(array: Bound<PyArray>) -> PyResult<Bound<Self>> {
        PyArray::init_encoding(array, &DateTimePartsEncoding, PyDateTimePartsArray)
    }
}

/// Concrete class for arrays with `vortex.dict` encoding.
#[pyclass(name = "DictArray", module = "vortex", extends=PyArray, frozen)]
pub(crate) struct PyDictArray;

impl EncodingSubclass for PyDictArray {
    type Encoding = DictEncoding;
}

#[pymethods]
impl PyDictArray {
    #[new]
    fn new(array: Bound<PyArray>) -> PyResult<Bound<Self>> {
        PyArray::init_encoding(array, &DictEncoding, PyDictArray)
    }
}

/// Concrete class for arrays with `vortex.fsst` encoding.
#[pyclass(name = "FsstArray", module = "vortex", extends=PyArray, frozen)]
pub(crate) struct PyFsstArray;

impl EncodingSubclass for PyFsstArray {
    type Encoding = FSSTEncoding;
}

#[pymethods]
impl PyFsstArray {
    #[new]
    fn new(array: Bound<PyArray>) -> PyResult<Bound<Self>> {
        PyArray::init_encoding(array, &FSSTEncoding, PyFsstArray)
    }
}

/// Concrete class for arrays with `vortex.runend` encoding.
#[pyclass(name = "RunEndArray", module = "vortex", extends=PyArray, frozen)]
pub(crate) struct PyRunEndArray;

impl EncodingSubclass for PyRunEndArray {
    type Encoding = RunEndEncoding;
}

#[pymethods]
impl PyRunEndArray {
    #[new]
    fn new(array: Bound<PyArray>) -> PyResult<Bound<Self>> {
        PyArray::init_encoding(array, &RunEndEncoding, PyRunEndArray)
    }
}

/// Concrete class for arrays with `vortex.sparse` encoding.
#[pyclass(name = "SparseArray", module = "vortex", extends=PyArray, frozen)]
pub(crate) struct PySparseArray;

impl EncodingSubclass for PySparseArray {
    type Encoding = SparseEncoding;
}

#[pymethods]
impl PySparseArray {
    #[new]
    fn new(array: Bound<PyArray>) -> PyResult<Bound<Self>> {
        PyArray::init_encoding(array, &SparseEncoding, PySparseArray)
    }
}

/// Concrete class for arrays with `vortex.zigzag` encoding.
#[pyclass(name = "ZigZagArray", module = "vortex", extends=PyArray, frozen)]
pub(crate) struct PyZigZagArray;

impl EncodingSubclass for PyZigZagArray {
    type Encoding = ZigZagEncoding;
}

#[pymethods]
impl PyZigZagArray {
    #[new]
    fn new(array: Bound<PyArray>) -> PyResult<Bound<Self>> {
        PyArray::init_encoding(array, &ZigZagEncoding, PyZigZagArray)
    }
}
