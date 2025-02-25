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
#[pyclass(name = "AlpEncoding", module = "vortex", extends=PyArray, frozen)]
pub(crate) struct PyAlpEncoding;

impl EncodingSubclass for PyAlpEncoding {
    type Encoding = ALPEncoding;
}

#[pymethods]
impl PyAlpEncoding {
    #[new]
    fn new(array: Bound<PyArray>) -> PyResult<Bound<Self>> {
        PyArray::init_encoding(array, PyAlpEncoding)
    }
}

/// Concrete class for arrays with `vortex.alprd` encoding.
#[pyclass(name = "AlpRdEncoding", module = "vortex", extends=PyArray, frozen)]
pub(crate) struct PyAlpRdEncoding;

impl EncodingSubclass for PyAlpRdEncoding {
    type Encoding = ALPRDEncoding;
}

#[pymethods]
impl PyAlpRdEncoding {
    #[new]
    fn new(array: Bound<PyArray>) -> PyResult<Bound<Self>> {
        PyArray::init_encoding(array, PyAlpRdEncoding)
    }
}

/// Concrete class for arrays with `vortex.datetimeparts` encoding.
#[pyclass(name = "DateTimePartsEncoding", module = "vortex", extends=PyArray, frozen)]
pub(crate) struct PyDateTimePartsEncoding;

impl EncodingSubclass for PyDateTimePartsEncoding {
    type Encoding = DateTimePartsEncoding;
}

#[pymethods]
impl PyDateTimePartsEncoding {
    #[new]
    fn new(array: Bound<PyArray>) -> PyResult<Bound<Self>> {
        PyArray::init_encoding(array, PyDateTimePartsEncoding)
    }
}

/// Concrete class for arrays with `vortex.dict` encoding.
#[pyclass(name = "DictEncoding", module = "vortex", extends=PyArray, frozen)]
pub(crate) struct PyDictEncoding;

impl EncodingSubclass for PyDictEncoding {
    type Encoding = DictEncoding;
}

#[pymethods]
impl PyDictEncoding {
    #[new]
    fn new(array: Bound<PyArray>) -> PyResult<Bound<Self>> {
        PyArray::init_encoding(array, PyDictEncoding)
    }
}

/// Concrete class for arrays with `vortex.fsst` encoding.
#[pyclass(name = "FsstEncoding", module = "vortex", extends=PyArray, frozen)]
pub(crate) struct PyFsstEncoding;

impl EncodingSubclass for PyFsstEncoding {
    type Encoding = FSSTEncoding;
}

#[pymethods]
impl PyFsstEncoding {
    #[new]
    fn new(array: Bound<PyArray>) -> PyResult<Bound<Self>> {
        PyArray::init_encoding(array, PyFsstEncoding)
    }
}

/// Concrete class for arrays with `vortex.runend` encoding.
#[pyclass(name = "RunEndEncoding", module = "vortex", extends=PyArray, frozen)]
pub(crate) struct PyRunEndEncoding;

impl EncodingSubclass for PyRunEndEncoding {
    type Encoding = RunEndEncoding;
}

#[pymethods]
impl PyRunEndEncoding {
    #[new]
    fn new(array: Bound<PyArray>) -> PyResult<Bound<Self>> {
        PyArray::init_encoding(array, PyRunEndEncoding)
    }
}

/// Concrete class for arrays with `vortex.sparse` encoding.
#[pyclass(name = "SparseEncoding", module = "vortex", extends=PyArray, frozen)]
pub(crate) struct PySparseEncoding;

impl EncodingSubclass for PySparseEncoding {
    type Encoding = SparseEncoding;
}

#[pymethods]
impl PySparseEncoding {
    #[new]
    fn new(array: Bound<PyArray>) -> PyResult<Bound<Self>> {
        PyArray::init_encoding(array, PySparseEncoding)
    }
}

/// Concrete class for arrays with `vortex.zigzag` encoding.
#[pyclass(name = "ZigZagEncoding", module = "vortex", extends=PyArray, frozen)]
pub(crate) struct PyZigZagEncoding;

impl EncodingSubclass for PyZigZagEncoding {
    type Encoding = ZigZagEncoding;
}

#[pymethods]
impl PyZigZagEncoding {
    #[new]
    fn new(array: Bound<PyArray>) -> PyResult<Bound<Self>> {
        PyArray::init_encoding(array, PyZigZagEncoding)
    }
}
