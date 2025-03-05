use pyo3::prelude::*;
use vortex::encodings::alp::{ALPEncoding, ALPRDEncoding};
use vortex::encodings::datetime_parts::DateTimePartsEncoding;
use vortex::encodings::dict::DictEncoding;
use vortex::encodings::fsst::FSSTEncoding;
use vortex::encodings::runend::RunEndEncoding;
use vortex::encodings::sparse::SparseEncoding;
use vortex::encodings::zigzag::ZigZagEncoding;

use crate::arrays::native::{EncodingSubclass, PyNativeArray};

/// Concrete class for arrays with `vortex.alp` encoding.
#[pyclass(name = "AlpArray", module = "vortex", extends=PyNativeArray, frozen)]
pub(crate) struct PyAlpArray;

impl EncodingSubclass for PyAlpArray {
    type Encoding = ALPEncoding;
}

/// Concrete class for arrays with `vortex.alprd` encoding.
#[pyclass(name = "AlpRdArray", module = "vortex", extends=PyNativeArray, frozen)]
pub(crate) struct PyAlpRdArray;

impl EncodingSubclass for PyAlpRdArray {
    type Encoding = ALPRDEncoding;
}

/// Concrete class for arrays with `vortex.datetimeparts` encoding.
#[pyclass(name = "DateTimePartsArray", module = "vortex", extends=PyNativeArray, frozen)]
pub(crate) struct PyDateTimePartsArray;

impl EncodingSubclass for PyDateTimePartsArray {
    type Encoding = DateTimePartsEncoding;
}

/// Concrete class for arrays with `vortex.dict` encoding.
#[pyclass(name = "DictArray", module = "vortex", extends=PyNativeArray, frozen)]
pub(crate) struct PyDictArray;

impl EncodingSubclass for PyDictArray {
    type Encoding = DictEncoding;
}

/// Concrete class for arrays with `vortex.fsst` encoding.
#[pyclass(name = "FsstArray", module = "vortex", extends=PyNativeArray, frozen)]
pub(crate) struct PyFsstArray;

impl EncodingSubclass for PyFsstArray {
    type Encoding = FSSTEncoding;
}

/// Concrete class for arrays with `vortex.runend` encoding.
#[pyclass(name = "RunEndArray", module = "vortex", extends=PyNativeArray, frozen)]
pub(crate) struct PyRunEndArray;

impl EncodingSubclass for PyRunEndArray {
    type Encoding = RunEndEncoding;
}

/// Concrete class for arrays with `vortex.sparse` encoding.
#[pyclass(name = "SparseArray", module = "vortex", extends=PyNativeArray, frozen)]
pub(crate) struct PySparseArray;

impl EncodingSubclass for PySparseArray {
    type Encoding = SparseEncoding;
}

/// Concrete class for arrays with `vortex.zigzag` encoding.
#[pyclass(name = "ZigZagArray", module = "vortex", extends=PyNativeArray, frozen)]
pub(crate) struct PyZigZagArray;

impl EncodingSubclass for PyZigZagArray {
    type Encoding = ZigZagEncoding;
}
