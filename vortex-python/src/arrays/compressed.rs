use pyo3::prelude::*;
use vortex::encodings::alp::{ALPRDVTable, ALPVTable};
use vortex::encodings::datetime_parts::DateTimePartsVTable;
use vortex::encodings::dict::DictVTable;
use vortex::encodings::fsst::FSSTVTable;
use vortex::encodings::runend::RunEndVTable;
use vortex::encodings::sparse::SparseVTable;
use vortex::encodings::zigzag::ZigZagVTable;

use crate::arrays::native::{EncodingSubclass, PyNativeArray};

/// Concrete class for arrays with `vortex.alp` encoding.
#[pyclass(name = "AlpArray", module = "vortex", extends=PyNativeArray, frozen)]
pub(crate) struct PyAlpArray;

impl EncodingSubclass for PyAlpArray {
    type VTable = ALPVTable;
}

/// Concrete class for arrays with `vortex.alprd` encoding.
#[pyclass(name = "AlpRdArray", module = "vortex", extends=PyNativeArray, frozen)]
pub(crate) struct PyAlpRdArray;

impl EncodingSubclass for PyAlpRdArray {
    type VTable = ALPRDVTable;
}

/// Concrete class for arrays with `vortex.datetimeparts` encoding.
#[pyclass(name = "DateTimePartsArray", module = "vortex", extends=PyNativeArray, frozen)]
pub(crate) struct PyDateTimePartsArray;

impl EncodingSubclass for PyDateTimePartsArray {
    type VTable = DateTimePartsVTable;
}

/// Concrete class for arrays with `vortex.dict` encoding.
#[pyclass(name = "DictArray", module = "vortex", extends=PyNativeArray, frozen)]
pub(crate) struct PyDictArray;

impl EncodingSubclass for PyDictArray {
    type VTable = DictVTable;
}

/// Concrete class for arrays with `vortex.fsst` encoding.
#[pyclass(name = "FsstArray", module = "vortex", extends=PyNativeArray, frozen)]
pub(crate) struct PyFsstArray;

impl EncodingSubclass for PyFsstArray {
    type VTable = FSSTVTable;
}

/// Concrete class for arrays with `vortex.runend` encoding.
#[pyclass(name = "RunEndArray", module = "vortex", extends=PyNativeArray, frozen)]
pub(crate) struct PyRunEndArray;

impl EncodingSubclass for PyRunEndArray {
    type VTable = RunEndVTable;
}

/// Concrete class for arrays with `vortex.sparse` encoding.
#[pyclass(name = "SparseArray", module = "vortex", extends=PyNativeArray, frozen)]
pub(crate) struct PySparseArray;

impl EncodingSubclass for PySparseArray {
    type VTable = SparseVTable;
}

/// Concrete class for arrays with `vortex.zigzag` encoding.
#[pyclass(name = "ZigZagArray", module = "vortex", extends=PyNativeArray, frozen)]
pub(crate) struct PyZigZagArray;

impl EncodingSubclass for PyZigZagArray {
    type VTable = ZigZagVTable;
}
