use pyo3::prelude::*;

use crate::arrays::PyArray;

/// Concrete class for arrays with `vortex.alp` encoding.
#[pyclass(name = "AlpArray", module = "vortex", extends=PyArray, frozen)]
pub(crate) struct PyAlpArray;

/// Concrete class for arrays with `vortex.alprd` encoding.
#[pyclass(name = "AlpRdArray", module = "vortex", extends=PyArray, frozen)]
pub(crate) struct PyAlpRdArray;

/// Concrete class for arrays with `vortex.datetimeparts` encoding.
#[pyclass(name = "DateTimePartsArray", module = "vortex", extends=PyArray, frozen)]
pub(crate) struct PyDateTimePartsArray;

/// Concrete class for arrays with `vortex.dict` encoding.
#[pyclass(name = "DictArray", module = "vortex", extends=PyArray, frozen)]
pub(crate) struct PyDictArray;

/// Concrete class for arrays with `vortex.fsst` encoding.
#[pyclass(name = "FsstArray", module = "vortex", extends=PyArray, frozen)]
pub(crate) struct PyFsstArray;

/// Concrete class for arrays with `vortex.runend` encoding.
#[pyclass(name = "RunEndArray", module = "vortex", extends=PyArray, frozen)]
pub(crate) struct PyRunEndArray;

/// Concrete class for arrays with `vortex.sparse` encoding.
#[pyclass(name = "SparseArray", module = "vortex", extends=PyArray, frozen)]
pub(crate) struct PySparseArray;

/// Concrete class for arrays with `vortex.zigzag` encoding.
#[pyclass(name = "ZigZagArray", module = "vortex", extends=PyArray, frozen)]
pub(crate) struct PyZigZagArray;
