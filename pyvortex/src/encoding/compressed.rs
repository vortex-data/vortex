use pyo3::prelude::*;

use crate::arrays::PyArray;

/// Concrete class for arrays with `vortex.alp` encoding.
#[pyclass(name = "AlpEncoding", module = "vortex", extends=PyArray, frozen)]
pub(crate) struct PyAlpEncoding;

/// Concrete class for arrays with `vortex.alprd` encoding.
#[pyclass(name = "AlpRdEncoding", module = "vortex", extends=PyArray, frozen)]
pub(crate) struct PyAlpRdEncoding;

/// Concrete class for arrays with `vortex.datetimeparts` encoding.
#[pyclass(name = "DateTimePartsEncoding", module = "vortex", extends=PyArray, frozen)]
pub(crate) struct PyDateTimePartsEncoding;

/// Concrete class for arrays with `vortex.dict` encoding.
#[pyclass(name = "DictEncoding", module = "vortex", extends=PyArray, frozen)]
pub(crate) struct PyDictEncoding;

/// Concrete class for arrays with `vortex.fsst` encoding.
#[pyclass(name = "FsstEncoding", module = "vortex", extends=PyArray, frozen)]
pub(crate) struct PyFsstEncoding;

/// Concrete class for arrays with `vortex.runend` encoding.
#[pyclass(name = "RunEndEncoding", module = "vortex", extends=PyArray, frozen)]
pub(crate) struct PyRunEndEncoding;

/// Concrete class for arrays with `vortex.sparse` encoding.
#[pyclass(name = "SparseEncoding", module = "vortex", extends=PyArray, frozen)]
pub(crate) struct PySparseEncoding;

/// Concrete class for arrays with `vortex.zigzag` encoding.
#[pyclass(name = "ZigZagEncoding", module = "vortex", extends=PyArray, frozen)]
pub(crate) struct PyZigZagEncoding;
