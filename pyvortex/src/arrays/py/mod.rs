mod array;
mod encoding;

pub use encoding::*;
use pyo3::{pyclass, pymethods};

use crate::arrays::PyArray;

/// Base class for array encodings implemented in Python.
#[pyclass(name = "PyEncoding", module = "vortex", extends=PyArray, frozen, subclass)]
pub(crate) struct PyEncoding;

#[pymethods]
impl PyEncoding {}
