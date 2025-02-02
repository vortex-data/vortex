use pyo3::pyclass;
use vortex::scalar::Utf8Scalar;

use crate::scalar::{PyScalar, ScalarSubclass};

#[pyclass(name = "Utf8Scalar", module = "vortex", extends=PyScalar, frozen)]
pub(crate) struct PyUtf8Scalar;

impl ScalarSubclass for PyUtf8Scalar {
    type Scalar<'a> = Utf8Scalar<'a>;
}
