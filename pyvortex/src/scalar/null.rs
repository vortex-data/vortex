use pyo3::pyclass;

use crate::scalar::{PyScalar, ScalarSubclass};

#[pyclass(name = "NullScalar", module = "vortex", extends=PyScalar, frozen)]
pub(crate) struct PyNullScalar;

impl ScalarSubclass for PyNullScalar {
    type Scalar<'a> = ();
}
