use pyo3::pyclass;
use vortex::scalar::BoolScalar;

use crate::scalar::{PyScalar, ScalarSubclass};

/// Concrete class for boolean scalars.
#[pyclass(name = "BoolScalar", module = "vortex", extends=PyScalar, frozen)]
pub(crate) struct PyBoolScalar;

impl ScalarSubclass for PyBoolScalar {
    type Scalar<'a> = BoolScalar<'a>;
}
