use pyo3::pyclass;
use vortex::scalar::BinaryScalar;

use crate::scalar::{PyScalar, ScalarSubclass};

/// Concrete class for binary scalars.
#[pyclass(name = "BinaryScalar", module = "vortex", extends=PyScalar, frozen)]
pub(crate) struct PyBinaryScalar;

// TODO(ngates): implement buffer protocol
impl ScalarSubclass for PyBinaryScalar {
    type Scalar<'a> = BinaryScalar<'a>;
}
