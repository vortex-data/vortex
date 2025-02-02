use pyo3::pyclass;

use crate::scalar::{PyScalar, ScalarSubclass};

#[pyclass(name = "NullScalar", module = "vortex", extends=PyScalar, frozen)]
pub(crate) struct PyNullScalar;

/// Concrete class for null scalars.
impl ScalarSubclass for PyNullScalar {
    type Scalar<'a> = ();
}
