use pyo3::pyclass;
use vortex::scalar::DecimalScalar;

use crate::scalar::{PyScalar, ScalarSubclass};

/// Concrete class for primitive scalars.
#[pyclass(name = "DecimalScalar", module = "vortex", extends=PyScalar, frozen)]
pub(crate) struct PyDecimalScalar;

impl ScalarSubclass for PyDecimalScalar {
    type Scalar<'a> = DecimalScalar<'a>;
}
