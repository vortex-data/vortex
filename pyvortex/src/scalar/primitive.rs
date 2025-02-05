use pyo3::pyclass;
use vortex::scalar::PrimitiveScalar;

use crate::scalar::{PyScalar, ScalarSubclass};

/// Concrete class for primitive scalars.
#[pyclass(name = "PrimitiveScalar", module = "vortex", extends=PyScalar, frozen)]
pub(crate) struct PyPrimitiveScalar;

impl ScalarSubclass for PyPrimitiveScalar {
    type Scalar<'a> = PrimitiveScalar<'a>;
}
