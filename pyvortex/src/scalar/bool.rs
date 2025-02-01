use pyo3::{pyclass, pymethods, PyClass, PyRef};
use vortex::error::{VortexError, VortexExpect};
use vortex::scalar::{BoolScalar, Scalar};

use crate::scalar::PyScalar;

#[pyclass(name = "BoolScalar", module = "vortex", extends=PyScalar, frozen)]
pub struct PyBoolScalar;

#[pymethods]
impl PyBoolScalar {
    pub fn as_py(self_: PyRef<'_, Self>) -> Option<bool> {
        let bool: BoolScalar = self_.as_scalar_ref();
        bool.value()
    }
}

impl ScalarSubclass for PyBoolScalar {
    type Scalar<'a> = BoolScalar<'a>;
}

/// A marker trait indicating a PyO3 class is a subclass of Vortex `Array`.
pub trait ScalarSubclass: PyClass<BaseType = PyScalar> {
    type Scalar<'a>;
}

pub trait AsBorrowedRef<'a, T: 'a> {
    fn as_scalar_ref(&'a self) -> T;
}

impl<'a, T: ScalarSubclass> AsBorrowedRef<'a, <T as ScalarSubclass>::Scalar<'a>> for PyRef<'a, T>
where
    for<'b> <T as ScalarSubclass>::Scalar<'b>: TryFrom<&'b Scalar, Error = VortexError>,
{
    fn as_scalar_ref(&self) -> <T as ScalarSubclass>::Scalar<'_> {
        <<T as ScalarSubclass>::Scalar<'_>>::try_from(self.as_super().inner())
            .vortex_expect("Failed to downcast scalar")
    }
}
