use pyo3::{pyclass, pymethods, IntoPyObject, PyObject, PyRef, PyResult};
use vortex::scalar::StructScalar;

use crate::scalar::{AsScalarRef, PyScalar, ScalarSubclass};
use crate::PyVortex;

/// Concrete class for struct scalars.
#[pyclass(name = "StructScalar", module = "vortex", extends=PyScalar, frozen)]
pub(crate) struct PyStructScalar;

impl ScalarSubclass for PyStructScalar {
    type Scalar<'a> = StructScalar<'a>;
}

#[pymethods]
impl PyStructScalar {
    /// Return the child scalar with the given field name.
    pub fn field(self_: PyRef<'_, Self>, name: &str) -> PyResult<PyObject> {
        let scalar = self_.as_scalar_ref();
        let child = scalar.field(name)?;
        PyVortex(&child).into_pyobject(self_.py()).map(|v| v.into())
    }
}
