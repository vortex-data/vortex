use std::ops::Deref;

use pyo3::prelude::*;
use vortex::dtype::DType;
use vortex::error::vortex_panic;

use crate::dtype::PyDType;
use crate::dtype::ptype::PyPType;

/// Concrete class for primitive dtypes.
#[pyclass(name = "PrimitiveDType", module = "vortex", extends=PyDType, frozen)]
pub(crate) struct PyPrimitiveDType;

#[pymethods]
impl PyPrimitiveDType {
    /// The :class:`~vortex.PType` of the primitive dtype.
    #[getter]
    fn ptype(slf: PyRef<Self>) -> PyPType {
        let DType::Primitive(ptype, _) = slf.as_ref().deref() else {
            vortex_panic!("Not a primitive DType");
        };
        PyPType::from(*ptype)
    }
}
