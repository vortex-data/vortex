use std::ops::Deref;

use pyo3::{PyRef, PyResult, pyclass, pymethods};
use vortex::dtype::DType;
use vortex::error::vortex_panic;

use crate::dtype::PyDType;

/// Concrete class for primitive dtypes.
#[pyclass(name = "DecimalDType", module = "vortex", extends=PyDType, frozen)]
pub(crate) struct PyDecimalDType;

#[pymethods]
impl PyDecimalDType {
    /// The decimal precision.
    #[getter]
    fn precision(slf: PyRef<Self>) -> PyResult<u8> {
        let DType::Decimal(decimal_dtype, _) = slf.as_ref().deref() else {
            vortex_panic!("Not a decimal DType");
        };
        Ok(decimal_dtype.precision())
    }

    /// The decimal's scale. The last `scale` digits of the scalar are the digits after
    /// the decimal point.
    #[getter]
    fn scale(slf: PyRef<Self>) -> PyResult<i8> {
        let DType::Decimal(decimal_dtype, _) = slf.as_ref().deref() else {
            vortex_panic!("Not a decimal DType");
        };
        Ok(decimal_dtype.scale())
    }
}
