use pyo3::{pyclass, pymethods, IntoPy, PyObject, PyRef};
use vortex::dtype::half::f16;
use vortex::dtype::PType;
use vortex::scalar::PrimitiveScalar;

use crate::scalar::{AsScalarRef, PyScalar, ScalarSubclass};

#[pyclass(name = "PrimitiveScalar", module = "vortex", extends=PyScalar, frozen)]
pub(crate) struct PyPrimitiveScalar;

impl ScalarSubclass for PyPrimitiveScalar {
    type Scalar<'a> = PrimitiveScalar<'a>;
}

#[pymethods]
impl PyPrimitiveScalar {
    /// Return this value as a Python primitive integer or float.
    pub fn as_py(self_: PyRef<'_, Self>) -> PyObject {
        let scalar = self_.as_scalar_ref();
        match scalar.ptype() {
            PType::U8 => scalar.typed_value::<u8>().map(|v| v.into_py(self_.py())),
            PType::U16 => scalar.typed_value::<u16>().map(|v| v.into_py(self_.py())),
            PType::U32 => scalar.typed_value::<u32>().map(|v| v.into_py(self_.py())),
            PType::U64 => scalar.typed_value::<u64>().map(|v| v.into_py(self_.py())),
            PType::I8 => scalar.typed_value::<i8>().map(|v| v.into_py(self_.py())),
            PType::I16 => scalar.typed_value::<i16>().map(|v| v.into_py(self_.py())),
            PType::I32 => scalar.typed_value::<i32>().map(|v| v.into_py(self_.py())),
            PType::I64 => scalar.typed_value::<i64>().map(|v| v.into_py(self_.py())),
            PType::F16 => scalar
                .typed_value::<f16>()
                .map(|v| v.to_f32().into_py(self_.py())),
            PType::F32 => scalar.typed_value::<f32>().map(|v| v.into_py(self_.py())),
            PType::F64 => scalar.typed_value::<f64>().map(|v| v.into_py(self_.py())),
        }
        .unwrap_or_else(|| self_.py().None())
    }
}
