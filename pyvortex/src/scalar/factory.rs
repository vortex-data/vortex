use pyo3::prelude::PyAnyMethods;
use pyo3::types::{PyBool, PyList, PyNone};
use pyo3::{pyfunction, Bound, PyAny, PyResult};
use vortex::dtype::half::f16;
use vortex::dtype::{DType, Nullability, PType};
use vortex::scalar::Scalar;

use crate::dtype::PyDType;
use crate::scalar::PyScalar;

#[allow(unused_variables)]
#[pyfunction(name = "scalar")]
#[pyo3(signature = (value, dtype=None))]
pub fn scalar(value: Bound<'_, PyAny>, dtype: Option<PyDType>) -> PyResult<PyScalar> {
    Ok(PyScalar(Scalar::bool(true, Nullability::Nullable)))
}

pub fn scalar_helper(value: &Bound<'_, PyAny>, dtype: DType) -> PyResult<Scalar> {
    match dtype {
        DType::Null => {
            value.downcast::<PyNone>()?;
            Ok(Scalar::null(dtype))
        }
        DType::Bool(_) => {
            let value = value.downcast::<PyBool>()?;
            Ok(Scalar::from(value.extract::<bool>()?))
        }
        DType::Primitive(ptype, _) => match ptype {
            PType::I8 => Ok(Scalar::from(value.extract::<i8>()?)),
            PType::I16 => Ok(Scalar::from(value.extract::<i16>()?)),
            PType::I32 => Ok(Scalar::from(value.extract::<i32>()?)),
            PType::I64 => Ok(Scalar::from(value.extract::<i64>()?)),
            PType::U8 => Ok(Scalar::from(value.extract::<u8>()?)),
            PType::U16 => Ok(Scalar::from(value.extract::<u16>()?)),
            PType::U32 => Ok(Scalar::from(value.extract::<u32>()?)),
            PType::U64 => Ok(Scalar::from(value.extract::<u64>()?)),
            PType::F16 => {
                let float = value.extract::<f32>()?;
                Ok(Scalar::from(f16::from_f32(float)))
            }
            PType::F32 => Ok(Scalar::from(value.extract::<f32>()?)),
            PType::F64 => Ok(Scalar::from(value.extract::<f64>()?)),
        },
        DType::Utf8(_) => Ok(Scalar::from(value.extract::<String>()?)),
        DType::Binary(_) => Ok(Scalar::from(value.extract::<&[u8]>()?)),
        DType::Struct(..) => todo!(),
        DType::List(element_type, _) => {
            let list = value.downcast::<PyList>();
            let values = list
                .iter()
                .map(|element| scalar_helper(element, element_type.as_ref().clone()))
                .collect::<PyResult<Vec<_>>>()?;
            Ok(Scalar::list(element_type, values, Nullability::Nullable))
        }
        DType::Extension(..) => todo!(),
    }
}
