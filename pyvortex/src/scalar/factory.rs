use std::sync::Arc;

use pyo3::prelude::PyAnyMethods;
use pyo3::types::{PyBool, PyBytes, PyFloat, PyInt, PyString};
use pyo3::{pyfunction, Bound, PyAny, PyResult};
use vortex::buffer::ByteBuffer;
use vortex::dtype::{DType, Nullability};
use vortex::scalar::Scalar;

use crate::dtype::PyDType;
use crate::scalar::{bool, PyScalar};

#[allow(unused_variables)]
#[pyfunction(name = "scalar")]
#[pyo3(signature = (value, dtype=None))]
pub fn scalar(value: Bound<'_, PyAny>, dtype: Option<PyDType>) -> PyResult<PyScalar> {
    Ok(PyScalar(Scalar::bool(true, Nullability::Nullable)))
}

pub fn scalar_helper(value: &Bound<'_, PyAny>, dtype: Option<&DType>) -> PyResult<Scalar> {
    let scalar = scalar_helper_inner(value, dtype)?;

    // If a dtype was provided, attempt to  cast the scalar to that dtype.
    // This is a trivially cheap no-op if the scalar is already of the correct type.
    if let Some(dtype) = dtype {
        Ok(scalar.cast(&dtype)?)
    } else {
        Ok(scalar)
    }
}

// This function attempts to convert the python object to a scalar, with a hint of the expected
// dtype. It can assume that the scalar_helper function will perform a final cast to the correct
// dtype if necessary.
fn scalar_helper_inner(value: &Bound<'_, PyAny>, dtype: Option<&DType>) -> PyResult<Scalar> {
    // If it's already a scalar, return it
    if let Ok(value) = value.downcast::<PyScalar>() {
        return Ok(value.get().inner().clone());
    }

    // Otherwise, we start checking the known Python types.

    // None
    if value.is_none() {
        return Ok(Scalar::null(dtype.cloned().unwrap_or(DType::Null)));
    }

    // bool
    if let Ok(bool) = value.downcast::<PyBool>() {
        return Ok(Scalar::bool(
            bool.extract::<bool>()?,
            Nullability::NonNullable,
        ));
    }

    // int
    if let Ok(integer) = value.downcast::<PyInt>() {
        return Ok(Scalar::primitive(
            integer.extract::<i64>()?,
            Nullability::NonNullable,
        ));
    }

    // float
    if let Ok(float) = value.downcast::<PyFloat>() {
        return Ok(Scalar::primitive(
            float.extract::<f64>()?,
            Nullability::NonNullable,
        ));
    }

    // str
    if let Ok(string) = value.downcast::<PyString>() {
        return Ok(Scalar::utf8(
            string.extract::<String>()?,
            Nullability::NonNullable,
        ));
    }

    // bytes
    if let Ok(bytes) = value.downcast::<PyBytes>() {
        return Ok(Scalar::binary(
            Arc::new(ByteBuffer::from(bytes.extract::<Vec<u8>>()?)),
            Nullability::NonNullable,
        ));
    }

    Err(pyo3::exceptions::PyTypeError::new_err(
        "Invalid scalar type",
    ))
    //
    // match dtype {
    //     DType::Null => {
    //         value.downcast::<PyNone>()?;
    //         Ok(Scalar::null(dtype))
    //     }
    //     DType::Bool(_) => {
    //         let value = value.downcast::<PyBool>()?;
    //         Ok(Scalar::from(value.extract::<bool>()?))
    //     }
    //     DType::Primitive(ptype, _) => match ptype {
    //         PType::I8 => Ok(Scalar::from(value.extract::<i8>()?)),
    //         PType::I16 => Ok(Scalar::from(value.extract::<i16>()?)),
    //         PType::I32 => Ok(Scalar::from(value.extract::<i32>()?)),
    //         PType::I64 => Ok(Scalar::from(value.extract::<i64>()?)),
    //         PType::U8 => Ok(Scalar::from(value.extract::<u8>()?)),
    //         PType::U16 => Ok(Scalar::from(value.extract::<u16>()?)),
    //         PType::U32 => Ok(Scalar::from(value.extract::<u32>()?)),
    //         PType::U64 => Ok(Scalar::from(value.extract::<u64>()?)),
    //         PType::F16 => {
    //             let float = value.extract::<f32>()?;
    //             Ok(Scalar::from(f16::from_f32(float)))
    //         }
    //         PType::F32 => Ok(Scalar::from(value.extract::<f32>()?)),
    //         PType::F64 => Ok(Scalar::from(value.extract::<f64>()?)),
    //     },
    //     DType::Utf8(_) => Ok(Scalar::from(value.extract::<String>()?)),
    //     DType::Binary(_) => Ok(Scalar::from(value.extract::<&[u8]>()?)),
    //     DType::Struct(..) => todo!(),
    //     DType::List(element_type, _) => {
    //         let list = value.downcast::<PyList>();
    //         let values = list
    //             .iter()
    //             .map(|element| scalar_helper(element, element_type.as_ref().clone()))
    //             .collect::<PyResult<Vec<_>>>()?;
    //         Ok(Scalar::list(element_type, values, Nullability::Nullable))
    //     }
    //     DType::Extension(..) => todo!(),
    // }
}
