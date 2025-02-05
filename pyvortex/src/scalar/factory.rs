use std::sync::Arc;

use itertools::Itertools;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyBool, PyBytes, PyDict, PyFloat, PyInt, PyList, PyString};
use vortex::buffer::ByteBuffer;
use vortex::dtype::{DType, FieldName, FieldNames, Nullability, StructDType};
use vortex::scalar::Scalar;

use crate::dtype::PyDType;
use crate::scalar::{bool, PyScalar};

#[allow(unused_variables)]
#[pyfunction(name = "scalar")]
#[pyo3(signature = (value, *, dtype=None))]
pub fn scalar<'py>(
    py: Python<'py>,
    value: Bound<'py, PyAny>,
    dtype: Option<PyDType>,
) -> PyResult<Bound<'py, PyScalar>> {
    PyScalar::init(
        py,
        scalar_helper(&value, dtype.as_ref().map(|dtype| dtype.inner()))?,
    )
}

pub fn scalar_helper(value: &Bound<'_, PyAny>, dtype: Option<&DType>) -> PyResult<Scalar> {
    let scalar = scalar_helper_inner(value, dtype)?;

    // If a dtype was provided, attempt to  cast the scalar to that dtype.
    // This is a trivially cheap no-op if the scalar is already of the correct type.
    if let Some(dtype) = dtype {
        Ok(scalar.cast(dtype)?)
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

    // dict
    if let Ok(dict) = value.downcast::<PyDict>() {
        // Extract the field names from the dictionary keys
        let names: FieldNames = dict
            .keys()
            .iter()
            .map(|key| key.extract::<String>())
            .map_ok(Arc::from)
            .collect::<PyResult<Vec<FieldName>>>()?
            .into();

        if let Some(DType::Struct(dtype, nullability)) = dtype {
            if &names != dtype.names() {
                return Err(PyValueError::new_err(format!(
                    "Dictionary field names {:?} do not match target dtype names {:?}",
                    &names,
                    dtype.names()
                )));
            }

            return Ok(Scalar::struct_(
                DType::Struct(dtype.clone(), *nullability),
                dict.values()
                    .into_iter()
                    .map(|item| scalar_helper_inner(&item, None))
                    .try_collect()?,
            ));
        } else {
            let values: Vec<Scalar> = dict
                .values()
                .into_iter()
                .map(|value| scalar_helper_inner(&value, None))
                .try_collect()?;
            let dtype = DType::Struct(
                Arc::new(StructDType::new(
                    names,
                    values.iter().map(|value| value.dtype().clone()).collect(),
                )),
                Nullability::NonNullable,
            );
            return Ok(Scalar::struct_(dtype, values));
        };
    }

    if let Ok(list) = value.downcast::<PyList>() {
        if let Some(DType::List(element_dtype, ..)) = dtype {
            let elements = list
                .iter()
                .map(|e| scalar_helper_inner(&e, Some(element_dtype)))
                .try_collect()?;
            Scalar::list(element_dtype.clone(), elements, Nullability::NonNullable);
        } else {
            // If no dtype was provided, we need to infer the element dtype from the list contents.
            // We do this in a greedy way taking the first element dtype we find.
            let mut elements = Vec::with_capacity(list.len());
            let mut element_dtype = None;

            for element in list.iter() {
                let scalar = scalar_helper_inner(&element, element_dtype.as_ref())?;
                if element_dtype.is_none() {
                    element_dtype = Some(scalar.dtype().clone());
                }
                elements.push(scalar);
            }

            return Ok(Scalar::list(
                element_dtype
                    .map(Arc::new)
                    // Empty list defaults to Null dtype
                    .unwrap_or_else(|| Arc::new(DType::Null)),
                elements,
                Nullability::NonNullable,
            ));
        }
    }

    Err(pyo3::exceptions::PyTypeError::new_err(format!(
        "Cannot convert Python object to Vortex scalar: {}",
        value.get_type()
    )))
}
