// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::sync::Arc;

use itertools::Itertools;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyBool;
use pyo3::types::PyBytes;
use pyo3::types::PyDict;
use pyo3::types::PyFloat;
use pyo3::types::PyInt;
use pyo3::types::PyList;
use pyo3::types::PyString;
use vortex::dtype::DType;
use vortex::dtype::FieldName;
use vortex::dtype::FieldNames;
use vortex::dtype::Nullability;
use vortex::dtype::StructFields;
use vortex::scalar::DecimalValue;
use vortex::scalar::Scalar;

use crate::dtype::PyDType;
use crate::error::PyVortexResult;
use crate::scalar::PyScalar;
use crate::scalar::bool;

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

pub fn scalar_helper(value: &Bound<'_, PyAny>, dtype: Option<&DType>) -> PyVortexResult<Scalar> {
    let scalar = scalar_helper_inner(value, dtype)?;

    // If a dtype was provided, attempt to  cast the scalar to that dtype.
    // This is a trivially cheap no-op if the scalar is already of the correct type.
    if let Some(dtype) = dtype {
        Ok(scalar.cast(dtype)?)
    } else {
        Ok(scalar)
    }
}

/// Attempts to convert the python object to a scalar, with a hint of the expected
/// dtype. It can assume that the scalar_helper function will perform a final cast to the correct
/// dtype if necessary.
fn scalar_helper_inner(value: &Bound<'_, PyAny>, dtype: Option<&DType>) -> PyResult<Scalar> {
    // If it's already a scalar, return it
    if let Ok(value) = value.cast::<PyScalar>() {
        return Ok(value.get().inner().clone());
    }

    // Otherwise, we start checking the known Python types.

    // None
    if value.is_none() {
        return Ok(Scalar::null(dtype.cloned().unwrap_or(DType::Null)));
    }

    // bool
    if let Ok(bool) = value.cast::<PyBool>() {
        return Ok(Scalar::bool(
            bool.extract::<bool>()?,
            Nullability::NonNullable,
        ));
    }

    // decimal
    if let Some(decimal_dtype) = dtype.and_then(|d| d.as_decimal_opt()) {
        let value = if let Ok(v) = value.extract::<i8>() {
            DecimalValue::I8(v)
        } else if let Ok(v) = value.extract::<i16>() {
            DecimalValue::I16(v)
        } else if let Ok(v) = value.extract::<i32>() {
            DecimalValue::I32(v)
        } else if let Ok(v) = value.extract::<i64>() {
            DecimalValue::I64(v)
        } else if let Ok(v) = value.extract::<i128>() {
            DecimalValue::I128(v)
        } else {
            return Err(PyValueError::new_err(
                "Value can't be represented as decimal",
            ));
        };
        return Ok(Scalar::decimal(
            value,
            *decimal_dtype,
            Nullability::NonNullable,
        ));
    }

    if let Ok(integer) = value.cast::<PyInt>() {
        return Ok(Scalar::primitive(
            integer.extract::<i64>()?,
            Nullability::NonNullable,
        ));
    }

    // float
    if let Ok(float) = value.cast::<PyFloat>() {
        return Ok(Scalar::primitive(
            float.extract::<f64>()?,
            Nullability::NonNullable,
        ));
    }

    // str
    if let Ok(string) = value.cast::<PyString>() {
        return Ok(Scalar::utf8(
            string.extract::<String>()?,
            Nullability::NonNullable,
        ));
    }

    // bytes
    if let Ok(bytes) = value.cast::<PyBytes>() {
        return Ok(Scalar::binary(
            bytes.extract::<Vec<u8>>()?,
            Nullability::NonNullable,
        ));
    }

    // dict
    if let Ok(dict) = value.cast::<PyDict>() {
        // Extract the field names from the dictionary keys
        let names: FieldNames = dict
            .keys()
            .iter()
            .map(|key| key.extract::<String>())
            .map_ok(FieldName::from)
            .collect::<PyResult<Vec<FieldName>>>()?
            .into();

        if let Some(DType::Struct(dtype, nullability)) = dtype {
            if names != dtype.names() {
                return Err(PyValueError::new_err(format!(
                    "Dictionary field names {:?} do not match target dtype names {:?}",
                    &names,
                    dtype.names()
                )));
            }

            let children: Vec<Scalar> = dict
                .values()
                .into_iter()
                .map(|item| scalar_helper_inner(&item, None))
                .try_collect()?;
            return Ok(Scalar::struct_(
                DType::Struct(dtype.clone(), *nullability),
                children,
            ));
        } else {
            let values: Vec<Scalar> = dict
                .values()
                .into_iter()
                .map(|value| scalar_helper_inner(&value, None))
                .try_collect()?;
            let dtype = DType::Struct(
                StructFields::new(
                    names,
                    values.iter().map(|value| value.dtype().clone()).collect(),
                ),
                Nullability::NonNullable,
            );
            return Ok(Scalar::struct_(dtype, values));
        };
    }

    if let Ok(list) = value.cast::<PyList>() {
        if let Some(DType::List(element_dtype, ..)) = dtype {
            let elements = list
                .iter()
                .map(|e| scalar_helper_inner(&e, Some(element_dtype)))
                .try_collect()?;
            Scalar::list(
                Arc::clone(element_dtype),
                elements,
                Nullability::NonNullable,
            );
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
