use pyo3::exceptions::PyValueError;
use pyo3::prelude::PyAnyMethods;
use pyo3::types::PyDict;
use pyo3::{pyfunction, Bound, Py, PyResult, Python};
use vortex::dtype::{DType, FieldName, PType, StructDType};

use crate::dtype::PyDType;

#[pyfunction(name = "null")]
#[pyo3(signature = ())]
pub(super) fn dtype_null(py: Python<'_>) -> PyResult<Py<PyDType>> {
    PyDType::wrap(py, DType::Null)
}

#[pyfunction(name = "bool_")]
#[pyo3(signature = (*, nullable = false))]
pub(super) fn dtype_bool(py: Python<'_>, nullable: bool) -> PyResult<Py<PyDType>> {
    PyDType::wrap(py, DType::Bool(nullable.into()))
}

#[pyfunction(name = "int_")]
#[pyo3(signature = (width = 64, *, nullable = false))]
pub(super) fn dtype_int(py: Python<'_>, width: u16, nullable: bool) -> PyResult<Py<PyDType>> {
    let dtype = match width {
        8 => DType::Primitive(PType::I8, nullable.into()),
        16 => DType::Primitive(PType::I16, nullable.into()),
        32 => DType::Primitive(PType::I32, nullable.into()),
        64 => DType::Primitive(PType::I64, nullable.into()),
        _ => return Err(PyValueError::new_err("Invalid int width")),
    };
    PyDType::wrap(py, dtype)
}

#[pyfunction(name = "uint")]
#[pyo3(signature = (width = 64, *, nullable = false))]
pub(super) fn dtype_uint(py: Python<'_>, width: u16, nullable: bool) -> PyResult<Py<PyDType>> {
    let dtype = match width {
        8 => DType::Primitive(PType::U8, nullable.into()),
        16 => DType::Primitive(PType::U16, nullable.into()),
        32 => DType::Primitive(PType::U32, nullable.into()),
        64 => DType::Primitive(PType::U64, nullable.into()),
        _ => return Err(PyValueError::new_err("Invalid uint width")),
    };
    PyDType::wrap(py, dtype)
}

#[pyfunction(name = "float_")]
#[pyo3(signature = (width = 64, *, nullable = false))]
pub(super) fn dtype_float(py: Python<'_>, width: i8, nullable: bool) -> PyResult<Py<PyDType>> {
    let dtype = match width {
        16 => DType::Primitive(PType::F16, nullable.into()),
        32 => DType::Primitive(PType::F32, nullable.into()),
        64 => DType::Primitive(PType::F64, nullable.into()),
        _ => return Err(PyValueError::new_err("Invalid float width")),
    };
    PyDType::wrap(py, dtype)
}

#[pyfunction(name = "utf8")]
#[pyo3(signature = (*, nullable = false))]
pub(super) fn dtype_utf8(py: Python<'_>, nullable: bool) -> PyResult<Py<PyDType>> {
    PyDType::wrap(py, DType::Utf8(nullable.into()))
}

#[pyfunction(name = "binary")]
#[pyo3(signature = (*, nullable = false))]
pub(super) fn dtype_binary(py: Python<'_>, nullable: bool) -> PyResult<Py<PyDType>> {
    PyDType::wrap(py, DType::Binary(nullable.into()))
}

// TODO(ngates): return a StructDType to allow inspection of fields
#[pyfunction(name = "struct")]
#[pyo3(signature = (fields = None, *, nullable = false))]
pub(super) fn dtype_struct(
    py: Python<'_>,
    fields: Option<&Bound<'_, PyDict>>,
    nullable: bool,
) -> PyResult<Py<PyDType>> {
    if let Some(fields) = fields {
        let nfields = fields.len()?;
        let mut names = Vec::with_capacity(nfields);
        let mut dtypes = Vec::with_capacity(nfields);

        for (name, field) in fields.into_iter() {
            let field_name = FieldName::from(name.to_string());
            let field_dtype: PyDType = field.extract()?;
            names.push(field_name);
            dtypes.push(field_dtype.unwrap().clone());
        }

        PyDType::wrap(
            py,
            DType::Struct(
                StructDType::new(names.into(), dtypes).into(),
                nullable.into(),
            ),
        )
    } else {
        PyDType::wrap(
            py,
            DType::Struct(
                StructDType::new(vec![].into(), vec![]).into(),
                nullable.into(),
            ),
        )
    }
}
