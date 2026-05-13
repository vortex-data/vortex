// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

mod binary;
mod bool;
mod decimal;
mod extension;
mod factory;
mod fixed_size_list;
mod list;
mod null;
mod primitive;
mod ptype;
mod struct_;
mod utf8;

use std::ops::Deref;

use arrow_schema::DataType;
use arrow_schema::Field;
pub(crate) use ptype::*;
use pyo3::Bound;
use pyo3::Py;
use pyo3::PyAny;
use pyo3::PyClass;
use pyo3::PyClassInitializer;
use pyo3::PyResult;
use pyo3::Python;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::PyModule;
use pyo3::prelude::PyModuleMethods;
use pyo3::pyclass;
use pyo3::pymethods;
use pyo3::types::PyType;
use pyo3::wrap_pyfunction;
use vortex::dtype::DType;
use vortex::dtype::arrow::FromArrowType;

use crate::arrow::FromPyArrow;
use crate::arrow::ToPyArrow;
use crate::dtype::binary::PyBinaryDType;
use crate::dtype::bool::PyBoolDType;
use crate::dtype::decimal::PyDecimalDType;
use crate::dtype::extension::PyExtensionDType;
use crate::dtype::fixed_size_list::PyFixedSizeListDType;
use crate::dtype::list::PyListDType;
use crate::dtype::null::PyNullDType;
use crate::dtype::primitive::PyPrimitiveDType;
use crate::dtype::struct_::PyStructDType;
use crate::dtype::utf8::PyUtf8DType;
use crate::error::PyVortexResult;
use crate::install_module;
use crate::python_repr::PythonRepr;

/// Register DType functions and classes.
pub(crate) fn init(py: Python, parent: &Bound<PyModule>) -> PyResult<()> {
    let m = PyModule::new(py, "dtype")?;
    parent.add_submodule(&m)?;
    install_module("vortex._lib.dtype", &m)?;

    // Register the DType class.
    m.add_class::<PyDType>()?;
    m.add_class::<PyPType>()?;
    m.add_class::<PyNullDType>()?;
    m.add_class::<PyBoolDType>()?;
    m.add_class::<PyPrimitiveDType>()?;
    m.add_class::<PyDecimalDType>()?;
    m.add_class::<PyUtf8DType>()?;
    m.add_class::<PyBinaryDType>()?;
    m.add_class::<PyStructDType>()?;
    m.add_class::<PyListDType>()?;
    m.add_class::<PyFixedSizeListDType>()?;
    m.add_class::<PyExtensionDType>()?;

    // Register factory functions.
    m.add_function(wrap_pyfunction!(factory::dtype_null, &m)?)?;
    m.add_function(wrap_pyfunction!(factory::dtype_bool, &m)?)?;
    m.add_function(wrap_pyfunction!(factory::dtype_int, &m)?)?;
    m.add_function(wrap_pyfunction!(factory::dtype_decimal, &m)?)?;
    m.add_function(wrap_pyfunction!(factory::dtype_uint, &m)?)?;
    m.add_function(wrap_pyfunction!(factory::dtype_float, &m)?)?;
    m.add_function(wrap_pyfunction!(factory::dtype_utf8, &m)?)?;
    m.add_function(wrap_pyfunction!(factory::dtype_binary, &m)?)?;
    m.add_function(wrap_pyfunction!(factory::dtype_struct, &m)?)?;
    m.add_function(wrap_pyfunction!(factory::dtype_list, &m)?)?;
    m.add_function(wrap_pyfunction!(factory::dtype_fixed_size_list, &m)?)?;
    m.add_function(wrap_pyfunction!(factory::dtype_date, &m)?)?;
    m.add_function(wrap_pyfunction!(factory::dtype_time, &m)?)?;
    m.add_function(wrap_pyfunction!(factory::dtype_timestamp, &m)?)?;

    Ok(())
}

/// Base class for all Vortex data types.
#[pyclass(
    name = "DType",
    module = "vortex",
    frozen,
    eq,
    hash,
    subclass,
    from_py_object
)]
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct PyDType(DType);

impl From<DType> for PyDType {
    fn from(dtype: DType) -> Self {
        Self(dtype)
    }
}

impl Deref for PyDType {
    type Target = DType;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl PyDType {
    /// Initialize a [`PyDType`] from a Vortex [`DType`], ensuring the correct subclass is
    /// returned.
    pub fn init(py: Python, dtype: DType) -> PyResult<Bound<PyDType>> {
        match dtype {
            DType::Null => Self::with_subclass(py, dtype, PyNullDType),
            DType::Bool(_) => Self::with_subclass(py, dtype, PyBoolDType),
            DType::Primitive(..) => Self::with_subclass(py, dtype, PyPrimitiveDType),
            DType::Decimal(..) => Self::with_subclass(py, dtype, PyDecimalDType),
            DType::Utf8(..) => Self::with_subclass(py, dtype, PyUtf8DType),
            DType::Binary(..) => Self::with_subclass(py, dtype, PyBinaryDType),
            DType::Struct(..) => Self::with_subclass(py, dtype, PyStructDType),
            DType::Union(..) => todo!("TODO(connor)[Union]: unimplemented"),
            DType::List(..) => Self::with_subclass(py, dtype, PyListDType),
            DType::FixedSizeList(..) => Self::with_subclass(py, dtype, PyFixedSizeListDType),
            DType::Extension(..) => Self::with_subclass(py, dtype, PyExtensionDType),
            DType::Variant(_) => Err(PyValueError::new_err(
                "Variant DType is not supported in Python yet",
            )),
        }
    }

    /// Initialize a [`PyDType`] from a Vortex [`DType`], with the given subclass.
    /// We keep this a private method to ensure we correctly match on the DType in init.
    fn with_subclass<S: PyClass<BaseType = PyDType>>(
        py: Python,
        dtype: DType,
        subclass: S,
    ) -> PyResult<Bound<PyDType>> {
        Ok(Bound::new(
            py,
            PyClassInitializer::from(PyDType(dtype)).add_subclass(subclass),
        )?
        .into_any()
        .cast_into::<PyDType>()?)
    }

    /// Return the inner [`DType`] value.
    pub fn inner(&self) -> &DType {
        &self.0
    }

    /// Return the inner [`DType`] value.
    pub fn into_inner(self) -> DType {
        self.0
    }
}

#[pymethods]
impl PyDType {
    fn to_arrow_type(&self, py: Python) -> PyVortexResult<Py<PyAny>> {
        Ok(self.0.to_arrow_dtype()?.to_pyarrow(py)?)
    }

    fn to_arrow_schema(&self, py: Python) -> PyVortexResult<Py<PyAny>> {
        Ok(self.0.to_arrow_schema()?.to_pyarrow(py)?)
    }

    fn __str__(&self) -> String {
        format!("{}", self.0)
    }

    fn __repr__(&self) -> String {
        self.0.python_repr().to_string()
    }

    /// Construct a Vortex data type from an Arrow data type.
    #[classmethod]
    #[pyo3(signature = (arrow_dtype, *, non_nullable = false))]
    fn from_arrow<'py>(
        cls: &'py Bound<'py, PyType>,
        #[pyo3(from_py_with = import_arrow_dtype)] arrow_dtype: DataType,
        non_nullable: bool,
    ) -> PyResult<Bound<'py, PyDType>> {
        Self::init(
            cls.py(),
            DType::from_arrow(&Field::new("_", arrow_dtype, !non_nullable)),
        )
    }
}

fn import_arrow_dtype(obj: &Bound<PyAny>) -> PyResult<DataType> {
    DataType::from_pyarrow(&obj.as_borrowed())
}
