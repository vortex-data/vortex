mod binary;
mod bool;
mod extension;
mod factory;
mod list;
mod null;
mod primitive;
mod struct_;
mod utf8;

use std::ops::Deref;

use arrow::datatypes::{DataType, Field};
use arrow::pyarrow::FromPyArrow;
use pyo3::prelude::{PyAnyMethods, PyModule, PyModuleMethods};
use pyo3::types::PyType;
use pyo3::{
    pyclass, pymethods, wrap_pyfunction, Bound, PyAny, PyClass, PyClassInitializer, PyResult,
    Python,
};
use vortex::arrow::FromArrowType;
use vortex::dtype::DType;

use crate::dtype::binary::PyBinaryDType;
use crate::dtype::bool::PyBoolDType;
use crate::dtype::extension::PyExtensionDType;
use crate::dtype::list::PyListDType;
use crate::dtype::null::PyNullDType;
use crate::dtype::primitive::PyPrimitiveDType;
use crate::dtype::struct_::PyStructDType;
use crate::dtype::utf8::PyUtf8DType;
use crate::install_module;
use crate::python_repr::PythonRepr;

/// Register DType functions and classes.
pub(crate) fn init(py: Python, parent: &Bound<PyModule>) -> PyResult<()> {
    let m = PyModule::new(py, "dtype")?;
    parent.add_submodule(&m)?;
    install_module("vortex._lib.dtype", &m)?;

    // Register the DType class.
    m.add_class::<PyDType>()?;
    m.add_class::<PyNullDType>()?;
    m.add_class::<PyBoolDType>()?;
    m.add_class::<PyPrimitiveDType>()?;
    m.add_class::<PyUtf8DType>()?;
    m.add_class::<PyBinaryDType>()?;
    m.add_class::<PyStructDType>()?;
    m.add_class::<PyListDType>()?;
    m.add_class::<PyExtensionDType>()?;

    // Register factory functions.
    m.add_function(wrap_pyfunction!(factory::dtype_null, &m)?)?;
    m.add_function(wrap_pyfunction!(factory::dtype_bool, &m)?)?;
    m.add_function(wrap_pyfunction!(factory::dtype_int, &m)?)?;
    m.add_function(wrap_pyfunction!(factory::dtype_uint, &m)?)?;
    m.add_function(wrap_pyfunction!(factory::dtype_float, &m)?)?;
    m.add_function(wrap_pyfunction!(factory::dtype_utf8, &m)?)?;
    m.add_function(wrap_pyfunction!(factory::dtype_binary, &m)?)?;
    m.add_function(wrap_pyfunction!(factory::dtype_struct, &m)?)?;
    m.add_function(wrap_pyfunction!(factory::dtype_list, &m)?)?;

    Ok(())
}

/// Base class for all Vortex data types.
#[pyclass(name = "DType", module = "vortex", frozen, eq, hash, subclass)]
#[derive(Clone, PartialEq, Eq, Hash)]
pub struct PyDType(DType);

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
            DType::Utf8(..) => Self::with_subclass(py, dtype, PyUtf8DType),
            DType::Binary(..) => Self::with_subclass(py, dtype, PyBinaryDType),
            DType::Struct(..) => Self::with_subclass(py, dtype, PyStructDType),
            DType::List(..) => Self::with_subclass(py, dtype, PyListDType),
            DType::Extension(..) => Self::with_subclass(py, dtype, PyExtensionDType),
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
        .downcast_into::<PyDType>()?)
    }

    /// Return the inner [`DType`] value.
    pub fn inner(&self) -> &DType {
        &self.0
    }

    /// Return the inner [`DType`] value.
    #[allow(dead_code)]
    pub fn into_inner(self) -> DType {
        self.0
    }
}

#[pymethods]
impl PyDType {
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
        #[pyo3(from_py_with = "import_arrow_dtype")] arrow_dtype: DataType,
        non_nullable: bool,
    ) -> PyResult<Bound<'py, PyDType>> {
        Self::init(
            cls.py(),
            DType::from_arrow(&Field::new("_", arrow_dtype, !non_nullable)),
        )
    }
}

fn import_arrow_dtype(obj: &Bound<PyAny>) -> PyResult<DataType> {
    DataType::from_pyarrow_bound(obj)
}
