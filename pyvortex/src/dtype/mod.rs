mod factory;

use arrow::datatypes::{DataType, Field};
use arrow::pyarrow::FromPyArrow;
use pyo3::prelude::{PyModule, PyModuleMethods};
use pyo3::types::PyType;
use pyo3::{pyclass, pymethods, wrap_pyfunction, Bound, Py, PyAny, PyResult, Python};
use vortex::arrow::FromArrowType;
use vortex::dtype::DType;

use crate::install_module;
use crate::python_repr::PythonRepr;

/// Register DType functions and classes.
pub(crate) fn init(py: Python, parent: &Bound<PyModule>) -> PyResult<()> {
    let m = PyModule::new_bound(py, "dtype")?;
    parent.add_submodule(&m)?;
    install_module("vortex._lib.dtype", &m)?;

    // Register the DType class.
    m.add_class::<PyDType>()?;

    // Register factory functions.
    m.add_function(wrap_pyfunction!(factory::dtype_null, &m)?)?;
    m.add_function(wrap_pyfunction!(factory::dtype_bool, &m)?)?;
    m.add_function(wrap_pyfunction!(factory::dtype_int, &m)?)?;
    m.add_function(wrap_pyfunction!(factory::dtype_uint, &m)?)?;
    m.add_function(wrap_pyfunction!(factory::dtype_float, &m)?)?;
    m.add_function(wrap_pyfunction!(factory::dtype_utf8, &m)?)?;
    m.add_function(wrap_pyfunction!(factory::dtype_binary, &m)?)?;
    m.add_function(wrap_pyfunction!(factory::dtype_struct, &m)?)?;

    Ok(())
}

/// Base class for all Vortex data types.
#[pyclass(name = "DType", module = "vortex")]
#[derive(Clone)]
pub struct PyDType {
    inner: DType,
}

impl PyDType {
    pub fn wrap(py: Python<'_>, inner: DType) -> PyResult<Py<Self>> {
        Py::new(py, Self { inner })
    }

    pub fn unwrap(&self) -> &DType {
        &self.inner
    }
}

#[pymethods]
impl PyDType {
    fn __str__(&self) -> String {
        format!("{}", self.inner)
    }

    fn __repr__(&self) -> String {
        self.inner.python_repr().to_string()
    }

    /// Construct a Vortex data type from an Arrow data type.
    #[classmethod]
    #[pyo3(signature = (arrow_dtype, *, non_nullable = false))]
    fn from_arrow(
        cls: &Bound<PyType>,
        #[pyo3(from_py_with = "import_arrow_dtype")] arrow_dtype: DataType,
        non_nullable: bool,
    ) -> PyResult<Py<Self>> {
        Self::wrap(
            cls.py(),
            DType::from_arrow(&Field::new("_", arrow_dtype, !non_nullable)),
        )
    }

    /// Return the names of the columns in a struct data type.
    // TODO(ngates): move this into StructDType class
    fn maybe_columns(&self) -> Option<Vec<String>> {
        match &self.inner {
            DType::Null => None,
            DType::Bool(_) => None,
            DType::Primitive(..) => None,
            DType::Utf8(_) => None,
            DType::Binary(_) => None,
            DType::Struct(child, _) => Some(child.names().iter().map(|x| x.to_string()).collect()),
            DType::List(..) => None,
            DType::Extension(..) => None,
        }
    }
}

fn import_arrow_dtype(obj: &Bound<PyAny>) -> PyResult<DataType> {
    DataType::from_pyarrow_bound(obj)
}
