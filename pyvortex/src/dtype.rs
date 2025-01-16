use std::sync::Arc;

use arrow::datatypes::{DataType, Field};
use arrow::pyarrow::FromPyArrow;
use pyo3::exceptions::PyValueError;
use pyo3::types::PyType;
use pyo3::{pyclass, pyfunction, pymethods, Bound, Py, PyAny, PyResult, Python};
use vortex::arrow::FromArrowType;
use vortex::dtype::dtypes::*;
use vortex::dtype::DType;

use crate::python_repr::PythonRepr;

#[pyclass(name = "DType", module = "vortex", subclass)]
/// A data type describes the set of operations available on a given column. These operations are
/// implemented by the column *encoding*. Each data type is implemented by one or more encodings.
pub struct PyDType {
    inner: Arc<DType>,
}

impl PyDType {
    pub fn wrap(py: Python<'_>, inner: Arc<DType>) -> PyResult<Py<Self>> {
        Py::new(py, Self { inner })
    }

    pub fn unwrap(&self) -> &Arc<DType> {
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

    #[classmethod]
    fn from_arrow(
        cls: &Bound<PyType>,
        #[pyo3(from_py_with = "import_arrow_dtype")] arrow_dtype: DataType,
        nullable: bool,
    ) -> PyResult<Py<Self>> {
        Self::wrap(
            cls.py(),
            <Arc<DType>>::from_arrow(&Field::new("_", arrow_dtype, nullable)),
        )
    }

    fn maybe_columns(&self) -> Option<Vec<String>> {
        match self.inner.as_ref() {
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

#[pyfunction(name = "null")]
#[pyo3(signature = ())]
/// Construct the data type for a column containing only the null value.
///
/// Returns
/// -------
/// :class:`vortex.dtype.DType`
///
/// Examples
/// --------
///
/// A data type permitting only :obj:`None`.
///
///     >>> vortex.dtype.null()
///     null()
pub fn dtype_null(py: Python<'_>) -> PyResult<Py<PyDType>> {
    PyDType::wrap(py, DTYPE_NULL.clone())
}

#[pyfunction(name = "bool")]
#[pyo3(signature = (nullable = false))]
/// Construct a Boolean data type.
///
/// Parameters
/// ----------
/// nullable : :class:`bool`
///     When :obj:`True`, :obj:`None` is a permissible value.
///
/// Returns
/// -------
/// :class:`vortex.dtype.DType`
///
/// Examples
/// --------
///
/// A data type permitting :obj:`None`, :obj:`True`, and :obj:`False`.
///
///     >>> vortex.dtype.bool(True)
///     bool(True)
///
/// A data type permitting just :obj:`True` and :obj:`False`.
///
///     >>> vortex.dtype.bool(False)
///     bool(False)
pub fn dtype_bool(py: Python<'_>, nullable: bool) -> PyResult<Py<PyDType>> {
    PyDType::wrap(
        py,
        match nullable {
            true => DTYPE_BOOL_NULL.clone(),
            false => DTYPE_BOOL_NONNULL.clone(),
        },
    )
}

#[pyfunction(name = "int")]
#[pyo3(signature = (width = None, nullable = false))]
/// Construct a signed integral data type.
///
/// Parameters
/// ----------
/// width : Literal[8, 16, 32, 64].
///     The bit width determines the span of valid values. If :obj:`None`, 64 is used.
///
/// nullable : :class:`bool`
///     When :obj:`True`, :obj:`None` is a permissible value.
///
/// Returns
/// -------
/// :class:`vortex.dtype.DType`
///
/// Examples
/// --------
///
/// A data type permitting :obj:`None` and the integers from -128 to 127, inclusive:
///
///     >>> vortex.dtype.int(8, True)
///     int(8, True)
///
/// A data type permitting just the integers from -2,147,483,648 to 2,147,483,647, inclusive:
///
///     >>> vortex.dtype.int(32, False)
///     int(32, False)
pub fn dtype_int(py: Python<'_>, width: Option<u16>, nullable: bool) -> PyResult<Py<PyDType>> {
    let dtype = if let Some(width) = width {
        match (width, nullable) {
            (8, false) => DTYPE_I8_NONNULL.clone(),
            (8, true) => DTYPE_I8_NULL.clone(),
            (16, false) => DTYPE_I16_NONNULL.clone(),
            (16, true) => DTYPE_I16_NULL.clone(),
            (32, false) => DTYPE_I32_NONNULL.clone(),
            (32, true) => DTYPE_I32_NULL.clone(),
            (64, false) => DTYPE_I64_NONNULL.clone(),
            (64, true) => DTYPE_I64_NULL.clone(),
            _ => return Err(PyValueError::new_err("Invalid int width")),
        }
    } else {
        DTYPE_I64_NONNULL.clone()
    };
    PyDType::wrap(py, dtype)
}

#[pyfunction(name = "uint")]
#[pyo3(signature = (width = None, nullable = false))]
/// Construct an unsigned integral data type.
///
/// Parameters
/// ----------
/// width : Literal[8, 16, 32, 64].
///     The bit width determines the span of valid values. If :obj:`None`, 64 is used.
///
/// nullable : :class:`bool`
///     When :obj:`True`, :obj:`None` is a permissible value.
///
/// Returns
/// -------
/// :class:`vortex.dtype.DType`
///
/// Examples
/// --------
///
/// A data type permitting :obj:`None` and the integers from 0 to 255, inclusive:
///
///     >>> vortex.dtype.uint(8, True)
///     uint(8, True)
///
/// A data type permitting just the integers from 0 to 4,294,967,296 inclusive:
///
///     >>> vortex.dtype.uint(32, False)
///     uint(32, False)
pub fn dtype_uint(py: Python<'_>, width: Option<u16>, nullable: bool) -> PyResult<Py<PyDType>> {
    let dtype = if let Some(width) = width {
        match (width, nullable) {
            (8, false) => DTYPE_U8_NONNULL.clone(),
            (8, true) => DTYPE_U8_NULL.clone(),
            (16, false) => DTYPE_U16_NONNULL.clone(),
            (16, true) => DTYPE_U16_NULL.clone(),
            (32, false) => DTYPE_U32_NONNULL.clone(),
            (32, true) => DTYPE_U32_NULL.clone(),
            (64, false) => DTYPE_U64_NONNULL.clone(),
            (64, true) => DTYPE_U64_NULL.clone(),
            _ => return Err(PyValueError::new_err("Invalid uint width")),
        }
    } else {
        DTYPE_U64_NONNULL.clone()
    };
    PyDType::wrap(py, dtype)
}

#[pyfunction(name = "float")]
#[pyo3(signature = (width = None, nullable = false))]
/// Construct an IEEE 754 binary floating-point data type.
///
/// Parameters
/// ----------
/// width : Literal[16, 32, 64].
///     The bit width determines the range and precision of the floating-point values. If
///     :obj:`None`, 64 is used.
///
/// nullable : :class:`bool`
///     When :obj:`True`, :obj:`None` is a permissible value.
///
/// Returns
/// -------
/// :class:`vortex.dtype.DType`
///
/// Examples
/// --------
///
/// A data type permitting :obj:`None` as well as IEEE 754 binary16 floating-point values. Values
/// larger than 65,520 or less than -65,520 will respectively round to positive and negative
/// infinity.
///
///     >>> vortex.dtype.float(16, False)
///     float(16, False)
pub fn dtype_float(py: Python<'_>, width: Option<i8>, nullable: bool) -> PyResult<Py<PyDType>> {
    let dtype = if let Some(width) = width {
        match (width, nullable) {
            (16, false) => DTYPE_F16_NONNULL.clone(),
            (16, true) => DTYPE_F16_NULL.clone(),
            (32, false) => DTYPE_F32_NONNULL.clone(),
            (32, true) => DTYPE_F32_NULL.clone(),
            (64, false) => DTYPE_F64_NONNULL.clone(),
            (64, true) => DTYPE_F64_NULL.clone(),
            _ => return Err(PyValueError::new_err("Invalid float width")),
        }
    } else {
        DTYPE_F64_NONNULL.clone()
    };
    PyDType::wrap(py, dtype)
}

#[pyfunction(name = "utf8")]
#[pyo3(signature = (nullable = false))]
/// Construct a UTF-8-encoded string data type.
///
/// Parameters
/// ----------
/// nullable : :class:`bool`
///     When :obj:`True`, :obj:`None` is a permissible value.
///
/// Returns
/// -------
/// :class:`vortex.dtype.DType`
///
/// Examples
/// --------
///
/// A data type permitting any UTF-8-encoded string, such as :code:`"Hello World"`, but not
/// permitting :obj:`None`.
///
///     >>> vortex.dtype.utf8(False)
///     utf8(False)
pub fn dtype_utf8(py: Python<'_>, nullable: bool) -> PyResult<Py<PyDType>> {
    PyDType::wrap(
        py,
        match nullable {
            true => DTYPE_STRING_NULL.clone(),
            false => DTYPE_STRING_NONNULL.clone(),
        },
    )
}

#[pyfunction(name = "binary")]
#[pyo3(signature = (nullable = false))]
/// Construct a data type for binary strings.
///
/// Parameters
/// ----------
/// nullable : :class:`bool`
///     When :obj:`True`, :obj:`None` is a permissible value.
///
/// Returns
/// -------
/// :class:`vortex.dtype.DType`
///
/// Examples
/// --------
///
/// A data type permitting any string of bytes but not permitting :obj:`None`.
///
///     >>> vortex.dtype.binary(False)
///     binary(False)
pub fn dtype_binary(py: Python<'_>, nullable: bool) -> PyResult<Py<PyDType>> {
    PyDType::wrap(
        py,
        match nullable {
            true => DTYPE_BINARY_NULL.clone(),
            false => DTYPE_BINARY_NONNULL.clone(),
        },
    )
}
