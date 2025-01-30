use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::*;
use vortex::dtype::half::f16;
use vortex::dtype::{DType, Nullability, PType};
use vortex::expr::{lit, BinaryExpr, ExprRef, GetItem, Operator};
use vortex::scalar::Scalar;

use crate::dtype::PyDType;
use crate::install_module;

pub(crate) fn init(py: Python, parent: &Bound<PyModule>) -> PyResult<()> {
    let m = PyModule::new_bound(py, "expr")?;
    parent.add_submodule(&m)?;
    install_module("vortex._lib.expr", &m)?;

    m.add_function(wrap_pyfunction!(column, &m)?)?;
    m.add_function(wrap_pyfunction!(ident, &m)?)?;
    m.add_function(wrap_pyfunction!(literal, &m)?)?;
    m.add_class::<PyExpr>()?;

    Ok(())
}

#[pyclass(name = "Expr", module = "vortex")]
#[derive(Clone)]
pub struct PyExpr {
    inner: ExprRef,
}

impl PyExpr {
    pub fn unwrap(&self) -> &ExprRef {
        &self.inner
    }
}

fn py_binary_opeartor<'py>(
    left: PyRef<'py, PyExpr>,
    operator: Operator,
    right: Bound<'py, PyExpr>,
) -> PyResult<Bound<'py, PyExpr>> {
    Bound::new(
        left.py(),
        PyExpr {
            inner: BinaryExpr::new_expr(left.inner.clone(), operator, right.borrow().inner.clone()),
        },
    )
}

fn coerce_expr<'py>(value: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyExpr>> {
    let nonnull = Nullability::NonNullable;
    if let Ok(value) = value.downcast::<PyExpr>() {
        Ok(value.clone())
    } else if let Ok(value) = value.downcast::<PyNone>() {
        scalar(DType::Null, value)
    } else if let Ok(value) = value.downcast::<PyLong>() {
        scalar(DType::Primitive(PType::I64, nonnull), value)
    } else if let Ok(value) = value.downcast::<PyFloat>() {
        scalar(DType::Primitive(PType::F64, nonnull), value)
    } else if let Ok(value) = value.downcast::<PyString>() {
        scalar(DType::Utf8(nonnull), value)
    } else if let Ok(value) = value.downcast::<PyBytes>() {
        scalar(DType::Binary(nonnull), value)
    } else {
        Err(PyValueError::new_err(format!(
            "expected None, int, float, str, or bytes but found: {}",
            value
        )))
    }
}

#[pymethods]
impl PyExpr {
    pub fn __str__(&self) -> String {
        format!("{:?}", self.inner)
    }

    fn __eq__<'py>(
        self_: PyRef<'py, Self>,
        right: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyExpr>> {
        py_binary_opeartor(self_, Operator::Eq, coerce_expr(right)?)
    }

    fn __ne__<'py>(
        self_: PyRef<'py, Self>,
        right: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyExpr>> {
        py_binary_opeartor(self_, Operator::NotEq, coerce_expr(right)?)
    }

    fn __gt__<'py>(
        self_: PyRef<'py, Self>,
        right: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyExpr>> {
        py_binary_opeartor(self_, Operator::Gt, coerce_expr(right)?)
    }

    fn __ge__<'py>(
        self_: PyRef<'py, Self>,
        right: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyExpr>> {
        py_binary_opeartor(self_, Operator::Gte, coerce_expr(right)?)
    }

    fn __lt__<'py>(
        self_: PyRef<'py, Self>,
        right: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyExpr>> {
        py_binary_opeartor(self_, Operator::Lt, coerce_expr(right)?)
    }

    fn __le__<'py>(
        self_: PyRef<'py, Self>,
        right: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyExpr>> {
        py_binary_opeartor(self_, Operator::Lte, coerce_expr(right)?)
    }

    fn __and__<'py>(
        self_: PyRef<'py, Self>,
        right: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyExpr>> {
        py_binary_opeartor(self_, Operator::And, coerce_expr(right)?)
    }

    fn __or__<'py>(
        self_: PyRef<'py, Self>,
        right: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyExpr>> {
        py_binary_opeartor(self_, Operator::Or, coerce_expr(right)?)
    }

    fn __getitem__(self_: PyRef<'_, Self>, field: String) -> PyResult<PyExpr> {
        get_item(field, self_.clone())
    }
}

// TODO(ngates): make dtype optional, casting if necessary.
#[pyfunction]
pub fn literal<'py>(
    dtype: &Bound<'py, PyDType>,
    value: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyExpr>> {
    scalar(dtype.borrow().unwrap().clone(), value)
}

#[pyfunction]
pub fn ident() -> PyExpr {
    PyExpr {
        inner: vortex::expr::ident(),
    }
}

#[pyfunction]
pub fn column<'py>(name: &Bound<'py, PyString>) -> PyResult<Bound<'py, PyExpr>> {
    let py = name.py();
    let name: String = name.extract()?;
    Bound::new(
        py,
        PyExpr {
            inner: vortex::expr::get_item(name, vortex::expr::ident()),
        },
    )
}

pub fn scalar<'py>(dtype: DType, value: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyExpr>> {
    let py = value.py();
    Bound::new(
        py,
        PyExpr {
            inner: lit(scalar_helper(dtype, value)?),
        },
    )
}

pub fn scalar_helper(dtype: DType, value: &Bound<'_, PyAny>) -> PyResult<Scalar> {
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
                .map(|element| scalar_helper(element_type.as_ref().clone(), element))
                .collect::<PyResult<Vec<_>>>()?;
            Ok(Scalar::list(element_type, values, Nullability::Nullable))
        }
        DType::Extension(..) => todo!(),
    }
}

pub fn get_item(field: String, child: PyExpr) -> PyResult<PyExpr> {
    Ok(PyExpr {
        inner: GetItem::new_expr(field, child.inner),
    })
}
