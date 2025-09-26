// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::ops::Deref;

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::*;
use vortex::dtype::{DType, Nullability, PType};
use vortex::expr::{BinaryExpr, ExprRef, GetItemExpr, IntoExpr, NotExpr, Operator, and, lit};

use crate::dtype::PyDType;
use crate::install_module;
use crate::scalar::factory::scalar_helper;

pub(crate) fn init(py: Python, parent: &Bound<PyModule>) -> PyResult<()> {
    let m = PyModule::new(py, "expr")?;
    parent.add_submodule(&m)?;
    install_module("vortex._lib.expr", &m)?;

    m.add_function(wrap_pyfunction!(column, &m)?)?;
    m.add_function(wrap_pyfunction!(root, &m)?)?;
    m.add_function(wrap_pyfunction!(literal, &m)?)?;
    m.add_function(wrap_pyfunction!(not_, &m)?)?;
    m.add_function(wrap_pyfunction!(and_, &m)?)?;
    m.add_class::<PyExpr>()?;

    Ok(())
}

/// An expression describes how to filter rows when reading an array from a file.
///
/// .. seealso::
///    :func:`.column`
///
#[pyclass(name = "Expr", module = "vortex", frozen)]
#[derive(Clone)]
pub struct PyExpr {
    inner: ExprRef,
}

impl From<ExprRef> for PyExpr {
    fn from(value: ExprRef) -> Self {
        Self { inner: value }
    }
}

impl Deref for PyExpr {
    type Target = ExprRef;

    fn deref(&self) -> &Self::Target {
        &self.inner
    }
}

impl PyExpr {
    pub fn inner(&self) -> &ExprRef {
        &self.inner
    }

    pub fn into_inner(self) -> ExprRef {
        self.inner
    }
}

fn py_binary_operator<'py>(
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
    } else if let Ok(value) = value.downcast::<PyInt>() {
        scalar(DType::Primitive(PType::I64, nonnull), value)
    } else if let Ok(value) = value.downcast::<PyFloat>() {
        scalar(DType::Primitive(PType::F64, nonnull), value)
    } else if let Ok(value) = value.downcast::<PyString>() {
        scalar(DType::Utf8(nonnull), value)
    } else if let Ok(value) = value.downcast::<PyBytes>() {
        scalar(DType::Binary(nonnull), value)
    } else {
        Err(PyValueError::new_err(format!(
            "expected None, int, float, str, or bytes but found: {value}"
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
        py_binary_operator(self_, Operator::Eq, coerce_expr(right)?)
    }

    fn __ne__<'py>(
        self_: PyRef<'py, Self>,
        right: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyExpr>> {
        py_binary_operator(self_, Operator::NotEq, coerce_expr(right)?)
    }

    fn __gt__<'py>(
        self_: PyRef<'py, Self>,
        right: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyExpr>> {
        py_binary_operator(self_, Operator::Gt, coerce_expr(right)?)
    }

    fn __ge__<'py>(
        self_: PyRef<'py, Self>,
        right: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyExpr>> {
        py_binary_operator(self_, Operator::Gte, coerce_expr(right)?)
    }

    fn __lt__<'py>(
        self_: PyRef<'py, Self>,
        right: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyExpr>> {
        py_binary_operator(self_, Operator::Lt, coerce_expr(right)?)
    }

    fn __le__<'py>(
        self_: PyRef<'py, Self>,
        right: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyExpr>> {
        py_binary_operator(self_, Operator::Lte, coerce_expr(right)?)
    }

    fn __and__<'py>(
        self_: PyRef<'py, Self>,
        right: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyExpr>> {
        py_binary_operator(self_, Operator::And, coerce_expr(right)?)
    }

    fn __or__<'py>(
        self_: PyRef<'py, Self>,
        right: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyExpr>> {
        py_binary_operator(self_, Operator::Or, coerce_expr(right)?)
    }

    fn __getitem__(self_: PyRef<'_, Self>, field: String) -> PyResult<PyExpr> {
        get_item(field, self_.clone())
    }
}

/// Create an expression that represents a literal value.
///
/// Parameters
/// ----------
/// dtype : :class:`vortex.DType`
///     The data type of the literal value.
/// value : :class:`Any`
///     The literal value.
///
/// Returns
/// -------
/// :class:`vortex.Expr`
///
/// Examples
/// --------
///
/// ```python
/// >>> import vortex.expr as ve
/// >>> ve.literal(vx.int_(), 42)
/// <vortex.Expr object at ...>
/// ```
// TODO(ngates): make dtype optional, casting if necessary.
#[pyfunction]
pub fn literal<'py>(
    dtype: &Bound<'py, PyDType>,
    value: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyExpr>> {
    scalar(dtype.borrow().inner().clone(), value)
}

/// Create an expression that refers to the identity scope.
///
/// That is, it returns the full input that the extension is run against.
///
/// Returns
/// -------
/// :class:`vortex.Expr`
///
/// Examples
/// --------
///
/// ```python
/// >>> import vortex.expr as ve
/// >>> ve.root()
/// <vortex.Expr object at ...>
/// ```
#[pyfunction]
pub fn root() -> PyExpr {
    PyExpr {
        inner: vortex::expr::root(),
    }
}

/// Create an expression that refers to a column by its name.
///
/// Parameters
/// ----------
/// name : :class:`str`
///     The name of the column.
///
/// Returns
/// -------
/// :class:`vortex.Expr`
///
/// Examples
/// --------
///
/// ```python
/// >>> import vortex.expr as ve
/// >>> ve.column("age")
/// <vortex.Expr object at ...>
/// ```
#[pyfunction]
pub fn column<'py>(name: &Bound<'py, PyString>) -> PyResult<Bound<'py, PyExpr>> {
    let py = name.py();
    let name: String = name.extract()?;
    Bound::new(
        py,
        PyExpr {
            inner: vortex::expr::get_item(name, vortex::expr::root()),
        },
    )
}

pub fn scalar<'py>(dtype: DType, value: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyExpr>> {
    let py = value.py();
    Bound::new(
        py,
        PyExpr {
            inner: lit(scalar_helper(value, Some(&dtype))?),
        },
    )
}

pub fn get_item(field: String, child: PyExpr) -> PyResult<PyExpr> {
    Ok(PyExpr {
        inner: GetItemExpr::new(field, child.inner).into_expr(),
    })
}

/// Negate a Boolean expression.
///
/// Parameters
/// ----------
/// child : :class:`Any`
///     A boolean expression.
///
/// Returns
/// -------
/// :class:`vortex.Expr`
///
/// Examples
/// --------
///
/// ```python
/// >>> import vortex.expr as ve
/// >>> import vortex as vx
/// >>> ve.not_(ve.literal(vx.int_(), 42) == ve.literal(vx.int_(), 42))
/// <vortex.Expr object at ...>
/// ```
#[pyfunction]
pub fn not_(child: PyExpr) -> PyResult<PyExpr> {
    Ok(PyExpr {
        inner: NotExpr::new_expr(child.inner),
    })
}

/// True if both arguments are true.
///
/// Parameters
/// ----------
/// child : :class:`Any`
///     A boolean expression.
///
/// Returns
/// -------
/// :class:`vortex.Expr`
///
/// Examples
/// --------
///
/// ```python
/// >>> import vortex.expr as ve
/// >>> import vortex as vx
/// >>> ve.and_(ve.literal(vx.bool_(), True), ve.literal(vx.bool_(), True))
/// <vortex.Expr object at ...>
/// ```
#[pyfunction]
pub fn and_(left: PyExpr, right: PyExpr) -> PyResult<PyExpr> {
    Ok(PyExpr {
        inner: and(left.inner, right.inner),
    })
}
