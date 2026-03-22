// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::ops::Deref;

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::*;
use vortex::dtype::DType;
use vortex::dtype::Nullability;
use vortex::dtype::PType;
use vortex::expr;
use vortex::expr::Expression;
use vortex::expr::and;
use vortex::expr::lit;
use vortex::expr::not;
use vortex::scalar_fn::ScalarFnVTableExt;
use vortex::scalar_fn::fns::binary::Binary;
use vortex::scalar_fn::fns::get_item::GetItem;
use vortex::scalar_fn::fns::operators::Operator;

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
    m.add_function(wrap_pyfunction!(cast, &m)?)?;
    m.add_function(wrap_pyfunction!(is_null, &m)?)?;
    m.add_class::<PyExpr>()?;

    Ok(())
}

/// An expression describes how to filter rows when reading an array from a file.
///
/// .. seealso::
///    :func:`.column`
///
#[pyclass(name = "Expr", module = "vortex", frozen, from_py_object)]
#[derive(Clone)]
pub struct PyExpr {
    inner: Expression,
}

impl From<Expression> for PyExpr {
    fn from(value: Expression) -> Self {
        Self { inner: value }
    }
}

impl Deref for PyExpr {
    type Target = Expression;

    fn deref(&self) -> &Self::Target {
        &self.inner
    }
}

impl PyExpr {
    pub fn inner(&self) -> &Expression {
        &self.inner
    }

    pub fn into_inner(self) -> Expression {
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
            inner: Binary.new_expr(operator, [left.inner.clone(), right.borrow().inner.clone()]),
        },
    )
}

fn coerce_expr<'py>(value: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyExpr>> {
    let nonnull = Nullability::NonNullable;
    if let Ok(value) = value.cast::<PyExpr>() {
        Ok(value.clone())
    } else if let Ok(value) = value.cast::<PyNone>() {
        scalar(DType::Null, value)
    } else if let Ok(value) = value.cast::<PyInt>() {
        scalar(DType::Primitive(PType::I64, nonnull), value)
    } else if let Ok(value) = value.cast::<PyFloat>() {
        scalar(DType::Primitive(PType::F64, nonnull), value)
    } else if let Ok(value) = value.cast::<PyString>() {
        scalar(DType::Utf8(nonnull), value)
    } else if let Ok(value) = value.cast::<PyBytes>() {
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

    fn __add__<'py>(
        self_: PyRef<'py, Self>,
        right: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyExpr>> {
        py_binary_operator(self_, Operator::Add, coerce_expr(right)?)
    }

    fn __sub__<'py>(
        self_: PyRef<'py, Self>,
        right: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyExpr>> {
        py_binary_operator(self_, Operator::Sub, coerce_expr(right)?)
    }

    fn __mul__<'py>(
        self_: PyRef<'py, Self>,
        right: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyExpr>> {
        py_binary_operator(self_, Operator::Mul, coerce_expr(right)?)
    }

    fn __truediv__<'py>(
        self_: PyRef<'py, Self>,
        right: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyExpr>> {
        py_binary_operator(self_, Operator::Div, coerce_expr(right)?)
    }

    // Special methods docstrings cannot be defined in Rust. Write a docstring in the corresponding
    // rST file. https://github.com/PyO3/pyo3/issues/4326
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
        inner: expr::root(),
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
///
/// .. seealso::
///
///    Use :meth:`.vortex.expr.Expr.__getitem__` to retrieve a field of a struct array.
#[pyfunction]
pub fn column<'py>(name: &Bound<'py, PyString>) -> PyResult<Bound<'py, PyExpr>> {
    let py = name.py();
    let name: String = name.extract()?;
    Bound::new(
        py,
        PyExpr {
            inner: expr::get_item(name, expr::root()),
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
        inner: GetItem.new_expr(field.into(), [child.inner]),
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
        inner: not(child.inner),
    })
}

/// True if both arguments are true.
///
/// Parameters
/// ----------
/// left : :class:`Expr`
///     A boolean expression.
///
/// right : :class:`Expr`
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

/// Cast an expression to a compatible type.
///
/// Parameters
/// ----------
/// child : :class:`Expr`
///     The expression to cast.
///
/// Returns
/// -------
/// :class:`vortex.Expr`
///
/// Examples
/// --------
///
/// Cast to a wider integer type:
///
/// ```python
/// >>> import vortex.expr as ve
/// >>> import vortex as vx
/// >>> ve.cast(ve.literal(vx.int_(8), 1), vx.int_(16))
/// <vortex.Expr object at ...>
/// ```
///
/// Cast to a wider floating-point type:
///
/// ```python
/// >>> import vortex.expr as ve
/// >>> import vortex as vx
/// >>> ve.cast(ve.literal(vx.float_(16), 3.145), vx.float_(64))
/// <vortex.Expr object at ...>
/// ```
#[pyfunction]
pub fn cast(child: PyExpr, dtype: PyDType) -> PyResult<PyExpr> {
    Ok(PyExpr {
        inner: expr::cast(child.into_inner(), dtype.into_inner()),
    })
}

/// Checks which elements of its child are null.
///
/// Parameters
/// ----------
/// child : :class:`Expr`
///     Any expression.
///
/// Returns
/// -------
/// :class:`vortex.Expr`
/// ```
#[pyfunction]
pub fn is_null(child: PyExpr) -> PyResult<PyExpr> {
    Ok(PyExpr {
        inner: expr::is_null(child.into_inner()),
    })
}
