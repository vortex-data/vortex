// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::sync::Arc;

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::*;
use vortex::dtype::DType;
use vortex::dtype::FieldName;
use vortex::dtype::Nullability;
use vortex::dtype::PType;
use vortex::error::VortexResult;
use vortex::expr;
use vortex::expr::BoundExpr;
use vortex::expr::lit;
use vortex::scalar::Scalar;
use vortex::scalar_fn::EmptyOptions;
use vortex::scalar_fn::ScalarFnVTableExt;
use vortex::scalar_fn::fns::binary::Binary;
use vortex::scalar_fn::fns::cast::Cast;
use vortex::scalar_fn::fns::get_item::GetItem;
use vortex::scalar_fn::fns::is_not_null::IsNotNull;
use vortex::scalar_fn::fns::is_null::IsNull;
use vortex::scalar_fn::fns::not::Not;
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
    m.add_function(wrap_pyfunction!(is_not_null, &m)?)?;
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
    inner: DeferredExpr,
}

impl From<DeferredExpr> for PyExpr {
    fn from(value: DeferredExpr) -> Self {
        Self { inner: value }
    }
}

impl PyExpr {
    pub(crate) fn inner(&self) -> &DeferredExpr {
        &self.inner
    }

    pub(crate) fn into_inner(self) -> DeferredExpr {
        self.inner
    }

    pub(crate) fn bind(&self, scope: &DType) -> VortexResult<BoundExpr> {
        self.inner.bind(scope)
    }
}

#[derive(Clone, Debug)]
pub(crate) enum DeferredExpr {
    Root,
    Literal(Scalar),
    GetItem {
        field: FieldName,
        child: Arc<DeferredExpr>,
    },
    Binary {
        operator: Operator,
        lhs: Arc<DeferredExpr>,
        rhs: Arc<DeferredExpr>,
    },
    Not(Arc<DeferredExpr>),
    Cast {
        child: Arc<DeferredExpr>,
        dtype: DType,
    },
    IsNull(Arc<DeferredExpr>),
    IsNotNull(Arc<DeferredExpr>),
}

impl DeferredExpr {
    pub(crate) fn bind(&self, scope: &DType) -> VortexResult<BoundExpr> {
        match self {
            Self::Root => Ok(expr::root(scope.clone())),
            Self::Literal(scalar) => Ok(lit(scalar.clone())),
            Self::GetItem { field, child } => {
                GetItem.try_new_expr(field.clone(), [child.bind(scope)?])
            }
            Self::Binary { operator, lhs, rhs } => {
                Binary.try_new_expr(*operator, [lhs.bind(scope)?, rhs.bind(scope)?])
            }
            Self::Not(child) => Not.try_new_expr(EmptyOptions, [child.bind(scope)?]),
            Self::Cast { child, dtype } => Cast.try_new_expr(dtype.clone(), [child.bind(scope)?]),
            Self::IsNull(child) => IsNull.try_new_expr(EmptyOptions, [child.bind(scope)?]),
            Self::IsNotNull(child) => IsNotNull.try_new_expr(EmptyOptions, [child.bind(scope)?]),
        }
    }
}

impl Drop for DeferredExpr {
    fn drop(&mut self) {
        let mut children_to_drop = Vec::new();
        self.take_children(&mut children_to_drop);

        while let Some(mut child) = children_to_drop.pop() {
            let Some(child) = Arc::get_mut(&mut child) else {
                continue;
            };
            child.take_children(&mut children_to_drop);
        }
    }
}

impl DeferredExpr {
    fn take_children(&mut self, children_to_drop: &mut Vec<Arc<Self>>) {
        match self {
            Self::Root | Self::Literal(_) => {}
            Self::GetItem { child, .. }
            | Self::Not(child)
            | Self::Cast { child, .. }
            | Self::IsNull(child)
            | Self::IsNotNull(child) => {
                children_to_drop.push(std::mem::replace(child, Self::drop_tombstone()));
            }
            Self::Binary { lhs, rhs, .. } => {
                children_to_drop.push(std::mem::replace(lhs, Self::drop_tombstone()));
                children_to_drop.push(std::mem::replace(rhs, Self::drop_tombstone()));
            }
        }
    }

    fn drop_tombstone() -> Arc<Self> {
        Arc::new(Self::Root)
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
            inner: DeferredExpr::Binary {
                operator,
                lhs: Arc::new(left.inner.clone()),
                rhs: Arc::new(right.borrow().inner.clone()),
            },
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
        inner: DeferredExpr::Root,
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
            inner: DeferredExpr::GetItem {
                field: FieldName::from(name),
                child: Arc::new(DeferredExpr::Root),
            },
        },
    )
}

pub fn scalar<'py>(dtype: DType, value: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyExpr>> {
    let py = value.py();
    Bound::new(
        py,
        PyExpr {
            inner: DeferredExpr::Literal(scalar_helper(value, Some(&dtype))?),
        },
    )
}

pub fn get_item(field: String, child: PyExpr) -> PyResult<PyExpr> {
    Ok(PyExpr {
        inner: DeferredExpr::GetItem {
            field: field.into(),
            child: Arc::new(child.inner),
        },
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
        inner: DeferredExpr::Not(Arc::new(child.inner)),
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
        inner: DeferredExpr::Binary {
            operator: Operator::And,
            lhs: Arc::new(left.inner),
            rhs: Arc::new(right.inner),
        },
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
        inner: DeferredExpr::Cast {
            child: Arc::new(child.into_inner()),
            dtype: dtype.into_inner(),
        },
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
        inner: DeferredExpr::IsNull(Arc::new(child.into_inner())),
    })
}

/// Creates an expression that checks for non-null values.
///
/// Parameters
/// ----------
/// child : :class:`vortex.Expr`
///
/// Returns
/// -------
/// :class:`vortex.Expr`
#[pyfunction]
pub fn is_not_null(child: PyExpr) -> PyResult<PyExpr> {
    Ok(PyExpr {
        inner: DeferredExpr::IsNotNull(Arc::new(child.into_inner())),
    })
}
