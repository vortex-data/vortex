// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::ops::Deref;

use prost::Message;
use pyo3::exceptions::PyTypeError;
use pyo3::exceptions::PyValueError;
use pyo3::intern;
use pyo3::prelude::*;
use pyo3::types::*;
use vortex::aggregate_fn::NumericalAggregateOpts;
use vortex::dtype::DType;
use vortex::dtype::FieldName;
use vortex::dtype::FieldNames;
use vortex::dtype::Nullability;
use vortex::expr;
use vortex::expr::Expression;
use vortex::expr::lit;
use vortex::expr::proto::ExprSerializeProtoExt;
use vortex::proto::expr as pb;
use vortex::scalar_fn::ScalarFnVTableExt;
use vortex::scalar_fn::fns::between::BetweenOptions;
use vortex::scalar_fn::fns::between::StrictComparison;
use vortex::scalar_fn::fns::binary::Binary;
use vortex::scalar_fn::fns::merge::DuplicateHandling;
use vortex::scalar_fn::fns::operators::Operator;
use vortex::scalar_fn::fns::variant_get::VariantPath;
use vortex::scalar_fn::fns::variant_get::VariantPathElement;

use crate::dtype::PyDType;
use crate::error::PyVortexResult;
use crate::install_module;
use crate::scalar::factory::scalar_helper;
use crate::session::session;

pub(crate) fn init(py: Python, parent: &Bound<PyModule>) -> PyResult<()> {
    let m = PyModule::new(py, "expr")?;
    parent.add_submodule(&m)?;
    install_module("vortex._lib.expr", &m)?;

    // Leaves and scope
    m.add_function(wrap_pyfunction!(root, &m)?)?;
    m.add_function(wrap_pyfunction!(column, &m)?)?;
    m.add_function(wrap_pyfunction!(literal, &m)?)?;
    m.add_function(wrap_pyfunction!(get_item, &m)?)?;

    // Boolean logic
    m.add_function(wrap_pyfunction!(not_, &m)?)?;
    m.add_function(wrap_pyfunction!(and_, &m)?)?;
    m.add_function(wrap_pyfunction!(or_, &m)?)?;
    m.add_function(wrap_pyfunction!(and_collect, &m)?)?;
    m.add_function(wrap_pyfunction!(or_collect, &m)?)?;

    // Comparisons and arithmetic
    m.add_function(wrap_pyfunction!(eq, &m)?)?;
    m.add_function(wrap_pyfunction!(not_eq, &m)?)?;
    m.add_function(wrap_pyfunction!(gt, &m)?)?;
    m.add_function(wrap_pyfunction!(gt_eq, &m)?)?;
    m.add_function(wrap_pyfunction!(lt, &m)?)?;
    m.add_function(wrap_pyfunction!(lt_eq, &m)?)?;
    m.add_function(wrap_pyfunction!(add, &m)?)?;
    m.add_function(wrap_pyfunction!(sub, &m)?)?;
    m.add_function(wrap_pyfunction!(mul, &m)?)?;
    m.add_function(wrap_pyfunction!(div, &m)?)?;
    m.add_function(wrap_pyfunction!(between, &m)?)?;

    // Nullability
    m.add_function(wrap_pyfunction!(is_null, &m)?)?;
    m.add_function(wrap_pyfunction!(is_not_null, &m)?)?;
    m.add_function(wrap_pyfunction!(fill_null, &m)?)?;

    // Strings
    m.add_function(wrap_pyfunction!(like, &m)?)?;
    m.add_function(wrap_pyfunction!(ilike, &m)?)?;
    m.add_function(wrap_pyfunction!(not_like, &m)?)?;
    m.add_function(wrap_pyfunction!(not_ilike, &m)?)?;
    m.add_function(wrap_pyfunction!(byte_length, &m)?)?;

    // Structs
    m.add_function(wrap_pyfunction!(select, &m)?)?;
    m.add_function(wrap_pyfunction!(select_exclude, &m)?)?;
    m.add_function(wrap_pyfunction!(pack, &m)?)?;
    m.add_function(wrap_pyfunction!(merge, &m)?)?;

    // Lists
    m.add_function(wrap_pyfunction!(list_contains, &m)?)?;
    m.add_function(wrap_pyfunction!(list_length, &m)?)?;
    m.add_function(wrap_pyfunction!(list_sum, &m)?)?;

    // Conditionals and misc
    m.add_function(wrap_pyfunction!(case_when, &m)?)?;
    m.add_function(wrap_pyfunction!(zip_, &m)?)?;
    m.add_function(wrap_pyfunction!(mask, &m)?)?;
    m.add_function(wrap_pyfunction!(cast, &m)?)?;
    m.add_function(wrap_pyfunction!(ext_storage, &m)?)?;
    m.add_function(wrap_pyfunction!(variant_get, &m)?)?;

    // Serialization
    m.add_function(wrap_pyfunction!(deserialize, &m)?)?;

    m.add_class::<PyExpr>()?;

    Ok(())
}

/// An expression describes how to filter rows when reading an array from a file.
///
/// .. seealso::
///    :func:`.column`
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

/// A Python value that can be coerced into an [`Expression`].
///
/// Accepts an existing [`PyExpr`], or any Python value convertible to a Vortex scalar (including
/// `None`, `bool`, `int`, `float`, `str`, `bytes`, `list`, `dict`, and `vortex.Scalar`), which is
/// wrapped in a literal expression.
pub struct PyIntoExpr(Expression);

impl PyIntoExpr {
    pub fn into_inner(self) -> Expression {
        self.0
    }
}

impl<'py> FromPyObject<'_, 'py> for PyIntoExpr {
    type Error = PyErr;

    fn extract(ob: Borrowed<'_, 'py, PyAny>) -> Result<Self, Self::Error> {
        coerce_expression(&ob).map(PyIntoExpr)
    }
}

/// Coerce an arbitrary Python object into an [`Expression`].
fn coerce_expression(value: &Bound<'_, PyAny>) -> PyResult<Expression> {
    if let Ok(value) = value.cast::<PyExpr>() {
        return Ok(value.get().inner.clone());
    }
    Ok(lit(scalar_helper(value, None).map_err(PyErr::from)?))
}

fn py_binary_operator<'py>(
    left: PyRef<'py, PyExpr>,
    operator: Operator,
    right: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyExpr>> {
    let right = coerce_expression(right)?;
    Bound::new(
        left.py(),
        PyExpr {
            inner: Binary.new_expr(operator, [left.inner.clone(), right]),
        },
    )
}

/// The reflected form of [`py_binary_operator`], used for `<literal> <op> <expr>`.
fn py_reflected_operator<'py>(
    right: PyRef<'py, PyExpr>,
    operator: Operator,
    left: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyExpr>> {
    let left = coerce_expression(left)?;
    Bound::new(
        right.py(),
        PyExpr {
            inner: Binary.new_expr(operator, [left, right.inner.clone()]),
        },
    )
}

fn field_names(fields: &Bound<'_, PyAny>) -> PyResult<FieldNames> {
    if let Ok(name) = fields.cast::<PyString>() {
        return Ok(FieldNames::from(vec![FieldName::from(
            name.extract::<String>()?,
        )]));
    }
    Ok(fields
        .try_iter()?
        .map(|field| field?.extract::<String>().map(FieldName::from))
        .collect::<PyResult<Vec<FieldName>>>()?
        .into())
}

/// Extract expressions from any iterable, so that a generator serves as well as a sequence.
///
/// A `Vec<PyIntoExpr>` parameter would reject anything that is not a sequence, and these functions
/// are documented to take an iterable — which is what a caller building expressions in a
/// comprehension naturally has.
fn into_exprs(exprs: &Bound<'_, PyAny>) -> PyResult<Vec<Expression>> {
    exprs
        .try_iter()?
        .map(|expr| coerce_expression(&expr?))
        .collect()
}

/// Extract `(name, expression)` pairs from either a mapping or an iterable of 2-tuples.
fn named_exprs(fields: &Bound<'_, PyAny>) -> PyResult<Vec<(FieldName, Expression)>> {
    let items: Vec<Bound<'_, PyAny>> = if let Ok(dict) = fields.cast::<PyDict>() {
        dict.items().iter().collect()
    } else {
        fields.try_iter()?.collect::<PyResult<Vec<_>>>()?
    };

    items
        .into_iter()
        .map(|item| {
            let (name, value): (String, Bound<'_, PyAny>) = item.extract()?;
            Ok((FieldName::from(name), coerce_expression(&value)?))
        })
        .collect()
}

fn nullability(nullable: bool) -> Nullability {
    if nullable {
        Nullability::Nullable
    } else {
        Nullability::NonNullable
    }
}

fn strictness(strict: bool) -> StrictComparison {
    if strict {
        StrictComparison::Strict
    } else {
        StrictComparison::NonStrict
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
        py_binary_operator(self_, Operator::Eq, right)
    }

    fn __ne__<'py>(
        self_: PyRef<'py, Self>,
        right: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyExpr>> {
        py_binary_operator(self_, Operator::NotEq, right)
    }

    fn __gt__<'py>(
        self_: PyRef<'py, Self>,
        right: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyExpr>> {
        py_binary_operator(self_, Operator::Gt, right)
    }

    fn __ge__<'py>(
        self_: PyRef<'py, Self>,
        right: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyExpr>> {
        py_binary_operator(self_, Operator::Gte, right)
    }

    fn __lt__<'py>(
        self_: PyRef<'py, Self>,
        right: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyExpr>> {
        py_binary_operator(self_, Operator::Lt, right)
    }

    fn __le__<'py>(
        self_: PyRef<'py, Self>,
        right: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyExpr>> {
        py_binary_operator(self_, Operator::Lte, right)
    }

    fn __and__<'py>(
        self_: PyRef<'py, Self>,
        right: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyExpr>> {
        py_binary_operator(self_, Operator::And, right)
    }

    fn __rand__<'py>(
        self_: PyRef<'py, Self>,
        left: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyExpr>> {
        py_reflected_operator(self_, Operator::And, left)
    }

    fn __or__<'py>(
        self_: PyRef<'py, Self>,
        right: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyExpr>> {
        py_binary_operator(self_, Operator::Or, right)
    }

    fn __ror__<'py>(
        self_: PyRef<'py, Self>,
        left: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyExpr>> {
        py_reflected_operator(self_, Operator::Or, left)
    }

    fn __invert__(self_: PyRef<'_, Self>) -> PyExpr {
        PyExpr {
            inner: expr::not(self_.inner.clone()),
        }
    }

    fn __add__<'py>(
        self_: PyRef<'py, Self>,
        right: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyExpr>> {
        py_binary_operator(self_, Operator::Add, right)
    }

    fn __radd__<'py>(
        self_: PyRef<'py, Self>,
        left: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyExpr>> {
        py_reflected_operator(self_, Operator::Add, left)
    }

    fn __sub__<'py>(
        self_: PyRef<'py, Self>,
        right: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyExpr>> {
        py_binary_operator(self_, Operator::Sub, right)
    }

    fn __rsub__<'py>(
        self_: PyRef<'py, Self>,
        left: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyExpr>> {
        py_reflected_operator(self_, Operator::Sub, left)
    }

    fn __mul__<'py>(
        self_: PyRef<'py, Self>,
        right: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyExpr>> {
        py_binary_operator(self_, Operator::Mul, right)
    }

    fn __rmul__<'py>(
        self_: PyRef<'py, Self>,
        left: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyExpr>> {
        py_reflected_operator(self_, Operator::Mul, left)
    }

    fn __truediv__<'py>(
        self_: PyRef<'py, Self>,
        right: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyExpr>> {
        py_binary_operator(self_, Operator::Div, right)
    }

    fn __rtruediv__<'py>(
        self_: PyRef<'py, Self>,
        left: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyExpr>> {
        py_reflected_operator(self_, Operator::Div, left)
    }

    // Special methods docstrings cannot be defined in Rust. Write a docstring in the corresponding
    // rST file. https://github.com/PyO3/pyo3/issues/4326
    fn __getitem__(self_: PyRef<'_, Self>, field: String) -> PyExpr {
        PyExpr {
            inner: expr::get_item(field, self_.inner.clone()),
        }
    }

    /// Serialize this expression to its Vortex protobuf wire format.
    ///
    /// The result can be sent to another process or machine and rebuilt with
    /// :func:`vortex.expr.deserialize`.
    ///
    /// Returns
    /// -------
    /// :class:`.bytes`
    ///
    /// Raises
    /// ------
    /// :class:`RuntimeError`
    ///     If the expression contains a scalar function that is not serializable.
    ///
    /// Examples
    /// --------
    ///
    /// ```python
    /// >>> import vortex.expr as ve
    /// >>> expr = ve.column("age") > 21
    /// >>> str(ve.deserialize(expr.serialize())) == str(expr)
    /// True
    /// ```
    fn serialize<'py>(self_: PyRef<'py, Self>) -> PyVortexResult<Bound<'py, PyBytes>> {
        let proto = self_.inner.serialize_proto()?;
        Ok(PyBytes::new(self_.py(), &proto.encode_to_vec()))
    }

    /// Support for Python's pickle protocol, backed by the protobuf wire format.
    ///
    /// This lets expressions cross process boundaries, for example as filter pushdown in a
    /// multiprocessing or Ray worker.
    fn __reduce__<'py>(
        self_: PyRef<'py, Self>,
    ) -> PyVortexResult<(Bound<'py, PyAny>, (Bound<'py, PyBytes>,))> {
        let py = self_.py();
        let proto = self_.inner.serialize_proto()?;
        let bytes = PyBytes::new(py, &proto.encode_to_vec());

        let module = PyModule::import(py, "vortex._lib.expr")?;
        let deserialize_fn = module.getattr(intern!(py, "deserialize"))?;

        Ok((deserialize_fn, (bytes,)))
    }
}

/// Rebuild an expression from its protobuf wire format.
///
/// Parameters
/// ----------
/// data : :class:`.bytes`
///     Bytes produced by :meth:`vortex.expr.Expr.serialize`.
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
/// >>> expr = ve.column("age") > 21
/// >>> str(ve.deserialize(expr.serialize())) == str(expr)
/// True
/// ```
#[pyfunction]
pub fn deserialize(data: &[u8]) -> PyVortexResult<PyExpr> {
    let proto = pb::Expr::decode(data)
        .map_err(|err| PyValueError::new_err(format!("invalid Vortex expression bytes: {err}")))?;
    Ok(PyExpr {
        inner: Expression::from_proto(&proto, session())?,
    })
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

/// Extract a named field from a struct expression.
///
/// Parameters
/// ----------
/// field : :class:`str`
///     The name of the field.
/// child : :class:`vortex.Expr`, optional
///     The struct expression to read from. Defaults to :func:`.root`.
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
/// >>> ve.get_item("yy", ve.column("y"))
/// <vortex.Expr object at ...>
/// ```
#[pyfunction]
#[pyo3(signature = (field, child = None))]
pub fn get_item(field: String, child: Option<PyIntoExpr>) -> PyExpr {
    let child = child.map_or_else(expr::root, PyIntoExpr::into_inner);
    PyExpr {
        inner: expr::get_item(field, child),
    }
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
pub fn not_(child: PyIntoExpr) -> PyExpr {
    PyExpr {
        inner: expr::not(child.into_inner()),
    }
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
pub fn and_(left: PyIntoExpr, right: PyIntoExpr) -> PyExpr {
    PyExpr {
        inner: expr::and(left.into_inner(), right.into_inner()),
    }
}

/// True if either argument is true.
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
/// >>> ve.or_(ve.literal(vx.bool_(), True), ve.literal(vx.bool_(), False))
/// <vortex.Expr object at ...>
/// ```
#[pyfunction]
pub fn or_(left: PyIntoExpr, right: PyIntoExpr) -> PyExpr {
    PyExpr {
        inner: expr::or(left.into_inner(), right.into_inner()),
    }
}

/// Combine expressions with logical AND using a balanced tree.
///
/// Parameters
/// ----------
/// exprs : :class:`Iterable`
///     The boolean expressions to combine.
///
/// Returns
/// -------
/// :class:`vortex.Expr` or ``None``
///     ``None`` if ``exprs`` is empty.
#[pyfunction]
pub fn and_collect(exprs: &Bound<'_, PyAny>) -> PyResult<Option<PyExpr>> {
    Ok(expr::and_collect(into_exprs(exprs)?).map(PyExpr::from))
}

/// Combine expressions with logical OR using a balanced tree.
///
/// Parameters
/// ----------
/// exprs : :class:`Iterable`
///     The boolean expressions to combine.
///
/// Returns
/// -------
/// :class:`vortex.Expr` or ``None``
///     ``None`` if ``exprs`` is empty.
#[pyfunction]
pub fn or_collect(exprs: &Bound<'_, PyAny>) -> PyResult<Option<PyExpr>> {
    Ok(expr::or_collect(into_exprs(exprs)?).map(PyExpr::from))
}

macro_rules! binary_fn {
    ($name:ident, $factory:path, $doc:literal) => {
        #[doc = $doc]
        /// Parameters
        /// ----------
        /// left : :class:`Any`
        /// right : :class:`Any`
        ///
        /// Returns
        /// -------
        /// :class:`vortex.Expr`
        #[pyfunction]
        pub fn $name(left: PyIntoExpr, right: PyIntoExpr) -> PyExpr {
            PyExpr {
                inner: $factory(left.into_inner(), right.into_inner()),
            }
        }
    };
}

binary_fn!(eq, expr::eq, "True where both arguments are equal.");
binary_fn!(
    not_eq,
    expr::not_eq,
    "True where the arguments are not equal."
);
binary_fn!(gt, expr::gt, "True where `left` is greater than `right`.");
binary_fn!(
    gt_eq,
    expr::gt_eq,
    "True where `left` is greater than or equal to `right`."
);
binary_fn!(lt, expr::lt, "True where `left` is less than `right`.");
binary_fn!(
    lt_eq,
    expr::lt_eq,
    "True where `left` is less than or equal to `right`."
);
binary_fn!(
    add,
    expr::checked_add,
    "The sum of the arguments, erroring on overflow."
);
binary_fn!(sub, sub_expr, "The difference between the arguments.");
binary_fn!(mul, mul_expr, "The product of the arguments.");
binary_fn!(div, div_expr, "`left` divided by `right`.");

fn sub_expr(left: Expression, right: Expression) -> Expression {
    Binary.new_expr(Operator::Sub, [left, right])
}

fn mul_expr(left: Expression, right: Expression) -> Expression {
    Binary.new_expr(Operator::Mul, [left, right])
}

fn div_expr(left: Expression, right: Expression) -> Expression {
    Binary.new_expr(Operator::Div, [left, right])
}

/// True where `child` lies between `lower` and `upper`.
///
/// Parameters
/// ----------
/// child : :class:`Any`
///     The expression to test.
/// lower : :class:`Any`
///     The lower bound.
/// upper : :class:`Any`
///     The upper bound.
/// lower_strict : :class:`bool`
///     If ``True``, compare the lower bound with ``<`` instead of ``<=``.
/// upper_strict : :class:`bool`
///     If ``True``, compare the upper bound with ``<`` instead of ``<=``.
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
/// >>> ve.between(ve.column("age"), 23, 55)
/// <vortex.Expr object at ...>
/// ```
#[pyfunction]
#[pyo3(signature = (child, lower, upper, *, lower_strict = false, upper_strict = false))]
pub fn between(
    child: PyIntoExpr,
    lower: PyIntoExpr,
    upper: PyIntoExpr,
    lower_strict: bool,
    upper_strict: bool,
) -> PyExpr {
    PyExpr {
        inner: expr::between(
            child.into_inner(),
            lower.into_inner(),
            upper.into_inner(),
            BetweenOptions {
                lower_strict: strictness(lower_strict),
                upper_strict: strictness(upper_strict),
            },
        ),
    }
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
#[pyfunction]
pub fn is_null(child: PyIntoExpr) -> PyExpr {
    PyExpr {
        inner: expr::is_null(child.into_inner()),
    }
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
pub fn is_not_null(child: PyIntoExpr) -> PyExpr {
    PyExpr {
        inner: expr::is_not_null(child.into_inner()),
    }
}

/// Replace null values with a fill value.
///
/// Parameters
/// ----------
/// child : :class:`Any`
/// fill_value : :class:`Any`
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
/// >>> ve.fill_null(ve.column("age"), 0)
/// <vortex.Expr object at ...>
/// ```
#[pyfunction]
pub fn fill_null(child: PyIntoExpr, fill_value: PyIntoExpr) -> PyExpr {
    PyExpr {
        inner: expr::fill_null(child.into_inner(), fill_value.into_inner()),
    }
}

/// A SQL ``LIKE`` expression.
///
/// Parameters
/// ----------
/// child : :class:`Any`
///     The string expression to match.
/// pattern : :class:`Any`
///     The SQL ``LIKE`` pattern, where ``%`` matches any run of characters and ``_`` matches any
///     single character.
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
/// >>> ve.like(ve.column("name"), "Ali%")
/// <vortex.Expr object at ...>
/// ```
#[pyfunction]
pub fn like(child: PyIntoExpr, pattern: PyIntoExpr) -> PyExpr {
    PyExpr {
        inner: expr::like(child.into_inner(), pattern.into_inner()),
    }
}

/// A case-insensitive SQL ``ILIKE`` expression.
///
/// Parameters
/// ----------
/// child : :class:`Any`
/// pattern : :class:`Any`
///
/// Returns
/// -------
/// :class:`vortex.Expr`
#[pyfunction]
pub fn ilike(child: PyIntoExpr, pattern: PyIntoExpr) -> PyExpr {
    PyExpr {
        inner: expr::ilike(child.into_inner(), pattern.into_inner()),
    }
}

/// A negated SQL ``NOT LIKE`` expression.
///
/// Parameters
/// ----------
/// child : :class:`Any`
/// pattern : :class:`Any`
///
/// Returns
/// -------
/// :class:`vortex.Expr`
#[pyfunction]
pub fn not_like(child: PyIntoExpr, pattern: PyIntoExpr) -> PyExpr {
    PyExpr {
        inner: expr::not_like(child.into_inner(), pattern.into_inner()),
    }
}

/// A negated case-insensitive SQL ``NOT ILIKE`` expression.
///
/// Parameters
/// ----------
/// child : :class:`Any`
/// pattern : :class:`Any`
///
/// Returns
/// -------
/// :class:`vortex.Expr`
#[pyfunction]
pub fn not_ilike(child: PyIntoExpr, pattern: PyIntoExpr) -> PyExpr {
    PyExpr {
        inner: expr::not_ilike(child.into_inner(), pattern.into_inner()),
    }
}

/// The byte length of each element, akin to SQL ``OCTET_LENGTH()``.
///
/// Parameters
/// ----------
/// child : :class:`Any`
///
/// Returns
/// -------
/// :class:`vortex.Expr`
#[pyfunction]
pub fn byte_length(child: PyIntoExpr) -> PyExpr {
    PyExpr {
        inner: expr::byte_length(child.into_inner()),
    }
}

/// Project only the named fields of a struct expression.
///
/// Parameters
/// ----------
/// fields : :class:`str` or :class:`Iterable` of :class:`str`
///     The field names to keep.
/// child : :class:`vortex.Expr`, optional
///     The struct expression to project. Defaults to :func:`.root`.
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
/// >>> ve.select(["name", "age"])
/// <vortex.Expr object at ...>
/// ```
#[pyfunction]
#[pyo3(signature = (fields, child = None))]
pub fn select(fields: &Bound<'_, PyAny>, child: Option<PyIntoExpr>) -> PyResult<PyExpr> {
    let child = child.map_or_else(expr::root, PyIntoExpr::into_inner);
    Ok(PyExpr {
        inner: expr::select(field_names(fields)?, child),
    })
}

/// Project every field of a struct expression except the named ones.
///
/// Parameters
/// ----------
/// fields : :class:`str` or :class:`Iterable` of :class:`str`
///     The field names to drop.
/// child : :class:`vortex.Expr`, optional
///     The struct expression to project. Defaults to :func:`.root`.
///
/// Returns
/// -------
/// :class:`vortex.Expr`
#[pyfunction]
#[pyo3(signature = (fields, child = None))]
pub fn select_exclude(fields: &Bound<'_, PyAny>, child: Option<PyIntoExpr>) -> PyResult<PyExpr> {
    let child = child.map_or_else(expr::root, PyIntoExpr::into_inner);
    Ok(PyExpr {
        inner: expr::select_exclude(field_names(fields)?, child),
    })
}

/// Pack expressions into a struct with named fields.
///
/// Parameters
/// ----------
/// fields : :class:`dict` or :class:`Iterable` of (:class:`str`, :class:`Any`)
///     The field names and their expressions.
/// nullable : :class:`bool`
///     Whether the resulting struct is nullable.
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
/// >>> ve.pack({"id": ve.column("user_id"), "constant": 42})
/// <vortex.Expr object at ...>
/// ```
#[pyfunction]
#[pyo3(signature = (fields, *, nullable = false))]
pub fn pack(fields: &Bound<'_, PyAny>, nullable: bool) -> PyResult<PyExpr> {
    Ok(PyExpr {
        inner: expr::pack(named_exprs(fields)?, nullability(nullable)),
    })
}

/// Merge struct expressions into a single struct.
///
/// Parameters
/// ----------
/// exprs : :class:`Iterable` of :class:`vortex.Expr`
///     The struct expressions to merge.
/// duplicate_handling : :class:`str`
///     Either ``"error"`` (the default) to reject duplicated field names, or ``"rightmost"`` to
///     take the value from the right-most expression.
///
/// Returns
/// -------
/// :class:`vortex.Expr`
#[pyfunction]
#[pyo3(signature = (exprs, *, duplicate_handling = "error"))]
pub fn merge(exprs: &Bound<'_, PyAny>, duplicate_handling: &str) -> PyResult<PyExpr> {
    let duplicate_handling = match duplicate_handling.to_ascii_lowercase().as_str() {
        "error" => DuplicateHandling::Error,
        "rightmost" | "right_most" => DuplicateHandling::RightMost,
        other => {
            return Err(PyValueError::new_err(format!(
                "duplicate_handling must be 'error' or 'rightmost', but found: {other}"
            )));
        }
    };
    Ok(PyExpr {
        inner: expr::merge_opts(into_exprs(exprs)?, duplicate_handling),
    })
}

/// True where the list contains the given value.
///
/// Parameters
/// ----------
/// child : :class:`Any`
///     A list expression.
/// value : :class:`Any`
///     The value to search for.
///
/// Returns
/// -------
/// :class:`vortex.Expr`
#[pyfunction]
pub fn list_contains(child: PyIntoExpr, value: PyIntoExpr) -> PyExpr {
    PyExpr {
        inner: expr::list_contains(child.into_inner(), value.into_inner()),
    }
}

/// The number of elements in each list, akin to SQL ``CARDINALITY()``.
///
/// Parameters
/// ----------
/// child : :class:`Any`
///     A list or fixed-size-list expression.
///
/// Returns
/// -------
/// :class:`vortex.Expr`
#[pyfunction]
pub fn list_length(child: PyIntoExpr) -> PyExpr {
    PyExpr {
        inner: expr::list_length(child.into_inner()),
    }
}

/// The sum of the elements of each list.
///
/// Follows SQL ``SUM`` semantics per list: null lists, empty lists, and lists whose elements are
/// all null yield null, and null elements are skipped.
///
/// Parameters
/// ----------
/// child : :class:`Any`
///     A list or fixed-size-list expression.
/// skip_nans : :class:`bool`
///     If ``True`` (the default), NaN float elements are skipped. Otherwise a single NaN poisons
///     the list's sum.
///
/// Returns
/// -------
/// :class:`vortex.Expr`
#[pyfunction]
#[pyo3(signature = (child, *, skip_nans = true))]
pub fn list_sum(child: PyIntoExpr, skip_nans: bool) -> PyExpr {
    PyExpr {
        inner: expr::list_sum_opts(child.into_inner(), NumericalAggregateOpts { skip_nans }),
    }
}

/// A ``CASE WHEN`` expression.
///
/// Parameters
/// ----------
/// when_then : :class:`Iterable` of (:class:`Any`, :class:`Any`)
///     One or more ``(condition, value)`` pairs, evaluated in order.
/// else_value : :class:`Any`, optional
///     The value to use when no condition matches. Defaults to null.
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
/// >>> ve.case_when([(ve.column("age") > 21, "adult")], else_value="minor")
/// <vortex.Expr object at ...>
/// ```
#[pyfunction]
#[pyo3(signature = (when_then, else_value = None))]
pub fn case_when(when_then: &Bound<'_, PyAny>, else_value: Option<PyIntoExpr>) -> PyResult<PyExpr> {
    let when_then: Vec<(Expression, Expression)> = when_then
        .try_iter()?
        .map(|pair| {
            let (condition, value): (Bound<'_, PyAny>, Bound<'_, PyAny>) = pair?.extract()?;
            Ok((coerce_expression(&condition)?, coerce_expression(&value)?))
        })
        .collect::<PyResult<_>>()?;
    if when_then.is_empty() {
        return Err(PyValueError::new_err(
            "case_when requires at least one (condition, value) pair",
        ));
    }
    Ok(PyExpr {
        inner: expr::nested_case_when(when_then, else_value.map(PyIntoExpr::into_inner)),
    })
}

/// Select element-wise between two expressions based on a boolean mask.
///
/// Parameters
/// ----------
/// mask : :class:`Any`
///     A boolean expression.
/// if_true : :class:`Any`
///     The value used where ``mask`` is true.
/// if_false : :class:`Any`
///     The value used where ``mask`` is false.
///
/// Returns
/// -------
/// :class:`vortex.Expr`
#[pyfunction(name = "zip_")]
pub fn zip_(mask: PyIntoExpr, if_true: PyIntoExpr, if_false: PyIntoExpr) -> PyExpr {
    PyExpr {
        inner: expr::zip_expr(
            mask.into_inner(),
            if_true.into_inner(),
            if_false.into_inner(),
        ),
    }
}

/// Null out the elements of an expression where the mask is true.
///
/// Parameters
/// ----------
/// child : :class:`Any`
///     The expression to mask.
/// mask : :class:`Any`
///     A boolean expression.
///
/// Returns
/// -------
/// :class:`vortex.Expr`
#[pyfunction]
pub fn mask(child: PyIntoExpr, mask: PyIntoExpr) -> PyExpr {
    PyExpr {
        inner: expr::mask(child.into_inner(), mask.into_inner()),
    }
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
pub fn cast(child: PyIntoExpr, dtype: PyDType) -> PyExpr {
    PyExpr {
        inner: expr::cast(child.into_inner(), dtype.into_inner()),
    }
}

/// Extract the storage values of an extension-typed expression.
///
/// Parameters
/// ----------
/// child : :class:`Any`
///
/// Returns
/// -------
/// :class:`vortex.Expr`
#[pyfunction]
pub fn ext_storage(child: PyIntoExpr) -> PyExpr {
    PyExpr {
        inner: expr::ext_storage(child.into_inner()),
    }
}

/// Extract a path from a Variant expression.
///
/// Missing paths, traversal mismatches, and failed casts all return null.
///
/// Parameters
/// ----------
/// child : :class:`Any`
///     A Variant expression.
/// path : :class:`str`, :class:`int`, or :class:`Iterable` of :class:`str` or :class:`int`
///     The path to extract. Strings select object fields, integers select list elements.
/// dtype : :class:`vortex.DType`, optional
///     The requested output type. When omitted, the result is a nullable Variant.
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
/// >>> ve.variant_get(ve.column("payload"), ["user", "id"])
/// <vortex.Expr object at ...>
/// ```
#[pyfunction]
#[pyo3(signature = (child, path, dtype = None))]
pub fn variant_get(
    child: PyIntoExpr,
    path: &Bound<'_, PyAny>,
    dtype: Option<PyDType>,
) -> PyResult<PyExpr> {
    Ok(PyExpr {
        inner: expr::variant_get(
            child.into_inner(),
            variant_path(path)?,
            dtype.map(PyDType::into_inner),
        ),
    })
}

fn variant_path(path: &Bound<'_, PyAny>) -> PyResult<VariantPath> {
    if let Ok(field) = path.cast::<PyString>() {
        return Ok(VariantPath::field(field.extract::<String>()?));
    }
    if let Ok(index) = path.cast::<PyInt>() {
        return Ok(VariantPath::new([VariantPathElement::index(
            index.extract::<u64>()?,
        )]));
    }
    let elements = path
        .try_iter()?
        .map(|element| {
            let element = element?;
            if let Ok(field) = element.cast::<PyString>() {
                Ok(VariantPathElement::field(field.extract::<String>()?))
            } else if let Ok(index) = element.cast::<PyInt>() {
                Ok(VariantPathElement::index(index.extract::<u64>()?))
            } else {
                Err(PyTypeError::new_err(format!(
                    "variant path elements must be str or int, but found: {element}"
                )))
            }
        })
        .collect::<PyResult<Vec<_>>>()?;
    Ok(VariantPath::new(elements))
}
