use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::*;
use vortex::dtype::{DType, Nullability, PType};
use vortex::expr::{BinaryExpr, ExprRef, GetItem, Operator, lit};

use crate::dtype::PyDType;
use crate::install_module;
use crate::scalar::factory::scalar_helper;

pub(crate) fn init(py: Python, parent: &Bound<PyModule>) -> PyResult<()> {
    let m = PyModule::new(py, "expr")?;
    parent.add_submodule(&m)?;
    install_module("vortex._lib.expr", &m)?;

    m.add_function(wrap_pyfunction!(column, &m)?)?;
    m.add_function(wrap_pyfunction!(ident, &m)?)?;
    m.add_function(wrap_pyfunction!(literal, &m)?)?;
    m.add_class::<PyExpr>()?;

    Ok(())
}

/// An expression describes how to filter rows when reading an array from a file.
///
/// .. seealso::
///    :func:`.column`
///
/// Examples
/// ========
///
/// All the examples read the following file.
///
///     >>> import vortex as vx
///     >>> a = vx.array([
///     ...     {'name': 'Joseph', 'age': 25},
///     ...     {'name': None, 'age': 31},
///     ...     {'name': 'Angela', 'age': None},
///     ...     {'name': 'Mikhail', 'age': 57},
///     ...     {'name': None, 'age': None},
///     ... ])
///     >>> vx.io.write_path(a, "a.vortex")
///
/// Read only those rows whose age column is greater than 35:
///
///     >>> e = vx.io.read_path("a.vortex", row_filter = vx.expr.column("age") > 35)
///     >>> e.to_arrow_array()
///     <pyarrow.lib.StructArray object at ...>
///     -- is_valid: all not null
///     -- child 0 type: int64
///       [
///         57
///       ]
///     -- child 1 type: string_view
///       [
///         "Mikhail"
///       ]
///
/// Read only those rows whose age column lies in (21, 33]. Notice that we must use parentheses
/// because of the Python precedence rules for ``&``:
///
///     >>> age = vx.expr.column("age")
///     >>> e = vx.io.read_path("a.vortex", row_filter = (age > 21) & (age <= 33))
///     >>> e.to_arrow_array()
///     <pyarrow.lib.StructArray object at ...>
///     -- is_valid: all not null
///     -- child 0 type: int64
///       [
///         25,
///         31
///       ]
///     -- child 1 type: string_view
///       [
///         "Joseph",
///         null
///       ]
///
/// Read only those rows whose name is `Joseph`:
///
///     >>> name = vx.expr.column("name")
///     >>> e = vx.io.read_path("a.vortex", row_filter = name == "Joseph")
///     >>> e.to_arrow_array()
///     <pyarrow.lib.StructArray object at ...>
///     -- is_valid: all not null
///     -- child 0 type: int64
///       [
///         25
///       ]
///     -- child 1 type: string_view
///       [
///         "Joseph"
///       ]
///
/// Read all the rows whose name is _not_ `Joseph`
///
///     >>> name = vx.expr.column("name")
///     >>> e = vx.io.read_path("a.vortex", row_filter = name != "Joseph")
///     >>> e.to_arrow_array()
///     <pyarrow.lib.StructArray object at ...>
///     -- is_valid: all not null
///     -- child 0 type: int64
///       [
///         null,
///         57
///       ]
///     -- child 1 type: string_view
///       [
///         "Angela",
///         "Mikhail"
///       ]
///
/// Read rows whose name is `Angela` or whose age is between 20 and 30, inclusive. Notice that the
/// Angela row is included even though its age is null. Under SQL / Kleene semantics, `true or
/// null` is `true`.
///
///     >>> name = vx.expr.column("name")
///     >>> e = vx.io.read_path("a.vortex", row_filter = (name == "Angela") | ((age >= 20) & (age <= 30)))
///     >>> e.to_arrow_array()
///     <pyarrow.lib.StructArray object at ...>
///     -- is_valid: all not null
///     -- child 0 type: int64
///       [
///         25,
///         null
///       ]
///     -- child 1 type: string_view
///       [
///         "Joseph",
///         "Angela"
///       ]
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
///     >>> import vortex.expr as ve
///     >>> ve.literal(ve.int_(), 42)
///     literal(int(), 42)
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
///     >>> import vortex.expr as ve
///     >>> ve.ident()
///     ident()
#[pyfunction]
pub fn ident() -> PyExpr {
    PyExpr {
        inner: vortex::expr::ident(),
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
///     >>> import vortex.expr as ve
///     >>> ve.column("age")
///     <vortex.Expr object at ...>
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
            inner: lit(scalar_helper(value, Some(&dtype))?),
        },
    )
}

pub fn get_item(field: String, child: PyExpr) -> PyResult<PyExpr> {
    Ok(PyExpr {
        inner: GetItem::new_expr(field, child.inner),
    })
}
