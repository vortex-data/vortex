use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::*;
use vortex::dtype::field::Field;
use vortex::dtype::half::f16;
use vortex::dtype::{DType, Nullability, PType};
use vortex::expr::{BinaryExpr, Column, ExprRef, Literal, Operator};
use vortex::scalar::Scalar;

use crate::dtype::PyDType;

/// An expression describes how to filter rows when reading an array from a file.
///
/// .. seealso::
///     :func:`.column`
///
/// Examples
/// ========
///
/// All the examples read the following file.
///
/// >>> a = vortex.array([
/// ...     {'name': 'Joseph', 'age': 25},
/// ...     {'name': None, 'age': 31},
/// ...     {'name': 'Angela', 'age': None},
/// ...     {'name': 'Mikhail', 'age': 57},
/// ...     {'name': None, 'age': None},
/// ... ])
/// >>> vortex.io.write_path(a, "a.vortex")
///
/// Read only those rows whose age column is greater than 35:
///
/// >>> e = vortex.io.read_path("a.vortex", row_filter = vortex.expr.column("age") > 35)
/// >>> e.to_arrow_array()
/// <pyarrow.lib.StructArray object at ...>
/// -- is_valid: all not null
/// -- child 0 type: int64
///   [
///     57
///   ]
/// -- child 1 type: string_view
///   [
///     "Mikhail"
///   ]
///
/// Read only those rows whose age column lies in (21, 33]. Notice that we must use parentheses
/// because of the Python precedence rules for ``&``:
///
/// >>> age = vortex.expr.column("age")
/// >>> e = vortex.io.read_path("a.vortex", row_filter = (age > 21) & (age <= 33))
/// >>> e.to_arrow_array()
/// <pyarrow.lib.StructArray object at ...>
/// -- is_valid: all not null
/// -- child 0 type: int64
///   [
///     25,
///     31
///   ]
/// -- child 1 type: string_view
///   [
///     "Joseph",
///     null
///   ]
///
/// Read only those rows whose name is `Joseph`:
///
/// >>> name = vortex.expr.column("name")
/// >>> e = vortex.io.read_path("a.vortex", row_filter = name == "Joseph")
/// >>> e.to_arrow_array()
/// <pyarrow.lib.StructArray object at ...>
/// -- is_valid: all not null
/// -- child 0 type: int64
///   [
///     25
///   ]
/// -- child 1 type: string_view
///   [
///     "Joseph"
///   ]
///
/// Read all the rows whose name is _not_ `Joseph`
///
/// >>> name = vortex.expr.column("name")
/// >>> e = vortex.io.read_path("a.vortex", row_filter = name != "Joseph")
/// >>> e.to_arrow_array()
/// <pyarrow.lib.StructArray object at ...>
/// -- is_valid: all not null
/// -- child 0 type: int64
///   [
///     null,
///     57
///   ]
/// -- child 1 type: string_view
///   [
///     "Angela",
///     "Mikhail"
///   ]
///
/// Read rows whose name is `Angela` or whose age is between 20 and 30, inclusive. Notice that the
/// Angela row is included even though its age is null. Under SQL / Kleene semantics, `true or
/// null` is `true`.
///
/// >>> name = vortex.expr.column("name")
/// >>> e = vortex.io.read_path("a.vortex", row_filter = (name == "Angela") | ((age >= 20) & (age <= 30)))
/// >>> e.to_arrow_array()
/// <pyarrow.lib.StructArray object at ...>
/// -- is_valid: all not null
/// -- child 0 type: int64
///   [
///     25,
///     null
///   ]
/// -- child 1 type: string_view
///   [
///     "Joseph",
///     "Angela"
///   ]
#[pyclass(name = "Expr", module = "vortex")]
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
}

/// A named column.
///
/// .. seealso::
///     :class:`.Expr`
///
/// Example
/// =======
///
/// A filter that selects only those rows whose name is `Joseph`:
///
/// >>> name = vortex.expr.column("name")
/// >>> filter = name == "Joseph"
///
/// See :class:`.Expr` for more examples.
///
#[pyfunction]
pub fn column<'py>(name: &Bound<'py, PyString>) -> PyResult<Bound<'py, PyExpr>> {
    let py = name.py();
    let name: String = name.extract()?;
    Bound::new(
        py,
        PyExpr {
            inner: Column::new_expr(Field::Name(name)),
        },
    )
}

#[pyfunction]
pub fn literal<'py>(
    dtype: &Bound<'py, PyDType>,
    value: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyExpr>> {
    scalar(dtype.borrow().unwrap().clone(), value)
}

pub fn scalar<'py>(dtype: DType, value: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyExpr>> {
    let py = value.py();
    Bound::new(
        py,
        PyExpr {
            inner: Literal::new_expr(scalar_helper(dtype, value)?),
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
            Ok(Scalar::list(element_type, values))
        }
        DType::Extension(..) => todo!(),
    }
}
