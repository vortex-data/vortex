use std::sync::Arc;

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3_polars::export::polars_plan::dsl::{Expr, Operator as PlOperator};
use pyo3_polars::export::polars_plan::plans::LiteralValue;
use vortex::buffer::ByteBuffer;
use vortex::dtype::DType;
use vortex::dtype::Nullability::NonNullable;
use vortex::expr::{BinaryExpr, ExprRef, GetItem, Literal, Operator, ident};
use vortex::scalar::{Scalar, ScalarValue};

use crate::expr::PyExpr;

trait TryFromPolars<T>
where
    Self: Sized,
{
    fn try_from_polars(value: T) -> PyResult<Self>;
}

#[pyfunction(name = "_expr_from_polars")]
pub fn expr_from_polars(expr: pyo3_polars::PyExpr) -> PyResult<PyExpr> {
    Ok(PyExpr::from(ExprRef::try_from_polars(&expr.0)?))
}

impl<'a> TryFromPolars<&'a Expr> for ExprRef {
    fn try_from_polars(expr: &'a Expr) -> PyResult<Self> {
        Ok(match expr {
            Expr::Column(column) => GetItem::new_expr(column.to_string(), ident()),
            Expr::Literal(literal) => ExprRef::try_from_polars(literal)?,
            Expr::BinaryExpr { left, op, right } => {
                let left = ExprRef::try_from_polars(left.as_ref())?;
                let right = ExprRef::try_from_polars(right.as_ref())?;
                match op {
                    PlOperator::Eq => BinaryExpr::new_expr(left, Operator::Eq, right),
                    PlOperator::NotEq => BinaryExpr::new_expr(left, Operator::NotEq, right),
                    PlOperator::Lt => BinaryExpr::new_expr(left, Operator::Lt, right),
                    PlOperator::LtEq => BinaryExpr::new_expr(left, Operator::Lte, right),
                    PlOperator::Gt => BinaryExpr::new_expr(left, Operator::Gt, right),
                    PlOperator::GtEq => BinaryExpr::new_expr(left, Operator::Gte, right),
                    PlOperator::And => BinaryExpr::new_expr(left, Operator::And, right),
                    PlOperator::Or => BinaryExpr::new_expr(left, Operator::Or, right),
                    PlOperator::EqValidity|
                    PlOperator::NotEqValidity|
                    PlOperator::Plus|
                    PlOperator::Minus|
                    PlOperator::Multiply|
                    PlOperator::Divide|
                    PlOperator::TrueDivide|
                    PlOperator::FloorDivide|
                    PlOperator::Modulus|
                    PlOperator::Xor|
                    PlOperator::LogicalAnd|
                    PlOperator::LogicalOr => {
                        return Err(PyValueError::new_err(format!(
                            "Unsupported binary operator {:?}",
                            op
                        )));
                    }
                }
            }
            Expr::Alias(..)
            | Expr::Columns(_)
            | Expr::DtypeColumn(_)
            | Expr::IndexColumn(_)
            | Expr::Cast { .. }
            | Expr::Sort { .. }
            | Expr::Gather { .. }
            | Expr::SortBy { .. }
            | Expr::Agg(_)
            | Expr::Ternary { .. }
            | Expr::Function { .. }
            | Expr::Explode(_)
            | Expr::Filter { .. }
            | Expr::Window { .. }
            | Expr::Wildcard
            | Expr::Slice { .. }
            | Expr::Exclude(..)
            | Expr::KeepName(_)
            | Expr::Len
            | Expr::Nth(_)
            | Expr::RenameAlias { .. }
            // | Expr::Field(_)
            | Expr::AnonymousFunction { .. }
            | Expr::SubPlan(..)
            | Expr::Selector(_) => {
                return Err(PyValueError::new_err(format!(
                    "Unsupported expression type {}",
                    expr
                )));
            }
        })
    }
}

impl TryFromPolars<&LiteralValue> for ExprRef {
    fn try_from_polars(value: &LiteralValue) -> PyResult<Self> {
        Ok(match value {
            LiteralValue::Null => Literal::new_expr(Scalar::new(DType::Null, ScalarValue::null())),
            LiteralValue::Boolean(b) => Literal::new_expr(Scalar::bool(*b, NonNullable)),
            LiteralValue::String(s) => Literal::new_expr(Scalar::utf8(s.to_string(), NonNullable)),
            LiteralValue::Binary(b) => {
                Literal::new_expr(Scalar::binary(Arc::new(ByteBuffer::copy_from(b)), NonNullable))
            }
            //LiteralValue::UInt8(v) => Literal::new_expr(Scalar::primitive(v, NonNullable)),
            //LiteralValue::UInt16(v) => Literal::new_expr(Scalar::primitive(v, NonNullable)),
            LiteralValue::UInt32(v) => Literal::new_expr(Scalar::primitive(*v, NonNullable)),
            LiteralValue::UInt64(v) => Literal::new_expr(Scalar::primitive(*v, NonNullable)),
            //LiteralValue::Int8(v) => Literal::new_expr(Scalar::primitive(v, NonNullable)),
            //LiteralValue::Int16(v) => Literal::new_expr(Scalar::primitive(v, NonNullable)),
            LiteralValue::Int32(v) => Literal::new_expr(Scalar::primitive(*v, NonNullable)),
            LiteralValue::Int64(v) => Literal::new_expr(Scalar::primitive(*v, NonNullable)),
            LiteralValue::Float32(v) => Literal::new_expr(Scalar::primitive(*v, NonNullable)),
            LiteralValue::Float64(v) => Literal::new_expr(Scalar::primitive(*v, NonNullable)),
            LiteralValue::Float(v) => Literal::new_expr(Scalar::primitive(*v, NonNullable)),
            LiteralValue::Int(v) => {
                Literal::new_expr(Scalar::primitive(i64::try_from(*v)?, NonNullable))
            }
            //LiteralValue::Int128(v) => {
            //    Literal::new_expr(Scalar::primitive(i64::try_from(v)?, NonNullable))
            //}
            //LiteralValue::Decimal(..)
            | LiteralValue::Range { .. }
            //| LiteralValue::Date(_)
            //| LiteralValue::DateTime(..)
            //| LiteralValue::Duration(..)
            //| LiteralValue::Time(_)
            | LiteralValue::Series(_)
            | LiteralValue::OtherScalar(_)
            | LiteralValue::StrCat(_) => {
                return Err(PyValueError::new_err(format!(
                    "Unsupported literal value {:?}",
                    value
                )));
            }
        })
    }
}
