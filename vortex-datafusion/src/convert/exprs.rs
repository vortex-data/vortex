// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::sync::Arc;

use arrow_schema::{DataType, Schema};
use datafusion_expr::Operator as DFOperator;
use datafusion_physical_expr::{PhysicalExpr, PhysicalExprRef};
use datafusion_physical_plan::expressions as df_expr;
use vortex::error::{VortexResult, vortex_bail, vortex_err};
use vortex::expr::{BinaryExpr, ExprRef, LikeExpr, Operator, and, get_item, lit, root};
use vortex::scalar::Scalar;

use crate::convert::{FromDataFusion, TryFromDataFusion};

const SUPPORTED_BINARY_OPS: &[DFOperator] = &[
    DFOperator::Eq,
    DFOperator::NotEq,
    DFOperator::Gt,
    DFOperator::GtEq,
    DFOperator::Lt,
    DFOperator::LtEq,
];

/// Tries to convert the expressions into a vortex conjunction. Will return Ok(None) iff the input conjunction is empty.
pub(crate) fn make_vortex_predicate(
    predicate: &[&Arc<dyn PhysicalExpr>],
) -> VortexResult<Option<ExprRef>> {
    let exprs = predicate
        .iter()
        .map(|e| ExprRef::try_from_df(e.as_ref()))
        .collect::<VortexResult<Vec<_>>>()?;

    Ok(exprs.into_iter().reduce(and))
}

// TODO(joe): Don't return an error when we have an unsupported node, bubble up "TRUE" as in keep
//  for that node, up to any `and` or `or` node.
impl TryFromDataFusion<dyn PhysicalExpr> for ExprRef {
    fn try_from_df(df: &dyn PhysicalExpr) -> VortexResult<Self> {
        if let Some(binary_expr) = df.as_any().downcast_ref::<df_expr::BinaryExpr>() {
            let left = ExprRef::try_from_df(binary_expr.left().as_ref())?;
            let right = ExprRef::try_from_df(binary_expr.right().as_ref())?;
            let operator = Operator::try_from_df(binary_expr.op())?;

            return Ok(BinaryExpr::new_expr(left, operator, right));
        }

        if let Some(col_expr) = df.as_any().downcast_ref::<df_expr::Column>() {
            return Ok(get_item(col_expr.name().to_owned(), root()));
        }

        if let Some(like) = df.as_any().downcast_ref::<df_expr::LikeExpr>() {
            let child = ExprRef::try_from_df(like.expr().as_ref())?;
            let pattern = ExprRef::try_from_df(like.pattern().as_ref())?;
            return Ok(LikeExpr::new_expr(
                child,
                pattern,
                like.negated(),
                like.case_insensitive(),
            ));
        }

        if let Some(literal) = df.as_any().downcast_ref::<df_expr::Literal>() {
            let value = Scalar::from_df(literal.value());
            return Ok(lit(value));
        }

        vortex_bail!("Couldn't convert DataFusion physical {df} expression to a vortex expression")
    }
}

impl TryFromDataFusion<DFOperator> for Operator {
    fn try_from_df(value: &DFOperator) -> VortexResult<Self> {
        match value {
            DFOperator::Eq => Ok(Operator::Eq),
            DFOperator::NotEq => Ok(Operator::NotEq),
            DFOperator::Lt => Ok(Operator::Lt),
            DFOperator::LtEq => Ok(Operator::Lte),
            DFOperator::Gt => Ok(Operator::Gt),
            DFOperator::GtEq => Ok(Operator::Gte),
            DFOperator::And => Ok(Operator::And),
            DFOperator::Or => Ok(Operator::Or),
            DFOperator::IsDistinctFrom
            | DFOperator::IsNotDistinctFrom
            | DFOperator::RegexMatch
            | DFOperator::RegexIMatch
            | DFOperator::RegexNotMatch
            | DFOperator::RegexNotIMatch
            | DFOperator::LikeMatch
            | DFOperator::ILikeMatch
            | DFOperator::NotLikeMatch
            | DFOperator::NotILikeMatch
            | DFOperator::BitwiseAnd
            | DFOperator::BitwiseOr
            | DFOperator::BitwiseXor
            | DFOperator::BitwiseShiftRight
            | DFOperator::BitwiseShiftLeft
            | DFOperator::StringConcat
            | DFOperator::AtArrow
            | DFOperator::ArrowAt
            | DFOperator::Plus
            | DFOperator::Minus
            | DFOperator::Multiply
            | DFOperator::Divide
            | DFOperator::Modulo
            | DFOperator::Arrow
            | DFOperator::LongArrow
            | DFOperator::HashArrow
            | DFOperator::HashLongArrow
            | DFOperator::AtAt
            | DFOperator::IntegerDivide
            | DFOperator::HashMinus
            | DFOperator::AtQuestion
            | DFOperator::Question
            | DFOperator::QuestionAnd
            | DFOperator::QuestionPipe => {
                Err(vortex_err!("Unsupported datafusion operator {value}"))
            }
        }
    }
}

pub(crate) fn can_be_pushed_down(expr: &PhysicalExprRef, schema: &Schema) -> bool {
    let expr = expr.as_any();
    if let Some(binary) = expr.downcast_ref::<df_expr::BinaryExpr>() {
        can_binary_be_pushed_down(binary, schema)
    } else if let Some(col) = expr.downcast_ref::<df_expr::Column>() {
        schema
            .field_with_name(col.name())
            .ok()
            .is_some_and(|field| supported_data_types(field.data_type()))
    } else if let Some(like) = expr.downcast_ref::<df_expr::LikeExpr>() {
        can_be_pushed_down(like.expr(), schema) && can_be_pushed_down(like.pattern(), schema)
    } else if let Some(lit) = expr.downcast_ref::<df_expr::Literal>() {
        supported_data_types(&lit.value().data_type())
    } else {
        log::debug!("DataFusion expression can't be pushed down: {expr:?}");
        false
    }
}

fn can_binary_be_pushed_down(binary: &df_expr::BinaryExpr, schema: &Schema) -> bool {
    let is_op_supported =
        binary.op().is_logic_operator() || SUPPORTED_BINARY_OPS.contains(binary.op());
    is_op_supported
        && can_be_pushed_down(binary.left(), schema)
        && can_be_pushed_down(binary.right(), schema)
}

fn supported_data_types(dt: &DataType) -> bool {
    use DataType::*;
    let is_supported = dt.is_null()
        || dt.is_numeric()
        || matches!(
            dt,
            Boolean
                | Utf8
                | Utf8View
                | Binary
                | BinaryView
                | Date32
                | Date64
                | Timestamp(_, _)
                | Time32(_)
                | Time64(_)
        );

    if !is_supported {
        log::debug!("DataFusion data type {dt:?} is not supported");
    }

    is_supported
}
