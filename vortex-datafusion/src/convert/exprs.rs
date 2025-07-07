// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use datafusion::logical_expr::Operator as DFOperator;
use datafusion::physical_expr::{PhysicalExpr, expressions};
use vortex::error::{VortexResult, vortex_bail, vortex_err};
use vortex::expr::{BinaryExpr, ExprRef, Like, Operator, get_item, lit, root};
use vortex::scalar::Scalar;

use crate::convert::{FromDataFusion, TryFromDataFusion};

// TODO(joe): Don't return an error when we have an unsupported node, bubble up "TRUE" as in keep
//  for that node, up to any `and` or `or` node.
impl TryFromDataFusion<dyn PhysicalExpr> for ExprRef {
    fn try_from_df(df: &dyn PhysicalExpr) -> VortexResult<Self> {
        if let Some(binary_expr) = df.as_any().downcast_ref::<expressions::BinaryExpr>() {
            let left = ExprRef::try_from_df(binary_expr.left().as_ref())?;
            let right = ExprRef::try_from_df(binary_expr.right().as_ref())?;
            let operator = Operator::try_from_df(binary_expr.op())?;

            return Ok(BinaryExpr::new_expr(left, operator, right));
        }

        if let Some(col_expr) = df.as_any().downcast_ref::<expressions::Column>() {
            return Ok(get_item(col_expr.name().to_owned(), root()));
        }

        if let Some(like) = df.as_any().downcast_ref::<expressions::LikeExpr>() {
            let child = ExprRef::try_from_df(like.expr().as_ref())?;
            let pattern = ExprRef::try_from_df(like.pattern().as_ref())?;
            return Ok(Like::new_expr(
                child,
                pattern,
                like.negated(),
                like.case_insensitive(),
            ));
        }

        if let Some(literal) = df.as_any().downcast_ref::<expressions::Literal>() {
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
