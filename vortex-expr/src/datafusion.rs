#![cfg(feature = "datafusion")]

use std::sync::Arc;

use datafusion_expr::Operator as DFOperator;
use datafusion_physical_expr::{expressions, PhysicalExpr};
use vortex_error::{vortex_bail, vortex_err, VortexError, VortexResult};
use vortex_scalar::Scalar;

use crate::{BinaryExpr, Column, ExprRef, Literal, Operator};
pub fn convert_expr_to_vortex(physical_expr: Arc<dyn PhysicalExpr>) -> VortexResult<ExprRef> {
    if let Some(binary_expr) = physical_expr
        .as_any()
        .downcast_ref::<expressions::BinaryExpr>()
    {
        let left = convert_expr_to_vortex(binary_expr.left().clone())?;
        let right = convert_expr_to_vortex(binary_expr.right().clone())?;
        let operator = *binary_expr.op();

        return Ok(BinaryExpr::new_expr(left, operator.try_into()?, right));
    }

    if let Some(col_expr) = physical_expr.as_any().downcast_ref::<expressions::Column>() {
        let expr = Column::from(col_expr.name().to_owned());

        return Ok(Arc::new(expr) as _);
    }

    if let Some(lit) = physical_expr
        .as_any()
        .downcast_ref::<expressions::Literal>()
    {
        let value = Scalar::from(lit.value().clone());
        return Ok(Literal::new_expr(value));
    }

    vortex_bail!(
        "Couldn't convert DataFusion physical {physical_expr} expression to a vortex expression"
    )
}

impl TryFrom<DFOperator> for Operator {
    type Error = VortexError;

    fn try_from(value: DFOperator) -> Result<Self, Self::Error> {
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
            | DFOperator::Modulo => Err(vortex_err!("Unsupported datafusion operator {value}")),
        }
    }
}
