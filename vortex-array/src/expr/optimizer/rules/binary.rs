// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_error::VortexResult;

use super::OptimizerRule;
use super::preserve_dtype;
use crate::expr::BoundExpression;
use crate::expr::ExpressionId;
use crate::expr::bound;
use crate::scalar::Scalar;
use crate::scalar_fn::ScalarFnVTable;
use crate::scalar_fn::fns::binary::Binary;
use crate::scalar_fn::fns::literal::Literal;
use crate::scalar_fn::fns::operators::Operator;

/// Simplifies `AND` and `OR` with literal operands using Kleene boolean semantics.
///
/// # Example
///
/// ```text
/// original: and(value, lit(true))
/// rewritten: value
/// ```
#[derive(Debug)]
pub(crate) struct BinaryBoolean;

impl OptimizerRule for BinaryBoolean {
    fn expression_id(&self) -> ExpressionId {
        Binary.id()
    }

    fn rewrite(&self, expr: &BoundExpression) -> VortexResult<Option<BoundExpression>> {
        let operator = expr.as_::<Binary>();
        let lhs = expr.child(0);
        let rhs = expr.child(1);
        let bool_literal = |expr: &BoundExpression| {
            expr.as_opt::<Literal>()?
                .as_bool_opt()
                .map(|value| value.value())
        };

        let replacement = match operator {
            Operator::And => match (bool_literal(lhs), bool_literal(rhs)) {
                (Some(Some(false)), _) | (_, Some(Some(false))) => {
                    Some(bound::lit(Scalar::bool(false, expr.dtype().nullability())))
                }
                (Some(Some(true)), _) => Some(preserve_dtype(rhs.clone(), expr.dtype())?),
                (_, Some(Some(true))) => Some(preserve_dtype(lhs.clone(), expr.dtype())?),
                (Some(None), Some(None)) => Some(lhs.clone()),
                _ => None,
            },
            Operator::Or => match (bool_literal(lhs), bool_literal(rhs)) {
                (Some(Some(true)), _) | (_, Some(Some(true))) => {
                    Some(bound::lit(Scalar::bool(true, expr.dtype().nullability())))
                }
                (Some(Some(false)), _) => Some(preserve_dtype(rhs.clone(), expr.dtype())?),
                (_, Some(Some(false))) => Some(preserve_dtype(lhs.clone(), expr.dtype())?),
                (Some(None), Some(None)) => Some(lhs.clone()),
                _ => None,
            },
            _ => None,
        };
        Ok(replacement)
    }
}

/// Replaces a comparison against a null literal with a null boolean literal.
///
/// # Example
///
/// ```text
/// original: eq(value, lit(Scalar::null(nullable_i32)))
/// rewritten: lit(Scalar::null(nullable_bool))
/// ```
#[derive(Debug)]
pub(crate) struct BinaryNullComparison;

impl OptimizerRule for BinaryNullComparison {
    fn expression_id(&self) -> ExpressionId {
        Binary.id()
    }

    fn rewrite(&self, expr: &BoundExpression) -> VortexResult<Option<BoundExpression>> {
        if !expr.as_::<Binary>().is_comparison() {
            return Ok(None);
        }
        let is_null_literal =
            |child: &BoundExpression| child.as_opt::<Literal>().is_some_and(Scalar::is_null);
        if !is_null_literal(expr.child(0)) && !is_null_literal(expr.child(1)) {
            return Ok(None);
        }

        Ok(Some(bound::lit(Scalar::null(expr.dtype().clone()))))
    }
}

#[cfg(test)]
mod tests {
    use vortex_error::VortexResult;

    use crate::dtype::DType;
    use crate::dtype::Nullability;
    use crate::dtype::PType;
    use crate::expr::BoundExpression;
    use crate::expr::bound;
    use crate::expr::optimizer::BoundExpressionOptimizer;
    use crate::scalar::Scalar;

    fn optimize(expr: &BoundExpression) -> VortexResult<BoundExpression> {
        BoundExpressionOptimizer::default().optimize(expr)
    }

    #[test]
    fn boolean_annihilator_preserves_nullable_dtype() -> VortexResult<()> {
        let nullable_bool = Scalar::bool(true, Nullability::Nullable);
        let expr = bound::and(bound::lit(false), bound::lit(nullable_bool));

        assert_eq!(
            optimize(&expr)?,
            bound::lit(Scalar::bool(false, Nullability::Nullable))
        );
        Ok(())
    }

    #[test]
    fn null_comparison_folds_to_nullable_null() -> VortexResult<()> {
        let nullable_i32 = DType::Primitive(PType::I32, Nullability::Nullable);
        let expr = bound::eq(
            bound::root(nullable_i32.clone()),
            bound::lit(Scalar::null(nullable_i32)),
        );

        assert_eq!(
            optimize(&expr)?,
            bound::lit(Scalar::null(DType::Bool(Nullability::Nullable)))
        );
        Ok(())
    }
}
