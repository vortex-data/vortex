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
use crate::scalar_fn::fns::literal::Literal;
use crate::scalar_fn::fns::mask::Mask;
use crate::scalar_fn::fns::zip::Zip;

/// Evaluates a mask whose mask argument is a non-null boolean literal.
///
/// # Example
///
/// ```text
/// original: mask(nullable_value, lit(true))
/// rewritten: nullable_value
/// ```
#[derive(Debug)]
pub(crate) struct ConstantMask;

impl OptimizerRule for ConstantMask {
    fn expression_id(&self) -> ExpressionId {
        Mask.id()
    }

    fn rewrite(&self, expr: &BoundExpression) -> VortexResult<Option<BoundExpression>> {
        let Some(mask) = expr
            .child(1)
            .as_opt::<Literal>()
            .and_then(|scalar| scalar.as_bool_opt())
            .and_then(|value| value.value())
        else {
            return Ok(None);
        };

        if mask {
            return Ok(Some(preserve_dtype(expr.child(0).clone(), expr.dtype())?));
        }
        Ok(Some(bound::lit(Scalar::null(expr.dtype().clone()))))
    }
}

/// Selects the reachable branch of a zip with a non-null literal mask.
///
/// # Example
///
/// ```text
/// original: zip_expr(lit(true), if_true, if_false)
/// rewritten: if_true
/// ```
#[derive(Debug)]
pub(crate) struct ConstantZip;

impl OptimizerRule for ConstantZip {
    fn expression_id(&self) -> ExpressionId {
        Zip.id()
    }

    fn rewrite(&self, expr: &BoundExpression) -> VortexResult<Option<BoundExpression>> {
        let Some(mask) = expr
            .child(2)
            .as_opt::<Literal>()
            .and_then(|scalar| scalar.as_bool_opt())
            .and_then(|value| value.value())
        else {
            return Ok(None);
        };
        let child = if mask { expr.child(0) } else { expr.child(1) };
        Ok(Some(preserve_dtype(child.clone(), expr.dtype())?))
    }
}

#[cfg(test)]
mod tests {
    use vortex_error::VortexResult;

    use crate::dtype::Nullability;
    use crate::expr::BoundExpression;
    use crate::expr::bound;
    use crate::expr::optimizer::BoundExpressionOptimizer;
    use crate::scalar::Scalar;

    fn optimize(expr: &BoundExpression) -> VortexResult<BoundExpression> {
        BoundExpressionOptimizer::default().optimize(expr)
    }

    #[test]
    fn constant_mask_and_zip_preserve_nullable_output() -> VortexResult<()> {
        let nullable_two = bound::lit(Scalar::primitive(2i32, Nullability::Nullable));
        let masked = bound::mask(bound::lit(1i32), bound::lit(true));
        let zipped = bound::zip_expr(bound::lit(true), bound::lit(1i32), nullable_two);
        let expected = bound::lit(Scalar::primitive(1i32, Nullability::Nullable));

        assert_eq!(optimize(&masked)?, expected);
        assert_eq!(optimize(&zipped)?, expected);
        Ok(())
    }
}
