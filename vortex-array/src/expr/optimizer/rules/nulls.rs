// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_error::VortexResult;

use super::OptimizerRule;
use super::preserve_dtype;
use crate::expr::BoundExpression;
use crate::expr::ExpressionId;
use crate::expr::bound;
use crate::scalar_fn::EmptyOptions;
use crate::scalar_fn::ScalarFnVTable;
use crate::scalar_fn::ScalarFnVTableExt;
use crate::scalar_fn::fns::case_when::CaseWhen;
use crate::scalar_fn::fns::fill_null::FillNull;
use crate::scalar_fn::fns::is_not_null::IsNotNull;
use crate::scalar_fn::fns::is_null::IsNull;
use crate::scalar_fn::fns::literal::Literal;

/// Removes `fill_null` when its input is already non-nullable.
///
/// # Example
///
/// ```text
/// original: fill_null(non_nullable_value, lit(0))
/// rewritten: non_nullable_value
/// ```
#[derive(Debug)]
pub(crate) struct RemoveRedundantFillNull;

impl OptimizerRule for RemoveRedundantFillNull {
    fn expression_id(&self) -> ExpressionId {
        FillNull.id()
    }

    fn rewrite(&self, expr: &BoundExpression) -> VortexResult<Option<BoundExpression>> {
        if expr.child(0).dtype().is_nullable() {
            return Ok(None);
        }
        Ok(Some(preserve_dtype(expr.child(0).clone(), expr.dtype())?))
    }
}

/// Lowers a single-branch null-checking `case_when` into `fill_null` or its input.
///
/// # Example
///
/// ```text
/// original: case_when(is_null(value), fill, value)
/// rewritten: fill_null(value, fill)
/// ```
#[derive(Debug)]
pub(crate) struct CaseWhenToFillNull;

impl OptimizerRule for CaseWhenToFillNull {
    fn expression_id(&self) -> ExpressionId {
        CaseWhen.id()
    }

    fn rewrite(&self, expr: &BoundExpression) -> VortexResult<Option<BoundExpression>> {
        let options = expr.as_::<CaseWhen>();
        if options.num_when_then_pairs != 1 || !options.has_else {
            return Ok(None);
        }

        let when = expr.child(0);
        let then = expr.child(1);
        let els = expr.child(2);
        let (value, fill) = if when.is::<IsNull>() && when.child(0) == els {
            (els, then)
        } else if when.is::<IsNotNull>() && when.child(0) == then {
            (then, els)
        } else {
            return Ok(None);
        };
        let Some(fill_scalar) = fill.as_opt::<Literal>() else {
            return Ok(None);
        };

        if fill_scalar.is_null() {
            return Ok(Some(preserve_dtype(value.clone(), expr.dtype())?));
        }
        let fill = if fill.dtype() == expr.dtype() {
            fill.clone()
        } else {
            bound::lit(fill_scalar.cast(expr.dtype())?)
        };
        Ok(Some(
            FillNull.try_new_bound_expr(EmptyOptions, [value.clone(), fill])?,
        ))
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
    use crate::scalar_fn::fns::fill_null::FillNull;

    fn optimize(expr: &BoundExpression) -> VortexResult<BoundExpression> {
        BoundExpressionOptimizer::default().optimize(expr)
    }

    #[test]
    fn nonnullable_fill_null_input_is_removed() -> VortexResult<()> {
        let expr = bound::fill_null(
            bound::lit(1i32),
            bound::lit(Scalar::primitive(0i32, Nullability::Nullable)),
        );

        assert_eq!(
            optimize(&expr)?,
            bound::lit(Scalar::primitive(1i32, Nullability::Nullable))
        );
        Ok(())
    }

    #[test]
    fn coalesce_shaped_case_lowers_to_fill_null() -> VortexResult<()> {
        let value_dtype = DType::Primitive(PType::I64, Nullability::Nullable);
        let value = bound::root(value_dtype);
        let expr = bound::case_when(
            bound::is_null(value.clone()),
            bound::lit(0i64),
            value.clone(),
        );
        let optimized = optimize(&expr)?;

        assert!(optimized.is::<FillNull>());
        assert_eq!(
            optimized,
            bound::fill_null(
                value,
                bound::lit(Scalar::primitive(0i64, Nullability::Nullable))
            )
        );
        Ok(())
    }
}
