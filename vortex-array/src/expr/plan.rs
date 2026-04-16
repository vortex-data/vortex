// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Expression planning against an input scope.

use vortex_error::VortexResult;
use vortex_error::vortex_bail;

use crate::dtype::DType;
use crate::expr::Expression;
use crate::expr::transform::coerce_expression;

/// Plan an expression against an input [`DType`].
///
/// Planning typeifies the expression by inserting casts required by scalar functions, simplifies
/// the resulting tree, and verifies that the planned expression has a valid return type for the
/// provided scope.
pub fn plan_expression(expr: Expression, scope: &DType) -> VortexResult<Expression> {
    let expr = coerce_expression(expr, scope)?;
    let expr = expr.optimize_recursive(scope)?;
    expr.return_dtype(scope)?;
    Ok(expr)
}

/// Plan a filter expression against an input [`DType`].
///
/// This performs the same planning pass as [`plan_expression`] and then requires the expression to
/// return a Boolean value.
pub fn plan_filter_expression(expr: Expression, scope: &DType) -> VortexResult<Expression> {
    let expr = plan_expression(expr, scope)?;
    let dtype = expr.return_dtype(scope)?;
    if !matches!(dtype, DType::Bool(_)) {
        vortex_bail!("filter expression must return bool, got {}", dtype);
    }
    Ok(expr)
}

#[cfg(test)]
mod tests {
    use vortex_error::VortexResult;

    use crate::dtype::DType;
    use crate::dtype::Nullability::NonNullable;
    use crate::dtype::Nullability::Nullable;
    use crate::dtype::PType;
    use crate::dtype::StructFields;
    use crate::expr::col;
    use crate::expr::lit;
    use crate::expr::plan_expression;
    use crate::expr::plan_filter_expression;
    use crate::scalar::Scalar;
    use crate::scalar_fn::ScalarFnVTableExt;
    use crate::scalar_fn::fns::binary::Binary;
    use crate::scalar_fn::fns::cast::Cast;
    use crate::scalar_fn::fns::operators::Operator;

    fn scope() -> DType {
        DType::Struct(
            StructFields::new(
                ["i32", "i64", "u8", "flag"].into(),
                vec![
                    DType::Primitive(PType::I32, NonNullable),
                    DType::Primitive(PType::I64, NonNullable),
                    DType::Primitive(PType::U8, NonNullable),
                    DType::Bool(NonNullable),
                ],
            ),
            NonNullable,
        )
    }

    #[test]
    fn mixed_numeric_comparison_inserts_cast() -> VortexResult<()> {
        let scope = scope();
        let expr = Binary.new_expr(Operator::Lt, [col("i32"), col("i64")]);

        let planned = plan_filter_expression(expr, &scope)?;

        assert!(planned.child(0).is::<Cast>());
        assert_eq!(
            planned.child(0).return_dtype(&scope)?,
            DType::Primitive(PType::I64, NonNullable)
        );
        assert!(!planned.child(1).is::<Cast>());
        Ok(())
    }

    #[test]
    fn mixed_numeric_arithmetic_inserts_casts() -> VortexResult<()> {
        let scope = scope();
        let expr = Binary.new_expr(Operator::Add, [col("u8"), col("i32")]);

        let planned = plan_expression(expr, &scope)?;

        assert!(planned.child(0).is::<Cast>());
        assert_eq!(
            planned.return_dtype(&scope)?,
            DType::Primitive(PType::I64, NonNullable)
        );
        Ok(())
    }

    #[test]
    fn literal_values_are_coerced_against_column_types() -> VortexResult<()> {
        let scope = scope();
        let expr = Binary.new_expr(Operator::Eq, [col("i32"), lit(1i64)]);

        let planned = plan_filter_expression(expr, &scope)?;

        assert!(!planned.child(0).is::<Cast>());
        assert!(planned.child(1).is::<Cast>());
        assert_eq!(
            planned.child(1).return_dtype(&scope)?,
            DType::Primitive(PType::I32, NonNullable)
        );
        Ok(())
    }

    #[test]
    fn null_literals_are_typed_from_context() -> VortexResult<()> {
        let scope = scope();
        let expr = Binary.new_expr(Operator::Eq, [col("i32"), lit(Scalar::null(DType::Null))]);

        let planned = plan_filter_expression(expr, &scope)?;

        assert!(planned.child(1).is::<Cast>());
        assert_eq!(
            planned.child(1).return_dtype(&scope)?,
            DType::Primitive(PType::I32, Nullable)
        );
        Ok(())
    }

    #[test]
    fn boolean_and_preserves_boolean_inputs() -> VortexResult<()> {
        let scope = scope();
        let expr = Binary.new_expr(Operator::And, [col("flag"), col("flag")]);

        let planned = plan_filter_expression(expr, &scope)?;

        assert_eq!(planned.return_dtype(&scope)?, DType::Bool(NonNullable));
        assert!(!planned.child(0).is::<Cast>());
        assert!(!planned.child(1).is::<Cast>());
        Ok(())
    }

    #[test]
    fn filter_planning_rejects_non_boolean_outputs() {
        let scope = scope();
        let expr = Binary.new_expr(Operator::Add, [col("i32"), lit(1i32)]);

        let err = plan_filter_expression(expr, &scope).unwrap_err();

        assert!(
            err.to_string()
                .contains("filter expression must return bool")
        );
    }

    #[test]
    fn logical_operators_reject_non_boolean_inputs() {
        let scope = scope();
        let expr = Binary.new_expr(Operator::And, [col("i32"), col("i64")]);

        let err = plan_filter_expression(expr, &scope).unwrap_err();

        assert!(
            err.to_string()
                .contains("logical operation requires boolean operands")
        );
    }
}
