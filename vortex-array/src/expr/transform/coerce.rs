// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Expression-level type coercion pass.

use vortex_error::VortexResult;

use crate::dtype::DType;
use crate::expr::Expression;
use crate::expr::cast;
use crate::expr::traversal::NodeExt;
use crate::expr::traversal::Transformed;
use crate::scalar_fn::fns::literal::Literal;
use crate::scalar_fn::fns::root::Root;

/// Rewrite an expression tree to insert casts where a scalar function's `coerce_args` demands
/// a different type than what the child currently produces.
///
/// The rewrite is bottom-up: children are coerced first, then each parent node checks whether
/// its children match the coerced argument types.
pub fn coerce_expression(expr: Expression, scope: &DType) -> VortexResult<Expression> {
    // We capture scope by reference for the closure.
    let scope = scope.clone();
    expr.transform_up(|node| {
        // Leaf nodes (Root, Literal) have no children to coerce.
        if node.is::<Root>() || node.is::<Literal>() || node.children().is_empty() {
            return Ok(Transformed::no(node));
        }

        // Compute the current child return types.
        let child_dtypes: Vec<DType> = node
            .children()
            .iter()
            .map(|c| c.return_dtype(&scope))
            .collect::<VortexResult<_>>()?;

        // Ask the scalar function what types it wants.
        let coerced_dtypes = node.scalar_fn().coerce_args(&child_dtypes)?;

        // If nothing changed, skip.
        if child_dtypes == coerced_dtypes {
            return Ok(Transformed::no(node));
        }

        // Build new children, inserting casts where needed.
        let new_children: Vec<Expression> = node
            .children()
            .iter()
            .zip(coerced_dtypes.iter())
            .map(|(child, target)| {
                let child_dtype = child.return_dtype(&scope)?;
                if child_dtype.eq_ignore_nullability(target)
                    && child_dtype.nullability() == target.nullability()
                {
                    Ok(child.clone())
                } else {
                    Ok(cast(child.clone(), target.clone()))
                }
            })
            .collect::<VortexResult<_>>()?;

        let new_expr = node.with_children(new_children)?;
        Ok(Transformed::yes(new_expr))
    })
    .map(|t| t.into_inner())
}

#[cfg(test)]
mod tests {
    use vortex_error::VortexResult;

    use crate::dtype::DType;
    use crate::dtype::DecimalDType;
    use crate::dtype::Nullability::NonNullable;
    use crate::dtype::PType;
    use crate::dtype::StructFields;
    use crate::expr::col;
    use crate::expr::lit;
    use crate::expr::transform::coerce::coerce_expression;
    use crate::scalar::Scalar;
    use crate::scalar_fn::ScalarFnVTableExt;
    use crate::scalar_fn::fns::binary::Binary;
    use crate::scalar_fn::fns::cast::Cast;
    use crate::scalar_fn::fns::operators::Operator;

    fn test_scope() -> DType {
        DType::Struct(
            StructFields::new(
                ["x", "y"].into(),
                vec![
                    DType::Primitive(PType::I32, NonNullable),
                    DType::Primitive(PType::I64, NonNullable),
                ],
            ),
            NonNullable,
        )
    }

    #[test]
    fn mixed_type_comparison_inserts_cast() -> VortexResult<()> {
        let scope = test_scope();
        // x (I32) < y (I64) => should cast x to I64
        let expr = Binary.new_expr(Operator::Lt, [col("x"), col("y")]);
        let coerced = coerce_expression(expr, &scope)?;

        // The LHS child should now be a cast expression
        assert!(coerced.child(0).is::<Cast>());
        // The coerced LHS should return I64
        assert_eq!(
            coerced.child(0).return_dtype(&scope)?,
            DType::Primitive(PType::I64, NonNullable)
        );
        // The RHS should be unchanged
        assert!(!coerced.child(1).is::<Cast>());
        Ok(())
    }

    #[test]
    fn same_type_comparison_no_cast() -> VortexResult<()> {
        let scope = test_scope();
        // x (I32) < x (I32) => no cast needed
        let expr = Binary.new_expr(Operator::Lt, [col("x"), col("x")]);
        let coerced = coerce_expression(expr, &scope)?;

        // Neither child should be a cast
        assert!(!coerced.child(0).is::<Cast>());
        assert!(!coerced.child(1).is::<Cast>());
        Ok(())
    }

    #[test]
    fn mixed_type_arithmetic_coerces_both() -> VortexResult<()> {
        let scope = DType::Struct(
            StructFields::new(
                ["a", "b"].into(),
                vec![
                    DType::Primitive(PType::U8, NonNullable),
                    DType::Primitive(PType::I32, NonNullable),
                ],
            ),
            NonNullable,
        );
        // a (U8) + b (I32) => both should be coerced to I32
        // U8 + I32: unsigned_signed_supertype(U8, I32) => max(1,4)=4 => I64
        let expr = Binary.new_expr(Operator::Add, [col("a"), col("b")]);
        let coerced = coerce_expression(expr, &scope)?;

        // LHS (U8) should be cast
        assert!(coerced.child(0).is::<Cast>());
        // Both should return the same supertype
        let lhs_dt = coerced.child(0).return_dtype(&scope)?;
        let rhs_dt = coerced.child(1).return_dtype(&scope)?;
        assert_eq!(lhs_dt, rhs_dt);
        Ok(())
    }

    #[test]
    fn mixed_decimal_arithmetic_preserves_input_types() -> VortexResult<()> {
        let lhs = DecimalDType::new(10, 2);
        let rhs = DecimalDType::new(5, 1);
        let scope = DType::Struct(
            StructFields::new(
                ["a", "b"].into(),
                vec![
                    DType::Decimal(lhs, NonNullable),
                    DType::Decimal(rhs, NonNullable),
                ],
            ),
            NonNullable,
        );
        let expr = Binary.new_expr(Operator::Add, [col("a"), col("b")]);
        let coerced = coerce_expression(expr, &scope)?;

        assert!(!coerced.child(0).is::<Cast>());
        assert!(!coerced.child(1).is::<Cast>());
        assert_eq!(
            coerced.return_dtype(&scope)?,
            DType::Decimal(DecimalDType::new(11, 2), NonNullable)
        );
        Ok(())
    }

    #[test]
    fn boolean_operators_no_coercion() -> VortexResult<()> {
        let scope = DType::Struct(
            StructFields::new(
                ["p", "q"].into(),
                vec![DType::Bool(NonNullable), DType::Bool(NonNullable)],
            ),
            NonNullable,
        );
        let expr = Binary.new_expr(Operator::And, [col("p"), col("q")]);
        let coerced = coerce_expression(expr, &scope)?;

        assert!(!coerced.child(0).is::<Cast>());
        assert!(!coerced.child(1).is::<Cast>());
        Ok(())
    }

    #[test]
    fn literal_coercion() -> VortexResult<()> {
        let scope = DType::Struct(
            StructFields::new(
                ["x"].into(),
                vec![DType::Primitive(PType::I64, NonNullable)],
            ),
            NonNullable,
        );
        // x (I64) + 1i32 => literal should be cast to I64
        let expr = Binary.new_expr(Operator::Add, [col("x"), lit(Scalar::from(1i32))]);
        let coerced = coerce_expression(expr, &scope)?;

        // The RHS (literal) should be cast to I64
        assert!(coerced.child(1).is::<Cast>());
        assert_eq!(
            coerced.child(1).return_dtype(&scope)?,
            DType::Primitive(PType::I64, NonNullable)
        );
        Ok(())
    }
}
