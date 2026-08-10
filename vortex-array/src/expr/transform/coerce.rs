// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Expression-level type coercion pass.

use vortex_error::VortexResult;

use crate::dtype::DType;
use crate::expr::Expression;
use crate::expr::Scope;
use crate::expr::cast;
use crate::expr::traversal::Transformed;
use crate::scalar_fn::fns::literal::Literal;

/// Rewrite an expression tree to insert casts where a scalar function's `coerce_args` demands
/// a different type than what the child currently produces.
///
/// The rewrite is bottom-up: children are coerced first, then each parent node checks whether
/// its children match the coerced argument types.
pub fn coerce_expression(expr: Expression, scope: impl Into<Scope>) -> VortexResult<Expression> {
    // A lambda is a coercion boundary. Its parameter dtypes come from whoever applies it, which an
    // unbound tree does not record, so there is no frame to push and the body's variables would not
    // resolve. Recursing explicitly rather than using `transform_up` is what makes skipping the
    // body possible.
    fn coerce_node(node: Expression, scope: &Scope) -> VortexResult<Transformed<Expression>> {
        if node.as_lambda().is_some() {
            return Ok(Transformed::no(node));
        }

        // Rebuild only when a child actually changed. Rebuilding unconditionally would allocate a
        // fresh children `Arc` for every node and break the pointer identity `ExactExpr` keys on.
        let mut changed = false;
        let mut coerced_children = Vec::with_capacity(node.children().len());
        for child in node.children() {
            let coerced = coerce_node(child.clone(), scope)?;
            changed |= coerced.changed;
            coerced_children.push(coerced.value);
        }

        let node = if changed {
            node.with_children(coerced_children)?
        } else {
            node
        };

        let coerced = coerce_one(node, scope)?;
        Ok(Transformed {
            changed: changed || coerced.changed,
            ..coerced
        })
    }

    fn coerce_one(node: Expression, scope: &Scope) -> VortexResult<Transformed<Expression>> {
        {
            // Leaf nodes (Root, Literal) have no children to coerce.
            if node.is_root() || node.is::<Literal>() || node.children().is_empty() {
                return Ok(Transformed::no(node));
            }

            // Compute the current child return types.
            let child_dtypes: Vec<DType> = node
                .children()
                .iter()
                .map(|c| c.return_dtype(scope))
                .collect::<VortexResult<_>>()?;

            // Ask the scalar function what types it wants.
            let Some(scalar_fn) = node.as_scalar() else {
                return Ok(Transformed::no(node));
            };
            let coerced_dtypes = scalar_fn.coerce_args(&child_dtypes)?;

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
                    let child_dtype = child.return_dtype(scope)?;
                    if child_dtype.eq_ignore_nullability(target)
                        && child_dtype.nullability() == target.nullability()
                    {
                        Ok(child.clone())
                    } else {
                        Ok(cast(child.clone(), target.clone()))
                    }
                })
                .collect::<VortexResult<_>>()?;

            node.with_children(new_children).map(Transformed::yes)
        }
    }

    coerce_node(expr, &scope.into()).map(Transformed::into_inner)
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
    fn decimal_arithmetic_coerces_precision_and_scale() -> VortexResult<()> {
        let common_dtype = DType::Decimal(DecimalDType::new(4, 2), NonNullable);
        let result_dtype = DType::Decimal(DecimalDType::new(5, 2), NonNullable);
        let scope = DType::Struct(
            StructFields::new(
                ["a", "b"].into(),
                vec![
                    DType::Decimal(DecimalDType::new(3, 1), NonNullable),
                    common_dtype,
                ],
            ),
            NonNullable,
        );
        let expr = Binary.new_expr(Operator::Add, [col("a"), col("b")]);

        let coerced = coerce_expression(expr, &scope)?;

        assert!(coerced.child(0).is::<Cast>());
        assert!(!coerced.child(1).is::<Cast>());
        assert_eq!(coerced.return_dtype(&scope)?, result_dtype);
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

#[cfg(test)]
mod lambda_tests {
    use vortex_error::VortexResult;

    use super::*;
    use crate::dtype::DType;
    use crate::dtype::Nullability::NonNullable;
    use crate::dtype::PType;
    use crate::expr::Expression;
    use crate::expr::Frame;
    use crate::expr::Scope;
    use crate::expr::checked_add;
    use crate::expr::col;
    use crate::expr::lambda;
    use crate::expr::lit;
    use crate::expr::test_harness::struct_dtype;
    use crate::expr::var;
    use crate::scalar_fn::ScalarFnVTableExt;
    use crate::scalar_fn::fns::binary::Binary;
    use crate::scalar_fn::fns::cast::Cast;
    use crate::scalar_fn::fns::operators::Operator;

    /// A lambda body types against a parameter frame, which this pass does not carry. Descending
    /// into one would try to type the variable against the root dtype and fail, so a lambda is a
    /// coercion boundary and is returned untouched.
    #[test]
    fn a_lambda_is_a_coercion_boundary() -> VortexResult<()> {
        let l = Expression::from(lambda(["x"], checked_add(var("x"), lit(1i32))));
        assert_eq!(coerce_expression(l.clone(), struct_dtype())?, l);
        Ok(())
    }

    /// Coercion that changes nothing must return the original tree, not a rebuilt copy: rebuilding
    /// allocates a fresh children `Arc` and breaks the pointer identity `ExactExpr` keys on.
    #[test]
    fn a_no_op_coercion_preserves_identity() -> VortexResult<()> {
        use crate::expr::ExactExpr;

        // Already well-typed, so nothing should be coerced.
        let expr = checked_add(col("a"), lit(1i32));
        let coerced = coerce_expression(expr.clone(), struct_dtype())?;

        assert_eq!(coerced, expr);
        assert_eq!(
            ExactExpr(coerced),
            ExactExpr(expr),
            "an unchanged tree should keep its identity"
        );
        Ok(())
    }

    /// The boundary must not stop coercion of everything around it.
    #[test]
    fn coercion_still_applies_outside_a_lambda() -> VortexResult<()> {
        let expr = checked_add(col("a"), lit(1i64));
        let coerced = coerce_expression(expr.clone(), struct_dtype())?;
        assert_ne!(
            coerced, expr,
            "an i32 column against an i64 literal should coerce"
        );
        Ok(())
    }

    /// A frame supplies the parameter dtypes that a bare root dtype cannot, so a variable is
    /// typeable here even though the tree is unbound.
    #[test]
    fn a_variable_bound_by_a_frame_is_coerced() -> VortexResult<()> {
        let scope = Scope::new(struct_dtype()).push_frame(Frame::try_new([
            ("a".into(), DType::Primitive(PType::I32, NonNullable)),
            ("b".into(), DType::Primitive(PType::I64, NonNullable)),
        ])?);

        let expr = Binary.new_expr(Operator::Lt, [var("a"), var("b")]);
        let coerced = coerce_expression(expr, &scope)?;

        assert!(coerced.child(0).is::<Cast>());
        assert_eq!(
            coerced.child(0).return_dtype(&scope)?,
            DType::Primitive(PType::I64, NonNullable)
        );
        assert!(!coerced.child(1).is::<Cast>());
        Ok(())
    }

    /// Without a frame there is nothing to resolve against, so the pass fails rather than
    /// mistyping the variable against the root dtype.
    #[test]
    fn a_variable_with_no_frame_is_rejected() {
        let expr = Binary.new_expr(Operator::Lt, [var("a"), lit(1i64)]);
        let err = coerce_expression(expr, struct_dtype()).unwrap_err();
        assert!(
            err.to_string().contains("unbound variable 'a'"),
            "unexpected error: {err}"
        );
    }
}
