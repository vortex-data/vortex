// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_array::arrays::{ConstantArray, ExprArray, ExprVTable};
use vortex_array::compute::Operator;
use vortex_array::expr::{Binary, Literal, Root, VTableExt, lit, root};
use vortex_array::transform::{ArrayParentReduceRule, ArrayRuleContext};
use vortex_array::{ArrayRef, IntoArray};
use vortex_error::VortexResult;
use vortex_scalar::{PrimitiveScalar, Scalar};

use super::compare_common::{EncodedComparison, encode_for_comparison};
use crate::{ALPArray, ALPFloat, ALPVTable, match_each_alp_float_ptype};

/// Rule to push down comparison operations into the ALP compressed domain.
///
/// When an `ExprArray` wraps an `ALPArray` with a comparison expression, this rule
/// transforms the comparison to work directly on the encoded integers, avoiding
/// decompression of the entire array.
///
/// This uses the same comparison logic as the eager `CompareKernel` implementation
/// in `compare.rs`, but creates a lazy `ExprArray` instead of computing the result
/// immediately.
///
/// # Conditions
///
/// This optimization applies when:
/// - The child array is an `ALPArray` with no patches
/// - The parent is an `ExprArray` with a comparison expression (Binary with comparison operator)
/// - The comparison is between the root array and a constant scalar
/// - Neither the array nor the scalar is nullable
///
/// # Transformation
///
/// For example, `ExprArray(ALPArray, $ > 5.0)` becomes:
/// - Encode `5.0` into the ALP domain using the same exponents as the array
/// - Create `ExprArray(encoded_integers, $ >= encoded_value)`
///
/// Note: The operator may change (e.g., `>` becomes `>=`) when the scalar doesn't
/// encode exactly in the ALP domain. This follows the exact same logic as
/// `alp_scalar_compare` in `compare.rs`.
pub struct ALPExprPushdownRule;

impl ArrayParentReduceRule<ALPVTable, ExprVTable> for ALPExprPushdownRule {
    fn reduce_parent(
        &self,
        alp: &ALPArray,
        parent: &ExprArray,
        _child_idx: usize,
        _ctx: &ArrayRuleContext,
    ) -> VortexResult<Option<ArrayRef>> {
        // Only optimize if there are no patches
        if alp.patches().is_some() {
            return Ok(None);
        }

        // Only optimize if the array is not nullable
        if alp.dtype().is_nullable() {
            return Ok(None);
        }

        // Check if the expression is a binary comparison
        let Some(binary_view) = parent.expr().as_opt::<Binary>() else {
            return Ok(None);
        };

        // Get the comparison operator - only handle comparison operators
        let operator = binary_view.operator();
        if !matches!(
            operator,
            vortex_array::expr::Operator::Eq
                | vortex_array::expr::Operator::NotEq
                | vortex_array::expr::Operator::Lt
                | vortex_array::expr::Operator::Lte
                | vortex_array::expr::Operator::Gt
                | vortex_array::expr::Operator::Gte
        ) {
            return Ok(None);
        }

        // Convert to compute operator
        let compute_op = match operator {
            vortex_array::expr::Operator::Eq => Operator::Eq,
            vortex_array::expr::Operator::NotEq => Operator::NotEq,
            vortex_array::expr::Operator::Lt => Operator::Lt,
            vortex_array::expr::Operator::Lte => Operator::Lte,
            vortex_array::expr::Operator::Gt => Operator::Gt,
            vortex_array::expr::Operator::Gte => Operator::Gte,
            _ => return Ok(None),
        };

        // Check if this is a comparison of root() with a literal
        // For simplicity, we only handle `root() op literal` (not swapped)
        if !binary_view.lhs().is::<Root>() {
            return Ok(None);
        }
        if !binary_view.rhs().is::<Literal>() {
            return Ok(None);
        }

        // Get the literal scalar - literals evaluate to a constant array with one element
        let literal_array = binary_view.rhs().evaluate(&alp.clone().into_array())?;
        let Some(literal_value) = literal_array.as_constant() else {
            return Ok(None);
        };

        // Don't optimize nullable comparisons
        if literal_value.dtype().is_nullable() {
            return Ok(None);
        }

        // Convert to primitive scalar
        let Ok(pscalar) = PrimitiveScalar::try_from(&literal_value) else {
            return Ok(None);
        };

        // Use the common comparison logic to determine how to compare
        match_each_alp_float_ptype!(pscalar.ptype(), |T| {
            match pscalar.typed_value::<T>() {
                Some(value) => encode_and_pushdown(alp, value, compute_op),
                None => Ok(None),
            }
        })
    }
}

/// Encode a scalar value and create a pushdown comparison expression.
///
/// Uses the common `encode_for_comparison` logic to determine how to handle the comparison,
/// then creates an `ExprArray` with the appropriate expression.
fn encode_and_pushdown<F: ALPFloat + Into<Scalar>>(
    alp: &ALPArray,
    value: F,
    operator: Operator,
) -> VortexResult<Option<ArrayRef>>
where
    F::ALPInt: Into<Scalar>,
{
    let exponents = alp.exponents();
    let encoded_array = alp.encoded();

    // Use the common comparison logic from compare_common.rs
    match encode_for_comparison(value, exponents, operator) {
        EncodedComparison::Encoded { value, operator } => {
            // Create an expression that compares the encoded array with the encoded value
            let expr = Binary.new_expr(operator.into(), [root(), lit(value)]);
            Ok(Some(
                ExprArray::new_infer_dtype(encoded_array.clone(), expr)?.into_array(),
            ))
        }
        EncodedComparison::Constant(result) => {
            // Return a constant array with the comparison result
            Ok(Some(ConstantArray::new(result, alp.len()).into_array()))
        }
    }
}

#[cfg(test)]
mod tests {
    use vortex_array::arrays::PrimitiveArray;
    use vortex_array::compute::{Operator as ComputeOp, compare};
    use vortex_array::expr::session::ExprSession;
    use vortex_array::expr::transform::ExprOptimizer;
    use vortex_array::expr::{gt, lit, root};
    use vortex_array::{Array, ArraySession, IntoArray, ToCanonical};

    use super::*;
    use crate::alp_encode;

    #[test]
    fn test_alp_pushdown_gt_encodable() {
        // Create an ALP array with values [1.234f32; 100]
        let array = PrimitiveArray::from_iter([1.234f32; 100]);
        let alp = alp_encode(&array, None).unwrap();
        assert!(alp.patches().is_none());

        // Verify the encoded values (1.234 * 10^3 = 1234)
        assert_eq!(
            alp.encoded().to_primitive().as_slice::<i32>(),
            vec![1234; 100]
        );

        // Create expression: $ > 1.0
        let expr = gt(root(), lit(1.0f32));
        let expr_array = ExprArray::new_infer_dtype(alp.clone().into_array(), expr).unwrap();

        // Before optimization: child should be ALPArray
        assert!(expr_array.child().is::<ALPVTable>());

        // Apply the optimization
        let session = ArraySession::default();
        crate::register_alp_rules(&session);
        let expr_session = ExprSession::default();
        let optimizer = session.optimizer(ExprOptimizer::new(&expr_session));
        let optimized = optimizer.optimize_array(expr_array.into_array()).unwrap();

        // Verify the pushdown happened: should be ExprArray wrapping encoded integers
        let optimized_expr = optimized.as_::<ExprVTable>();
        assert!(
            optimized_expr
                .child()
                .is::<vortex_array::arrays::PrimitiveVTable>(),
            "Pushdown failed: child is not PrimitiveArray, it's {:?}",
            optimized_expr.child().encoding().id()
        );

        // Verify the child is the encoded integers (1.0 * 10^3 = 1000)
        let encoded_child = optimized_expr.child().to_primitive();
        assert_eq!(encoded_child.as_slice::<i32>(), vec![1234; 100]);

        // Verify the expression is comparing against encoded value (1.0 * 10^3 = 1000)
        // The expression should be: $ > 1000 (since 1.0 encodes to 1000)
        println!("Optimized expression: {:?}", optimized_expr.expr());

        // Verify correctness by comparing with the eager comparison kernel
        let expected = compare(
            alp.as_ref(),
            ConstantArray::new(1.0f32, 100).as_ref(),
            ComputeOp::Gt,
        )
        .unwrap();
        let actual = optimized.to_canonical().into_array();

        assert_eq!(actual.len(), expected.len());
        for i in 0..actual.len() {
            assert_eq!(actual.scalar_at(i), expected.scalar_at(i));
        }

        // Result should be all true (1.234 > 1.0)
        for i in 0..actual.len() {
            assert_eq!(actual.scalar_at(i), true.into());
        }
    }

    #[test]
    fn test_alp_pushdown_eq() {
        let array = PrimitiveArray::from_iter([1.234f32; 100]);
        let alp = alp_encode(&array, None).unwrap();

        let expr = vortex_array::expr::eq(root(), lit(1.234f32));
        let expr_array = ExprArray::new_infer_dtype(alp.clone().into_array(), expr).unwrap();

        let session = ArraySession::default();
        crate::register_alp_rules(&session);
        let expr_session = ExprSession::default();
        let optimizer = session.optimizer(ExprOptimizer::new(&expr_session));
        let optimized = optimizer.optimize_array(expr_array.into_array()).unwrap();

        // Verify correctness matches the eager comparison
        let expected = compare(
            alp.as_ref(),
            ConstantArray::new(1.234f32, 100).as_ref(),
            ComputeOp::Eq,
        )
        .unwrap();
        let actual = optimized.to_canonical().into_array();

        assert_eq!(actual.len(), expected.len());
        for i in 0..actual.len() {
            assert_eq!(actual.scalar_at(i), expected.scalar_at(i));
        }
    }

    #[test]
    fn test_alp_pushdown_unencodable_value() {
        let array = PrimitiveArray::from_iter([1.234f32; 100]);
        let alp = alp_encode(&array, None).unwrap();

        // Use a value that doesn't encode cleanly
        #[allow(clippy::excessive_precision)]
        let expr = vortex_array::expr::eq(root(), lit(1.234444f32));
        let expr_array = ExprArray::new_infer_dtype(alp.clone().into_array(), expr).unwrap();

        // Before optimization: child should be ALPArray
        assert!(expr_array.child().is::<ALPVTable>());

        let session = ArraySession::default();
        crate::register_alp_rules(&session);
        let expr_session = ExprSession::default();
        let optimizer = session.optimizer(ExprOptimizer::new(&expr_session));
        let optimized = optimizer.optimize_array(expr_array.into_array()).unwrap();

        // For unencodable Eq comparison, pushdown returns ConstantArray (not ExprArray)
        // This is the optimization: we know the result without any computation
        assert!(
            optimized.is::<vortex_array::arrays::ConstantVTable>(),
            "Pushdown should have returned ConstantArray for unencodable Eq, got {:?}",
            optimized.encoding().id()
        );

        // Should return constant false (value can't be equal to any encoded value)
        let actual = optimized.to_canonical().into_array();
        assert_eq!(actual.len(), 100);
        for i in 0..actual.len() {
            assert_eq!(actual.scalar_at(i), false.into());
        }
    }

    #[test]
    fn test_alp_pushdown_with_patches_skips() {
        // Create an array with patches
        let array =
            PrimitiveArray::from_iter([1.234f32, 1.5, 19.0, std::f32::consts::E, 1_000_000.9]);
        let alp = alp_encode(&array, None).unwrap();
        assert!(alp.patches().is_some());

        let expr = gt(root(), lit(1.0f32));
        let expr_array = ExprArray::new_infer_dtype(alp.into_array(), expr.clone()).unwrap();

        let session = ArraySession::default();
        crate::register_alp_rules(&session);
        let expr_session = ExprSession::default();
        let optimizer = session.optimizer(ExprOptimizer::new(&expr_session));
        let optimized = optimizer.optimize_array(expr_array.into_array()).unwrap();

        // Optimization should not apply - expression should be unchanged
        let optimized_expr = optimized.as_::<ExprVTable>();
        assert_eq!(optimized_expr.expr(), &expr);
    }

    #[test]
    fn test_alp_pushdown_all_operators() {
        let array = PrimitiveArray::from_iter([0.0605f32; 10]);
        let alp = alp_encode(&array, None).unwrap();
        assert!(alp.patches().is_none());

        // Verify encoded value: 0.0605 * 10^4 = 605
        assert_eq!(
            alp.encoded().to_primitive().as_slice::<i32>(),
            vec![605; 10]
        );

        // Test all comparison operators - results should match the eager comparison kernel
        let operators = [
            (ComputeOp::Eq, vortex_array::expr::eq as fn(_, _) -> _),
            (
                ComputeOp::NotEq,
                vortex_array::expr::not_eq as fn(_, _) -> _,
            ),
            (ComputeOp::Lt, vortex_array::expr::lt as fn(_, _) -> _),
            (ComputeOp::Lte, vortex_array::expr::lt_eq as fn(_, _) -> _),
            (ComputeOp::Gt, gt as fn(_, _) -> _),
            (ComputeOp::Gte, vortex_array::expr::gt_eq as fn(_, _) -> _),
        ];

        let test_value = 0.06051f32;

        let session = ArraySession::default();
        crate::register_alp_rules(&session);
        let expr_session = ExprSession::default();
        let expr_optimizer = ExprOptimizer::new(&expr_session);

        let mut pushdown_count = 0;

        for (compute_op, expr_fn) in operators {
            let expr = expr_fn(root(), lit(test_value));
            let expr_array = ExprArray::new_infer_dtype(alp.clone().into_array(), expr).unwrap();

            // Before optimization: child should be ALPArray
            assert!(expr_array.child().is::<ALPVTable>());

            let optimizer = session.optimizer(expr_optimizer.clone());
            let optimized = optimizer.optimize_array(expr_array.into_array()).unwrap();

            // Verify pushdown happened - child should no longer be ALPArray
            // Note: For unencodable comparisons with Gt/Gte/Lt/Lte, we use encode_above/encode_below
            // and push down to ExprArray. For Eq/NotEq with unencodable, we return ConstantArray.
            let is_expr = optimized.is::<ExprVTable>();
            if is_expr {
                let opt_expr = optimized.as_::<ExprVTable>();
                if opt_expr
                    .child()
                    .is::<vortex_array::arrays::PrimitiveVTable>()
                {
                    println!(
                        "✓ Operator {:?}: Pushed down to PrimitiveArray with expr {:?}",
                        compute_op,
                        opt_expr.expr()
                    );
                    pushdown_count += 1;
                } else {
                    println!(
                        "✗ Operator {:?}: Still ExprArray but child is {:?}",
                        compute_op,
                        opt_expr.child().encoding().id()
                    );
                }
            } else {
                println!(
                    "✓ Operator {:?}: Optimized to {:?} (constant result)",
                    compute_op,
                    optimized.encoding().id()
                );
                pushdown_count += 1;
            }

            // Verify correctness matches the eager comparison kernel
            let expected = compare(
                alp.as_ref(),
                ConstantArray::new(test_value, 10).as_ref(),
                compute_op,
            )
            .unwrap();
            let actual = optimized.to_canonical().into_array();

            assert_eq!(
                actual.len(),
                expected.len(),
                "Failed for operator {:?}",
                compute_op
            );
            for i in 0..actual.len() {
                assert_eq!(
                    actual.scalar_at(i),
                    expected.scalar_at(i),
                    "Mismatch at index {} for operator {:?}",
                    i,
                    compute_op
                );
            }
        }

        // Verify that all operators were optimized
        assert_eq!(
            pushdown_count, 6,
            "Expected all 6 operators to be pushed down, but only {} were",
            pushdown_count
        );
    }
}
