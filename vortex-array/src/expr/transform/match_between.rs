// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use crate::expr::BoundExpression;
use crate::expr::bound;
use crate::scalar_fn::ScalarFnVTableExt;
use crate::scalar_fn::fns::between::Between;
use crate::scalar_fn::fns::between::BetweenOptions;
use crate::scalar_fn::fns::between::StrictComparison;
use crate::scalar_fn::fns::binary::Binary;
use crate::scalar_fn::fns::get_item::GetItem;
use crate::scalar_fn::fns::literal::Literal;
use crate::scalar_fn::fns::operators::Operator;

/// This pass looks for expression of the form
///      `x >= a && x < b` and converts them into x between a and b`
pub fn find_between(expr: BoundExpression) -> BoundExpression {
    // We search all pairs of cnfs to find any pair of expressions can be converted into a between
    // expression.
    //
    // Only leaf conjuncts (`GetItem op Literal`) can form a between; anything else is moved to the
    // rest unexamined so pairs of non-leaf conjuncts are never compared.
    let mut conjuncts = conjuncts(&expr)
        .into_iter()
        .map(|expression| {
            let is_leaf = is_leaf_conjunct(&expression);
            (expression, is_leaf)
        })
        .collect::<Vec<_>>();
    let mut rest = vec![];

    for idx in 0..conjuncts.len() {
        let Some((c, is_leaf)) = conjuncts.get(idx).cloned() else {
            continue;
        };
        if !is_leaf {
            rest.push(c);
            continue;
        }
        let mut matched = false;
        for idx2 in (idx + 1)..conjuncts.len() {
            // Since values are removed in iterations there might not be a value at idx2,
            // but all values will have been considered.
            let Some((c2, true)) = conjuncts.get(idx2) else {
                continue;
            };
            if let Some(expr) = maybe_match(&c, c2) {
                rest.push(expr);
                conjuncts.remove(idx2);
                matched = true;
                break;
            }
        }
        if !matched {
            rest.push(c.clone())
        }
    }

    bound::and_collect(rest).unwrap_or_else(|| bound::lit(true))
}

fn conjuncts(expr: &BoundExpression) -> Vec<BoundExpression> {
    let mut conjuncts = vec![];
    conjuncts_impl(expr, &mut conjuncts);
    conjuncts
}

fn conjuncts_impl(expr: &BoundExpression, conjuncts: &mut Vec<BoundExpression>) {
    if let Some(operator) = expr.as_opt::<Binary>()
        && *operator == Operator::And
    {
        conjuncts_impl(expr.child(0), conjuncts);
        conjuncts_impl(expr.child(1), conjuncts);
    } else {
        conjuncts.push(expr.clone())
    }
}

/// A leaf conjunct is a strict-comparison binary between a `GetItem` and a `Literal`, in either
/// order — the only shape that can be half of a between.
fn is_leaf_conjunct(expr: &BoundExpression) -> bool {
    let Some(operator) = expr.as_opt::<Binary>() else {
        return false;
    };
    if is_strict_comparison(*operator).is_none() {
        return false;
    }

    let lhs = expr.child(0);
    let rhs = expr.child(1);
    (lhs.is::<GetItem>() && rhs.is::<Literal>()) || (rhs.is::<GetItem>() && lhs.is::<Literal>())
}

fn maybe_match(lhs: &BoundExpression, rhs: &BoundExpression) -> Option<BoundExpression> {
    let (Some(lhs_op), Some(rhs_op)) = (lhs.as_opt::<Binary>(), rhs.as_opt::<Binary>()) else {
        return None;
    };

    // Extract the grandchildren
    let lhs_lhs = lhs.child(0);
    let lhs_rhs = lhs.child(1);
    let rhs_lhs = rhs.child(0);
    let rhs_rhs = rhs.child(1);

    // Cannot compare to self
    if lhs_lhs.eq(lhs_rhs) || rhs_lhs.eq(rhs_rhs) {
        return None;
    }

    // First, get both halves to have GetItem on the left
    let lhs = match (lhs_lhs.is::<GetItem>(), lhs_rhs.is::<GetItem>()) {
        (true, false) => lhs.clone(),
        (false, true) => Binary
            .try_new_bound_expr(lhs_op.swap()?, [lhs_rhs.clone(), lhs_lhs.clone()])
            .ok()?,
        _ => return None,
    };
    let lhs_op = lhs.as_::<Binary>();
    let lhs_lhs = lhs.child(0);

    let rhs = match (rhs_lhs.is::<GetItem>(), rhs_rhs.is::<GetItem>()) {
        (true, false) => rhs.clone(),
        (false, true) => Binary
            .try_new_bound_expr(rhs_op.swap()?, [rhs_rhs.clone(), rhs_lhs.clone()])
            .ok()?,
        _ => return None,
    };
    let rhs_op = rhs.as_::<Binary>();
    let rhs_lhs = rhs.child(0);

    // Both conjuncts must reference the same GetItem column
    if !lhs_lhs.eq(rhs_lhs) {
        return None;
    }

    let target = lhs_lhs.clone();

    // Find the lower bound
    let (lower, upper) = match (lhs_op, rhs_op) {
        (Operator::Lt | Operator::Lte, Operator::Gt | Operator::Gte) => (rhs, lhs),
        (Operator::Gt | Operator::Gte, Operator::Lt | Operator::Lte) => (lhs, rhs),
        _ => return None,
    };
    let lower_op = lower.as_::<Binary>();
    let lower_rhs = lower.child(1);
    let upper_op = upper.as_::<Binary>();
    let upper_rhs = upper.child(1);

    // Ensure bounds are literals
    let _ = lower_rhs.as_opt::<Literal>()?;
    let _ = upper_rhs.as_opt::<Literal>()?;

    let lower_strict = is_strict_comparison(*lower_op)?;
    let upper_strict = is_strict_comparison(*upper_op)?;

    // A between that fails to type-check leaves the comparisons in place.
    Between
        .try_new_bound_expr(
            BetweenOptions {
                lower_strict,
                upper_strict,
            },
            [target, lower_rhs.clone(), upper_rhs.clone()],
        )
        .ok()
}

fn is_strict_comparison(op: Operator) -> Option<StrictComparison> {
    match op {
        Operator::Lt | Operator::Gt => Some(StrictComparison::Strict),
        Operator::Lte | Operator::Gte => Some(StrictComparison::NonStrict),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use vortex_buffer::buffer;
    use vortex_error::VortexResult;

    use super::find_between;
    use crate::IntoArray;
    use crate::VortexSessionExecute;
    use crate::array_session;
    use crate::arrays::BoolArray;
    use crate::arrays::StructArray;
    use crate::dtype::DType;
    use crate::dtype::Nullability;
    use crate::dtype::PType;
    use crate::dtype::StructFields;
    use crate::expr::BoundExpression;
    use crate::expr::Expression;
    use crate::expr::and;
    use crate::expr::between;
    use crate::expr::col;
    use crate::expr::gt;
    use crate::expr::gt_eq;
    use crate::expr::lit;
    use crate::expr::lt;
    use crate::expr::lt_eq;
    use crate::scalar::Scalar;
    use crate::scalar_fn::fns::between::BetweenOptions;
    use crate::scalar_fn::fns::between::StrictComparison;

    fn scope() -> DType {
        let i32_dtype = DType::Primitive(PType::I32, Nullability::NonNullable);
        DType::Struct(
            StructFields::new(
                ["x", "y", "z"].into(),
                vec![i32_dtype.clone(), i32_dtype.clone(), i32_dtype],
            ),
            Nullability::NonNullable,
        )
    }

    fn bind(expr: Expression) -> VortexResult<BoundExpression> {
        expr.bind(&scope())
    }

    /// A null literal bound must not change the values of the rewritten expression. Kleene `AND`
    /// keeps a row false when the surviving comparison is false, so the rewrite cannot null it.
    #[test]
    fn test_null_literal_bound_is_value_preserving() -> VortexResult<()> {
        let session = array_session();
        let ctx = &mut session.create_execution_ctx();
        let data = StructArray::from_fields(&[("x", buffer![10, 1].into_array())])?.into_array();

        let null_lit = lit(Scalar::null(DType::Primitive(
            PType::I32,
            Nullability::Nullable,
        )));
        let expr = and(gt_eq(col("x"), null_lit), lt_eq(col("x"), lit(5i32)));

        let before = data
            .clone()
            .apply(&expr)?
            .execute::<BoolArray>(ctx)?
            .opt_bool_vec(ctx);

        let rewritten = find_between(expr.bind(data.dtype())?);
        let after = data
            .apply_bound(&rewritten)?
            .execute::<BoolArray>(ctx)?
            .opt_bool_vec(ctx);

        // Row 0 is false rather than null because `$.x <= 5` falsifies it on its own.
        assert_eq!(before, [Some(false), None]);
        assert_eq!(before, after);

        Ok(())
    }

    #[test]
    fn test_bad_match() -> VortexResult<()> {
        // An impossible expression
        let expr = and(lt_eq(lit(100), col("x")), gt(lit(-100), col("x")));
        let find = find_between(bind(expr)?);

        assert_eq!(
            &find,
            &bind(between(
                col("x"),
                lit(100),
                lit(-100),
                BetweenOptions {
                    lower_strict: StrictComparison::NonStrict,
                    upper_strict: StrictComparison::Strict,
                }
            ))?
        );
        Ok(())
    }

    #[test]
    fn test_match_between() -> VortexResult<()> {
        let expr = and(lt(lit(2), col("x")), gt_eq(lit(5), col("x")));
        let find = find_between(bind(expr)?);

        // 2 < x <= 5
        assert_eq!(
            &bind(between(
                col("x"),
                lit(2),
                lit(5),
                BetweenOptions {
                    lower_strict: StrictComparison::Strict,
                    upper_strict: StrictComparison::NonStrict,
                }
            ))?,
            &find
        );
        Ok(())
    }

    #[test]
    fn test_match_2_between() -> VortexResult<()> {
        let expr = and(gt_eq(col("x"), lit(2)), lt(col("x"), lit(5)));
        let find = find_between(bind(expr)?);

        // 2 <= x < 5
        assert_eq!(
            &bind(between(
                col("x"),
                lit(2),
                lit(5),
                BetweenOptions {
                    lower_strict: StrictComparison::NonStrict,
                    upper_strict: StrictComparison::Strict,
                }
            ))?,
            &find
        );
        Ok(())
    }

    #[test]
    fn test_match_3_between() -> VortexResult<()> {
        let expr = and(gt_eq(col("x"), lit(2)), gt_eq(lit(5), col("x")));
        let find = find_between(bind(expr)?);

        // 2 <= x < 5
        assert_eq!(
            &bind(between(
                col("x"),
                lit(2),
                lit(5),
                BetweenOptions {
                    lower_strict: StrictComparison::NonStrict,
                    upper_strict: StrictComparison::NonStrict,
                }
            ))?,
            &find
        );
        Ok(())
    }

    #[test]
    fn test_match_4_between() -> VortexResult<()> {
        let expr = and(gt_eq(lit(5), col("x")), lt(lit(2), col("x")));
        let find = find_between(bind(expr)?);

        // 2 < x <= 5
        assert_eq!(
            &bind(between(
                col("x"),
                lit(2),
                lit(5),
                BetweenOptions {
                    lower_strict: StrictComparison::Strict,
                    upper_strict: StrictComparison::NonStrict,
                }
            ))?,
            &find
        );
        Ok(())
    }

    #[test]
    fn test_match_5_between() -> VortexResult<()> {
        let expr = and(
            and(gt_eq(col("y"), lit(10)), gt_eq(lit(5), col("x"))),
            lt(lit(2), col("x")),
        );
        let find = find_between(bind(expr)?);

        // $.y >= 10 /\ 2 < $.x <= 5
        assert_eq!(
            &bind(and(
                gt_eq(col("y"), lit(10)),
                between(
                    col("x"),
                    lit(2),
                    lit(5),
                    BetweenOptions {
                        lower_strict: StrictComparison::Strict,
                        upper_strict: StrictComparison::NonStrict,
                    }
                )
            ))?,
            &find
        );
        Ok(())
    }

    #[test]
    fn test_match_6_between() -> VortexResult<()> {
        let expr = and(
            and(gt_eq(lit(5), col("x")), gt_eq(col("y"), lit(10))),
            lt(lit(2), col("x")),
        );
        let find = find_between(bind(expr)?);

        // $.y >= 10 /\ 2 < $.x <= 5
        assert_eq!(
            &bind(and(
                between(
                    col("x"),
                    lit(2),
                    lit(5),
                    BetweenOptions {
                        lower_strict: StrictComparison::Strict,
                        upper_strict: StrictComparison::NonStrict,
                    }
                ),
                gt_eq(col("y"), lit(10)),
            ))?,
            &find
        );
        Ok(())
    }

    #[test]
    fn test_match_between_skips_non_leaf_conjuncts() -> VortexResult<()> {
        let lower = gt_eq(col("x"), lit(2));
        let non_leaf = gt(col("y"), col("z"));
        let upper = lt(col("x"), lit(5));
        assert!(super::is_leaf_conjunct(&bind(lower.clone())?));
        assert!(!super::is_leaf_conjunct(&bind(non_leaf.clone())?));
        assert!(super::is_leaf_conjunct(&bind(upper.clone())?));

        let find = find_between(bind(and(and(lower, non_leaf.clone()), upper))?);

        assert_eq!(
            &bind(and(
                between(
                    col("x"),
                    lit(2),
                    lit(5),
                    BetweenOptions {
                        lower_strict: StrictComparison::NonStrict,
                        upper_strict: StrictComparison::Strict,
                    }
                ),
                non_leaf,
            ))?,
            &find
        );
        Ok(())
    }
}
