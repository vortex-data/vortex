// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use crate::expr::BoundExpression;
use crate::expr::bound::and_collect as bound_and_collect;
use crate::expr::bound::between as bound_between;
use crate::expr::bound::binary as bound_binary;
use crate::expr::bound::lit as bound_lit;
use crate::scalar_fn::fns::between::BetweenOptions;
use crate::scalar_fn::fns::between::StrictComparison;
use crate::scalar_fn::fns::binary::Binary;
use crate::scalar_fn::fns::get_item::GetItem;
use crate::scalar_fn::fns::literal::Literal;
use crate::scalar_fn::fns::operators::Operator;

/// Look for `x >= a AND x < b` and replace it with a bound `between` expression.
pub(crate) fn find_between_bound(expr: BoundExpression) -> BoundExpression {
    let mut conjuncts = bound_conjuncts(&expr);
    let mut rest = vec![];

    for idx in 0..conjuncts.len() {
        let Some(conjunct) = conjuncts.get(idx).cloned() else {
            continue;
        };
        let mut matched = false;
        for idx2 in (idx + 1)..conjuncts.len() {
            let Some(other) = conjuncts.get(idx2) else {
                continue;
            };
            if let Some(expr) = maybe_match_bound(&conjunct, other) {
                rest.push(expr);
                conjuncts.remove(idx2);
                matched = true;
                break;
            }
        }
        if !matched {
            rest.push(conjunct);
        }
    }

    bound_and_collect(rest).unwrap_or_else(|| bound_lit(true))
}

fn bound_conjuncts(expr: &BoundExpression) -> Vec<BoundExpression> {
    let mut conjuncts = vec![];
    bound_conjuncts_impl(expr, &mut conjuncts);
    conjuncts
}

fn bound_conjuncts_impl(expr: &BoundExpression, conjuncts: &mut Vec<BoundExpression>) {
    if expr
        .as_opt::<Binary>()
        .is_some_and(|operator| *operator == Operator::And)
    {
        bound_conjuncts_impl(expr.child(0), conjuncts);
        bound_conjuncts_impl(expr.child(1), conjuncts);
    } else {
        conjuncts.push(expr.clone());
    }
}

fn maybe_match_bound(lhs: &BoundExpression, rhs: &BoundExpression) -> Option<BoundExpression> {
    let (Some(lhs_op), Some(rhs_op)) = (lhs.as_opt::<Binary>(), rhs.as_opt::<Binary>()) else {
        return None;
    };

    let lhs_lhs = lhs.child(0);
    let lhs_rhs = lhs.child(1);
    let rhs_lhs = rhs.child(0);
    let rhs_rhs = rhs.child(1);

    if lhs_lhs.eq(lhs_rhs) || rhs_lhs.eq(rhs_rhs) {
        return None;
    }

    let lhs = match (lhs_lhs.is::<GetItem>(), lhs_rhs.is::<GetItem>()) {
        (true, false) => lhs.clone(),
        (false, true) => bound_binary(lhs_op.swap()?, lhs_rhs.clone(), lhs_lhs.clone()),
        _ => return None,
    };
    let lhs_op = lhs.as_::<Binary>();
    let lhs_lhs = lhs.child(0);

    let rhs = match (rhs_lhs.is::<GetItem>(), rhs_rhs.is::<GetItem>()) {
        (true, false) => rhs.clone(),
        (false, true) => bound_binary(rhs_op.swap()?, rhs_rhs.clone(), rhs_lhs.clone()),
        _ => return None,
    };
    let rhs_op = rhs.as_::<Binary>();
    let rhs_lhs = rhs.child(0);

    if !lhs_lhs.eq(rhs_lhs) {
        return None;
    }

    let target = lhs_lhs.clone();
    let (lower, upper) = match (lhs_op, rhs_op) {
        (Operator::Lt | Operator::Lte, Operator::Gt | Operator::Gte) => (rhs, lhs),
        (Operator::Gt | Operator::Gte, Operator::Lt | Operator::Lte) => (lhs, rhs),
        _ => return None,
    };
    let lower_op = lower.as_::<Binary>();
    let lower_rhs = lower.child(1);
    let upper_op = upper.as_::<Binary>();
    let upper_rhs = upper.child(1);

    lower_rhs.as_opt::<Literal>()?;
    upper_rhs.as_opt::<Literal>()?;

    let lower_strict = is_strict_comparison(*lower_op)?;
    let upper_strict = is_strict_comparison(*upper_op)?;

    Some(bound_between(
        target,
        lower_rhs.clone(),
        upper_rhs.clone(),
        BetweenOptions {
            lower_strict,
            upper_strict,
        },
    ))
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

    use crate::IntoArray;
    use crate::VortexSessionExecute;
    use crate::array_session;
    use crate::arrays::BoolArray;
    use crate::arrays::StructArray;
    use crate::dtype::DType;
    use crate::dtype::Nullability;
    use crate::dtype::PType;
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

    fn scope(fields: &[&str]) -> DType {
        DType::struct_(
            fields.iter().map(|name| {
                (
                    *name,
                    DType::Primitive(PType::I32, Nullability::NonNullable),
                )
            }),
            Nullability::NonNullable,
        )
    }

    fn optimize(expr: Expression, scope: &DType) -> VortexResult<BoundExpression> {
        expr.bind(scope)?.optimize_recursive()
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

        let optimized = optimize(expr, data.dtype())?;
        let after = data
            .apply_bound(&optimized)?
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
        let scope = scope(&["x"]);
        let find = optimize(expr, &scope)?;

        assert_eq!(
            find,
            between(
                col("x"),
                lit(100),
                lit(-100),
                BetweenOptions {
                    lower_strict: StrictComparison::NonStrict,
                    upper_strict: StrictComparison::Strict,
                }
            )
            .bind(&scope)?
        );
        Ok(())
    }

    #[test]
    fn test_match_between() -> VortexResult<()> {
        let expr = and(lt(lit(2), col("x")), gt_eq(lit(5), col("x")));
        let scope = scope(&["x"]);
        let find = optimize(expr, &scope)?;

        // 2 < x <= 5
        assert_eq!(
            between(
                col("x"),
                lit(2),
                lit(5),
                BetweenOptions {
                    lower_strict: StrictComparison::Strict,
                    upper_strict: StrictComparison::NonStrict,
                }
            )
            .bind(&scope)?,
            find
        );
        Ok(())
    }

    #[test]
    fn test_match_2_between() -> VortexResult<()> {
        let expr = and(gt_eq(col("x"), lit(2)), lt(col("x"), lit(5)));
        let scope = scope(&["x"]);
        let find = optimize(expr, &scope)?;

        // 2 <= x < 5
        assert_eq!(
            between(
                col("x"),
                lit(2),
                lit(5),
                BetweenOptions {
                    lower_strict: StrictComparison::NonStrict,
                    upper_strict: StrictComparison::Strict,
                }
            )
            .bind(&scope)?,
            find
        );
        Ok(())
    }

    #[test]
    fn test_match_3_between() -> VortexResult<()> {
        let expr = and(gt_eq(col("x"), lit(2)), gt_eq(lit(5), col("x")));
        let scope = scope(&["x"]);
        let find = optimize(expr, &scope)?;

        // 2 <= x < 5
        assert_eq!(
            between(
                col("x"),
                lit(2),
                lit(5),
                BetweenOptions {
                    lower_strict: StrictComparison::NonStrict,
                    upper_strict: StrictComparison::NonStrict,
                }
            )
            .bind(&scope)?,
            find
        );
        Ok(())
    }

    #[test]
    fn test_match_4_between() -> VortexResult<()> {
        let expr = and(gt_eq(lit(5), col("x")), lt(lit(2), col("x")));
        let scope = scope(&["x"]);
        let find = optimize(expr, &scope)?;

        // 2 < x <= 5
        assert_eq!(
            between(
                col("x"),
                lit(2),
                lit(5),
                BetweenOptions {
                    lower_strict: StrictComparison::Strict,
                    upper_strict: StrictComparison::NonStrict,
                }
            )
            .bind(&scope)?,
            find
        );
        Ok(())
    }

    #[test]
    fn test_match_5_between() -> VortexResult<()> {
        let expr = and(
            and(gt_eq(col("y"), lit(10)), gt_eq(lit(5), col("x"))),
            lt(lit(2), col("x")),
        );
        let scope = scope(&["x", "y"]);
        let find = optimize(expr, &scope)?;

        // $.y >= 10 /\ 2 < $.x <= 5
        assert_eq!(
            and(
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
            )
            .bind(&scope)?,
            find
        );
        Ok(())
    }

    #[test]
    fn test_match_6_between() -> VortexResult<()> {
        let expr = and(
            and(gt_eq(lit(5), col("x")), gt_eq(col("y"), lit(10))),
            lt(lit(2), col("x")),
        );
        let scope = scope(&["x", "y"]);
        let find = optimize(expr, &scope)?;

        // $.y >= 10 /\ 2 < $.x <= 5
        assert_eq!(
            and(
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
            )
            .bind(&scope)?,
            find
        );
        Ok(())
    }
}
