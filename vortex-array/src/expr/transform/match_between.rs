// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use crate::expr::BetweenOptions;
use crate::expr::Expression;
use crate::expr::StrictComparison;
use crate::expr::VTableExt;
use crate::expr::and_collect;
use crate::expr::exprs::between::Between;
use crate::expr::exprs::binary::Binary;
use crate::expr::exprs::get_item::GetItem;
use crate::expr::exprs::literal::Literal;
use crate::expr::exprs::literal::lit;
use crate::expr::exprs::operators::Operator;
use crate::expr::forms::conjuncts;

/// This pass looks for expression of the form
///      `x >= a && x < b` and converts them into x between a and b`
pub fn find_between(expr: Expression) -> Expression {
    // We search all pairs of cnfs to find any pair of expressions can be converted into a between
    // expression.
    let mut conjuncts = conjuncts(&expr);
    let mut rest = vec![];

    for idx in 0..conjuncts.len() {
        let Some(c) = conjuncts.get(idx).cloned() else {
            continue;
        };
        let mut matched = false;
        for idx2 in (idx + 1)..conjuncts.len() {
            // Since values are removed in iterations there might not be a value at idx2,
            // but all values will have been considered.
            let Some(c2) = conjuncts.get(idx2) else {
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

    and_collect(rest).unwrap_or_else(|| lit(true))
}

fn maybe_match(lhs: &Expression, rhs: &Expression) -> Option<Expression> {
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
        (false, true) => Binary.new_expr(lhs_op.swap()?, [lhs_rhs.clone(), lhs_lhs.clone()]),
        _ => return None,
    };
    let lhs_op = lhs.as_::<Binary>();
    let lhs_lhs = lhs.child(0);

    let rhs = match (rhs_lhs.is::<GetItem>(), rhs_rhs.is::<GetItem>()) {
        (true, false) => rhs.clone(),
        (false, true) => Binary.new_expr(rhs_op.swap()?, [rhs_rhs.clone(), rhs_lhs.clone()]),
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

    Some(Between.new_expr(
        BetweenOptions {
            lower_strict,
            upper_strict,
        },
        [target, lower_rhs.clone(), upper_rhs.clone()],
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
    use super::find_between;
    use crate::expr::BetweenOptions;
    use crate::expr::StrictComparison;
    use crate::expr::exprs::between::between;
    use crate::expr::exprs::binary::and;
    use crate::expr::exprs::binary::gt;
    use crate::expr::exprs::binary::gt_eq;
    use crate::expr::exprs::binary::lt;
    use crate::expr::exprs::binary::lt_eq;
    use crate::expr::exprs::get_item::col;
    use crate::expr::exprs::literal::lit;

    #[test]
    fn test_bad_match() {
        // An impossible expression
        let expr = and(lt_eq(lit(100), col("x")), gt(lit(-100), col("x")));
        let find = find_between(expr);

        assert_eq!(
            &find,
            &between(
                col("x"),
                lit(100),
                lit(-100),
                BetweenOptions {
                    lower_strict: StrictComparison::NonStrict,
                    upper_strict: StrictComparison::Strict,
                }
            )
        );
    }

    #[test]
    fn test_match_between() {
        let expr = and(lt(lit(2), col("x")), gt_eq(lit(5), col("x")));
        let find = find_between(expr);

        // 2 < x <= 5
        assert_eq!(
            &between(
                col("x"),
                lit(2),
                lit(5),
                BetweenOptions {
                    lower_strict: StrictComparison::Strict,
                    upper_strict: StrictComparison::NonStrict,
                }
            ),
            &find
        );
    }

    #[test]
    fn test_match_2_between() {
        let expr = and(gt_eq(col("x"), lit(2)), lt(col("x"), lit(5)));
        let find = find_between(expr);

        // 2 <= x < 5
        assert_eq!(
            &between(
                col("x"),
                lit(2),
                lit(5),
                BetweenOptions {
                    lower_strict: StrictComparison::NonStrict,
                    upper_strict: StrictComparison::Strict,
                }
            ),
            &find
        );
    }

    #[test]
    fn test_match_3_between() {
        let expr = and(gt_eq(col("x"), lit(2)), gt_eq(lit(5), col("x")));
        let find = find_between(expr);

        // 2 <= x < 5
        assert_eq!(
            &between(
                col("x"),
                lit(2),
                lit(5),
                BetweenOptions {
                    lower_strict: StrictComparison::NonStrict,
                    upper_strict: StrictComparison::NonStrict,
                }
            ),
            &find
        );
    }

    #[test]
    fn test_match_4_between() {
        let expr = and(gt_eq(lit(5), col("x")), lt(lit(2), col("x")));
        let find = find_between(expr);

        // 2 < x <= 5
        assert_eq!(
            &between(
                col("x"),
                lit(2),
                lit(5),
                BetweenOptions {
                    lower_strict: StrictComparison::Strict,
                    upper_strict: StrictComparison::NonStrict,
                }
            ),
            &find
        );
    }

    #[test]
    fn test_match_5_between() {
        let expr = and(
            and(gt_eq(col("y"), lit(10)), gt_eq(lit(5), col("x"))),
            lt(lit(2), col("x")),
        );
        let find = find_between(expr);

        // $.y >= 10 /\ 2 < $.x <= 5
        assert_eq!(
            &and(
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
            ),
            &find
        );
    }

    #[test]
    fn test_match_6_between() {
        let expr = and(
            and(gt_eq(lit(5), col("x")), gt_eq(col("y"), lit(10))),
            lt(lit(2), col("x")),
        );
        let find = find_between(expr);

        // $.y >= 10 /\ 2 < $.x <= 5
        assert_eq!(
            &and(
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
            ),
            &find
        );
    }
}
