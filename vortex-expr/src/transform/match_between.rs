// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_array::compute::{BetweenOptions, StrictComparison};

use crate::exprs::between::Between;
use crate::exprs::binary::{Binary, and};
use crate::exprs::get_item::GetItem;
use crate::exprs::literal::{Literal, lit};
use crate::exprs::operators::Operator;
use crate::forms::conjuncts;
use crate::{Expression, VTableExt};

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

    rest.into_iter().reduce(and).unwrap_or_else(|| lit(true))
}

fn maybe_match(lhs: &Expression, rhs: &Expression) -> Option<Expression> {
    let (Some(lhs_e), Some(rhs_e)) = (lhs.as_opt::<Binary>(), rhs.as_opt::<Binary>()) else {
        return None;
    };

    // Cannot compare to self
    if lhs_e.lhs().eq(lhs_e.rhs()) || rhs_e.lhs().eq(rhs_e.rhs()) {
        return None;
    }

    // First, get both halves to have GetItem on the left
    let lhs = match (lhs_e.lhs().is::<GetItem>(), lhs_e.rhs().is::<GetItem>()) {
        (true, false) => lhs.clone(),
        (false, true) => Binary.new_expr(
            lhs_e.operator().swap()?,
            [lhs_e.rhs().clone(), lhs_e.lhs().clone()],
        ),
        _ => return None,
    };
    let lhs_e = lhs.as_::<Binary>();

    let rhs = match (rhs_e.lhs().is::<GetItem>(), rhs_e.rhs().is::<GetItem>()) {
        (true, false) => rhs.clone(),
        (false, true) => Binary.new_expr(
            rhs_e.operator().swap()?,
            [rhs_e.rhs().clone(), rhs_e.lhs().clone()],
        ),
        _ => return None,
    };
    let rhs_e = rhs.as_::<Binary>();

    // Both conjuncts must reference the same GetItem column
    if !lhs_e.lhs().eq(rhs_e.lhs()) {
        return None;
    }

    let target = lhs_e.lhs().clone();

    // Find the lower bound
    let (lower, upper) = match (lhs_e.operator(), rhs_e.operator()) {
        (Operator::Lt | Operator::Lte, Operator::Gt | Operator::Gte) => (rhs, lhs),
        (Operator::Gt | Operator::Gte, Operator::Lt | Operator::Lte) => (lhs, rhs),
        _ => return None,
    };
    let lower_e = lower.as_::<Binary>();
    let upper_e = upper.as_::<Binary>();

    // Ensure bounds are literals
    let _ = lower_e.rhs().as_opt::<Literal>()?;
    let _ = upper_e.rhs().as_opt::<Literal>()?;

    let lower_strict = is_strict_comparison(lower_e.operator())?;
    let upper_strict = is_strict_comparison(upper_e.operator())?;

    Some(Between.new_expr(
        BetweenOptions {
            lower_strict,
            upper_strict,
        },
        [target, lower_e.rhs().clone(), upper_e.rhs().clone()],
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
    use vortex_array::compute::{BetweenOptions, StrictComparison};

    use super::find_between;
    use crate::exprs::between::between;
    use crate::exprs::binary::{and, gt, gt_eq, lt, lt_eq};
    use crate::exprs::get_item::col;
    use crate::exprs::literal::lit;

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
