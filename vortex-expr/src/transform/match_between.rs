// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_array::compute::{BetweenOptions, StrictComparison};

use crate::forms::conjuncts;
use crate::{
    BetweenExpr, BinaryExpr, BinaryVTable, ExprRef, GetItemVTable, IntoExpr, LiteralVTable,
    Operator, and, lit,
};

/// This pass looks for expression of the form
///      `x >= a && x < b` and converts them into x between a and b`
pub fn find_between(expr: ExprRef) -> ExprRef {
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

fn maybe_match(lhs: &ExprRef, rhs: &ExprRef) -> Option<ExprRef> {
    let (Some(lhs), Some(rhs)) = (lhs.as_opt::<BinaryVTable>(), rhs.as_opt::<BinaryVTable>())
    else {
        return None;
    };

    // Cannot compare to self
    if lhs.lhs().eq(lhs.rhs()) || rhs.lhs().eq(rhs.rhs()) {
        return None;
    }

    // First, get both halves to have GetItem on the left
    let lhs = match (
        lhs.lhs().is::<GetItemVTable>(),
        lhs.rhs().is::<GetItemVTable>(),
    ) {
        (true, false) => lhs.clone(),
        (false, true) => BinaryExpr::new(lhs.rhs().clone(), lhs.op().swap()?, lhs.lhs().clone()),
        _ => return None,
    };

    let rhs = match (
        rhs.lhs().is::<GetItemVTable>(),
        rhs.rhs().is::<GetItemVTable>(),
    ) {
        (true, false) => rhs.clone(),
        (false, true) => BinaryExpr::new(rhs.rhs().clone(), rhs.op().swap()?, rhs.lhs().clone()),
        _ => return None,
    };

    // Both conjuncts must reference the same GetItem column
    if !lhs.lhs().eq(rhs.lhs()) {
        return None;
    }

    let target = lhs.lhs().clone();

    // Find the lower bound
    let (lower, upper) = match (lhs.op(), rhs.op()) {
        (Operator::Lt | Operator::Lte, Operator::Gt | Operator::Gte) => (rhs, lhs),
        (Operator::Gt | Operator::Gte, Operator::Lt | Operator::Lte) => (lhs, rhs),
        _ => return None,
    };

    let lower_lit = lower.rhs().as_opt::<LiteralVTable>()?.to_expr();
    let upper_lit = upper.rhs().as_opt::<LiteralVTable>()?.to_expr();

    let lower_strict = is_strict_comparison(lower.op())?;
    let upper_strict = is_strict_comparison(upper.op())?;

    let expr = BetweenExpr::new(
        target.clone(),
        lower_lit,
        upper_lit,
        BetweenOptions {
            lower_strict,
            upper_strict,
        },
    );
    Some(expr.into_expr())
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

    use crate::transform::match_between::find_between;
    use crate::{and, between, col, gt, gt_eq, lit, lt, lt_eq};

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
