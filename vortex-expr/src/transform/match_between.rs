use vortex_array::compute::{BetweenOptions, StrictComparison};

use crate::between::Between;
use crate::forms::cnf::cnf;
use crate::{and, lit, BinaryExpr, ExprRef, GetItem, Literal, Operator};

/// This pass looks for expression of the form
///      `x >= a && x < b` and converts them into x between a and b`
pub fn find_between(expr: ExprRef) -> ExprRef {
    // We search all pairs of cnfs to find any pair of expressions can be converted into a between
    // expression.
    let mut conjuncts = cnf(expr.clone());
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
    let (Some(lhs), Some(rhs)) = (
        lhs.as_any().downcast_ref::<BinaryExpr>(),
        rhs.as_any().downcast_ref::<BinaryExpr>(),
    ) else {
        return None;
    };

    // Cannot compare to self
    if lhs.lhs().eq(lhs.rhs()) || rhs.lhs().eq(rhs.rhs()) {
        return None;
    }

    // Extract pairs of comparison of the form (left left_op eq) and (eq right_op right)
    let (eq, left, left_op, right, right_op) = if GetItem::is(lhs.lhs()) && lhs.lhs().eq(rhs.lhs())
    {
        (
            lhs.lhs().clone(),
            lhs.rhs().clone(),
            lhs.op().swap(),
            rhs.rhs().clone(),
            rhs.op(),
        )
    } else if GetItem::is(lhs.lhs()) && lhs.lhs().eq(rhs.rhs()) {
        (
            lhs.lhs().clone(),
            lhs.rhs().clone(),
            lhs.op().swap(),
            rhs.lhs().clone(),
            rhs.op().swap(),
        )
    } else if GetItem::is(lhs.rhs()) && lhs.rhs().eq(rhs.lhs()) {
        (
            lhs.rhs().clone(),
            lhs.lhs().clone(),
            lhs.op(),
            rhs.rhs().clone(),
            rhs.op(),
        )
    } else if GetItem::is(lhs.rhs()) && lhs.rhs().eq(rhs.rhs()) {
        (
            lhs.rhs().clone(),
            lhs.lhs().clone(),
            lhs.op(),
            rhs.lhs().clone(),
            rhs.op().swap(),
        )
    } else {
        return None;
    };

    // Find the greater op.
    let (Some(left_lit), Some(right_lit)) =
        (Literal::maybe_from(&left), Literal::maybe_from(&right))
    else {
        return None;
    };

    let (left, left_op, right, right_op) = if left_lit.value() > right_lit.value() {
        (right, right_op, left, left_op)
    } else {
        (left, left_op, right, right_op)
    };

    // Check if the operators form an inequality.
    let (left_op, right_op) = if let (Some(left_op), Some(right_op)) = (
        maybe_strict_comparison(left_op),
        maybe_strict_comparison(right_op),
    ) {
        (left_op, right_op)
    } else if let (Some(left_op), Some(right_op)) = (
        maybe_strict_comparison(left_op.swap()),
        maybe_strict_comparison(right_op.swap()),
    ) {
        (left_op, right_op)
    } else {
        return None;
    };

    let expr = Between::between(
        eq.clone(),
        left,
        right,
        BetweenOptions {
            lower_strict: left_op,
            upper_strict: right_op,
        },
    );
    Some(expr)
}

fn maybe_strict_comparison(op: Operator) -> Option<StrictComparison> {
    match op {
        Operator::Lt => Some(StrictComparison::Strict),
        Operator::Lte => Some(StrictComparison::NonStrict),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use vortex_array::compute::{BetweenOptions, StrictComparison};

    use crate::between::Between;
    use crate::transform::match_between::find_between;
    use crate::{and, col, gt_eq, lit, lt};

    #[test]
    fn test_match_between() {
        let expr = and(lt(lit(2), col("x")), gt_eq(lit(5), col("x")));
        let find = find_between(expr);

        // 2 < x <= 5
        assert_eq!(
            &Between::between(
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
            &Between::between(
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
            &Between::between(
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
            &Between::between(
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

        println!("{}", find);

        // $.y >= 10 /\ 2 < $.x <= 5
        assert_eq!(
            &and(
                gt_eq(col("y"), lit(10)),
                Between::between(
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

        println!("{}", find);

        // $.y >= 10 /\ 2 < $.x <= 5
        assert_eq!(
            &and(
                Between::between(
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
