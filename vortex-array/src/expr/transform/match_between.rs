// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use crate::expr;
use crate::expr::BoundExpression;
use crate::scalar_fn::ScalarFnVTableExt;
use crate::scalar_fn::fns::between::Between;
use crate::scalar_fn::fns::between::BetweenOptions;
use crate::scalar_fn::fns::between::StrictComparison;
use crate::scalar_fn::fns::binary::Binary;
use crate::scalar_fn::fns::get_item::GetItem;
use crate::scalar_fn::fns::literal::Literal;
use crate::scalar_fn::fns::operators::Operator;

/// Combine compatible, already-typed comparisons without discarding their checked dtypes.
pub fn find_between_bound(expr: BoundExpression) -> BoundExpression {
    let mut conjuncts = Vec::new();
    split_bound_conjunction(&expr, &mut conjuncts);
    let mut rest = Vec::with_capacity(conjuncts.len());
    for idx in 0..conjuncts.len() {
        let Some(current) = conjuncts.get(idx).cloned() else {
            continue;
        };
        if !is_leaf_conjunct_bound(&current) {
            rest.push(current);
            continue;
        }
        let mut replacement = None;
        for next in (idx + 1)..conjuncts.len() {
            let Some(candidate) = conjuncts.get(next) else {
                continue;
            };
            if !is_leaf_conjunct_bound(candidate) {
                continue;
            }
            if let Some(between) = maybe_match_bound(&current, candidate) {
                replacement = Some((next, between));
                break;
            }
        }
        if let Some((next, between)) = replacement {
            rest.push(between);
            conjuncts.remove(next);
        } else {
            rest.push(current);
        }
    }
    expr::and_collect(rest).unwrap_or(expr)
}

fn split_bound_conjunction(expr: &BoundExpression, out: &mut Vec<BoundExpression>) {
    if expr.as_opt::<Binary>() == Some(&Operator::And) {
        split_bound_conjunction(expr.child(0), out);
        split_bound_conjunction(expr.child(1), out);
    } else {
        out.push(expr.clone());
    }
}

fn is_leaf_conjunct_bound(expr: &BoundExpression) -> bool {
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

fn normalized_bound_comparison(
    expr: &BoundExpression,
) -> Option<(Operator, BoundExpression, BoundExpression)> {
    let operator = *expr.as_opt::<Binary>()?;
    let lhs = expr.child(0);
    let rhs = expr.child(1);
    if lhs.is::<GetItem>() && rhs.is::<Literal>() {
        Some((operator, lhs.clone(), rhs.clone()))
    } else if rhs.is::<GetItem>() && lhs.is::<Literal>() {
        Some((operator.swap()?, rhs.clone(), lhs.clone()))
    } else {
        None
    }
}

fn maybe_match_bound(lhs: &BoundExpression, rhs: &BoundExpression) -> Option<BoundExpression> {
    let (lhs_op, lhs_target, lhs_literal) = normalized_bound_comparison(lhs)?;
    let (rhs_op, rhs_target, rhs_literal) = normalized_bound_comparison(rhs)?;
    if lhs_target != rhs_target {
        return None;
    }
    let (lower_op, lower, upper_op, upper) = match (lhs_op, rhs_op) {
        (Operator::Gt | Operator::Gte, Operator::Lt | Operator::Lte) => {
            (lhs_op, lhs_literal, rhs_op, rhs_literal)
        }
        (Operator::Lt | Operator::Lte, Operator::Gt | Operator::Gte) => {
            (rhs_op, rhs_literal, lhs_op, lhs_literal)
        }
        _ => return None,
    };
    // A null bound cannot be combined: `null AND false` is false, while BETWEEN's
    // strict validity would make the result null.
    if lower.as_opt::<Literal>()?.is_null() || upper.as_opt::<Literal>()?.is_null() {
        return None;
    }
    Between
        .try_new_bound_expr(
            BetweenOptions {
                lower_strict: is_strict_comparison(lower_op)?,
                upper_strict: is_strict_comparison(upper_op)?,
            },
            [lhs_target, lower, upper],
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

    use super::find_between_bound;
    use crate::IntoArray;
    use crate::VortexSessionExecute;
    use crate::array_session;
    use crate::arrays::BoolArray;
    use crate::arrays::StructArray;
    use crate::dtype::DType;
    use crate::dtype::Nullability;
    use crate::dtype::PType;
    use crate::expr;
    use crate::scalar::Scalar;
    use crate::scalar_fn::fns::between::BetweenOptions;
    use crate::scalar_fn::fns::between::StrictComparison;

    #[test]
    fn compatible_comparisons_form_between() -> VortexResult<()> {
        let data = StructArray::from_fields(&[("x", buffer![1, 2].into_array())])?.into_array();
        let column = expr::col("x", data.dtype().clone());
        let predicate = expr::and(
            expr::lt(expr::lit(2i32), column.clone()),
            expr::gt_eq(expr::lit(5i32), column.clone()),
        );
        let expected = expr::between(
            column,
            expr::lit(2i32),
            expr::lit(5i32),
            BetweenOptions {
                lower_strict: StrictComparison::Strict,
                upper_strict: StrictComparison::NonStrict,
            },
        );
        assert_eq!(find_between_bound(predicate), expected);
        Ok(())
    }

    #[test]
    fn null_literal_bound_keeps_kleene_and_semantics() -> VortexResult<()> {
        let session = array_session();
        let ctx = &mut session.create_execution_ctx();
        let data = StructArray::from_fields(&[("x", buffer![10, 1].into_array())])?.into_array();
        let column = expr::col("x", data.dtype().clone());
        let null = expr::lit(Scalar::null(DType::Primitive(
            PType::I32,
            Nullability::Nullable,
        )));
        let predicate = expr::and(
            expr::gt_eq(column.clone(), null),
            expr::lt_eq(column, expr::lit(5i32)),
        );

        let before = data
            .clone()
            .apply(&predicate)?
            .execute::<BoolArray>(ctx)?
            .opt_bool_vec(ctx);
        let after = data
            .apply(&find_between_bound(predicate))?
            .execute::<BoolArray>(ctx)?
            .opt_bool_vec(ctx);

        assert_eq!(before, [Some(false), None]);
        assert_eq!(before, after);
        Ok(())
    }
}
