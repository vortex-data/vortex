// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Rewrites `x >= a AND x < b` into `x BETWEEN a AND b`.

use std::ptr;

use vortex_array::expr::BoundExpression;
use vortex_array::scalar_fn::ScalarFnVTableExt;
use vortex_array::scalar_fn::builtin::binary::Binary;
use vortex_array::scalar_fn::builtin::get_item::GetItem;
use vortex_array::scalar_fn::builtin::literal::Literal;
use vortex_array::scalar_fn::builtin::operators::Operator;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_functions::between::Between;
use vortex_functions::between::BetweenOptions;
use vortex_functions::between::StrictComparison;

/// Merge pairs of range comparisons on the same column within any conjunction of `expr` into a
/// single [`Between`].
///
/// This is not an optimizer rule because it is worth running only once, on a scan's filter.
/// Subtrees without a match keep their identity, so an expression without any match is returned
/// unchanged.
pub fn find_between(expr: BoundExpression) -> VortexResult<BoundExpression> {
    Ok(try_find_between(&expr)?.unwrap_or(expr))
}

fn try_find_between(expr: &BoundExpression) -> VortexResult<Option<BoundExpression>> {
    let children = expr.children();
    let mut new_children: Option<Vec<BoundExpression>> = None;
    for (idx, child) in children.iter().enumerate() {
        if let Some(rewritten) = try_find_between(child)? {
            new_children
                .get_or_insert_with(|| children[..idx].to_vec())
                .push(rewritten);
        } else if let Some(new_children) = new_children.as_mut() {
            new_children.push(child.clone());
        }
    }
    let rewritten = new_children
        .map(|children| expr.clone().with_children(children))
        .transpose()?;

    let current = rewritten.as_ref().unwrap_or(expr);
    if is_and(current)
        && let Some(merged) = merge_straddling_pairs(current)?
    {
        return Ok(Some(merged));
    }
    Ok(rewritten)
}

/// Merge pairs of range comparisons that sit on opposite sides of the `AND` node `expr`.
///
/// Children are rewritten before their parents, so a pair within one side was already merged at
/// the `AND` node joining it. Only considering straddling pairs keeps the pass linear in the size
/// of the conjunction instead of re-pairing every conjunct at every level. The rest of the tree
/// keeps its shape: the lhs conjunct becomes the between and the rhs conjunct is removed.
fn merge_straddling_pairs(expr: &BoundExpression) -> VortexResult<Option<BoundExpression>> {
    let mut current: Option<BoundExpression> = None;

    loop {
        let expr = current.as_ref().unwrap_or(expr);
        if !is_and(expr) {
            break;
        }
        let lhs = expr.child(0);
        let rhs = expr.child(1);

        let Some((lower, upper, between)) = find_conjunct(rhs, &mut |rhs_conjunct| {
            let rhs_cmp = RangeComparison::try_new(rhs_conjunct)?;
            find_conjunct(lhs, &mut |lhs_conjunct| {
                let lhs_cmp = RangeComparison::try_new(lhs_conjunct)?;
                let between = lhs_cmp.between(&rhs_cmp)?;
                Some((lhs_conjunct, rhs_conjunct, between))
            })
        }) else {
            break;
        };

        tracing::debug!(%lower, %upper, "merged range comparisons into {between}");

        let new_lhs = rewrite_conjunct(lhs, lower, Some(&between))?
            .vortex_expect("matched conjunct is part of the lhs")
            .vortex_expect("replaced conjunct is never removed");
        let new_rhs = rewrite_conjunct(rhs, upper, None)?
            .vortex_expect("matched conjunct is part of the rhs");

        current = Some(match new_rhs {
            Some(new_rhs) => expr.clone().with_children([new_lhs, new_rhs])?,
            None => new_lhs,
        });
    }

    Ok(current)
}

fn is_and(expr: &BoundExpression) -> bool {
    expr.as_opt::<Binary>() == Some(&Operator::And)
}

/// Return the first `Some` produced by `f` over the conjuncts of the `AND` tree rooted at `expr`.
fn find_conjunct<'a, T>(
    expr: &'a BoundExpression,
    f: &mut impl FnMut(&'a BoundExpression) -> Option<T>,
) -> Option<T> {
    if is_and(expr) {
        find_conjunct(expr.child(0), f).or_else(|| find_conjunct(expr.child(1), f))
    } else {
        f(expr)
    }
}

/// Replace the conjunct `target`, identified by address, in the `AND` tree rooted at `expr`, or
/// remove it when `replacement` is `None`.
///
/// Returns `None` when `target` is not a conjunct of `expr`, otherwise the rewritten tree, which
/// is itself `None` when removing `target` leaves nothing behind.
fn rewrite_conjunct(
    expr: &BoundExpression,
    target: &BoundExpression,
    replacement: Option<&BoundExpression>,
) -> VortexResult<Option<Option<BoundExpression>>> {
    if ptr::eq(expr, target) {
        return Ok(Some(replacement.cloned()));
    }
    if !is_and(expr) {
        return Ok(None);
    }

    for idx in 0..2 {
        let Some(new_child) = rewrite_conjunct(expr.child(idx), target, replacement)? else {
            continue;
        };
        let sibling = expr.child(1 - idx).clone();
        let rewritten = match new_child {
            Some(new_child) if idx == 0 => expr.clone().with_children([new_child, sibling])?,
            Some(new_child) => expr.clone().with_children([sibling, new_child])?,
            None => sibling,
        };
        return Ok(Some(Some(rewritten)));
    }

    Ok(None)
}

/// A conjunct `column op literal` (in either order) that bounds `column` from one side.
struct RangeComparison<'a> {
    column: &'a BoundExpression,
    literal: &'a BoundExpression,
    is_lower: bool,
    strict: StrictComparison,
}

impl<'a> RangeComparison<'a> {
    fn try_new(expr: &'a BoundExpression) -> Option<Self> {
        let operator = *expr.as_opt::<Binary>()?;
        let lhs = expr.child(0);
        let rhs = expr.child(1);

        // Normalize the column to the left of the operator.
        let (column, literal, operator) = if lhs.is::<GetItem>() && rhs.is::<Literal>() {
            (lhs, rhs, operator)
        } else if rhs.is::<GetItem>() && lhs.is::<Literal>() {
            (rhs, lhs, operator.swap()?)
        } else {
            return None;
        };

        let (is_lower, strict) = match operator {
            Operator::Gt => (true, StrictComparison::Strict),
            Operator::Gte => (true, StrictComparison::NonStrict),
            Operator::Lt => (false, StrictComparison::Strict),
            Operator::Lte => (false, StrictComparison::NonStrict),
            _ => return None,
        };

        Some(Self {
            column,
            literal,
            is_lower,
            strict,
        })
    }

    /// Combine two comparisons bounding the same column from opposite sides.
    fn between(&self, other: &Self) -> Option<BoundExpression> {
        if self.is_lower == other.is_lower || self.column != other.column {
            return None;
        }
        let (lower, upper) = if self.is_lower {
            (self, other)
        } else {
            (other, self)
        };

        // A between that fails to type-check leaves the comparisons in place.
        Between
            .try_new_bound_expr(
                BetweenOptions {
                    lower_strict: lower.strict,
                    upper_strict: upper.strict,
                },
                [
                    self.column.clone(),
                    lower.literal.clone(),
                    upper.literal.clone(),
                ],
            )
            .ok()
    }
}

#[cfg(test)]
mod tests {
    use std::sync::LazyLock;

    use rstest::rstest;
    use vortex_array::IntoArray;
    use vortex_array::VortexSessionExecute;
    use vortex_array::arrays::BoolArray;
    use vortex_array::arrays::StructArray;
    use vortex_array::dtype::DType;
    use vortex_array::dtype::Nullability;
    use vortex_array::dtype::PType;
    use vortex_array::dtype::StructFields;
    use vortex_array::expr::Expression;
    use vortex_array::expr::and;
    use vortex_array::expr::col;
    use vortex_array::expr::gt;
    use vortex_array::expr::gt_eq;
    use vortex_array::expr::lit;
    use vortex_array::expr::lt;
    use vortex_array::expr::lt_eq;
    use vortex_array::expr::or;
    use vortex_array::scalar::Scalar;
    use vortex_buffer::buffer;
    use vortex_error::VortexResult;
    use vortex_functions::between::BetweenOptions;
    use vortex_functions::between::StrictComparison;
    use vortex_functions::expr::between;
    use vortex_session::VortexSession;

    use super::find_between;

    static SESSION: LazyLock<VortexSession> = LazyLock::new(VortexSession::empty);

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

    fn between_x(lower: i32, upper: i32, lower_strict: bool, upper_strict: bool) -> Expression {
        let strictness = |strict| {
            if strict {
                StrictComparison::Strict
            } else {
                StrictComparison::NonStrict
            }
        };
        between(
            col("x"),
            lit(lower),
            lit(upper),
            BetweenOptions {
                lower_strict: strictness(lower_strict),
                upper_strict: strictness(upper_strict),
            },
        )
    }

    #[rstest]
    // An impossible range is still a between.
    #[case::bad_match(
        and(lt_eq(lit(100), col("x")), gt(lit(-100), col("x"))),
        between_x(100, -100, false, true)
    )]
    #[case::literals_on_left(
        and(lt(lit(2), col("x")), gt_eq(lit(5), col("x"))),
        between_x(2, 5, true, false)
    )]
    #[case::columns_on_left(
        and(gt_eq(col("x"), lit(2)), lt(col("x"), lit(5))),
        between_x(2, 5, false, true)
    )]
    #[case::mixed_sides(
        and(gt_eq(col("x"), lit(2)), gt_eq(lit(5), col("x"))),
        between_x(2, 5, false, false)
    )]
    #[case::upper_first(
        and(gt_eq(lit(5), col("x")), lt(lit(2), col("x"))),
        between_x(2, 5, true, false)
    )]
    #[case::other_conjunct_first(
        and(
            and(gt_eq(col("y"), lit(10)), gt_eq(lit(5), col("x"))),
            lt(lit(2), col("x")),
        ),
        and(gt_eq(col("y"), lit(10)), between_x(2, 5, true, false))
    )]
    #[case::other_conjunct_last(
        and(
            and(gt_eq(lit(5), col("x")), gt_eq(col("y"), lit(10))),
            lt(lit(2), col("x")),
        ),
        and(between_x(2, 5, true, false), gt_eq(col("y"), lit(10)))
    )]
    #[case::skips_non_leaf_conjuncts(
        and(
            and(gt_eq(col("x"), lit(2)), gt(col("y"), col("z"))),
            lt(col("x"), lit(5))
        ),
        and(between_x(2, 5, false, true), gt(col("y"), col("z")))
    )]
    #[case::pair_within_one_side(
        and(
            and(gt_eq(col("x"), lit(2)), lt(col("x"), lit(5))),
            gt(col("y"), col("z"))
        ),
        and(between_x(2, 5, false, true), gt(col("y"), col("z")))
    )]
    #[case::under_or(
        or(
            and(gt_eq(col("x"), lit(2)), lt(col("x"), lit(5))),
            gt(col("y"), col("z"))
        ),
        or(between_x(2, 5, false, true), gt(col("y"), col("z")))
    )]
    #[case::no_pair(
        and(gt_eq(col("x"), lit(2)), lt(col("y"), lit(5))),
        and(gt_eq(col("x"), lit(2)), lt(col("y"), lit(5)))
    )]
    #[case::same_direction(
        and(gt_eq(col("x"), lit(2)), gt(col("x"), lit(5))),
        and(gt_eq(col("x"), lit(2)), gt(col("x"), lit(5)))
    )]
    fn finds_between(#[case] expr: Expression, #[case] expected: Expression) -> VortexResult<()> {
        let scope = scope();
        assert_eq!(find_between(expr.bind(&scope)?)?, expected.bind(&scope)?);
        Ok(())
    }

    /// Every pair straddling a single `AND` node is merged.
    #[test]
    fn merges_many_pairs() -> VortexResult<()> {
        let scope = scope();
        let lowers = ["x", "y", "z"].map(|name| gt_eq(col(name), lit(0)));
        let uppers = ["x", "y", "z"].map(|name| lt(col(name), lit(10)));
        let [lx, ly, lz] = lowers;
        let [ux, uy, uz] = uppers;
        let expr = and(and(and(lx, ly), lz), and(and(uz, uy), ux));

        let range = |name| {
            between(
                col(name),
                lit(0),
                lit(10),
                BetweenOptions {
                    lower_strict: StrictComparison::NonStrict,
                    upper_strict: StrictComparison::Strict,
                },
            )
        };
        let expected = and(and(range("x"), range("y")), range("z"));

        assert_eq!(find_between(expr.bind(&scope)?)?, expected.bind(&scope)?);
        Ok(())
    }

    /// A null literal bound must not change the values of the rewritten expression. Kleene `AND`
    /// keeps a row false when the surviving comparison is false, so the rewrite cannot null it.
    #[test]
    fn null_literal_bound_is_value_preserving() -> VortexResult<()> {
        let ctx = &mut SESSION.create_execution_ctx();
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

        let rewritten = find_between(expr.bind(data.dtype())?)?;
        let after = data
            .apply_bound(&rewritten)?
            .execute::<BoolArray>(ctx)?
            .opt_bool_vec(ctx);

        // Row 0 is false rather than null because `$.x <= 5` falsifies it on its own.
        assert_eq!(before, [Some(false), None]);
        assert_eq!(before, after);

        Ok(())
    }
}
