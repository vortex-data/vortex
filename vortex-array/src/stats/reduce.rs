// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Helpers for reducing `stat(expr, aggregate)` nodes through scalar functions.
//!
//! A `stat` node summarizes its input over the current scope (a zone, a file) and is only ever a
//! sound approximation: `min` is a lower bound and `max` an upper bound of the input's non-null
//! values. That lets a scalar function push a `stat` parent down to its children with
//! [`ScalarFnVTable::reduce_parent`](crate::scalar_fn::ScalarFnVTable::reduce_parent), e.g.
//! `stat(a + b, max)` becomes `stat(a, max) + stat(b, max)`.
//!
//! Booleans order `false < true`, so predicate proofs are bounds too: `stat(p, max) = false`
//! proves that no row of `p` is true, and `stat(p, min) = true` proves that no row is false.

use std::slice;

use vortex_error::VortexExpect;
use vortex_error::VortexResult;

use crate::aggregate_fn::AggregateFnRef;
use crate::expr::stats::Stat;
use crate::scalar_fn::EmptyOptions;
use crate::scalar_fn::ReduceNode;
use crate::scalar_fn::ScalarFnVTableExt;
use crate::scalar_fn::fns::binary::Binary;
use crate::scalar_fn::fns::not::Not;
use crate::scalar_fn::fns::operators::Operator;
use crate::scalar_fn::fns::stat::StatFn;
use crate::scalar_fn::fns::stat::StatOptions;

/// Which side of the input's values a `min` or `max` stat bounds.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Bound {
    /// `stat(_, min)`: at most every non-null value.
    Lower,
    /// `stat(_, max)`: at least every non-null value.
    Upper,
}

impl Bound {
    /// The bound requested by `parent` if it is a `min` or `max` stat node.
    pub(crate) fn of_parent<T: ReduceNode>(parent: &T) -> Option<Self> {
        match Stat::from_aggregate_fn(stat_aggregate(parent)?)? {
            Stat::Min => Some(Self::Lower),
            Stat::Max => Some(Self::Upper),
            _ => None,
        }
    }

    /// The opposite bound, for inputs the function is decreasing in.
    pub(crate) fn flip(self) -> Self {
        match self {
            Self::Lower => Self::Upper,
            Self::Upper => Self::Lower,
        }
    }

    /// `stat(input, min)` or `stat(input, max)`, or `None` if `input`'s dtype has no min/max.
    pub(crate) fn of<T: ReduceNode>(self, input: &T) -> VortexResult<Option<T>> {
        let stat = match self {
            Self::Lower => Stat::Min,
            Self::Upper => Stat::Max,
        };
        stat_node(
            input,
            stat.aggregate_fn()
                .vortex_expect("min and max are aggregate-backed stats"),
        )
    }
}

/// The aggregate of `parent` if it is a `stat` node.
pub(crate) fn stat_aggregate<T: ReduceNode>(parent: &T) -> Option<&AggregateFnRef> {
    parent
        .scalar_fn()?
        .as_opt::<StatFn>()
        .map(StatOptions::aggregate_fn)
}

/// `stat(input, aggregate_fn)`, or `None` if the aggregate does not support `input`'s dtype.
pub(crate) fn stat_node<T: ReduceNode>(
    input: &T,
    aggregate_fn: AggregateFnRef,
) -> VortexResult<Option<T>> {
    if aggregate_fn.state_dtype(&input.node_dtype()?).is_none() {
        return Ok(None);
    }
    input
        .new_node(
            StatFn.bind(StatOptions::new(aggregate_fn)),
            slice::from_ref(input),
        )
        .map(Some)
}

/// `operator(lhs, rhs)`.
pub(crate) fn binary_node<T: ReduceNode>(operator: Operator, lhs: T, rhs: T) -> VortexResult<T> {
    lhs.new_node(Binary.bind(operator), &[lhs.clone(), rhs])
}

/// `not(input)`.
pub(crate) fn not_node<T: ReduceNode>(input: &T) -> VortexResult<T> {
    input.new_node(Not.bind(EmptyOptions), slice::from_ref(input))
}

#[cfg(test)]
mod tests {
    use vortex_error::VortexExpect;
    use vortex_error::VortexResult;

    use crate::dtype::DType;
    use crate::dtype::Nullability;
    use crate::dtype::PType;
    use crate::dtype::StructFields;
    use crate::expr::Expression;
    use crate::expr::and;
    use crate::expr::cast;
    use crate::expr::checked_add;
    use crate::expr::col;
    use crate::expr::eq;
    use crate::expr::gt;
    use crate::expr::is_not_null;
    use crate::expr::lit;
    use crate::expr::lt;
    use crate::expr::lt_eq;
    use crate::expr::not;
    use crate::expr::stats::Stat;
    use crate::scalar::Scalar;
    use crate::stats::all_non_null;
    use crate::stats::all_null;
    use crate::stats::stat;

    fn scope() -> DType {
        DType::Struct(
            StructFields::from_iter([
                ("a", DType::Primitive(PType::I32, Nullability::NonNullable)),
                ("b", DType::Primitive(PType::I32, Nullability::NonNullable)),
                ("c", DType::Primitive(PType::I32, Nullability::Nullable)),
                ("d", DType::Primitive(PType::I32, Nullability::Nullable)),
                ("f", DType::Primitive(PType::F64, Nullability::NonNullable)),
            ]),
            Nullability::NonNullable,
        )
    }

    fn min(expr: Expression) -> Expression {
        stat(expr, Stat::Min.aggregate_fn().vortex_expect("min"))
    }

    fn max(expr: Expression) -> Expression {
        stat(expr, Stat::Max.aggregate_fn().vortex_expect("max"))
    }

    /// The `min`/`max` of a literal is the literal itself, as a nullable stat value.
    fn nlit(value: i32) -> Expression {
        lit(Scalar::primitive(value, Nullability::Nullable))
    }

    fn assert_reduces(input: Expression, expected: Expression) -> VortexResult<()> {
        let scope = scope();
        assert_eq!(
            input.bind(&scope)?.optimize_recursive()?,
            expected.bind(&scope)?
        );
        Ok(())
    }

    #[test]
    fn max_of_add() -> VortexResult<()> {
        assert_reduces(
            max(checked_add(col("a"), lit(1i32))),
            checked_add(max(col("a")), nlit(1)),
        )
    }

    #[test]
    fn falsify_add_comparison() -> VortexResult<()> {
        // `x + 1 > y` has no true row when `max(x + 1) > min(y)` is false.
        assert_reduces(
            max(gt(checked_add(col("a"), lit(1i32)), col("b"))),
            gt(checked_add(max(col("a")), nlit(1)), min(col("b"))),
        )
    }

    #[test]
    fn satisfy_negated_conjunction() -> VortexResult<()> {
        // `min(not p) = not max(p)` and `max(p and q) = max(p) and max(q)`.
        assert_reduces(
            min(not(and(gt(col("a"), lit(10i32)), lt(col("a"), lit(50i32))))),
            not(and(
                gt(max(col("a")), nlit(10)),
                lt(min(col("a")), nlit(50)),
            )),
        )
    }

    #[test]
    fn equality_overlap() -> VortexResult<()> {
        assert_reduces(
            max(eq(col("a"), lit(5i32))),
            and(
                lt_eq(min(col("a")), nlit(5)),
                lt_eq(nlit(5), max(col("a"))),
            ),
        )
    }

    #[test]
    fn is_not_null_bounds() -> VortexResult<()> {
        assert_reduces(min(is_not_null(col("c"))), all_non_null(col("c")))?;
        assert_reduces(max(is_not_null(col("c"))), not(all_null(col("c"))))
    }

    #[test]
    fn all_non_null_of_expression() -> VortexResult<()> {
        // `validity(c + d) = is_not_null(c) and is_not_null(d)`, whose lower bound is then reduced.
        assert_reduces(
            all_non_null(checked_add(col("c"), col("d"))),
            and(all_non_null(col("c")), all_non_null(col("d"))),
        )?;
        assert_reduces(
            all_non_null(checked_add(col("a"), col("b"))),
            lit(Scalar::bool(true, Nullability::Nullable)),
        )
    }

    #[test]
    fn max_of_cast() -> VortexResult<()> {
        let i64_dtype = DType::Primitive(PType::I64, Nullability::NonNullable);
        assert_reduces(
            max(cast(col("a"), i64_dtype.clone())),
            cast(max(col("a")), i64_dtype.as_nullable()),
        )
    }

    #[test]
    fn float_comparison_is_not_reduced() -> VortexResult<()> {
        let expr = max(gt(col("f"), lit(1.0f64)));
        assert_reduces(expr.clone(), expr)
    }
}
