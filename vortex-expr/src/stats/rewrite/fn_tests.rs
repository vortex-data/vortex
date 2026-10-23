// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Falsification tests for individual built-in scalar functions.

use std::sync::Arc;
use std::sync::LazyLock;

use vortex_array::dtype::DType;
use vortex_array::dtype::Nullability;
use vortex_array::dtype::PType::I32;
use vortex_array::dtype::StructFields;
use vortex_array::expr::Expression;
use vortex_array::expr::and;
use vortex_array::expr::col;
use vortex_array::expr::eq;
use vortex_array::expr::gt;
use vortex_array::expr::is_not_null;
use vortex_array::expr::is_null;
use vortex_array::expr::lit;
use vortex_array::expr::lt;
use vortex_array::expr::or;
use vortex_array::expr::stats::Stat;
use vortex_array::expr::test_harness;
use vortex_array::scalar::Scalar;
use vortex_array::scalar_fn::EmptyOptions;
use vortex_array::scalar_fn::ScalarFnVTableExt;
use vortex_array::scalar_fn::internal::row_count::RowCount;
use vortex_error::VortexResult;
use vortex_functions::expr::list_contains;
use vortex_session::VortexSession;

use crate::stats::StatsSession;
use crate::stats::all_non_null;
use crate::stats::all_null;
use crate::stats::null_count;
use crate::stats::rewrite::BoundExpressionPruneExt;
use crate::stats::stat as stat_expr;

static STATS_SESSION: LazyLock<VortexSession> =
    LazyLock::new(|| VortexSession::empty().with::<StatsSession>());

fn stat(expr: Expression, stat: Stat) -> Expression {
    stat_expr(expr, stat.aggregate_fn().unwrap())
}

#[test]
fn test_is_null_falsification() -> VortexResult<()> {
    let expr = is_null(col("a"));
    let dtype = test_harness::struct_dtype();

    assert_eq!(
        expr.bind(&dtype)?.falsify(&STATS_SESSION)?,
        Some(or(eq(null_count(col("a")), lit(0u64)), all_non_null(col("a")),).bind(&dtype)?)
    );
    Ok(())
}

#[test]
fn test_is_not_null_falsification() -> VortexResult<()> {
    let expr = is_not_null(col("a"));
    let dtype = test_harness::struct_dtype();

    assert_eq!(
        expr.bind(&dtype)?.falsify(&STATS_SESSION)?,
        Some(
            or(
                eq(null_count(col("a")), RowCount.new_expr(EmptyOptions, []),),
                all_null(col("a")),
            )
            .bind(&dtype)?
        )
    );
    Ok(())
}

#[test]
pub fn list_falsification() -> VortexResult<()> {
    let expr = list_contains(
        lit(Scalar::list(
            Arc::new(DType::Primitive(I32, Nullability::NonNullable)),
            vec![1.into(), 2.into(), 3.into()],
            Nullability::NonNullable,
        )),
        col("a"),
    );
    let scope = DType::Struct(
        StructFields::new(
            ["a"].into(),
            vec![DType::Primitive(I32, Nullability::NonNullable)],
        ),
        Nullability::NonNullable,
    );

    assert_eq!(
        expr.bind(&scope)?.falsify(&STATS_SESSION)?,
        Some(
            and(
                and(
                    or(
                        lt(stat(col("a"), Stat::Max), lit(1i32)),
                        gt(stat(col("a"), Stat::Min), lit(1i32)),
                    ),
                    or(
                        lt(stat(col("a"), Stat::Max), lit(2i32)),
                        gt(stat(col("a"), Stat::Min), lit(2i32)),
                    )
                ),
                or(
                    lt(stat(col("a"), Stat::Max), lit(3i32)),
                    gt(stat(col("a"), Stat::Min), lit(3i32)),
                )
            )
            .bind(&scope)?
        )
    );
    Ok(())
}
