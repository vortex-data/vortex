// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Falsification tests for individual built-in scalar functions.

use std::sync::LazyLock;

use vortex_array::expr::col;
use vortex_array::expr::eq;
use vortex_array::expr::is_not_null;
use vortex_array::expr::is_null;
use vortex_array::expr::lit;
use vortex_array::expr::or;
use vortex_array::expr::test_harness;
use vortex_array::scalar_fn::EmptyOptions;
use vortex_array::scalar_fn::ScalarFnVTableExt;
use vortex_array::scalar_fn::internal::row_count::RowCount;
use vortex_error::VortexResult;
use vortex_session::VortexSession;

use crate::stats::StatsSession;
use crate::stats::all_non_null;
use crate::stats::all_null;
use crate::stats::null_count;
use crate::stats::rewrite::BoundExpressionPruneExt;

static STATS_SESSION: LazyLock<VortexSession> =
    LazyLock::new(|| VortexSession::empty().with::<StatsSession>());

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
