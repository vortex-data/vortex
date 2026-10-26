// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_error::VortexExpect;

use crate::ArrayRef;
use crate::ExecutionCtx;
use crate::expr::stats::Stat;
use crate::scalar::Scalar;

pub mod binary_numeric;
pub mod cast;
pub mod consistency;
pub mod filter;
pub mod mask;
pub mod search_sorted;
pub mod take;

/// Computes `stat` for `array` through the session's aggregate function registry.
///
/// # Panics
///
/// Panics if no aggregate function computing `stat` is registered, so that a conformance suite run
/// with an incomplete session fails rather than silently skipping its statistics checks. Register
/// the aggregate functions with `vortex_aggregates::initialize`.
pub(crate) fn conformance_stat(
    array: &ArrayRef,
    stat: Stat,
    ctx: &mut ExecutionCtx,
) -> Option<Scalar> {
    let registered = stat
        .resolve_aggregate_fn(ctx.session())
        .vortex_expect("aggregate function should resolve in conformance test")
        .is_some();
    assert!(
        registered,
        "conformance tests require the aggregate function {} to be registered in the session",
        stat.aggregate_fn_id()
    );
    array
        .statistics()
        .compute_stat(stat, ctx)
        .vortex_expect("statistic should compute in conformance test")
}
