// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_array::dtype::DType;
use vortex_array::expr::BoundExpression;
use vortex_array::expr::bound::and_collect;
use vortex_scan::DynamicFilter;

pub mod arrow;
mod execution;
mod filter;
pub mod layout;
mod limit;
pub mod multi;
pub mod repeated_scan;
pub mod scan_builder;
pub mod split_by;
mod splits;
mod tasks;
#[cfg(test)]
mod test;
pub mod v2;

/// A heuristic for an ideal split size.
///
/// We don't actually know if this is right, but it is probably a good estimate.
const IDEAL_SPLIT_SIZE: u64 = 100_000;

/// Snapshot advisory predicates without allowing unavailable or unsupported hints to fail scans.
fn current_filter(
    exact: Option<BoundExpression>,
    dynamic_filter: Option<&dyn DynamicFilter>,
    dtype: &DType,
) -> Option<BoundExpression> {
    let current = dynamic_filter
        .and_then(|hint| hint.current())
        .and_then(|expr| expr.bind(dtype).ok())
        .filter(|expr| matches!(expr.dtype(), DType::Bool(_)))
        .and_then(|expr| expr.optimize_recursive().ok());
    and_collect(exact.into_iter().chain(current))
}
