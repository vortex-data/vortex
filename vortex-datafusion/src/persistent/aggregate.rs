// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Aggregation pushdown for Vortex scans.
//!
//! Vortex files store exact per-column statistics (row count, `min`, `max`,
//! `null_count`, and `sum`) in their footer. For an ungrouped aggregation whose
//! inputs are exact, the result can be computed directly from those statistics
//! without scanning any data.
//!
//! DataFusion already ships an [`AggregateStatistics`] rule that folds ungrouped
//! `COUNT`, `MIN`, and `MAX` from exact statistics, but its built-in `SUM`
//! aggregate does not implement statistics-based resolution. This rule extends
//! that behavior for Vortex scans so that ungrouped `SUM` (and any mix of
//! `COUNT`/`MIN`/`MAX`/`SUM`) is also answered from the file statistics.
//!
//! Aggregations over a filtered scan are intentionally left untouched: a pushed
//! filter makes the scan statistics inexact (see
//! [`FileScanConfig::statistics`]), so the rule does not fire and DataFusion
//! computes the aggregate over the filtered, projected Vortex scan as usual.
//!
//! [`AggregateStatistics`]: https://docs.rs/datafusion/latest/datafusion/physical_optimizer/aggregate_statistics/struct.AggregateStatistics.html
//! [`FileScanConfig::statistics`]: datafusion_datasource::file_scan_config::FileScanConfig

use std::sync::Arc;

use datafusion_common::Result;
use datafusion_common::ScalarValue;
use datafusion_common::Statistics;
use datafusion_common::config::ConfigOptions;
use datafusion_common::stats::Precision;
use datafusion_common::tree_node::Transformed;
use datafusion_common::tree_node::TransformedResult;
use datafusion_common::tree_node::TreeNode;
use datafusion_datasource::file_scan_config::FileScanConfig;
use datafusion_datasource::source::DataSourceExec;
use datafusion_physical_expr::PhysicalExpr;
use datafusion_physical_expr::expressions::CastExpr;
use datafusion_physical_expr::expressions::Column;
use datafusion_physical_optimizer::PhysicalOptimizerRule;
use datafusion_physical_plan::ExecutionPlan;
use datafusion_physical_plan::aggregates::AggregateExec;
use datafusion_physical_plan::aggregates::AggregateMode;
use datafusion_physical_plan::expressions::lit;
use datafusion_physical_plan::placeholder_row::PlaceholderRowExec;
use datafusion_physical_plan::projection::ProjectionExec;
use datafusion_physical_plan::projection::ProjectionExpr;
use datafusion_physical_plan::udaf::AggregateFunctionExpr;
use datafusion_physical_plan::udaf::StatisticsArgs;

use crate::persistent::source::VortexSource;

/// Physical optimizer rule that answers ungrouped aggregations over a Vortex
/// scan from the file statistics, eliminating the scan.
///
/// Register the rule on a [`SessionStateBuilder`] to enable it:
///
/// ```no_run
/// use std::sync::Arc;
///
/// use datafusion::execution::SessionStateBuilder;
/// use vortex_datafusion::VortexAggregatePushdown;
///
/// let state = SessionStateBuilder::new()
///     .with_default_features()
///     .with_physical_optimizer_rule(Arc::new(VortexAggregatePushdown::new()))
///     .build();
/// # let _ = state;
/// ```
///
/// The rule folds ungrouped, non-distinct `COUNT`, `MIN`, `MAX`, and `SUM`
/// aggregations into a single literal row when every required statistic is
/// [`Precision::Exact`]. It only rewrites plans whose scan is a [`VortexSource`]
/// and leaves all other aggregations unchanged.
///
/// [`SessionStateBuilder`]: https://docs.rs/datafusion/latest/datafusion/execution/session_state/struct.SessionStateBuilder.html
#[derive(Debug, Default)]
pub struct VortexAggregatePushdown {}

impl VortexAggregatePushdown {
    /// Creates a new [`VortexAggregatePushdown`] rule.
    pub fn new() -> Self {
        Self {}
    }
}

impl PhysicalOptimizerRule for VortexAggregatePushdown {
    fn optimize(
        &self,
        plan: Arc<dyn ExecutionPlan>,
        _config: &ConfigOptions,
    ) -> Result<Arc<dyn ExecutionPlan>> {
        plan.transform_down(|node| match try_rewrite(&node)? {
            Some(rewritten) => Ok(Transformed::yes(rewritten)),
            None => Ok(Transformed::no(node)),
        })
        .data()
    }

    fn name(&self) -> &str {
        "vortex_aggregate_pushdown"
    }

    /// The rewritten projection may report different nullability than the
    /// original aggregate output, so skip the post-rule schema equality check
    /// (mirroring DataFusion's own `AggregateStatistics` rule).
    fn schema_check(&self) -> bool {
        false
    }
}

/// Attempts to rewrite a single ungrouped aggregate over a Vortex scan into a
/// literal row computed from statistics. Returns `None` when the node is not a
/// foldable aggregate or when any required statistic is not exact.
fn try_rewrite(node: &Arc<dyn ExecutionPlan>) -> Result<Option<Arc<dyn ExecutionPlan>>> {
    let Some(agg) = node.as_any().downcast_ref::<AggregateExec>() else {
        return Ok(None);
    };
    if !agg.group_expr().is_empty() {
        return Ok(None);
    }

    // Locate the aggregate that reads raw (un-aggregated) input. For `Single`
    // modes that is the node itself; for `Final` modes it is the `Partial`
    // aggregate sitting below it.
    let raw_agg_plan = match agg.mode() {
        AggregateMode::Single | AggregateMode::SinglePartitioned => Arc::clone(node),
        AggregateMode::Final | AggregateMode::FinalPartitioned => {
            match find_partial_aggregate(agg.input()) {
                Some(partial) => partial,
                None => return Ok(None),
            }
        }
        _ => return Ok(None),
    };
    let Some(raw_agg) = raw_agg_plan.as_any().downcast_ref::<AggregateExec>() else {
        return Ok(None);
    };

    if !reads_from_vortex(raw_agg.input()) {
        return Ok(None);
    }

    // The aggregate inputs are indexed against the schema directly below the
    // raw aggregate, so resolve statistics there.
    let stats = raw_agg.input().partition_statistics(None)?;

    let output_schema = agg.schema();
    let mut projections = Vec::with_capacity(raw_agg.aggr_expr().len());
    for (idx, expr) in raw_agg.aggr_expr().iter().enumerate() {
        let Some(value) = resolve_from_stats(expr, &stats) else {
            return Ok(None);
        };
        projections.push(ProjectionExpr {
            expr: lit(value),
            alias: output_schema.field(idx).name().clone(),
        });
    }

    let placeholder = Arc::new(PlaceholderRowExec::new(output_schema));
    let projection = ProjectionExec::try_new(projections, placeholder)?;
    Ok(Some(Arc::new(projection)))
}

/// Walks a single-child chain looking for a `Partial`, ungrouped
/// [`AggregateExec`]. Returns `None` if a branching node or a different
/// aggregate is encountered first.
fn find_partial_aggregate(plan: &Arc<dyn ExecutionPlan>) -> Option<Arc<dyn ExecutionPlan>> {
    let mut current = Arc::clone(plan);
    loop {
        if let Some(agg) = current.as_any().downcast_ref::<AggregateExec>() {
            return (agg.group_expr().is_empty() && matches!(agg.mode(), AggregateMode::Partial))
                .then_some(Arc::clone(&current));
        }
        current = match current.children().as_slice() {
            [child] => Arc::clone(child),
            _ => return None,
        };
    }
}

/// Returns `true` if the single-child chain rooted at `plan` bottoms out in a
/// [`DataSourceExec`] backed by a [`VortexSource`].
fn reads_from_vortex(plan: &Arc<dyn ExecutionPlan>) -> bool {
    let mut current = Arc::clone(plan);
    loop {
        if let Some(data_source) = current.as_any().downcast_ref::<DataSourceExec>() {
            return data_source
                .data_source()
                .as_any()
                .downcast_ref::<FileScanConfig>()
                .map(|config| {
                    config
                        .file_source()
                        .as_any()
                        .downcast_ref::<VortexSource>()
                        .is_some()
                })
                .unwrap_or(false);
        }
        current = match current.children().as_slice() {
            [child] => Arc::clone(child),
            _ => return false,
        };
    }
}

/// Resolves a single ungrouped aggregate to a scalar value using exact
/// statistics, or `None` if it cannot be resolved.
fn resolve_from_stats(agg: &AggregateFunctionExpr, stats: &Statistics) -> Option<ScalarValue> {
    let field = agg.field();
    let exprs = agg.expressions();
    let args = StatisticsArgs {
        statistics: stats,
        return_type: field.data_type(),
        is_distinct: agg.is_distinct(),
        exprs: &exprs,
    };

    // `COUNT`, `MIN`, and `MAX` resolve themselves from statistics.
    if let Some(value) = agg.fun().value_from_stats(&args) {
        return Some(value);
    }

    // The built-in `SUM` aggregate does not resolve from statistics, so handle
    // it here using the column's exact `sum_value`.
    if agg.is_distinct() || exprs.len() != 1 || !agg.fun().name().eq_ignore_ascii_case("sum") {
        return None;
    }
    // `SUM` coerces its input, so the argument is typically a widening cast over
    // a column. The stored `sum_value` is already widened, so the underlying
    // column's sum cast to the aggregate return type is the correct result.
    let column = leaf_column(&exprs[0])?;
    match &stats.column_statistics.get(column.index())?.sum_value {
        Precision::Exact(value) => value.cast_to(field.data_type()).ok(),
        _ => None,
    }
}

/// Peels any wrapping [`CastExpr`] layers to find a leaf [`Column`].
fn leaf_column(expr: &Arc<dyn PhysicalExpr>) -> Option<&Column> {
    let mut current = expr;
    loop {
        if let Some(column) = current.as_any().downcast_ref::<Column>() {
            return Some(column);
        }
        current = current.as_any().downcast_ref::<CastExpr>()?.expr();
    }
}
