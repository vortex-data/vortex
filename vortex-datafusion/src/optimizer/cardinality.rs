// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Row-count estimates for the relations a join reorder chooses between.
//!
//! DataFusion's `FilterExec` falls back to a fixed selectivity for predicates its interval
//! analysis does not handle, such as an equality on a string column. These estimates use the
//! column statistics instead: an equality keeps `1 / NDV` of the rows, an `IN` list `k / NDV`,
//! a range the matching fraction of `[min, max]`.

use std::sync::Arc;

use arrow_schema::DataType;
use datafusion_common::ColumnStatistics;
use datafusion_common::ScalarValue;
use datafusion_common::Statistics;
use datafusion_expr::Operator;
use datafusion_physical_expr::PhysicalExpr;
use datafusion_physical_expr::expressions::BinaryExpr;
use datafusion_physical_expr::expressions::Column;
use datafusion_physical_expr::expressions::InListExpr;
use datafusion_physical_expr::expressions::IsNotNullExpr;
use datafusion_physical_expr::expressions::IsNullExpr;
use datafusion_physical_expr::expressions::LikeExpr;
use datafusion_physical_expr::expressions::Literal;
use datafusion_physical_expr::expressions::NotExpr;
use datafusion_physical_plan::ExecutionPlan;
use datafusion_physical_plan::StatisticsArgs;
use datafusion_physical_plan::StatisticsContext;
use datafusion_physical_plan::filter::FilterExec;
use datafusion_physical_plan::projection::ProjectionExec;

/// The selectivity of a predicate whose shape gives nothing better, matching DataFusion's.
const DEFAULT_SELECTIVITY: f64 = 0.2;
/// The selectivity of a `LIKE` pattern.
const LIKE_SELECTIVITY: f64 = 0.1;
/// The selectivity of an open range whose bounds cannot be compared to the column's.
const RANGE_SELECTIVITY: f64 = 1.0 / 3.0;

/// The estimated rows and per-column distinct counts of a relation.
#[derive(Clone, Debug)]
pub(crate) struct RelationEstimate {
    pub rows: f64,
    /// The estimated distinct count of each output column, capped at `rows`.
    pub ndv: Vec<f64>,
}

/// Estimate the output of `plan`.
///
/// A `FilterExec`, under any column-only projections, is estimated from its input's statistics
/// and the selectivity of its predicate. Any other plan uses its own statistics.
pub(crate) fn estimate_relation(plan: &Arc<dyn ExecutionPlan>) -> RelationEstimate {
    let stats = plan_statistics(plan);
    let width = plan.schema().fields().len();
    let rows = match filtered_rows(plan) {
        Some(rows) => rows,
        None => stats
            .as_ref()
            .and_then(|s| s.num_rows.get_value().copied())
            .map_or(f64::MAX, |r| r as f64),
    };
    let rows = rows.max(1.0);
    let ndv = (0..width)
        .map(|idx| {
            stats
                .as_ref()
                .and_then(|s| s.column_statistics.get(idx))
                .and_then(|c| column_ndv(c, base_rows(stats.as_deref())))
                .unwrap_or(rows)
                .clamp(1.0, rows)
        })
        .collect();
    RelationEstimate { rows, ndv }
}

fn plan_statistics(plan: &Arc<dyn ExecutionPlan>) -> Option<Arc<Statistics>> {
    StatisticsContext::new()
        .compute(plan.as_ref(), &StatisticsArgs::new())
        .ok()
}

fn base_rows(stats: Option<&Statistics>) -> Option<f64> {
    stats
        .and_then(|s| s.num_rows.get_value().copied())
        .map(|r| r as f64)
}

/// The rows a `FilterExec` at the top of `plan` keeps, looking through column-only projections.
fn filtered_rows(plan: &Arc<dyn ExecutionPlan>) -> Option<f64> {
    if let Some(projection) = plan.downcast_ref::<ProjectionExec>() {
        let column_only = projection
            .expr()
            .iter()
            .all(|e| e.expr.downcast_ref::<Column>().is_some());
        return column_only
            .then(|| filtered_rows(projection.input()))
            .flatten();
    }
    let filter = plan.downcast_ref::<FilterExec>()?;
    let input_stats = plan_statistics(filter.input())?;
    let input_rows = *input_stats.num_rows.get_value()? as f64;
    let input_rows = match filtered_rows(filter.input()) {
        Some(rows) => rows.min(input_rows),
        None => input_rows,
    };
    let selectivity = selectivity(filter.predicate(), &input_stats);
    Some((input_rows * selectivity).max(1.0))
}

/// The distinct count of a column: its NDV statistic, else its integer range, else unknown.
fn column_ndv(stats: &ColumnStatistics, rows: Option<f64>) -> Option<f64> {
    if let Some(ndv) = stats.distinct_count.get_value() {
        return Some(*ndv as f64);
    }
    let (min, max) = (stats.min_value.get_value()?, stats.max_value.get_value()?);
    let span = scalar_to_f64(max)? - scalar_to_f64(min)? + 1.0;
    let integer = min.data_type().is_integer() || matches!(min.data_type(), DataType::Date32);
    (integer && span > 0.0).then(|| rows.map_or(span, |rows| span.min(rows)))
}

/// The fraction of rows of a relation with statistics `stats` that `predicate` keeps.
pub(crate) fn selectivity(predicate: &Arc<dyn PhysicalExpr>, stats: &Statistics) -> f64 {
    let rows = stats
        .num_rows
        .get_value()
        .map(|r| (*r).max(1) as f64)
        .unwrap_or(1.0);
    selectivity_inner(predicate, stats).clamp(1.0 / rows, 1.0)
}

fn selectivity_inner(expr: &Arc<dyn PhysicalExpr>, stats: &Statistics) -> f64 {
    if let Some(binary) = expr.downcast_ref::<BinaryExpr>() {
        let (left, right) = (binary.left(), binary.right());
        return match binary.op() {
            Operator::And => selectivity_inner(left, stats) * selectivity_inner(right, stats),
            Operator::Or => {
                let (l, r) = (
                    selectivity_inner(left, stats),
                    selectivity_inner(right, stats),
                );
                (l + r - l * r).min(1.0)
            }
            op => comparison_selectivity(*op, left, right, stats),
        };
    }
    if let Some(not) = expr.downcast_ref::<NotExpr>() {
        return 1.0 - selectivity_inner(not.arg(), stats);
    }
    if let Some(in_list) = expr.downcast_ref::<InListExpr>() {
        let Some(col) = column_stats(in_list.expr(), stats) else {
            return DEFAULT_SELECTIVITY;
        };
        let matched = match column_ndv(col, base_rows(Some(stats))) {
            Some(ndv) => (in_list.list().len() as f64 / ndv.max(1.0)).min(1.0),
            None => DEFAULT_SELECTIVITY,
        };
        return if in_list.negated() {
            1.0 - matched
        } else {
            matched
        };
    }
    if let Some(like) = expr.downcast_ref::<LikeExpr>() {
        return if like.negated() {
            1.0 - LIKE_SELECTIVITY
        } else {
            LIKE_SELECTIVITY
        };
    }
    if let Some(is_null) = expr.downcast_ref::<IsNullExpr>() {
        return null_fraction(is_null.arg(), stats).unwrap_or(DEFAULT_SELECTIVITY);
    }
    if let Some(is_not_null) = expr.downcast_ref::<IsNotNullExpr>() {
        return null_fraction(is_not_null.arg(), stats)
            .map_or(1.0 - DEFAULT_SELECTIVITY, |f| 1.0 - f);
    }
    DEFAULT_SELECTIVITY
}

fn comparison_selectivity(
    op: Operator,
    left: &Arc<dyn PhysicalExpr>,
    right: &Arc<dyn PhysicalExpr>,
    stats: &Statistics,
) -> f64 {
    // Normalize to `column op literal`.
    let (column, literal, op) = match (literal_value(left), literal_value(right)) {
        (None, Some(lit)) => (left, lit, op),
        (Some(lit), None) => match op.swap() {
            Some(op) => (right, lit, op),
            None => return DEFAULT_SELECTIVITY,
        },
        _ => return DEFAULT_SELECTIVITY,
    };
    let Some(col) = column_stats(column, stats) else {
        return DEFAULT_SELECTIVITY;
    };
    let ndv = column_ndv(col, base_rows(Some(stats)));
    match op {
        Operator::Eq => ndv.map_or(DEFAULT_SELECTIVITY, |ndv| 1.0 / ndv.max(1.0)),
        Operator::NotEq => ndv.map_or(1.0 - DEFAULT_SELECTIVITY, |ndv| 1.0 - 1.0 / ndv.max(1.0)),
        Operator::Lt | Operator::LtEq | Operator::Gt | Operator::GtEq => {
            range_selectivity(op, col, literal).unwrap_or(RANGE_SELECTIVITY)
        }
        _ => DEFAULT_SELECTIVITY,
    }
}

/// The fraction of `[min, max]` that `column op literal` keeps, assuming uniform values.
fn range_selectivity(op: Operator, col: &ColumnStatistics, literal: &ScalarValue) -> Option<f64> {
    let min = scalar_to_f64(col.min_value.get_value()?)?;
    let max = scalar_to_f64(col.max_value.get_value()?)?;
    let value = scalar_to_f64(literal)?;
    if max <= min {
        return None;
    }
    let below = ((value - min) / (max - min)).clamp(0.0, 1.0);
    Some(match op {
        Operator::Lt | Operator::LtEq => below,
        _ => 1.0 - below,
    })
}

fn scalar_to_f64(value: &ScalarValue) -> Option<f64> {
    match value {
        ScalarValue::Date32(Some(v)) => Some(f64::from(*v)),
        ScalarValue::Date64(Some(v))
        | ScalarValue::TimestampSecond(Some(v), _)
        | ScalarValue::TimestampMillisecond(Some(v), _)
        | ScalarValue::TimestampMicrosecond(Some(v), _)
        | ScalarValue::TimestampNanosecond(Some(v), _) => Some(*v as f64),
        v if v.data_type().is_numeric() => match v.cast_to(&DataType::Float64).ok()? {
            ScalarValue::Float64(Some(f)) => Some(f),
            _ => None,
        },
        _ => None,
    }
}

fn literal_value(expr: &Arc<dyn PhysicalExpr>) -> Option<&ScalarValue> {
    expr.downcast_ref::<Literal>().map(Literal::value)
}

fn column_stats<'a>(
    expr: &Arc<dyn PhysicalExpr>,
    stats: &'a Statistics,
) -> Option<&'a ColumnStatistics> {
    let column = expr.downcast_ref::<Column>()?;
    stats.column_statistics.get(column.index())
}

fn null_fraction(expr: &Arc<dyn PhysicalExpr>, stats: &Statistics) -> Option<f64> {
    let col = column_stats(expr, stats)?;
    let nulls = *col.null_count.get_value()? as f64;
    let rows = *stats.num_rows.get_value()? as f64;
    (rows > 0.0).then(|| nulls / rows)
}
