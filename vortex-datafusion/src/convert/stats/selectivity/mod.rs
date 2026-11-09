// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Estimate a scan's output statistics after it applies the filters it absorbed.
//!
//! DataFusion drops the `FilterExec` for every filter a scan accepts, so the scan has to report
//! the filtered row count itself. Each conjunct is estimated on its own and the selectivities are
//! multiplied, assuming the columns are independent:
//!
//! - Conjuncts that DataFusion's interval analysis supports (numeric and temporal comparisons)
//!   are analyzed together, as [`FilterExec`] does. A range filter keeps the fraction of the
//!   `[min, max]` range that survives, and an equality filter keeps `1 / distinct_count`.
//! - `column = literal` on any other type, such as strings, keeps
//!   `non_null_fraction / distinct_count` of the rows, or none if the literal lies outside the
//!   exact `[min, max]` bounds.
//! - Every other conjunct together keeps the configured default selectivity, which is what
//!   [`FilterExec`] would have applied.
//!
//! [`FilterExec`]: datafusion_physical_plan::filter::FilterExec

use std::sync::Arc;

use arrow_schema::Schema;
use arrow_schema::SchemaRef;
use datafusion_common::ColumnStatistics;
use datafusion_common::ScalarValue;
use datafusion_common::Statistics;
use datafusion_common::stats::Precision;
use datafusion_expr::Operator;
use datafusion_physical_expr::AnalysisContext;
use datafusion_physical_expr::ExprBoundaries;
use datafusion_physical_expr::PhysicalExpr;
use datafusion_physical_expr::analyze;
use datafusion_physical_expr::conjunction;
use datafusion_physical_expr::expressions::BinaryExpr;
use datafusion_physical_expr::expressions::Column;
use datafusion_physical_expr::expressions::Literal;
use datafusion_physical_expr::intervals::utils::check_support;
use datafusion_physical_expr::split_conjunction;
use datafusion_physical_expr::utils::collect_columns;

/// The statistics of the rows of `statistics` that satisfy every expression in `filters`.
///
/// `statistics` and `filters` are both over `schema`. `unestimated_filters` marks that the scan
/// applies further filters that are not in `filters`; they keep `default_selectivity` of the rows.
pub(crate) fn filtered_statistics(
    statistics: Statistics,
    schema: &SchemaRef,
    filters: &[Arc<dyn PhysicalExpr>],
    unestimated_filters: bool,
    default_selectivity: f64,
) -> Statistics {
    if statistics.num_rows == Precision::Exact(0) {
        return statistics;
    }

    let Statistics {
        num_rows,
        total_byte_size,
        mut column_statistics,
    } = statistics;

    let mut selectivity = 1.0;
    let mut unestimated = unestimated_filters;
    let mut interval_conjuncts = Vec::new();
    for conjunct in filters.iter().flat_map(split_conjunction) {
        if check_support(conjunct, schema) {
            interval_conjuncts.push(Arc::clone(conjunct));
        } else if let Some((index, value)) = column_eq_literal(conjunct) {
            let Some(stats) = column_statistics.get_mut(index) else {
                unestimated = true;
                continue;
            };
            match equality_selectivity(stats, value, num_rows) {
                Some(s) => {
                    selectivity *= s;
                    *stats = equality_statistics(stats, value, schema, index);
                }
                None => unestimated = true,
            }
        } else {
            unestimated = true;
        }
    }

    // Interval analysis only involves supported column types, so it never touches a column that
    // an equality above already rewrote.
    if !interval_conjuncts.is_empty() {
        match analyze_intervals(&conjunction(interval_conjuncts), schema, &column_statistics) {
            Some((s, boundaries)) => {
                selectivity *= s;
                for boundary in boundaries {
                    let index = boundary.column.index();
                    column_statistics[index] =
                        interval_statistics(&column_statistics[index], boundary);
                }
            }
            None => unestimated = true,
        }
    }

    if unestimated {
        selectivity *= default_selectivity;
    }

    let num_rows = num_rows.with_estimated_selectivity(selectivity);
    for stats in &mut column_statistics {
        *stats = std::mem::take(stats).to_inexact();
        stats.null_count = cap_at_rows(stats.null_count, num_rows);
        stats.distinct_count = cap_at_rows(stats.distinct_count, num_rows);
        stats.byte_size = stats.byte_size.with_estimated_selectivity(selectivity);
        stats.sum_value = Precision::Absent;
    }

    Statistics {
        num_rows,
        total_byte_size: total_byte_size.with_estimated_selectivity(selectivity),
        column_statistics,
    }
}

/// Match `column = literal` or `literal = column`.
fn column_eq_literal(expr: &Arc<dyn PhysicalExpr>) -> Option<(usize, &ScalarValue)> {
    let binary = expr.downcast_ref::<BinaryExpr>()?;
    if *binary.op() != Operator::Eq {
        return None;
    }
    let (left, right) = (binary.left(), binary.right());
    match (
        left.downcast_ref::<Column>(),
        right.downcast_ref::<Literal>(),
        left.downcast_ref::<Literal>(),
        right.downcast_ref::<Column>(),
    ) {
        (Some(column), Some(literal), ..) | (.., Some(literal), Some(column)) => {
            Some((column.index(), literal.value()))
        }
        _ => None,
    }
}

/// The fraction of rows where the column equals `value`, assuming the non-null values are spread
/// evenly over the distinct ones.
fn equality_selectivity(
    stats: &ColumnStatistics,
    value: &ScalarValue,
    num_rows: Precision<usize>,
) -> Option<f64> {
    if value.is_null() || outside_exact_bounds(stats, value) {
        return Some(0.0);
    }

    let distinct_count = *stats.distinct_count.get_value()?;
    if distinct_count == 0 {
        return Some(0.0);
    }

    let non_null_fraction = match (stats.null_count.get_value(), num_rows.get_value()) {
        (Some(&nulls), Some(&rows)) if rows > 0 => 1.0 - (nulls.min(rows) as f64 / rows as f64),
        _ => 1.0,
    };
    Some(non_null_fraction / distinct_count as f64)
}

/// Whether exact bounds prove that no row equals `value`.
fn outside_exact_bounds(stats: &ColumnStatistics, value: &ScalarValue) -> bool {
    let below = matches!(
        &stats.min_value,
        Precision::Exact(min) if value.partial_cmp(min).is_some_and(|o| o.is_lt())
    );
    let above = matches!(
        &stats.max_value,
        Precision::Exact(max) if value.partial_cmp(max).is_some_and(|o| o.is_gt())
    );
    below || above
}

/// The statistics of a column after `column = value` keeps only that value.
fn equality_statistics(
    stats: &ColumnStatistics,
    value: &ScalarValue,
    schema: &Schema,
    index: usize,
) -> ColumnStatistics {
    // The literal only becomes the bound if it already has the column's type.
    let bound = if schema.field(index).data_type() == &value.data_type() {
        Precision::Inexact(value.clone())
    } else {
        Precision::Absent
    };
    ColumnStatistics {
        null_count: Precision::Exact(0),
        min_value: bound.clone(),
        max_value: bound,
        sum_value: Precision::Absent,
        distinct_count: Precision::Inexact(1),
        byte_size: stats.byte_size,
    }
}

/// Run DataFusion's interval analysis over the columns `predicate` references, returning its
/// selectivity and the narrowed boundaries of those columns.
fn analyze_intervals(
    predicate: &Arc<dyn PhysicalExpr>,
    schema: &Schema,
    column_statistics: &[ColumnStatistics],
) -> Option<(f64, Vec<ExprBoundaries>)> {
    let boundaries = collect_columns(predicate)
        .into_iter()
        .map(|column| {
            let stats = column_statistics.get(column.index())?;
            let mut boundary =
                ExprBoundaries::try_from_column(schema, stats, column.index()).ok()?;
            boundary.column = column;
            Some(boundary)
        })
        .collect::<Option<Vec<_>>>()?;
    let analysis = analyze(predicate, AnalysisContext::new(boundaries), schema).ok()?;
    Some((analysis.selectivity.unwrap_or(1.0), analysis.boundaries))
}

/// The statistics of a column narrowed to `boundary` by interval analysis.
fn interval_statistics(stats: &ColumnStatistics, boundary: ExprBoundaries) -> ColumnStatistics {
    let Some(interval) = boundary.interval else {
        // The predicate cannot hold, so no rows survive.
        return ColumnStatistics {
            null_count: Precision::Inexact(0),
            min_value: Precision::Absent,
            max_value: Precision::Absent,
            sum_value: Precision::Absent,
            distinct_count: Precision::Inexact(0),
            byte_size: Precision::Inexact(0),
        };
    };
    let (lower, upper) = interval.into_bounds();
    let singleton = !lower.is_null() && lower == upper;
    let bound = |value: ScalarValue| {
        if value.is_null() {
            Precision::Absent
        } else {
            Precision::Inexact(value)
        }
    };
    ColumnStatistics {
        null_count: Precision::Exact(0),
        min_value: bound(lower),
        max_value: bound(upper),
        sum_value: Precision::Absent,
        distinct_count: if singleton {
            Precision::Inexact(1)
        } else {
            boundary.distinct_count.to_inexact()
        },
        byte_size: stats.byte_size,
    }
}

/// Cap a count of rows or values at the estimated number of surviving rows.
fn cap_at_rows(value: Precision<usize>, num_rows: Precision<usize>) -> Precision<usize> {
    match num_rows {
        Precision::Absent => value.to_inexact(),
        rows => value.to_inexact().min(&rows),
    }
}

#[cfg(test)]
mod tests;
