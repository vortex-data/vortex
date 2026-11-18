// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::sync::Arc;

use arrow_schema::DataType;
use arrow_schema::Field;
use arrow_schema::Schema;
use arrow_schema::SchemaRef;
use datafusion_common::ColumnStatistics;
use datafusion_common::ScalarValue;
use datafusion_common::Statistics;
use datafusion_common::stats::Precision;
use datafusion_expr::Operator;
use datafusion_physical_expr::PhysicalExpr;
use datafusion_physical_expr::expressions::BinaryExpr;
use datafusion_physical_expr::expressions::Column;
use datafusion_physical_expr::expressions::IsNotNullExpr;
use datafusion_physical_expr::expressions::lit;
use rstest::rstest;

use super::filtered_statistics;

const DEFAULT_SELECTIVITY: f64 = 0.2;

fn schema() -> SchemaRef {
    Arc::new(Schema::new(vec![
        Field::new("keyword", DataType::Utf8, true),
        Field::new("id", DataType::Int64, false),
    ]))
}

/// 1,000 rows. `keyword` spans `["b", "y"]` with 100 nulls and 900 distinct values. `id` spans
/// `[0, 999]` with 250 distinct values.
fn statistics() -> Statistics {
    Statistics {
        num_rows: Precision::Exact(1_000),
        total_byte_size: Precision::Exact(10_000),
        column_statistics: vec![
            ColumnStatistics {
                null_count: Precision::Exact(100),
                min_value: Precision::Exact(ScalarValue::from("b")),
                max_value: Precision::Exact(ScalarValue::from("y")),
                sum_value: Precision::Absent,
                distinct_count: Precision::Inexact(900),
                byte_size: Precision::Absent,
            },
            ColumnStatistics {
                null_count: Precision::Exact(0),
                min_value: Precision::Exact(ScalarValue::Int64(Some(0))),
                max_value: Precision::Exact(ScalarValue::Int64(Some(999))),
                sum_value: Precision::Exact(ScalarValue::Int64(Some(499_500))),
                distinct_count: Precision::Inexact(250),
                byte_size: Precision::Absent,
            },
        ],
    }
}

fn keyword() -> Arc<dyn PhysicalExpr> {
    Arc::new(Column::new("keyword", 0))
}

fn id() -> Arc<dyn PhysicalExpr> {
    Arc::new(Column::new("id", 1))
}

fn binary(
    left: Arc<dyn PhysicalExpr>,
    op: Operator,
    right: Arc<dyn PhysicalExpr>,
) -> Arc<dyn PhysicalExpr> {
    Arc::new(BinaryExpr::new(left, op, right))
}

fn rows(filters: &[Arc<dyn PhysicalExpr>]) -> Precision<usize> {
    filtered_statistics(statistics(), &schema(), filters, false, DEFAULT_SELECTIVITY).num_rows
}

#[rstest]
// 900 non-null rows over 900 distinct values.
#[case::string_eq(binary(keyword(), Operator::Eq, lit("m")), 1)]
#[case::string_eq_reversed(binary(lit("m"), Operator::Eq, keyword()), 1)]
#[case::string_eq_below_min(binary(keyword(), Operator::Eq, lit("a")), 0)]
#[case::string_eq_above_max(binary(keyword(), Operator::Eq, lit("z")), 0)]
#[case::string_eq_null(binary(keyword(), Operator::Eq, lit(ScalarValue::Utf8(None))), 0)]
// Interval analysis: 1,000 rows over 250 distinct values.
#[case::int_eq(binary(id(), Operator::Eq, lit(7_i64)), 4)]
// Interval analysis: `[0, 99]` is a tenth of `[0, 999]`.
#[case::int_range(binary(id(), Operator::Lt, lit(100_i64)), 100)]
#[case::int_range_outside(binary(id(), Operator::Gt, lit(5_000_i64)), 0)]
#[case::unsupported(Arc::new(IsNotNullExpr::new(keyword())), 200)]
fn single_filter_rows(#[case] filter: Arc<dyn PhysicalExpr>, #[case] expected: usize) {
    assert_eq!(rows(&[filter]), Precision::Inexact(expected));
}

#[test]
fn range_conjuncts_on_one_column_are_analyzed_together() {
    // `[100, 199]` is a tenth of `[0, 999]`; estimating each side alone would keep 0.9 × 0.2.
    let filter = binary(
        binary(id(), Operator::GtEq, lit(100_i64)),
        Operator::And,
        binary(id(), Operator::Lt, lit(200_i64)),
    );
    assert_eq!(rows(&[filter]), Precision::Inexact(100));
}

#[test]
fn conjuncts_multiply() {
    // A tenth of the ids, and one in 1,000 rows of the keywords.
    let filters = [
        binary(id(), Operator::Lt, lit(100_i64)),
        binary(keyword(), Operator::Eq, lit("m")),
    ];
    assert_eq!(rows(&filters), Precision::Inexact(1));
}

#[test]
fn unestimated_filters_apply_the_default_once() {
    let filters = [
        Arc::new(IsNotNullExpr::new(keyword())) as _,
        binary(keyword(), Operator::NotEq, lit("m")),
    ];
    assert_eq!(rows(&filters), Precision::Inexact(200));
    assert_eq!(
        filtered_statistics(statistics(), &schema(), &[], true, DEFAULT_SELECTIVITY).num_rows,
        Precision::Inexact(200)
    );
}

#[test]
fn equality_without_distinct_count_uses_the_default() {
    let mut statistics = statistics();
    statistics.column_statistics[0].distinct_count = Precision::Absent;
    let filtered = filtered_statistics(
        statistics,
        &schema(),
        &[binary(keyword(), Operator::Eq, lit("m"))],
        false,
        DEFAULT_SELECTIVITY,
    );
    assert_eq!(filtered.num_rows, Precision::Inexact(200));
}

#[test]
fn empty_input_is_unchanged() {
    let mut statistics = statistics();
    statistics.num_rows = Precision::Exact(0);
    let filtered = filtered_statistics(
        statistics.clone(),
        &schema(),
        &[binary(keyword(), Operator::Eq, lit("m"))],
        false,
        DEFAULT_SELECTIVITY,
    );
    assert_eq!(filtered, statistics);
}

#[test]
fn column_statistics_follow_the_filters() {
    let filtered = filtered_statistics(
        statistics(),
        &schema(),
        &[
            binary(keyword(), Operator::Eq, lit("m")),
            binary(id(), Operator::Lt, lit(500_i64)),
        ],
        false,
        DEFAULT_SELECTIVITY,
    );
    assert_eq!(filtered.num_rows, Precision::Inexact(1));
    assert_eq!(filtered.total_byte_size, Precision::Inexact(5));

    let keyword = &filtered.column_statistics[0];
    assert_eq!(keyword.null_count, Precision::Inexact(0));
    assert_eq!(
        keyword.min_value,
        Precision::Inexact(ScalarValue::from("m"))
    );
    assert_eq!(
        keyword.max_value,
        Precision::Inexact(ScalarValue::from("m"))
    );
    assert_eq!(keyword.distinct_count, Precision::Inexact(1));

    let id = &filtered.column_statistics[1];
    assert_eq!(
        id.min_value,
        Precision::Inexact(ScalarValue::Int64(Some(0)))
    );
    assert_eq!(
        id.max_value,
        Precision::Inexact(ScalarValue::Int64(Some(499)))
    );
    // Capped at the one surviving row.
    assert_eq!(id.distinct_count, Precision::Inexact(1));
    assert_eq!(id.sum_value, Precision::Absent);
}
