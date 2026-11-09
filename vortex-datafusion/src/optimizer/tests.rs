// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::sync::Arc;

use arrow_array::Int64Array;
use arrow_array::RecordBatch;
use arrow_array::StringArray;
use arrow_schema::DataType;
use arrow_schema::Field;
use arrow_schema::Schema;
use datafusion::arrow::util::pretty::pretty_format_batches;
use datafusion::datasource::MemTable;
use datafusion::execution::SessionStateBuilder;
use datafusion::physical_optimizer::optimizer::PhysicalOptimizer;
use datafusion::prelude::SessionContext;
use datafusion_common::ColumnStatistics;
use datafusion_common::ScalarValue;
use datafusion_common::Statistics;
use datafusion_common::stats::Precision;
use datafusion_expr::Operator;
use datafusion_physical_expr::PhysicalExpr;
use datafusion_physical_expr::expressions::BinaryExpr;
use datafusion_physical_expr::expressions::Column;
use datafusion_physical_expr::expressions::InListExpr;
use datafusion_physical_expr::expressions::IsNullExpr;
use datafusion_physical_expr::expressions::LikeExpr;
use datafusion_physical_expr::expressions::Literal;
use datafusion_physical_expr::expressions::NotExpr;
use datafusion_physical_plan::ExecutionPlan;
use datafusion_physical_plan::collect;
use datafusion_physical_plan::displayable;
use datafusion_physical_plan::joins::HashJoinExec;
use rstest::rstest;

use super::cardinality::selectivity;
use super::insert_join_reorder;

/// One integer column with 1000 rows, 100 distinct values in `[0, 99]` and 50 nulls.
fn stats() -> Statistics {
    Statistics {
        num_rows: Precision::Exact(1000),
        total_byte_size: Precision::Absent,
        column_statistics: vec![ColumnStatistics {
            null_count: Precision::Exact(50),
            max_value: Precision::Exact(ScalarValue::Int64(Some(99))),
            min_value: Precision::Exact(ScalarValue::Int64(Some(0))),
            sum_value: Precision::Absent,
            distinct_count: Precision::Inexact(100),
            byte_size: Precision::Absent,
        }],
    }
}

fn col() -> Arc<dyn PhysicalExpr> {
    Arc::new(Column::new("a", 0))
}

fn lit(value: i64) -> Arc<dyn PhysicalExpr> {
    Arc::new(Literal::new(ScalarValue::Int64(Some(value))))
}

fn binary(
    left: Arc<dyn PhysicalExpr>,
    op: Operator,
    right: Arc<dyn PhysicalExpr>,
) -> Arc<dyn PhysicalExpr> {
    Arc::new(BinaryExpr::new(left, op, right))
}

#[rstest]
#[case::equality(binary(col(), Operator::Eq, lit(7)), 0.01)]
#[case::equality_flipped(binary(lit(7), Operator::Eq, col()), 0.01)]
#[case::not_equal(binary(col(), Operator::NotEq, lit(7)), 0.99)]
#[case::less_than(binary(col(), Operator::Lt, lit(25)), 25.0 / 99.0)]
#[case::greater_than_flipped(binary(lit(25), Operator::Lt, col()), 74.0 / 99.0)]
#[case::and(
    binary(binary(col(), Operator::Eq, lit(1)), Operator::And, binary(col(), Operator::Lt, lit(50))),
    0.01 * 50.0 / 99.0
)]
#[case::or(
    binary(binary(col(), Operator::Eq, lit(1)), Operator::Or, binary(col(), Operator::Eq, lit(2))),
    0.01 + 0.01 - 0.0001
)]
#[case::in_list(Arc::new(InListExpr::try_new(col(), vec![lit(1), lit(2), lit(3)], false, &Schema::new(vec![Field::new("a", DataType::Int64, true)])).unwrap()), 0.03)]
#[case::not(Arc::new(NotExpr::new(binary(col(), Operator::Eq, lit(7)))), 0.99)]
#[case::is_null(Arc::new(IsNullExpr::new(col())), 0.05)]
#[case::like(
    Arc::new(LikeExpr::new(false, false, col(), Arc::new(Literal::new(ScalarValue::from("x%"))))),
    0.1
)]
fn predicate_selectivity(#[case] predicate: Arc<dyn PhysicalExpr>, #[case] expected: f64) {
    let actual = selectivity(&predicate, &stats());
    assert!(
        (actual - expected).abs() < 1e-9,
        "selectivity {actual}, expected {expected}"
    );
}

fn table(ids: impl Iterator<Item = i64>, tag: impl Fn(i64) -> &'static str) -> MemTable {
    let ids: Vec<i64> = ids.collect();
    let tags: Vec<&str> = ids.iter().map(|&id| tag(id)).collect();
    let schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int64, false),
        Field::new("tag", DataType::Utf8, false),
    ]));
    let batch = RecordBatch::try_new(
        Arc::clone(&schema),
        vec![
            Arc::new(Int64Array::from(ids)),
            Arc::new(StringArray::from(tags)),
        ],
    )
    .unwrap();
    MemTable::try_new(schema, vec![vec![batch]]).unwrap()
}

fn context(reorder: bool) -> anyhow::Result<SessionContext> {
    let mut builder = SessionStateBuilder::new().with_default_features();
    if reorder {
        let mut rules = PhysicalOptimizer::new().rules;
        insert_join_reorder(&mut rules);
        builder = builder.with_physical_optimizer_rules(rules);
    }
    let ctx = SessionContext::new_with_state(builder.build());
    ctx.register_table("big_a", Arc::new(table(0..2000, |_| "a")))?;
    ctx.register_table("big_b", Arc::new(table(0..2000, |_| "b")))?;
    ctx.register_table(
        "small",
        Arc::new(table(0..20, |id| if id == 7 { "x" } else { "y" })),
    )?;
    Ok(ctx)
}

/// The joins with no join beneath them, which the plan executes first.
fn innermost_joins(plan: &Arc<dyn ExecutionPlan>, out: &mut Vec<Arc<dyn ExecutionPlan>>) -> bool {
    let mut below = false;
    for child in plan.children() {
        below |= innermost_joins(child, out);
    }
    let is_join = plan.downcast_ref::<HashJoinExec>().is_some();
    if is_join && !below {
        out.push(Arc::clone(plan));
    }
    is_join || below
}

/// Whether `plan` scans the `small` table, recognized by its filter on `tag`.
fn reads_filtered_small(plan: &Arc<dyn ExecutionPlan>) -> bool {
    let text = displayable(plan.as_ref()).indent(false).to_string();
    text.contains("tag@1 = x")
}

/// The written order joins the two large tables first; the reorder starts from the filtered
/// small table instead, and the result does not change.
#[tokio::test]
async fn reorder_joins_filtered_relation_first() -> anyhow::Result<()> {
    let sql = "SELECT count(*) AS n FROM big_a, big_b, small \
               WHERE big_a.id = big_b.id AND big_b.id = small.id AND small.tag = 'x'";

    let mut results = Vec::new();
    for reorder in [false, true] {
        let ctx = context(reorder)?;
        let df = ctx.sql(sql).await?;
        let plan = df.create_physical_plan().await?;
        let mut joins = Vec::new();
        innermost_joins(&plan, &mut joins);
        assert_eq!(joins.len(), 1, "one join runs first");
        assert_eq!(reads_filtered_small(&joins[0]), reorder);
        results.push(collect(plan, ctx.task_ctx()).await?);
    }
    assert_eq!(
        pretty_format_batches(&results[0])?.to_string(),
        pretty_format_batches(&results[1])?.to_string()
    );
    Ok(())
}
