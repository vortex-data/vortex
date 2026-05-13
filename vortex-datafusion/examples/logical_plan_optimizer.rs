// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Build `join(t1, join(t2, t3)) -> filter(t1.a == 2)` as a DataFusion
//! logical plan, print it pre-optimization, run the optimizer, print the
//! rewritten plan, then execute.
//!
//! The interesting bit is what the optimizer does with the top-level
//! `t1.a == 2` filter: predicate pushdown drives it into the `t1` scan,
//! projection pushdown trims the columns each join carries up, and the
//! cross-references between filter and join keys turn into a smaller
//! intermediate result.

use std::sync::Arc;

use datafusion::arrow::array::{Int32Array, StringArray};
use datafusion::arrow::datatypes::{DataType, Field, Schema};
use datafusion::arrow::record_batch::RecordBatch;
use datafusion::common::Column;
use datafusion::datasource::MemTable;
use datafusion::logical_expr::{JoinType, LogicalPlan, LogicalPlanBuilder, col, lit};
use datafusion::optimizer::{Analyzer, Optimizer, OptimizerContext};
use datafusion::prelude::SessionContext;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let ctx = SessionContext::new();
    register_tables(&ctx).await?;

    // Grab unoptimized TableScan plans for each registered table.
    let t1 = ctx.table("t1").await?.into_unoptimized_plan();
    let t2 = ctx.table("t2").await?.into_unoptimized_plan();
    let t3 = ctx.table("t3").await?.into_unoptimized_plan();

    // Structurally:
    //   Projection: t1.a, t2.b, t3.c
    //     Filter: t1.a = 2
    //       Inner Join: t2.b = t3.b
    //         Inner Join: t1.a = t2.a
    //           TableScan: t1
    //           TableScan: t2
    //         TableScan: t3
    let plan = LogicalPlanBuilder::from(t1)
        .join(
            t2,
            JoinType::Inner,
            (
                vec![Column::from_qualified_name("t1.a")],
                vec![Column::from_qualified_name("t2.a")],
            ),
            None,
        )?
        .join(
            t3,
            JoinType::Inner,
            (
                vec![Column::from_qualified_name("t2.b")],
                vec![Column::from_qualified_name("t3.b")],
            ),
            None,
        )?
        .filter(col("t1.a").eq(lit(2i32)))?
        .project(vec![col("t1.a"), col("t2.b"), col("t3.c")])?
        .build()?;

    println!("=== Pre-optimization logical plan ===");
    println!("{}", plan.display_indent());

    let optimized = optimize_with_logging(&plan, &ctx)?;

    println!("\n=== Optimized logical plan ===");
    println!("{}", optimized.display_indent());

    println!("\n=== Results ===");
    ctx.execute_logical_plan(optimized).await?.show().await?;

    Ok(())
}

/// Run the analyzer + optimizer manually, logging the input and output plans
/// for every rule that actually changes the plan. Replicates what
/// `SessionState::optimize` does internally, but with an observer hook on
/// each pass.
fn optimize_with_logging(plan: &LogicalPlan, ctx: &SessionContext) -> anyhow::Result<LogicalPlan> {
    let state = ctx.state();
    let options = state.config_options();

    println!("\n=== Analyzer rules ===");
    let mut last = plan.clone();
    let analyzed = Analyzer::new().execute_and_check(plan.clone(), options, |after, rule| {
        log_rule("analyzer", rule.name(), &last, after);
        last = after.clone();
    })?;

    println!("\n=== Optimizer rules ===");
    let mut last = analyzed.clone();
    let optimized = Optimizer::new().optimize(
        analyzed,
        &OptimizerContext::new().with_max_passes(options.optimizer.max_passes as u8),
        |after, rule| {
            log_rule("optimizer", rule.name(), &last, after);
            last = after.clone();
        },
    )?;

    Ok(optimized)
}

fn log_rule(stage: &str, rule: &str, before: &LogicalPlan, after: &LogicalPlan) {
    let before_s = format!("{}", before.display_indent());
    let after_s = format!("{}", after.display_indent());
    if before_s == after_s {
        println!("(no-op) [{stage}] {rule}");
        return;
    }
    println!("\n--- [{stage}] {rule} ---");
    println!("input:");
    for line in before_s.lines() {
        println!("  {line}");
    }
    println!("output:");
    for line in after_s.lines() {
        println!("  {line}");
    }
}

async fn register_tables(ctx: &SessionContext) -> anyhow::Result<()> {
    let t1_schema = Arc::new(Schema::new(vec![Field::new("a", DataType::Int32, false)]));
    let t1 = MemTable::try_new(
        t1_schema.clone(),
        vec![vec![RecordBatch::try_new(
            t1_schema,
            vec![Arc::new(Int32Array::from(vec![1, 2, 2, 3]))],
        )?]],
    )?;
    ctx.register_table("t1", Arc::new(t1))?;

    let t2_schema = Arc::new(Schema::new(vec![
        Field::new("a", DataType::Int32, false),
        Field::new("b", DataType::Utf8, false),
    ]));
    let t2 = MemTable::try_new(
        t2_schema.clone(),
        vec![vec![RecordBatch::try_new(
            t2_schema,
            vec![
                Arc::new(Int32Array::from(vec![1, 2, 3])),
                Arc::new(StringArray::from(vec!["x", "y", "z"])),
            ],
        )?]],
    )?;
    ctx.register_table("t2", Arc::new(t2))?;

    let t3_schema = Arc::new(Schema::new(vec![
        Field::new("b", DataType::Utf8, false),
        Field::new("c", DataType::Int32, false),
    ]));
    let t3 = MemTable::try_new(
        t3_schema.clone(),
        vec![vec![RecordBatch::try_new(
            t3_schema,
            vec![
                Arc::new(StringArray::from(vec!["x", "y", "z", "y"])),
                Arc::new(Int32Array::from(vec![10, 20, 30, 40])),
            ],
        )?]],
    )?;
    ctx.register_table("t3", Arc::new(t3))?;

    Ok(())
}
