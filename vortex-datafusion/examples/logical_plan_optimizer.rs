// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Build a plan with several `TableScan`s of each table — some with the same
//! column projections, some with different ones — joined together with a
//! filter on top, then run DataFusion's analyzer + optimizer with per-rule
//! observers so you can watch projection pushdown, predicate pushdown and
//! aliasing get unwound step by step.
//!
//! The shape we build:
//!
//! ```text
//! Filter: t1ab1.a = 2
//!   Join (t1ab1.a = t1ab2.a)
//!     Join (t1ab1.a = t2xy.x)          -- left subtree
//!       SubqueryAlias t1ab1
//!         Projection [a, b]
//!           TableScan t1
//!       SubqueryAlias t2xy
//!         Projection [x, y]
//!           TableScan t2
//!     Join (t1ab2.a = t2xz.x)          -- right subtree
//!       Join (t1ab2.a = t1ac.a)
//!         SubqueryAlias t1ab2
//!           Projection [a, b]          -- same cols as t1ab1
//!             TableScan t1
//!         SubqueryAlias t1ac
//!           Projection [a, c]          -- different cols
//!             TableScan t1
//!       Join (t2xz.x = t3a.m)
//!         SubqueryAlias t2xz
//!           Projection [x, z]          -- shares x with t2xy
//!             TableScan t2
//!         SubqueryAlias t3a
//!           TableScan t3                -- full scan
//! ```
//!
//! Per-table scan counts:
//! - `t1` : 3 scans — two with `[a, b]`, one with `[a, c]`. Column `d` never touched.
//! - `t2` : 2 scans — `[x, y]` and `[x, z]`. Share `x`, otherwise disjoint.
//! - `t3` : 1 scan, full schema.

use std::sync::Arc;

use datafusion::arrow::array::Int32Array;
use datafusion::arrow::datatypes::{DataType, Field, Schema};
use datafusion::arrow::record_batch::RecordBatch;
use datafusion::datasource::MemTable;
use datafusion::logical_expr::{JoinType, LogicalPlan, LogicalPlanBuilder, col, lit};
use datafusion::optimizer::{Analyzer, Optimizer, OptimizerContext};
use datafusion::prelude::SessionContext;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let ctx = SessionContext::new();
    register_tables(&ctx).await?;

    // Three aliased projections of t1 — two share the same column set,
    // one picks a different column.
    let t1ab1 = aliased_projection(&ctx, "t1", &["a", "b"], "t1ab1").await?;
    let t1ab2 = aliased_projection(&ctx, "t1", &["a", "b"], "t1ab2").await?;
    let t1ac = aliased_projection(&ctx, "t1", &["a", "c"], "t1ac").await?;

    // Two aliased projections of t2 — share `x`, otherwise disjoint.
    let t2xy = aliased_projection(&ctx, "t2", &["x", "y"], "t2xy").await?;
    let t2xz = aliased_projection(&ctx, "t2", &["x", "z"], "t2xz").await?;

    // t3 in full, just aliased.
    let t3a = LogicalPlanBuilder::from(ctx.table("t3").await?.into_unoptimized_plan())
        .alias("t3a")?
        .build()?;

    let left = LogicalPlanBuilder::from(t1ab1)
        .join_on(
            t2xy,
            JoinType::Inner,
            vec![col("t1ab1.a").eq(col("t2xy.x"))],
        )?
        .build()?;

    let right_left = LogicalPlanBuilder::from(t1ab2)
        .join_on(
            t1ac,
            JoinType::Inner,
            vec![col("t1ab2.a").eq(col("t1ac.a"))],
        )?
        .build()?;

    let right_right = LogicalPlanBuilder::from(t2xz)
        .join_on(t3a, JoinType::Inner, vec![col("t2xz.x").eq(col("t3a.m"))])?
        .build()?;

    let right = LogicalPlanBuilder::from(right_left)
        .join_on(
            right_right,
            JoinType::Inner,
            vec![col("t1ab2.a").eq(col("t2xz.x"))],
        )?
        .build()?;

    let plan = LogicalPlanBuilder::from(left)
        .join_on(
            right,
            JoinType::Inner,
            vec![col("t1ab1.a").eq(col("t1ab2.a"))],
        )?
        .filter(col("t1ab1.a").eq(lit(2i32)))?
        .project(vec![
            col("t1ab1.a"),
            col("t1ab1.b"),
            col("t1ac.c"),
            col("t2xy.y"),
            col("t2xz.z"),
            col("t3a.n"),
        ])?
        .build()?;

    println!("=== Pre-optimization logical plan ===");
    println!("{}", plan.display_indent_schema());

    let optimized = optimize_with_logging(&plan, &ctx)?;

    println!("\n=== Optimized logical plan ===");
    println!("{}", optimized.display_indent_schema());

    println!("\n=== Results ===");
    ctx.execute_logical_plan(optimized).await?.show().await?;

    Ok(())
}

/// `SubqueryAlias(alias, Projection(cols, TableScan(table)))`.
async fn aliased_projection(
    ctx: &SessionContext,
    table: &str,
    cols: &[&str],
    alias: &str,
) -> anyhow::Result<LogicalPlan> {
    let scan = ctx.table(table).await?.into_unoptimized_plan();
    let plan = LogicalPlanBuilder::from(scan)
        .project(cols.iter().map(|c| col(*c)))?
        .alias(alias)?
        .build()?;
    Ok(plan)
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
    let before_s = format!("{}", before.display_indent_schema());
    let after_s = format!("{}", after.display_indent_schema());
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
    // t1: 4 columns; only a/b/c are referenced anywhere, d should never be scanned.
    let t1_schema = Arc::new(Schema::new(vec![
        Field::new("a", DataType::Int32, false),
        Field::new("b", DataType::Int32, false),
        Field::new("c", DataType::Int32, false),
        Field::new("d", DataType::Int32, false),
    ]));
    let t1 = MemTable::try_new(
        t1_schema.clone(),
        vec![vec![RecordBatch::try_new(
            t1_schema,
            vec![
                Arc::new(Int32Array::from(vec![1, 2, 2, 3])),
                Arc::new(Int32Array::from(vec![10, 20, 30, 40])),
                Arc::new(Int32Array::from(vec![100, 200, 300, 400])),
                Arc::new(Int32Array::from(vec![1000, 2000, 3000, 4000])),
            ],
        )?]],
    )?;
    ctx.register_table("t1", Arc::new(t1))?;

    let t2_schema = Arc::new(Schema::new(vec![
        Field::new("x", DataType::Int32, false),
        Field::new("y", DataType::Int32, false),
        Field::new("z", DataType::Int32, false),
    ]));
    let t2 = MemTable::try_new(
        t2_schema.clone(),
        vec![vec![RecordBatch::try_new(
            t2_schema,
            vec![
                Arc::new(Int32Array::from(vec![1, 2, 3])),
                Arc::new(Int32Array::from(vec![11, 22, 33])),
                Arc::new(Int32Array::from(vec![111, 222, 333])),
            ],
        )?]],
    )?;
    ctx.register_table("t2", Arc::new(t2))?;

    let t3_schema = Arc::new(Schema::new(vec![
        Field::new("m", DataType::Int32, false),
        Field::new("n", DataType::Int32, false),
    ]));
    let t3 = MemTable::try_new(
        t3_schema.clone(),
        vec![vec![RecordBatch::try_new(
            t3_schema,
            vec![
                Arc::new(Int32Array::from(vec![1, 2, 3])),
                Arc::new(Int32Array::from(vec![7, 8, 9])),
            ],
        )?]],
    )?;
    ctx.register_table("t3", Arc::new(t3))?;

    Ok(())
}
