use arrow_array::RecordBatch;
use datafusion::prelude::SessionContext;
use datafusion_common::Result;
use datafusion_physical_plan::collect;

use crate::tpch::Format;

pub async fn run_tpch_query(
    ctx: &SessionContext,
    queries: &[String],
    idx: usize,
    format: Format,
) -> usize {
    if idx == 15 {
        let mut result: usize = 0;
        for (i, q) in queries.iter().enumerate() {
            if i == 1 {
                result = execute_query(ctx, q)
                    .await
                    .map_err(|e| println!("Failed to execute {q} {format}: {e}"))
                    .unwrap()
                    .iter()
                    .map(|r| r.num_rows())
                    .sum();
            } else {
                execute_query(ctx, q)
                    .await
                    .map_err(|e| println!("Failed to execute {q} {format}: {e}"))
                    .unwrap();
            }
        }
        result
    } else {
        let q = &queries[0];
        execute_query(ctx, q)
            .await
            .map_err(|e| println!("Failed to execute {q} {format}: {e}"))
            .unwrap()
            .iter()
            .map(|r| r.num_rows())
            .sum()
    }
}

pub async fn execute_query(ctx: &SessionContext, query: &str) -> Result<Vec<RecordBatch>> {
    let plan = ctx.sql(query).await?;
    let (state, plan) = plan.into_parts();
    let optimized = state.optimize(&plan)?;
    let physical_plan = state.create_physical_plan(&optimized).await?;
    let result = collect(physical_plan.clone(), state.task_ctx()).await?;
    Ok(result)
}
