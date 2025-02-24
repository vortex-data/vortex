use std::sync::Arc;

use datafusion::prelude::SessionContext;
use datafusion_physical_plan::ExecutionPlan;

use crate::execute_query;

pub async fn run_tpch_query(
    ctx: &SessionContext,
    queries: &[String],
    idx: usize,
) -> (usize, Arc<dyn ExecutionPlan>) {
    if idx == 15 {
        let mut result = None;
        for (i, q) in queries.iter().enumerate() {
            if i == 1 {
                let (record_batches, metrics) = execute_query(ctx, q).await.unwrap();
                result = Some((record_batches.iter().map(|r| r.num_rows()).sum(), metrics));
            } else {
                execute_query(ctx, q).await.unwrap();
            }
        }
        result.expect("Must have had a result in 2nd sql statement for query 15")
    } else {
        let q = &queries[0];
        let (record_batches, metrics) = execute_query(ctx, q).await.unwrap();
        (record_batches.iter().map(|r| r.num_rows()).sum(), metrics)
    }
}
