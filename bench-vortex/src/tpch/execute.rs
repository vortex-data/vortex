use datafusion::prelude::SessionContext;

use crate::{execute_query, Format};

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
            .unwrap()
            .iter()
            .map(|r| r.num_rows())
            .sum()
    }
}
