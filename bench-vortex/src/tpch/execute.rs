use datafusion::prelude::SessionContext;

use crate::execute_query;

pub async fn run_tpch_query(ctx: &SessionContext, queries: &[String], idx: usize) -> usize {
    if idx == 15 {
        let mut result = None;
        for (i, q) in queries.iter().enumerate() {
            if i == 1 {
                result = Some(
                    execute_query(ctx, q)
                        .await
                        .unwrap()
                        .iter()
                        .map(|r| r.num_rows())
                        .sum(),
                );
            } else {
                execute_query(ctx, q).await.unwrap();
            }
        }
        result.expect("Must have had a result in 2nd sql statement for query 15")
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
