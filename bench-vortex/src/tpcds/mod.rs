use std::fs;
use std::path::Path;
use std::sync::Arc;
use std::time::Duration;

use datafusion::prelude::SessionContext;
use datafusion_physical_plan::ExecutionPlan;
use url::Url;

use crate::ddb::DuckDBExecutor;
use crate::df::{execute_query, get_session_context};
use crate::{Format, ddb, vortex_panic};

pub async fn load_datasets(_base_dir: &Url, _format: Format) -> anyhow::Result<SessionContext> {
    let context = get_session_context(true);
    Ok(context)
}

pub fn tpcds_queries() -> impl Iterator<Item = (usize, String)> {
    (1..=99).map(|idx| (idx, tpch_query(idx)))
}

// A few tpch queries have multiple statements, this handles that
fn tpch_query(query_idx: usize) -> String {
    let manifest_dir = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tpcds")
        .join(format!("{:02}", query_idx))
        .with_extension("sql");
    fs::read_to_string(manifest_dir).unwrap()
}

pub fn benchmark_duckdb_query(
    query_idx: usize,
    query: &str,
    iterations: usize,
    duckdb_executor: &DuckDBExecutor,
) -> Duration {
    (0..iterations).fold(Duration::from_millis(u64::MAX), |fastest, _| {
        let duration = ddb::execute_tpcds_query(query, duckdb_executor)
            .unwrap_or_else(|err| vortex_panic!("query: {query_idx} failed with: {err}"));

        fastest.min(duration)
    })
}

pub async fn run_datafusion_tpcds_query(
    ctx: &SessionContext,
    query: &str,
) -> (usize, Arc<dyn ExecutionPlan>) {
    let (record_batches, metrics) = execute_query(ctx, query).await.unwrap();
    (record_batches.iter().map(|r| r.num_rows()).sum(), metrics)
}
