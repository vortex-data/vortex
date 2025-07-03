// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

pub mod ddb;
pub mod df;

use std::future::Future;
use std::time::Duration;

use datafusion::prelude::SessionContext;
use vortex::error::VortexExpect;

pub use crate::Format;
use crate::{BenchmarkDataset, Engine, vortex_panic};

pub enum EngineCtx {
    DataFusion(df::DataFusionCtx),
    DuckDB(ddb::DuckDBCtx),
}

impl EngineCtx {
    pub fn new_with_datafusion(session_ctx: SessionContext, emit_plan: bool) -> Self {
        EngineCtx::DataFusion(df::DataFusionCtx {
            execution_plans: Vec::new(),
            metrics: Vec::new(),
            session: session_ctx,
            emit_plan,
        })
    }

    pub fn new_with_duckdb(dataset: BenchmarkDataset, format: Format) -> anyhow::Result<Self> {
        Ok(EngineCtx::DuckDB(ddb::DuckDBCtx::new(dataset, format)?))
    }

    pub fn to_engine(&self) -> Engine {
        match &self {
            EngineCtx::DuckDB(_) => Engine::DuckDB,
            EngineCtx::DataFusion(_) => Engine::DataFusion,
        }
    }
}

pub fn benchmark_duckdb_query(
    query_idx: usize,
    query_string: &str,
    iterations: usize,
    duckdb_ctx: &ddb::DuckDBCtx,
) -> (Duration, usize) {
    let mut fastest = Duration::from_millis(u64::MAX);
    let mut row_count = 0;

    for _ in 0..iterations {
        let (duration, current_row_count) = duckdb_ctx
            .execute_query(query_string)
            .unwrap_or_else(|err| vortex_panic!("query: {query_idx} failed with: {err}"));

        if duration < fastest {
            fastest = duration;
            row_count = current_row_count;
        }
    }

    (fastest, row_count)
}

pub async fn benchmark_datafusion_query<T, F, Fut>(iterations: usize, mut f: F) -> (Duration, T)
where
    F: FnMut() -> Fut,
    Fut: Future<Output = T>,
{
    let mut result = None;
    let mut fastest = Duration::from_millis(u64::MAX);

    for _ in 0..iterations {
        let start = std::time::Instant::now();
        let iter_result = f().await;
        let duration = start.elapsed();

        if result.is_none() {
            result = Some(iter_result);
        }

        fastest = fastest.min(duration);
    }

    (fastest, result.vortex_expect("Result must be set"))
}
