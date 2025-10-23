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

    pub fn new_with_duckdb(
        dataset: BenchmarkDataset,
        format: Format,
        delete_duckdb_database: bool,
        threads: Option<usize>,
    ) -> anyhow::Result<Self> {
        Ok(EngineCtx::DuckDB(ddb::DuckDBCtx::new(
            dataset,
            format,
            delete_duckdb_database,
            threads,
        )?))
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
) -> (Vec<Duration>, usize) {
    let mut runs = Vec::with_capacity(iterations);
    let mut row_count = None;

    for _ in 0..iterations {
        let (duration, current_row_count) = duckdb_ctx
            .execute_query(query_string)
            .unwrap_or_else(|err| vortex_panic!("query: {query_idx} failed with: {err}"));

        runs.push(duration);
        row_count.inspect(|rc| assert_eq!(*rc, current_row_count));
        row_count = Some(current_row_count);
    }

    (runs, row_count.vortex_expect("cannot have zero runs"))
}

pub async fn benchmark_datafusion_query<T, F, Fut>(
    iterations: usize,
    mut f: F,
) -> (Vec<Duration>, T)
where
    F: FnMut() -> Fut,
    Fut: Future<Output = T>,
{
    let mut result = None;
    let mut runs = Vec::with_capacity(iterations);

    for _ in 0..iterations {
        let start = std::time::Instant::now();
        let iter_result = f().await;
        let duration = start.elapsed();

        runs.push(duration);
        if result.is_none() {
            result = Some(iter_result);
        }
    }

    (runs, result.vortex_expect("Result must be set"))
}
