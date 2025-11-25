// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Simple benchmark trait interface focused on core operations

use anyhow::Result;
use url::Url;

use crate::engines::EngineCtx;
use crate::{BenchmarkDataset, Engine, Format, Target, df};

/// Core benchmark operations that all benchmark types implement
pub trait Benchmark {
    /// Get all available queries for this benchmark
    fn queries(&self) -> Result<Vec<(usize, String)>>;

    fn setup_engine_context(
        &self,
        target: &Target,
        disable_datafusion_cache: bool,
        emit_plan: bool,
        delete_duckdb_database: bool,
        threads: Option<usize>,
    ) -> Result<EngineCtx> {
        let engine = target.engine();
        let format = target.format();

        match engine {
            Engine::DataFusion => {
                let session_ctx = df::get_session_context(disable_datafusion_cache);
                df::make_object_store(&session_ctx, self.data_url())?;
                Ok(EngineCtx::new_with_datafusion(session_ctx, emit_plan))
            }
            Engine::DuckDB => {
                // Create a generic dataset for DuckDB context creation
                // This will be properly configured when tables are registered
                Ok(EngineCtx::new_with_duckdb(
                    self.dataset(),
                    format,
                    delete_duckdb_database,
                    threads,
                )?)
            }
            _ => unreachable!("engine not supported"),
        }
    }

    /// Generate or prepare data for the benchmark at the specified URL for a specific target
    /// This should be idempotent - safe to call multiple times for the same target
    fn generate_data(&self, target: &Target) -> Result<()>;

    /// Register tables with the engine context
    #[allow(async_fn_in_trait)]
    async fn register_tables(&self, engine_ctx: &EngineCtx, format: Format) -> Result<()>;

    /// Get the benchmark dataset identifier
    fn dataset(&self) -> BenchmarkDataset;

    /// Get expected row counts for validation (optional)
    /// If None, no validation will be performed
    fn expected_row_counts(&self) -> Option<&[usize]> {
        None
    }

    /// Get the name of the benchmark dataset
    fn dataset_name(&self) -> &str;

    /// Get the table names for this dataset (used for TPC benchmarks)
    fn tables(&self) -> &[&'static str] {
        &[] // Default empty for benchmarks that don't need this
    }

    /// Format a path for the given format and base URL
    fn format_path(&self, format: Format, base_url: &Url) -> Result<Url> {
        Ok(base_url.join(&format!("{}/", format.name()))?)
    }

    /// Get display string for the dataset (used in measurements)
    fn dataset_display(&self) -> String;

    fn validate_result(&self, _queries: Vec<usize>) -> Result<()> {
        Ok(())
    }

    fn data_url(&self) -> &Url;
}
