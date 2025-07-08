// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Simple benchmark trait interface focused on core operations

use anyhow::Result;
use url::Url;

use crate::engines::EngineCtx;
use crate::{BenchmarkDataset, Format, Target};

/// Core benchmark operations that all benchmark types implement
pub trait Benchmark {
    /// Get all available queries for this benchmark
    fn queries(&self) -> Result<Vec<(usize, String)>>;

    /// Generate or prepare data for the benchmark at the specified URL for a specific target
    /// This should be idempotent - safe to call multiple times for the same target
    fn generate_data(&self, data_url: &Url, target: &Target) -> Result<()>;

    /// Register tables with the engine context
    #[allow(async_fn_in_trait)]
    async fn register_tables(
        &self,
        engine_ctx: &EngineCtx,
        data_url: &Url,
        format: Format,
    ) -> Result<()>;

    /// Get the benchmark dataset identifier
    fn get_dataset(&self) -> BenchmarkDataset;

    /// Get expected row counts for validation (optional)
    /// If None, no validation will be performed
    fn get_expected_row_counts(&self) -> Option<&[usize]> {
        None
    }
}
