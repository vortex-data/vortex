// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Per-iteration scan driver.
//!
//! Each iteration re-opens every `.vortex` shard fresh (so the segment cache is re-primed
//! per run), pushes the cosine-similarity filter through the scan, and drains the resulting
//! [`vortex_array::stream::ArrayStream`]. The wall-clock around the entire per-iteration
//! pass is the headline number; we track best-of-N and median across iterations.

use std::path::Path;
use std::path::PathBuf;
use std::time::Duration;
use std::time::Instant;

use anyhow::Context;
use anyhow::Result;
use futures::TryStreamExt;
use vortex::array::ArrayRef;
use vortex::file::OpenOptionsSessionExt;

use crate::compression::VortexCompression;
use crate::expression::similarity_filter;
use crate::prepare::CompressionResult;
use crate::session::SESSION;

/// Inputs to a scan run.
#[derive(Debug, Clone)]
pub struct ScanConfig {
    /// Number of timed iterations (best-of-N).
    pub iterations: usize,
    /// Cosine threshold passed to the filter expression.
    pub threshold: f32,
}

/// Aggregate timing + counters for one `(flavor)` scan.
#[derive(Debug, Clone)]
pub struct ScanTiming {
    /// Which compression flavor's `.vortex` files were scanned.
    pub flavor: VortexCompression,
    /// Best (minimum) wall-clock across iterations.
    pub best_of: Duration,
    /// Per-iteration wall times in run order.
    pub all_runs: Vec<Duration>,
    /// Number of rows that survived the filter (constant across iterations because the
    /// filter is deterministic).
    pub matches: u64,
    /// Total rows scanned (sum of file row counts) — a sanity check that the iteration
    /// actually walked the files.
    pub rows_scanned: u64,
}

impl ScanTiming {
    /// Median wall time across iterations.
    pub fn median(&self) -> Duration {
        crate::scan_util::median(&self.all_runs)
    }
}

/// Scan every shard in a [`CompressionResult`] under the given config.
pub async fn run_scan(
    result: &CompressionResult,
    query: &[f32],
    config: &ScanConfig,
) -> Result<ScanTiming> {
    anyhow::ensure!(
        config.iterations > 0,
        "scan iterations must be >= 1, got {}",
        config.iterations
    );

    let mut all_runs = Vec::with_capacity(config.iterations);
    let mut matches = 0u64;
    let mut rows_scanned = 0u64;

    for iter_idx in 0..config.iterations {
        let (wall, iter_matches, iter_rows) =
            run_one_iteration(&result.vortex_files, query, config.threshold).await?;
        tracing::debug!(
            "{} iter {} -> {:?} ({} matches, {} rows)",
            result.flavor.label(),
            iter_idx,
            wall,
            iter_matches,
            iter_rows,
        );
        // Matches and row counts are deterministic across iterations; reset rather than
        // accumulate so the reported value matches a single pass.
        matches = iter_matches;
        rows_scanned = iter_rows;
        all_runs.push(wall);
    }

    let best_of = all_runs
        .iter()
        .copied()
        .min()
        .context("scan produced no iterations")?;

    Ok(ScanTiming {
        flavor: result.flavor,
        best_of,
        all_runs,
        matches,
        rows_scanned,
    })
}

async fn run_one_iteration(
    vortex_files: &[PathBuf],
    query: &[f32],
    threshold: f32,
) -> Result<(Duration, u64, u64)> {
    let started = Instant::now();
    let mut matches = 0u64;
    let mut rows_scanned = 0u64;
    for path in vortex_files {
        let (m, r) = scan_one_file(path, query, threshold).await?;
        matches = matches.saturating_add(m);
        rows_scanned = rows_scanned.saturating_add(r);
    }
    Ok((started.elapsed(), matches, rows_scanned))
}

async fn scan_one_file(path: &Path, query: &[f32], threshold: f32) -> Result<(u64, u64)> {
    let session = &*SESSION;
    let file = session
        .open_options()
        .open_path(path)
        .await
        .with_context(|| format!("open {}", path.display()))?;
    let total_rows = file.row_count();
    let filter = similarity_filter(query, threshold)?;
    let chunks: Vec<ArrayRef> = file
        .scan()?
        .with_filter(filter)
        .into_array_stream()?
        .try_collect()
        .await?;
    let matches: u64 = chunks.iter().map(|c| c.len() as u64).sum();
    Ok((matches, total_rows))
}
