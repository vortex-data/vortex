// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Per-iteration scan driver.
//!
//! Each iteration re-opens every `.vortex` shard fresh (so the segment cache is re-primed
//! per run), pushes the cosine-similarity filter through the scan, and drains the resulting
//! [`vortex::array::stream::ArrayStream`]. The wall-clock around the entire per-iteration
//! pass is the headline number; we track the mean and median across iterations.

use std::path::Path;
use std::path::PathBuf;
use std::time::Duration;
use std::time::Instant;

use anyhow::Context;
use anyhow::Result;
use futures::TryStreamExt;
use vortex::array::ArrayRef;
use vortex::file::OpenOptionsSessionExt;

use crate::SESSION;
use crate::compression::VectorFlavor;
use crate::expression::similarity_filter;
use crate::prepare::CompressedVortexDataset;

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
    pub flavor: VectorFlavor,
    /// Arithmetic mean of the per-iteration wall times.
    pub mean: Duration,
    /// Median of the per-iteration wall times.
    pub median: Duration,
    /// Per-iteration wall times in run order.
    pub all_runs: Vec<Duration>,
    /// Number of rows that survived the filter (constant across iterations because the
    /// filter is deterministic).
    pub matches: u64,
    /// Total rows scanned (sum of file row counts) as a sanity check that the iteration
    /// actually walked the files.
    pub rows_scanned: u64,
    /// Total on-disk size of the scanned `.vortex` files, in bytes.
    pub bytes_scanned: u64,
}

/// Scan every shard in a [`CompressedVortexDataset`] under the given config.
pub async fn run_search_scan(
    dataset: &CompressedVortexDataset,
    query: &[f32],
    config: &ScanConfig,
) -> Result<ScanTiming> {
    anyhow::ensure!(
        config.iterations > 0,
        "scan iterations must be >= 1, got {}",
        config.iterations
    );

    let bytes_scanned = total_file_size(&dataset.vortex_files)?;

    let mut all_runs = Vec::with_capacity(config.iterations);
    let mut matches = 0u64;
    let mut rows_scanned = 0u64;

    for iter_idx in 0..config.iterations {
        let (wall, iter_matches, iter_rows) =
            run_one_iteration(&dataset.vortex_files, query, config.threshold).await?;
        tracing::debug!(
            "{} iter {} -> {:?} ({} matches, {} rows)",
            dataset.flavor.label(),
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

    Ok(ScanTiming {
        flavor: dataset.flavor,
        mean: mean(&all_runs),
        median: median(&all_runs),
        all_runs,
        matches,
        rows_scanned,
        bytes_scanned,
    })
}

/// Sum the on-disk sizes of the given files.
fn total_file_size(paths: &[PathBuf]) -> Result<u64> {
    let mut total = 0u64;
    for path in paths {
        let meta =
            std::fs::metadata(path).with_context(|| format!("stat {} for size", path.display()))?;
        total = total.saturating_add(meta.len());
    }
    Ok(total)
}

async fn run_one_iteration(
    vortex_files: &[PathBuf],
    query: &[f32],
    threshold: f32,
) -> Result<(Duration, u64, u64)> {
    let mut matches = 0u64;
    let mut rows_scanned = 0u64;

    let started = Instant::now();
    for path in vortex_files {
        let (m, r) = scan_one_file(path, query, threshold).await?;
        matches = matches.saturating_add(m);
        rows_scanned = rows_scanned.saturating_add(r);
    }

    Ok((started.elapsed(), matches, rows_scanned))
}

async fn scan_one_file(path: &Path, query: &[f32], threshold: f32) -> Result<(u64, u64)> {
    let file = SESSION
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

/// Arithmetic mean of a list of [`Duration`]s. Empty lists return [`Duration::ZERO`].
pub fn mean(runs: &[Duration]) -> Duration {
    if runs.is_empty() {
        return Duration::ZERO;
    }
    let total_nanos: u128 = runs.iter().map(|d| d.as_nanos()).sum();
    let avg_nanos = total_nanos / runs.len() as u128;
    Duration::from_nanos(u64::try_from(avg_nanos).unwrap_or(u64::MAX))
}

/// Median of a list of [`Duration`]s. Empty lists return [`Duration::ZERO`].
pub fn median(runs: &[Duration]) -> Duration {
    if runs.is_empty() {
        return Duration::ZERO;
    }
    let mut sorted = runs.to_vec();
    sorted.sort();
    let mid = sorted.len() / 2;
    if sorted.len() % 2 == 1 {
        sorted[mid]
    } else {
        let total_nanos = sorted[mid - 1].as_nanos() + sorted[mid].as_nanos();
        Duration::from_nanos(u64::try_from(total_nanos / 2).unwrap_or(u64::MAX))
    }
}
