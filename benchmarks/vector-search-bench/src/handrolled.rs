// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Hand-rolled `&[f32]` parquet baseline.
//!
//! Intentionally minimal — `parquet-rs` decodes the `emb` column to a flat `Vec<f32>`,
//! then a plain scalar Rust loop computes cosine similarity row-by-row and applies the
//! threshold. This is the *compute-cost floor* the Vortex variants are measured against,
//! **not** a realistic parquet-on-DBMS baseline (a real engine would pay substantial
//! dispatch / planner / row-iterator cost this loop skips).
//!
//! Sequential across shards — matches the default-sequential Vortex scan, so the wall-clock
//! comparison is apples-to-apples.

use std::path::Path;
use std::path::PathBuf;
use std::time::Duration;
use std::time::Instant;

use anyhow::Context;
use anyhow::Result;
use anyhow::bail;

use crate::handrolled_decode::decode_parquet_emb;

/// Aggregate timing + counters for one handrolled scan.
#[derive(Debug, Clone)]
pub struct HandrolledTiming {
    /// Best (minimum) wall-clock across iterations.
    pub best_of: Duration,
    /// Per-iteration wall times in run order.
    pub all_runs: Vec<Duration>,
    /// Number of rows that survived the filter.
    pub matches: u64,
    /// Total rows scanned.
    pub rows_scanned: u64,
    /// Sum of input parquet shard sizes — emitted as the handrolled "size" measurement.
    pub total_input_bytes: u64,
}

impl HandrolledTiming {
    /// Median wall time across iterations.
    pub fn median(&self) -> Duration {
        crate::scan_util::median(&self.all_runs)
    }
}

/// Run the hand-rolled cosine-similarity scan over a list of parquet shards.
pub fn run_handrolled_scan(
    parquet_files: &[PathBuf],
    query: &[f32],
    threshold: f32,
    iterations: usize,
) -> Result<HandrolledTiming> {
    anyhow::ensure!(
        iterations > 0,
        "run_handrolled_scan requires iterations >= 1"
    );

    let mut total_input_bytes = 0u64;
    for path in parquet_files {
        total_input_bytes = total_input_bytes.saturating_add(
            std::fs::metadata(path)
                .with_context(|| format!("stat {}", path.display()))?
                .len(),
        );
    }

    let mut all_runs = Vec::with_capacity(iterations);
    let mut matches = 0u64;
    let mut rows_scanned = 0u64;

    for iter_idx in 0..iterations {
        let started = Instant::now();
        let (m, r) = scan_iteration(parquet_files, query, threshold)?;
        let wall = started.elapsed();
        all_runs.push(wall);
        matches = m;
        rows_scanned = r;
        tracing::debug!(
            "handrolled iter {} -> {:?} ({} matches, {} rows)",
            iter_idx,
            wall,
            m,
            r
        );
    }

    let best_of = all_runs.iter().copied().min().unwrap_or(Duration::ZERO);

    Ok(HandrolledTiming {
        best_of,
        all_runs,
        matches,
        rows_scanned,
        total_input_bytes,
    })
}

fn scan_iteration(parquet_files: &[PathBuf], query: &[f32], threshold: f32) -> Result<(u64, u64)> {
    let mut matches = 0u64;
    let mut rows = 0u64;
    for path in parquet_files {
        let (m, r) = scan_shard(path, query, threshold)?;
        matches = matches.saturating_add(m);
        rows = rows.saturating_add(r);
    }
    Ok((matches, rows))
}

fn scan_shard(path: &Path, query: &[f32], threshold: f32) -> Result<(u64, u64)> {
    let data = decode_parquet_emb(path)?;
    if data.dim != query.len() {
        bail!(
            "handrolled: shard {} has dim {} but query has dim {}",
            path.display(),
            data.dim,
            query.len()
        );
    }
    let scores = cosine_loop(&data.elements, data.num_rows, data.dim, query);
    let matches = scores.iter().filter(|&&s| s > threshold).count();
    Ok((matches as u64, data.num_rows as u64))
}

/// Plain scalar 4-way unrolled dot product, divided by `||query||` (data is assumed unit
/// norm — the prepare step does L2 normalization implicitly via TurboQuant on the Vortex
/// side; here we re-normalize per row to keep the comparison fair).
pub fn cosine_loop(elements: &[f32], num_rows: usize, dim: usize, query: &[f32]) -> Vec<f32> {
    assert_eq!(query.len(), dim);
    assert_eq!(elements.len(), num_rows * dim);

    let query_norm = query.iter().map(|&q| q * q).sum::<f32>().sqrt();
    let mut out = Vec::with_capacity(num_rows);
    if query_norm == 0.0 {
        out.resize(num_rows, 0.0);
        return out;
    }
    let inv_query_norm = 1.0 / query_norm;

    for slice in elements.chunks_exact(dim) {
        let row_norm_sq: f32 = slice.iter().map(|&v| v * v).sum();
        let row_norm = row_norm_sq.sqrt();
        if row_norm == 0.0 {
            out.push(0.0);
            continue;
        }
        let inv_row_norm = 1.0 / row_norm;

        let mut dot0 = 0.0f32;
        let mut dot1 = 0.0f32;
        let mut dot2 = 0.0f32;
        let mut dot3 = 0.0f32;
        let chunks = slice.chunks_exact(4);
        let q_chunks = query.chunks_exact(4);
        let rem = chunks.remainder();
        let q_rem = q_chunks.remainder();
        for (s, q) in chunks.zip(q_chunks) {
            dot0 += s[0] * q[0];
            dot1 += s[1] * q[1];
            dot2 += s[2] * q[2];
            dot3 += s[3] * q[3];
        }
        for (&s, &q) in rem.iter().zip(q_rem.iter()) {
            dot0 += s * q;
        }

        out.push((dot0 + dot1 + dot2 + dot3) * inv_query_norm * inv_row_norm);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cosine_self_match_is_one() {
        let elements = vec![1.0, 0.0, 0.0, 0.0];
        let query = vec![1.0f32, 0.0, 0.0, 0.0];
        let scores = cosine_loop(&elements, 1, 4, &query);
        assert!((scores[0] - 1.0).abs() < 1e-6, "got {}", scores[0]);
    }

    #[test]
    fn cosine_orthogonal_is_zero() {
        let elements = vec![0.0, 1.0, 0.0, 0.0];
        let query = vec![1.0f32, 0.0, 0.0, 0.0];
        let scores = cosine_loop(&elements, 1, 4, &query);
        assert!(scores[0].abs() < 1e-6, "got {}", scores[0]);
    }
}
