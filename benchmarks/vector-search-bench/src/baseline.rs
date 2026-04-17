// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Hand-rolled non-Vortex cosine-similarity baseline.
//!
//! The baseline shard format is a flat binary file: `num_rows * dim * 4` little-endian f32 bytes
//! with no header or validity. Each row is L2-normalized at write time, so cosine similarity at
//! scan time collapses to a single dot product against a pre-normalized query.
//!
//! This path intentionally uses raw Arrow + `std::fs` I/O instead of any Vortex primitive so it
//! can serve as a "theoretical minimum" ceiling to compare the Vortex flavors against.
//!
//! The current implementation is deliberately simple: a straight `iter().zip().map().sum()` dot
//! product loop, single-threaded, without any manual SIMD. A future revision will hand-tune the
//! inner loop (target-specific intrinsics or `std::simd`) once we are happy with the API shape.

use std::fs;
use std::fs::File as StdFile;
use std::io::BufWriter;
use std::io::Write as _;
use std::path::Path;

use anyhow::Context;
use anyhow::Result;
use anyhow::bail;
use anyhow::ensure;
use arrow_array::Array as _;
use arrow_array::Float32Array;
use arrow_array::Float64Array;
use arrow_array::ListArray;
use futures::StreamExt;
use parquet::arrow::ParquetRecordBatchStreamBuilder;
use tokio::fs::File as TokioFile;
use vortex::dtype::PType;

/// L2-normalize a vector in place.
///
/// Zero-magnitude vectors are left as zero so their dot product against any normalized query is
/// `0.0` and cannot match a positive threshold.
pub fn l2_normalize_in_place(v: &mut [f32]) {
    let norm_sq: f32 = v.iter().map(|&x| x * x).sum();
    if norm_sq > 0.0 {
        let inv_norm = norm_sq.sqrt().recip();
        for x in v.iter_mut() {
            *x *= inv_norm;
        }
    }
}

/// Return an L2-normalized copy of `v`.
pub fn l2_normalize_copy(v: &[f32]) -> Vec<f32> {
    let mut out = v.to_vec();
    l2_normalize_in_place(&mut out);
    out
}

/// Dot product of two equal-length f32 slices.
///
/// Kept deliberately simple for v1: the straight-line zip+map+sum pattern. LLVM may not fully
/// auto-vectorize this under strict IEEE ordering; future work will introduce manual SIMD.
#[inline]
pub fn dot_product(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());
    a.iter().zip(b.iter()).map(|(&x, &y)| x * y).sum()
}

/// Decode a row of `dim` little-endian f32s from `bytes` into `out`.
#[inline]
fn decode_row_le(bytes: &[u8], out: &mut [f32]) {
    debug_assert_eq!(bytes.len(), out.len() * 4);
    for (dst, src) in out.iter_mut().zip(bytes.chunks_exact(4)) {
        *dst = f32::from_le_bytes([src[0], src[1], src[2], src[3]]);
    }
}

/// Stream a parquet shard through arrow-rs and write it as a flat `.f32` file of pre-L2-normalized
/// contiguous little-endian f32 rows.
///
/// The source `emb` column is expected to be an arrow `ListArray` of either f32 or f64 (f64 is
/// narrowed to f32 here, matching the Vortex ingest path). Every row must have exactly `dim`
/// elements.
pub async fn write_shard_raw_f32(
    parquet_path: &Path,
    output_path: &Path,
    dim: usize,
    src_ptype: PType,
) -> Result<()> {
    let file = TokioFile::open(parquet_path)
        .await
        .with_context(|| format!("open parquet {}", parquet_path.display()))?;
    let builder = ParquetRecordBatchStreamBuilder::new(file).await?;
    let mut stream = builder.build()?;

    let out_file = StdFile::create(output_path)
        .with_context(|| format!("create {}", output_path.display()))?;
    let mut writer = BufWriter::with_capacity(1 << 20, out_file);

    let mut row: Vec<f32> = vec![0.0; dim];
    while let Some(batch) = stream.next().await {
        let batch = batch?;
        let emb_col = batch
            .column_by_name("emb")
            .context("parquet batch missing `emb` column")?;
        let list = emb_col
            .as_any()
            .downcast_ref::<ListArray>()
            .context("expected `emb` to be a ListArray")?;

        for row_idx in 0..list.len() {
            let values = list.value(row_idx);
            match src_ptype {
                PType::F32 => {
                    let arr = values
                        .as_any()
                        .downcast_ref::<Float32Array>()
                        .context("expected f32 list elements")?;
                    ensure!(
                        arr.len() == dim,
                        "row has wrong dim: got {}, expected {dim}",
                        arr.len()
                    );
                    row.copy_from_slice(arr.values());
                }
                PType::F64 => {
                    let arr = values
                        .as_any()
                        .downcast_ref::<Float64Array>()
                        .context("expected f64 list elements")?;
                    ensure!(
                        arr.len() == dim,
                        "row has wrong dim: got {}, expected {dim}",
                        arr.len()
                    );
                    #[expect(
                        clippy::cast_possible_truncation,
                        reason = "intentional f64 -> f32 narrowing, matches the Vortex ingest path"
                    )]
                    for (dst, &src) in row.iter_mut().zip(arr.values().iter()) {
                        *dst = src as f32;
                    }
                }
                other => bail!("unsupported emb element ptype {other}, expected f32 or f64"),
            }
            l2_normalize_in_place(&mut row);
            for &value in &row {
                writer.write_all(&value.to_le_bytes())?;
            }
        }
    }
    writer.flush()?;
    Ok(())
}

/// Scan a flat `.f32` shard and return `(matches, rows_scanned)`.
///
/// Loads the whole shard into RAM with a single `std::fs::read`, then walks it row by row
/// computing `dot_product(row, query_normalized)` and counting rows above `threshold`.
/// `query_normalized` must already be L2-normalized by the caller.
pub fn scan_shard_raw_f32(
    path: &Path,
    query_normalized: &[f32],
    threshold: f32,
    dim: usize,
) -> Result<(u64, u64)> {
    ensure!(
        query_normalized.len() == dim,
        "query dim mismatch: query is {}, shard dim is {dim}",
        query_normalized.len()
    );

    let bytes = fs::read(path).with_context(|| format!("read {}", path.display()))?;
    let row_bytes = dim.checked_mul(4).context("dim * 4 overflowed")?;
    ensure!(
        bytes.len() % row_bytes == 0,
        "file length {} not a multiple of row size {row_bytes}",
        bytes.len()
    );
    let row_count = (bytes.len() / row_bytes) as u64;

    let mut row: Vec<f32> = vec![0.0; dim];
    let mut matches = 0u64;
    let mut offset = 0;
    while offset < bytes.len() {
        decode_row_le(&bytes[offset..offset + row_bytes], &mut row);
        if dot_product(&row, query_normalized) > threshold {
            matches = matches.saturating_add(1);
        }
        offset += row_bytes;
    }

    Ok((matches, row_count))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn normalize_unit_vector_is_idempotent() {
        let mut v = vec![1.0f32, 0.0, 0.0];
        l2_normalize_in_place(&mut v);
        assert_eq!(v, vec![1.0, 0.0, 0.0]);
    }

    #[test]
    fn normalize_scales_to_unit_length() {
        let mut v = vec![3.0f32, 4.0];
        l2_normalize_in_place(&mut v);
        let norm: f32 = v.iter().map(|&x| x * x).sum::<f32>().sqrt();
        assert!((norm - 1.0).abs() < 1e-6, "got norm {norm}");
    }

    #[test]
    fn normalize_leaves_zero_vector_untouched() {
        let mut v = vec![0.0f32; 4];
        l2_normalize_in_place(&mut v);
        assert_eq!(v, vec![0.0; 4]);
    }

    #[test]
    fn dot_product_matches_manual_sum() {
        let a = [1.0_f32, 2.0, 3.0];
        let b = [4.0_f32, 5.0, 6.0];
        assert_eq!(dot_product(&a, &b), 1.0 * 4.0 + 2.0 * 5.0 + 3.0 * 6.0);
    }

    #[test]
    fn scan_counts_above_threshold() -> Result<()> {
        let dim = 4;
        let query = [1.0_f32, 0.0, 0.0, 0.0];
        let query_n = l2_normalize_copy(&query);

        // Three rows: first two are parallel to the query (dot=1.0), last is orthogonal (dot=0.0).
        let rows: [[f32; 4]; 3] = [
            [1.0, 0.0, 0.0, 0.0],
            [2.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
        ];

        let tmp = tempfile::NamedTempFile::new()?;
        {
            let mut w = BufWriter::new(tmp.reopen()?);
            for r in &rows {
                let mut rn = r.to_vec();
                l2_normalize_in_place(&mut rn);
                for &v in &rn {
                    w.write_all(&v.to_le_bytes())?;
                }
            }
            w.flush()?;
        }

        let (matches, scanned) = scan_shard_raw_f32(tmp.path(), &query_n, 0.5, dim)?;
        assert_eq!(scanned, 3);
        assert_eq!(matches, 2);
        Ok(())
    }
}
