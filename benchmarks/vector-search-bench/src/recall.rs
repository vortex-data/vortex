// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Recall@K vs ground-truth `neighbors.parquet`.
//!
//! For each sampled query row:
//!
//! 1. Read the corresponding row of `neighbors.parquet` to get the upstream top-K ids.
//! 2. Scan every `.vortex` shard (no filter), running cosine-similarity against the query
//!    and keeping a running top-K heap of `(score, ord_id)` pairs across the dataset.
//! 3. Recall = |intersect(predicted_top_k, ground_truth_top_k)| / K.
//!
//! Lossless flavors are trivially 1.0 by construction, so they're skipped unless the caller
//! explicitly asks. Only TurboQuant actually needs measurement.
//!
//! `ord_id` is the row's position in the train-split scan order — i.e. the row index that
//! upstream `neighbors.parquet` rows already use. Both Cohere/OpenAI/Bioasq partitioned
//! neighbor files reference the canonical scan order of the train shards, which is
//! exactly what we walk here.

use std::cmp::Ordering;
use std::collections::BinaryHeap;
use std::path::Path;

use anyhow::Context;
use anyhow::Result;
use anyhow::bail;
use futures::TryStreamExt;
use rand::RngExt;
use rand::SeedableRng;
use rand::rngs::StdRng;
use vortex::array::ArrayRef;
use vortex::array::IntoArray;
use vortex::array::VortexSessionExecute;
use vortex::array::arrays::ExtensionArray;
use vortex::array::arrays::FixedSizeListArray;
use vortex::array::arrays::PrimitiveArray;
use vortex::array::arrays::StructArray;
use vortex::array::arrays::extension::ExtensionArrayExt;
use vortex::array::arrays::fixed_size_list::FixedSizeListArrayExt;
use vortex::array::arrays::struct_::StructArrayExt;
use vortex::dtype::DType;
use vortex::dtype::PType;
use vortex::file::OpenOptionsSessionExt;
use vortex_bench::conversions::list_to_vector_ext;
use vortex_bench::conversions::parquet_to_vortex_chunks;

use crate::compression::VortexCompression;
use crate::expression::emb_projection;
use crate::prepare::CompressionResult;
use crate::session::SESSION;

/// Inputs to a recall measurement.
#[derive(Debug, Clone)]
pub struct RecallConfig {
    /// `K` in Recall@K.
    pub k: usize,
    /// Number of query rows to sample from `test.parquet`.
    pub num_queries: usize,
    /// Seed for picking query rows. Distinct seeds produce different but reproducible runs.
    pub query_seed: u64,
}

/// Per-flavor recall statistics across `num_queries` sampled queries.
#[derive(Debug, Clone)]
pub struct RecallResult {
    pub flavor: VortexCompression,
    pub k: usize,
    pub queries_run: usize,
    /// Mean of per-query recall@K values, range `[0.0, 1.0]`.
    pub mean_recall: f64,
    /// 5th-percentile (worst-case-ish) recall across queries.
    pub p05_recall: f64,
}

/// Compute recall@K for one prepared flavor against `neighbors.parquet`.
pub async fn measure_recall(
    result: &CompressionResult,
    test_parquet: &Path,
    neighbors_parquet: &Path,
    src_ptype: PType,
    config: &RecallConfig,
) -> Result<RecallResult> {
    anyhow::ensure!(config.k > 0, "recall k must be >= 1");
    anyhow::ensure!(config.num_queries > 0, "recall num_queries must be >= 1");

    let queries = load_test_queries(test_parquet, src_ptype, config).await?;
    let neighbors = load_neighbors(neighbors_parquet, config.k).await?;
    if neighbors.num_rows < queries.len() as u64 {
        bail!(
            "neighbors.parquet has {} rows, expected at least {}",
            neighbors.num_rows,
            queries.len()
        );
    }

    let mut per_query: Vec<f64> = Vec::with_capacity(queries.len());
    for q in &queries {
        let predicted = top_k_for_query(&result.vortex_files, &q.values, config.k).await?;
        let truth = neighbors.row_at(q.row_idx, config.k);
        per_query.push(recall_intersection(&predicted, truth));
    }

    let mean_recall = per_query.iter().sum::<f64>() / per_query.len() as f64;
    let p05_recall = percentile(&per_query, 0.05);

    Ok(RecallResult {
        flavor: result.flavor,
        k: config.k,
        queries_run: per_query.len(),
        mean_recall,
        p05_recall,
    })
}

#[derive(Debug, Clone)]
struct SampledQuery {
    row_idx: u64,
    values: Vec<f32>,
}

async fn load_test_queries(
    test_parquet: &Path,
    src_ptype: PType,
    config: &RecallConfig,
) -> Result<Vec<SampledQuery>> {
    let chunked = parquet_to_vortex_chunks(test_parquet.to_path_buf())
        .await
        .with_context(|| format!("read test parquet {}", test_parquet.display()))?;
    let mut ctx = SESSION.create_execution_ctx();
    let materialized: StructArray = chunked.into_array().execute(&mut ctx)?;
    let emb = materialized
        .unmasked_field_by_name("emb")
        .context("test parquet missing `emb` column")?
        .clone();
    let emb_ext: ExtensionArray = list_to_vector_ext(emb)?.execute(&mut ctx)?;
    let fsl: FixedSizeListArray = emb_ext.storage_array().clone().execute(&mut ctx)?;
    let dim = match fsl.dtype() {
        DType::FixedSizeList(_, dim, _) => *dim as usize,
        other => bail!("test parquet emb dtype is not FSL: {other}"),
    };
    let elements: PrimitiveArray = fsl.elements().clone().execute(&mut ctx)?;
    let num_test = u64::try_from(fsl.len()).unwrap_or(u64::MAX);
    if num_test == 0 {
        bail!("test parquet has zero rows");
    }

    let mut rng = StdRng::seed_from_u64(config.query_seed);
    let mut samples = Vec::with_capacity(config.num_queries);
    for _ in 0..config.num_queries {
        let row = rng.random_range(0..num_test);
        let row_usize = usize::try_from(row).unwrap_or(usize::MAX);
        let start = row_usize * dim;
        let end = start + dim;
        let values = match src_ptype {
            PType::F32 => elements.as_slice::<f32>()[start..end].to_vec(),
            PType::F64 =>
            {
                #[expect(clippy::cast_possible_truncation)]
                elements.as_slice::<f64>()[start..end]
                    .iter()
                    .map(|&v| v as f32)
                    .collect()
            }
            other => bail!("unsupported test ptype {other}"),
        };
        samples.push(SampledQuery {
            row_idx: row,
            values,
        });
    }
    Ok(samples)
}

#[derive(Debug, Clone)]
struct GroundTruth {
    /// Flat row-major buffer: `ids[query_idx * width + k] = neighbor_id`.
    ids: Vec<i64>,
    /// Width of each row (= upstream K, not necessarily our recall K).
    width: usize,
    /// Total rows in `neighbors.parquet`.
    num_rows: u64,
}

impl GroundTruth {
    fn row_at(&self, query_idx: u64, k: usize) -> &[i64] {
        let start = usize::try_from(query_idx).unwrap_or(usize::MAX) * self.width;
        let take = k.min(self.width);
        &self.ids[start..start + take]
    }
}

async fn load_neighbors(neighbors_parquet: &Path, k: usize) -> Result<GroundTruth> {
    let chunked = parquet_to_vortex_chunks(neighbors_parquet.to_path_buf())
        .await
        .with_context(|| format!("read neighbors parquet {}", neighbors_parquet.display()))?;
    let mut ctx = SESSION.create_execution_ctx();
    let materialized: StructArray = chunked.into_array().execute(&mut ctx)?;
    // Some neighbor parquets are FixedSizeList<i64, K>, some are List<i64> or LargeList<i64>;
    // `list_to_vector_ext` doesn't apply (it requires float elements), so we read manually.
    // Try common column names in order; the upstream files use `neighbors_id` or `id`.
    let neighbors_col = materialized
        .unmasked_field_by_name("neighbors_id")
        .or_else(|_| materialized.unmasked_field_by_name("id"))
        .context("neighbors parquet missing `neighbors_id`/`id` column")?
        .clone();

    // Collapse the column into a flat (ids, width) buffer.
    let (ids, width, num_rows) = flatten_neighbors(neighbors_col, &mut ctx)?;
    if width < k {
        bail!("neighbors.parquet K={width} is smaller than requested recall K={k}",);
    }
    Ok(GroundTruth {
        ids,
        width,
        num_rows,
    })
}

fn flatten_neighbors(
    col: ArrayRef,
    ctx: &mut vortex::array::ExecutionCtx,
) -> Result<(Vec<i64>, usize, u64)> {
    use vortex::array::arrays::List;
    use vortex::array::arrays::ListView;
    use vortex::array::arrays::list::ListArrayExt;
    use vortex::array::arrays::listview::recursive_list_from_list_view;

    let canon = if col.as_opt::<ListView>().is_some() {
        recursive_list_from_list_view(col)?
    } else {
        col
    };
    if let Some(list) = canon.as_opt::<List>() {
        let elements: PrimitiveArray = list.sliced_elements()?.execute(ctx)?;
        if elements.ptype() != PType::I64 {
            bail!("neighbors elements must be i64, got {}", elements.ptype());
        }
        let num_rows = list.len();
        if num_rows == 0 {
            bail!("neighbors parquet has zero rows");
        }
        let total = elements.len();
        if !total.is_multiple_of(num_rows) {
            bail!("neighbors rows are not all the same length");
        }
        let width = total / num_rows;
        Ok((elements.as_slice::<i64>().to_vec(), width, num_rows as u64))
    } else if let Some(fsl) = canon.as_opt::<vortex::array::arrays::FixedSizeList>() {
        let elements: PrimitiveArray = fsl.elements().clone().execute(ctx)?;
        if elements.ptype() != PType::I64 {
            bail!("neighbors elements must be i64, got {}", elements.ptype());
        }
        let width = match fsl.dtype() {
            DType::FixedSizeList(_, dim, _) => *dim as usize,
            other => bail!("expected FSL, got {other}"),
        };
        Ok((elements.as_slice::<i64>().to_vec(), width, fsl.len() as u64))
    } else {
        bail!("neighbors column has unsupported dtype {}", canon.dtype());
    }
}

#[derive(Clone, Copy)]
struct HeapItem {
    score: f32,
    id: i64,
}

impl PartialEq for HeapItem {
    fn eq(&self, other: &Self) -> bool {
        self.score == other.score
    }
}
impl Eq for HeapItem {}
impl PartialOrd for HeapItem {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}
impl Ord for HeapItem {
    fn cmp(&self, other: &Self) -> Ordering {
        // Reverse min-heap: smaller scores at the top so we can pop the lowest.
        other
            .score
            .partial_cmp(&self.score)
            .unwrap_or(Ordering::Equal)
    }
}

/// Brute-force top-K cosine similarity over every shard. Returns the K winning ids.
async fn top_k_for_query(
    vortex_files: &[std::path::PathBuf],
    query: &[f32],
    k: usize,
) -> Result<Vec<i64>> {
    let mut heap: BinaryHeap<HeapItem> = BinaryHeap::with_capacity(k + 1);
    let session = &*SESSION;
    let query_norm = query.iter().map(|&q| q * q).sum::<f32>().sqrt();
    let inv_query_norm = if query_norm == 0.0 {
        0.0
    } else {
        1.0 / query_norm
    };

    let mut next_ord_id: i64 = 0;
    for path in vortex_files {
        let file = session
            .open_options()
            .open_path(path)
            .await
            .with_context(|| format!("open {}", path.display()))?;
        let chunks: Vec<ArrayRef> = file
            .scan()?
            .with_projection(emb_projection())
            .into_array_stream()?
            .try_collect()
            .await?;

        let mut ctx = session.create_execution_ctx();
        for chunk in chunks {
            let emb_ext: ExtensionArray = chunk.execute(&mut ctx)?;
            let fsl: FixedSizeListArray = emb_ext.storage_array().clone().execute(&mut ctx)?;
            let dim = match fsl.dtype() {
                DType::FixedSizeList(_, dim, _) => *dim as usize,
                other => bail!("expected FSL emb, got {other}"),
            };
            if dim != query.len() {
                bail!("query dim {} != emb dim {}", query.len(), dim);
            }
            let elements: PrimitiveArray = fsl.elements().clone().execute(&mut ctx)?;
            let slice = elements.as_slice::<f32>();
            for row in slice.chunks_exact(dim) {
                let row_norm_sq: f32 = row.iter().map(|&v| v * v).sum();
                let row_norm = row_norm_sq.sqrt();
                let score = if row_norm == 0.0 {
                    0.0
                } else {
                    let dot = row.iter().zip(query).map(|(&a, &b)| a * b).sum::<f32>();
                    dot * inv_query_norm / row_norm
                };
                let id = next_ord_id;
                next_ord_id += 1;
                heap.push(HeapItem { score, id });
                if heap.len() > k {
                    heap.pop();
                }
            }
        }
    }

    let mut ids: Vec<i64> = heap.into_iter().map(|h| h.id).collect();
    ids.sort();
    Ok(ids)
}

fn recall_intersection(predicted: &[i64], truth: &[i64]) -> f64 {
    if truth.is_empty() {
        return 0.0;
    }
    let truth_set: vortex::utils::aliases::hash_set::HashSet<i64> = truth.iter().copied().collect();
    let hits = predicted
        .iter()
        .filter(|id| truth_set.contains(*id))
        .count();
    hits as f64 / truth.len() as f64
}

fn percentile(values: &[f64], p: f64) -> f64 {
    if values.is_empty() {
        return 0.0;
    }
    let mut sorted = values.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(Ordering::Equal));
    let n = sorted.len();
    // Index is bounded to [0, n-1] by the .min() below; the cast is safe by construction.
    #[expect(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
    let idx = ((p * (n - 1) as f64).round() as usize).min(n - 1);
    sorted[idx]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn recall_full_overlap_is_one() {
        let pred = [1, 2, 3];
        let truth = [3, 1, 2];
        assert_eq!(recall_intersection(&pred, &truth), 1.0);
    }

    #[test]
    fn recall_no_overlap_is_zero() {
        let pred = [1, 2, 3];
        let truth = [4, 5, 6];
        assert_eq!(recall_intersection(&pred, &truth), 0.0);
    }

    #[test]
    fn recall_partial_overlap_is_fraction() {
        let pred = [1, 2, 3];
        let truth = [1, 2, 4];
        assert!((recall_intersection(&pred, &truth) - 2.0 / 3.0).abs() < 1e-9);
    }

    #[test]
    fn percentile_picks_index() {
        let v = [0.0, 0.1, 0.5, 0.7, 0.9, 1.0];
        assert!((percentile(&v, 0.0) - 0.0).abs() < 1e-9);
        assert!((percentile(&v, 1.0) - 1.0).abs() < 1e-9);
    }
}
