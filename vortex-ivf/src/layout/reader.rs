// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! [`IvfReader`]: the read-time logic for [`IvfLayout`](super::IvfLayout).
//!
//! The key novel behavior is in [`IvfReader::pruning_evaluation`]: when the incoming
//! expression is a cosine similarity comparison against a constant query vector, the reader
//! fetches the centroids once, computes which clusters are closest to the query, and returns
//! a mask that eliminates rows belonging to non-probed clusters.

use std::collections::BTreeSet;
use std::ops::BitAnd;
use std::ops::Range;
use std::sync::Arc;
use std::sync::OnceLock;

use futures::FutureExt;
use futures::TryFutureExt;
use futures::future::BoxFuture;
use futures::future::Shared;
use vortex_array::ArrayRef;
use vortex_array::MaskFuture;
use vortex_array::VortexSessionExecute;
use vortex_array::arrays::Extension;
use vortex_array::arrays::FixedSizeListArray;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::arrays::extension::ExtensionArrayExt;
use vortex_array::arrays::fixed_size_list::FixedSizeListArrayExt;
use vortex_array::dtype::DType;
use vortex_array::dtype::FieldMask;
use vortex_array::expr::Expression;
use vortex_array::expr::root;
use vortex_buffer::BitBufferMut;
use vortex_error::SharedVortexResult;
use vortex_error::VortexError;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_layout::LayoutReader;
use vortex_layout::LayoutReaderRef;
use vortex_layout::LayoutRef;
use vortex_layout::segments::SegmentSource;
use vortex_mask::Mask;
use vortex_session::VortexSession;
use vortex_tensor::utils::cast_to_f32;

use super::IvfLayout;
use crate::layout::query::extract_cosine_query;

type SharedCentroids = Shared<BoxFuture<'static, SharedVortexResult<Arc<Vec<f32>>>>>;

/// The read-side implementation of [`IvfLayout`].
pub struct IvfReader {
    layout: IvfLayout,
    name: Arc<str>,
    #[allow(dead_code)]
    session: VortexSession,

    data_reader: LayoutReaderRef,
    centroids_reader: LayoutReaderRef,

    /// Lazily-materialized centroids as a flat `[K * dim]` f32 array.
    /// Loaded on first pruning-evaluation call.
    centroids_cache: OnceLock<SharedCentroids>,
}

impl IvfReader {
    pub(super) fn try_new(
        layout: IvfLayout,
        name: Arc<str>,
        segment_source: Arc<dyn SegmentSource>,
        session: VortexSession,
    ) -> VortexResult<Self> {
        let data_reader = layout.data().new_reader(
            format!("{name}.data").into(),
            Arc::clone(&segment_source),
            &session,
        )?;
        let centroids_reader = layout.centroids().new_reader(
            format!("{name}.centroids").into(),
            Arc::clone(&segment_source),
            &session,
        )?;
        Ok(Self {
            layout,
            name,
            session,
            data_reader,
            centroids_reader,
            centroids_cache: OnceLock::new(),
        })
    }

    /// Load the centroids child into a flat `[K * dim]` f32 buffer.
    fn centroids_future(&self) -> SharedCentroids {
        let centroids_reader: LayoutReaderRef = Arc::clone(&self.centroids_reader);
        let nrows_u64 = u64::from(self.layout.num_clusters());
        let nrows_usize = self.layout.num_clusters() as usize;

        self.centroids_cache
            .get_or_init(move || {
                async move {
                    let array = centroids_reader
                        .projection_evaluation(
                            &(0..nrows_u64),
                            &root(),
                            MaskFuture::new_true(nrows_usize),
                        )?
                        .await?;
                    Ok(Arc::new(extract_centroid_buffer(array)?))
                }
                .map_err(Arc::new)
                .boxed()
                .shared()
            })
            .clone()
    }

    /// Given a query vector and the centroids, compute which clusters to probe.
    /// Returns a boolean mask (true = probed) of length `num_clusters`.
    fn probe_clusters(
        centroids: &[f32],
        dim: usize,
        num_clusters: usize,
        query: &[f32],
        nprobes: usize,
    ) -> Vec<bool> {
        let nprobes = nprobes.min(num_clusters);
        let query_norm = l2_norm(query);
        let mut similarities: Vec<(usize, f32)> = (0..num_clusters)
            .map(|i| {
                let centroid = &centroids[i * dim..(i + 1) * dim];
                let centroid_norm = l2_norm(centroid);
                let denom = query_norm * centroid_norm;
                let dot: f32 = query
                    .iter()
                    .zip(centroid.iter())
                    .map(|(&q, &c)| q * c)
                    .sum();
                let sim = if denom == 0.0 { 0.0 } else { dot / denom };
                (i, sim)
            })
            .collect();
        similarities
            .sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        let mut probed = vec![false; num_clusters];
        for (cluster_idx, _) in similarities.iter().take(nprobes) {
            probed[*cluster_idx] = true;
        }
        probed
    }

    /// Get the chunk row ranges for the data child. These define the cluster boundaries.
    fn chunk_ranges(&self) -> Vec<Range<u64>> {
        chunk_ranges_from_layout(self.layout.data())
    }
}

impl LayoutReader for IvfReader {
    fn name(&self) -> &Arc<str> {
        &self.name
    }

    fn dtype(&self) -> &DType {
        self.layout.dtype()
    }

    fn row_count(&self) -> u64 {
        self.layout.row_count()
    }

    fn register_splits(
        &self,
        field_mask: &[FieldMask],
        row_range: &Range<u64>,
        splits: &mut BTreeSet<u64>,
    ) -> VortexResult<()> {
        self.data_reader
            .register_splits(field_mask, row_range, splits)
    }

    fn pruning_evaluation(
        &self,
        row_range: &Range<u64>,
        expr: &Expression,
        mask: Mask,
    ) -> VortexResult<MaskFuture> {
        tracing::debug!("IVF pruning evaluation: {} - {}", &self.name, expr);

        // Forward the existing pruning eval to the data child — we might still benefit from zone-
        // map or other pruning inside the data chunks.
        let inner_pruning = self
            .data_reader
            .pruning_evaluation(row_range, expr, mask.clone())?;

        // Check if this expression is a cosine-similarity > threshold we can exploit.
        let Some(query) = extract_cosine_query(expr) else {
            tracing::debug!("IVF pruning: expression is not a cosine-similarity query");
            return Ok(inner_pruning);
        };

        let dim = self.layout.dim() as usize;
        if query.len() != dim {
            tracing::debug!(
                "IVF pruning: query dim {} != layout dim {dim}; skipping IVF prune",
                query.len()
            );
            return Ok(inner_pruning);
        }

        let nprobes = self.layout.nprobes() as usize;
        let num_clusters = self.layout.num_clusters() as usize;
        let chunk_ranges = self.chunk_ranges();
        let input_row_range = row_range.clone();
        let centroids_future = self.centroids_future();
        let name = Arc::clone(&self.name);

        Ok(MaskFuture::new(mask.len(), async move {
            // Fetch centroids (once per file).
            let centroids = centroids_future.await.map_err(VortexError::from)?;

            // Compute probed clusters.
            let probed =
                IvfReader::probe_clusters(centroids.as_ref(), dim, num_clusters, &query, nprobes);

            // Build a per-row mask over the input row_range.
            // Row i is kept if its cluster is probed; otherwise it is pruned.
            let mut keep = BitBufferMut::new_unset(mask.len());
            for (chunk_idx, chunk_range) in chunk_ranges.iter().enumerate() {
                if chunk_idx >= probed.len() || !probed[chunk_idx] {
                    continue;
                }
                // Intersect the chunk range with the input row range.
                let intersect_start = chunk_range.start.max(input_row_range.start);
                let intersect_end = chunk_range.end.min(input_row_range.end);
                if intersect_start >= intersect_end {
                    continue;
                }
                let local_start = usize::try_from(intersect_start - input_row_range.start)?;
                let local_end = usize::try_from(intersect_end - input_row_range.start)?;
                keep.fill_range(local_start, local_end, true);
            }
            let ivf_mask = Mask::from(keep.freeze());

            // Intersect with inner pruning result.
            let combined = mask.bitand(&ivf_mask);
            let combined = if combined.all_false() {
                combined
            } else {
                let inner = inner_pruning.await?;
                combined.bitand(&inner)
            };

            tracing::debug!(
                "IVF pruning {name}: kept {} of {} rows (density={:.3})",
                combined.true_count(),
                combined.len(),
                combined.density()
            );
            Ok(combined)
        }))
    }

    fn filter_evaluation(
        &self,
        row_range: &Range<u64>,
        expr: &Expression,
        mask: MaskFuture,
    ) -> VortexResult<MaskFuture> {
        self.data_reader.filter_evaluation(row_range, expr, mask)
    }

    fn projection_evaluation(
        &self,
        row_range: &Range<u64>,
        expr: &Expression,
        mask: MaskFuture,
    ) -> VortexResult<BoxFuture<'static, VortexResult<ArrayRef>>> {
        self.data_reader
            .projection_evaluation(row_range, expr, mask)
    }
}

/// Compute row ranges for each chunk of a layout by walking `child_type()`.
fn chunk_ranges_from_layout(layout: &LayoutRef) -> Vec<Range<u64>> {
    let nchildren = layout.nchildren();
    let mut offsets = Vec::with_capacity(nchildren + 1);
    offsets.push(0u64);
    for idx in 0..nchildren {
        let child = layout.child(idx).vortex_expect("child");
        offsets.push(offsets[idx] + child.row_count());
    }
    (0..nchildren).map(|i| offsets[i]..offsets[i + 1]).collect()
}

/// Extract a flat `[K * dim]` f32 buffer from the centroids child array.
///
/// The input is a `Vector<dim, f32>` extension array; we extract the FSL elements and cast to
/// f32.
fn extract_centroid_buffer(array: ArrayRef) -> VortexResult<Vec<f32>> {
    let ext = array
        .as_opt::<Extension>()
        .ok_or_else(|| vortex_error::vortex_err!("centroids must be a Vector extension array"))?;
    let storage = ext.storage_array();
    // Use LEGACY_SESSION-equivalent path: materialize directly.
    let fsl: FixedSizeListArray = storage.clone().execute(&mut legacy_ctx())?;
    let elements: PrimitiveArray = fsl.elements().clone().execute(&mut legacy_ctx())?;
    let buf = cast_to_f32(elements)?;
    Ok(buf.as_ref().to_vec())
}

fn legacy_ctx() -> vortex_array::ExecutionCtx {
    vortex_array::LEGACY_SESSION.create_execution_ctx()
}

fn l2_norm(v: &[f32]) -> f32 {
    v.iter().map(|&x| x * x).sum::<f32>().sqrt()
}
