// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! IVF (Inverted File) vector index layout for Vortex.
//!
//! This crate provides an IVF index that clusters vectors into K groups using k-means,
//! then at query time only searches the `nprobes` most promising clusters. It trades a
//! small amount of recall for a large speedup by avoiding brute-force comparison against
//! every vector.
//!
//! # What's in this crate
//!
//! Three layers, each usable on its own:
//!
//! - **In-memory index** ([`IvfIndex`], [`IvfBuildConfig`]). k-means clustering and probe
//!   selection without any layout/file machinery. Good for experimentation.
//! - **Layout integration** ([`layout`]). `IvfLayout` is a first-class Vortex layout that
//!   stores data sorted by cluster plus an auxiliary centroid child. `IvfStrategy` writes
//!   that layout; `IvfReader` transparently prunes chunks at read time when the filter is
//!   a cosine-similarity expression.
//! - **TurboQuant integration** ([`tq`]). Builds the IVF index directly over TQ-compressed
//!   data: centroids live in the SORF-rotated quantized space, so the query is rotated
//!   once instead of decompressing every database vector.
//!
//! # Production write/read workflow
//!
//! ## Session setup (do this once)
//!
//! An IVF-indexed file needs three registrations:
//!
//! ```rust,no_run
//! use vortex_array::scalar_fn::session::ScalarFnSession;
//! use vortex_array::session::ArraySession;
//! use vortex_io::session::RuntimeSession;
//! use vortex_layout::session::LayoutSession;
//! use vortex_session::VortexSession;
//!
//! let session = VortexSession::empty()
//!     .with::<ArraySession>()
//!     .with::<ScalarFnSession>()
//!     .with::<LayoutSession>()
//!     .with::<RuntimeSession>();
//!
//! // 1. Default encodings (dict, alp, runend, pco, …) and file format plumbing.
//! vortex_file::register_default_encodings(&session);
//! // 2. Tensor scalar functions (CosineSimilarity, InnerProduct, L2Norm/Denorm, SorfTransform).
//! //    Required both for query expressions and for reading TurboQuant-encoded data.
//! vortex_tensor::initialize(&session);
//! // 3. This crate's layout encoding (`vortex.ivf`).
//! vortex_ivf::layout::register_ivf_layout(&session);
//! ```
//!
//! ## Ingest: turn your data into a `Vector<dim, f32>` column
//!
//! Vortex represents vectors as a [`Vector`][vortex_tensor::vector::Vector] extension type
//! wrapping a `FixedSizeList<f32>` storage. You build one per column:
//!
//! ```rust,no_run
//! use vortex_array::IntoArray;
//! use vortex_array::arrays::ExtensionArray;
//! use vortex_array::arrays::FixedSizeListArray;
//! use vortex_array::arrays::PrimitiveArray;
//! use vortex_array::arrays::fixed_size_list::FixedSizeListArrayExt;
//! use vortex_array::dtype::extension::ExtDType;
//! use vortex_array::extension::EmptyMetadata;
//! use vortex_array::validity::Validity;
//! use vortex_buffer::BufferMut;
//! use vortex_tensor::vector::Vector;
//!
//! # fn build(raw_vectors: Vec<f32>) -> vortex_error::VortexResult<vortex_array::ArrayRef> {
//! let dim: u32 = 768; // typical embedding size
//! let num_rows = raw_vectors.len() / dim as usize;
//!
//! // 1. Flat f32 elements buffer.
//! let mut buf = BufferMut::<f32>::with_capacity(raw_vectors.len());
//! for v in raw_vectors { buf.push(v); }
//! let elements = PrimitiveArray::new::<f32>(buf.freeze(), Validity::NonNullable);
//!
//! // 2. Wrap as FixedSizeList<f32> with per-row dimension `dim`.
//! let fsl = FixedSizeListArray::try_new(
//!     elements.into_array(), dim, Validity::NonNullable, num_rows,
//! )?;
//!
//! // 3. Wrap as a Vector<dim, f32> extension array.
//! let ext_dtype = ExtDType::<Vector>::try_new(EmptyMetadata, fsl.dtype().clone())?.erased();
//! let column = ExtensionArray::new(ext_dtype, fsl.into_array()).into_array();
//! # Ok(column)
//! # }
//! ```
//!
//! If your source data is f64 (e.g. OpenAI embeddings), cast to f32 during ingest. Vortex's
//! cosine-similarity fast paths and the SORF transform both operate in f32.
//!
//! ## Write: encode the column with `IvfStrategy`
//!
//! Pick a cluster count that's ~`sqrt(num_rows)` for general-purpose workloads, or tune via the
//! recall tests in [`recall_tests`](../src/recall_tests.rs) for your specific corpus. `nprobes`
//! is the *default* number of clusters a query reads; readers can override it per query in
//! future APIs, but today it's fixed at write time.
//!
//! ```rust,no_run
//! # async fn write_file(
//! #     session: &vortex_session::VortexSession,
//! #     column: vortex_array::ArrayRef,
//! #     output: &mut impl vortex_io::VortexWrite,
//! # ) -> vortex_error::VortexResult<()> {
//! use std::sync::Arc;
//!
//! use vortex_array::stream::ArrayStreamExt;
//! use vortex_file::WriteOptionsSessionExt;
//! use vortex_ivf::layout::writer::{IvfLayoutOptions, IvfStrategy};
//! use vortex_layout::layouts::flat::writer::FlatLayoutStrategy;
//!
//! let strategy = Arc::new(IvfStrategy::new(
//!     // Inside each cluster's chunk, use the default flat writer. To also TurboQuant-compress
//!     // each cluster, wrap FlatLayoutStrategy in a CompressedLayoutStrategy with TQ enabled.
//!     FlatLayoutStrategy::default(),
//!     // Centroids go through a plain flat writer — there's only one row per cluster so there's
//!     // nothing to compress.
//!     FlatLayoutStrategy::default(),
//!     IvfLayoutOptions {
//!         num_clusters: 64,
//!         max_iterations: 20,
//!         seed: 42,
//!         nprobes: 8,
//!     },
//! ));
//!
//! session
//!     .write_options()
//!     .with_strategy(strategy)
//!     .write(output, column.to_array_stream())
//!     .await?;
//! # Ok(())
//! # }
//! ```
//!
//! Under the hood the writer:
//!
//! 1. Buffers the entire column in memory (IVF needs a global view to cluster).
//! 2. Runs k-means++ to find K centroids.
//! 3. Reorders rows by cluster assignment.
//! 4. Writes one chunk per cluster via the `data` strategy, plus a `Vector<dim, f32>` centroid
//!    array via the `centroids` strategy, wrapped in an [`IvfLayout`](layout::IvfLayout).
//!
//! For columns that don't fit in memory, partition your input stream into multiple IVF layouts
//! under a parent chunked layout — each partition is clustered independently.
//!
//! ## Read: scan with a cosine-similarity filter
//!
//! Any filter of the form `CosineSimilarity(col, literal_query) > threshold` (with `literal_query`
//! carrying a `Vector<dim, f32>` extension scalar) automatically triggers IVF pruning.
//! `IvfReader::pruning_evaluation` walks the filter, extracts the query vector, probes the
//! centroids, and returns a mask that zeros out every row in non-probed clusters before any data
//! is read.
//!
//! ```rust,no_run
//! # async fn query(
//! #     session: &vortex_session::VortexSession,
//! #     file_bytes: vortex_buffer::ByteBuffer,
//! # ) -> vortex_error::VortexResult<()> {
//! use futures::pin_mut;
//! use futures::stream::StreamExt;
//! use vortex_array::dtype::{DType, Nullability, PType};
//! use vortex_array::dtype::extension::ExtDType;
//! use vortex_array::expr::{gt, lit, root};
//! use vortex_array::extension::EmptyMetadata;
//! use vortex_array::scalar::Scalar;
//! use vortex_array::scalar_fn::ScalarFnVTableExt;
//! use vortex_file::OpenOptionsSessionExt;
//! use vortex_tensor::scalar_fns::cosine_similarity::CosineSimilarity;
//! use vortex_tensor::vector::Vector;
//!
//! # let query_vec: Vec<f32> = vec![];
//! // 1. Wrap the query as a Vector<dim, f32> literal scalar.
//! let element_dtype = DType::Primitive(PType::F32, Nullability::NonNullable);
//! let children: Vec<Scalar> = query_vec
//!     .iter()
//!     .map(|&v| Scalar::primitive(v, Nullability::NonNullable))
//!     .collect();
//! let storage = Scalar::fixed_size_list(element_dtype, children, Nullability::NonNullable);
//! let ext_dtype = ExtDType::<Vector>::try_new(EmptyMetadata, storage.dtype().clone())?.erased();
//! let query_scalar = Scalar::extension_ref(ext_dtype, storage);
//!
//! // 2. Build the filter: CosineSimilarity(col, query) > threshold.
//! let cosine = CosineSimilarity
//!     .try_new_expr(
//!         vortex_array::scalar_fn::EmptyOptions,
//!         [root(), lit(query_scalar)],
//!     )?;
//! let filter = gt(cosine, lit(0.5f32));
//!
//! // 3. Scan with the filter — IVF pruning is automatic.
//! let file = session.open_options().open_buffer(file_bytes)?;
//! let stream = file.scan()?.with_filter(filter).into_array_stream()?;
//! pin_mut!(stream);
//! while let Some(chunk) = stream.next().await {
//!     let chunk = chunk?;
//!     // chunk contains only rows whose cosine similarity to the query exceeds 0.5,
//!     // restricted to the nprobes clusters closest to the query.
//! }
//! # Ok(())
//! # }
//! ```
//!
//! ## Recall tuning
//!
//! The built-in quality regression in [`recall_tests`](../src/recall_tests.rs) measures
//! recall@K across a sweep of `nprobes` on clustered synthetic data. Observed numbers with
//! `dim=128, 2000 rows, 16 clusters, K=10, 50 queries, seed=42`:
//!
//! | nprobes | clusters read | avg recall@10 | scan fraction |
//! |---------|---------------|---------------|---------------|
//! |     2   |     2/16      |      1.000    |     0.139     |
//! |     4   |     4/16      |      1.000    |     0.282     |
//! |     8   |     8/16      |      1.000    |     0.533     |
//! |    16   |    16/16      |      1.000    |     1.000     |
//!
//! On real embedding corpora with less natural cluster separation, expect to probe roughly
//! `sqrt(num_clusters)` to reach recall > 0.95. Start there and sweep down; the
//! `recall_at_k` helper in the tests is a good template for a production recall benchmark
//! against your ground-truth top-K.
//!
//! # Layering with TurboQuant
//!
//! When the column is already TurboQuant-compressed (see [`vortex_tensor::encodings::turboquant`])
//! the compressed representation is a dict-encoded fixed-size list in the SORF-rotated space.
//! Use [`tq::build_ivf_from_turboquant`] to skip SORF inversion: centroids are clustered
//! directly from the rotated coordinates, and the query is rotated once at read time via
//! [`tq::rotate_query`]. This composes with the [`vortex_tensor::scalar_fns::inner_product`]
//! dict+constant fast path inside each cluster chunk, so the total work per query is
//! `O(nprobes * cluster_size * padded_dim)` dict lookups — no float multiplies on the database
//! side.

mod kmeans;
pub mod layout;
pub mod partitioned;
pub mod search;
pub mod tq;

#[cfg(test)]
mod file_tests;
#[cfg(test)]
mod recall_tests;
#[cfg(test)]
mod tests;

use vortex_error::VortexResult;
use vortex_error::vortex_ensure;

/// Configuration for building an IVF index.
#[derive(Clone, Debug)]
pub struct IvfBuildConfig {
    /// Number of clusters (K). Must be >= 1.
    pub num_clusters: u32,
    /// Maximum number of k-means iterations.
    pub max_iterations: u32,
    /// Random seed for k-means initialization.
    pub seed: u64,
}

impl Default for IvfBuildConfig {
    fn default() -> Self {
        Self {
            num_clusters: 64,
            max_iterations: 20,
            seed: 42,
        }
    }
}

/// An IVF (Inverted File) index for approximate nearest neighbor search.
///
/// Contains K cluster centroids and the assignment of each database vector to a cluster.
/// At query time, only the `nprobes` clusters nearest to the query vector are searched,
/// providing a significant speedup over brute-force search at the cost of some recall.
#[derive(Clone, Debug)]
pub struct IvfIndex {
    /// Cluster centroids stored as a flat `[K * dim]` array in row-major order.
    centroids: Vec<f32>,
    /// The vector dimensionality.
    dim: usize,
    /// Number of clusters (K).
    num_clusters: usize,
    /// The cluster assignment for each row in the *original* (unsorted) order.
    assignments: Vec<u32>,
}

impl IvfIndex {
    /// Build an IVF index from a flat f32 vector array.
    ///
    /// `vectors` is a flat row-major `[num_vectors * dim]` buffer. Every contiguous
    /// group of `dim` values represents one vector.
    ///
    /// # Errors
    ///
    /// Returns an error if the input dimensions are inconsistent or if `num_clusters` is 0.
    pub fn build(vectors: &[f32], dim: usize, config: &IvfBuildConfig) -> VortexResult<Self> {
        vortex_ensure!(dim > 0, "IVF dimension must be > 0");
        vortex_ensure!(
            config.num_clusters >= 1,
            "IVF num_clusters must be >= 1, got {}",
            config.num_clusters
        );
        vortex_ensure!(
            vectors.len().is_multiple_of(dim),
            "vectors length {} is not a multiple of dim {}",
            vectors.len(),
            dim
        );

        let num_vectors = vectors.len() / dim;
        if num_vectors == 0 {
            return Ok(Self {
                centroids: vec![0.0; config.num_clusters as usize * dim],
                dim,
                num_clusters: config.num_clusters as usize,
                assignments: Vec::new(),
            });
        }

        // Clamp num_clusters to num_vectors (can't have more clusters than vectors).
        let k = (config.num_clusters as usize).min(num_vectors);

        let result = kmeans::kmeans(vectors, dim, k, config.max_iterations as usize, config.seed);

        Ok(Self {
            centroids: result.centroids,
            dim,
            num_clusters: k,
            assignments: result.assignments,
        })
    }

    /// Returns the cluster centroids as a flat `[K * dim]` slice in row-major order.
    pub fn centroids(&self) -> &[f32] {
        &self.centroids
    }

    /// Returns the dimensionality of the indexed vectors.
    pub fn dim(&self) -> usize {
        self.dim
    }

    /// Returns the number of clusters (K).
    pub fn num_clusters(&self) -> usize {
        self.num_clusters
    }

    /// Returns the cluster assignment for each vector in the original order.
    pub fn assignments(&self) -> &[u32] {
        &self.assignments
    }

    /// Returns the number of indexed vectors.
    pub fn num_vectors(&self) -> usize {
        self.assignments.len()
    }

    /// Find the `nprobes` clusters whose centroids are most similar to the query vector
    /// (by cosine similarity).
    ///
    /// Returns the cluster indices sorted by descending similarity. If `nprobes` exceeds
    /// the number of clusters, all clusters are returned.
    pub fn probe(&self, query: &[f32], nprobes: usize) -> VortexResult<Vec<usize>> {
        vortex_ensure!(
            query.len() == self.dim,
            "query dimension {} does not match index dimension {}",
            query.len(),
            self.dim
        );

        let nprobes = nprobes.min(self.num_clusters);
        let query_norm = l2_norm(query);

        // Compute cosine similarity of query to each centroid.
        let mut similarities: Vec<(usize, f32)> = (0..self.num_clusters)
            .map(|i| {
                let centroid = &self.centroids[i * self.dim..(i + 1) * self.dim];
                let centroid_norm = l2_norm(centroid);
                let dot = dot_product(query, centroid);
                let denom = query_norm * centroid_norm;
                let sim = if denom == 0.0 { 0.0 } else { dot / denom };
                (i, sim)
            })
            .collect();

        // Sort by descending similarity and take the top nprobes.
        similarities
            .sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        Ok(similarities.iter().take(nprobes).map(|(i, _)| *i).collect())
    }

    /// Build a boolean mask where `true` indicates the row belongs to one of the probed clusters.
    ///
    /// The mask has length equal to `self.num_vectors()`. Rows assigned to clusters in
    /// `probed_clusters` are marked `true`; all others are `false`.
    pub fn build_probe_mask(&self, probed_clusters: &[usize]) -> Vec<bool> {
        let mut cluster_set = vec![false; self.num_clusters];
        for &c in probed_clusters {
            if c < self.num_clusters {
                cluster_set[c] = true;
            }
        }

        self.assignments
            .iter()
            .map(|&a| cluster_set[a as usize])
            .collect()
    }

    /// Convenience: probe the index for the given query and return a boolean mask of rows to scan.
    ///
    /// Combines [`probe`](Self::probe) and [`build_probe_mask`](Self::build_probe_mask).
    pub fn query_mask(&self, query: &[f32], nprobes: usize) -> VortexResult<Vec<bool>> {
        let probed = self.probe(query, nprobes)?;
        Ok(self.build_probe_mask(&probed))
    }

    /// Returns the number of vectors in each cluster, indexed by cluster ID.
    pub fn cluster_sizes(&self) -> Vec<usize> {
        let mut sizes = vec![0usize; self.num_clusters];
        for &a in &self.assignments {
            sizes[a as usize] += 1;
        }
        sizes
    }
}

/// Compute the L2 norm of a vector.
fn l2_norm(v: &[f32]) -> f32 {
    v.iter().map(|&x| x * x).sum::<f32>().sqrt()
}

/// Compute the dot product of two vectors.
fn dot_product(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b.iter()).map(|(&x, &y)| x * y).sum()
}
