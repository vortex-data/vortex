// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Writing IVF layouts.
//!
//! Provides [`IvfStrategy`], a [`LayoutStrategy`] that:
//!
//! 1. Collects all input chunks into memory (IVF requires a global view of the data to
//!    cluster it).
//! 2. Runs k-means to find cluster centroids.
//! 3. Reorders the data by cluster assignment and writes one chunk per cluster using the
//!    caller-provided child strategy (e.g. `FlatLayoutStrategy` or a TurboQuant-enabled
//!    compressed strategy).
//! 4. Writes the centroids as a second child layout.
//!
//! The resulting [`IvfLayout`](super::IvfLayout) can be pruned at read time by the
//! [`IvfReader`](super::reader) when a cosine-similarity query is applied.
//!
//! Because IVF fundamentally requires buffering the entire column to cluster it, the strategy
//! is best suited for moderately-sized columns that fit in memory during writes. Columns that
//! do not fit can be chunked and indexed independently (multiple IVF layouts combined via a
//! parent chunked layout).

use std::sync::Arc;

use async_trait::async_trait;
use futures::StreamExt;
use vortex_array::ArrayContext;
use vortex_array::ArrayRef;
use vortex_array::IntoArray;
use vortex_array::VortexSessionExecute;
use vortex_array::arrays::Extension;
use vortex_array::arrays::ExtensionArray;
use vortex_array::arrays::FixedSizeListArray;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::arrays::extension::ExtensionArrayExt;
use vortex_array::arrays::fixed_size_list::FixedSizeListArrayExt;
use vortex_array::dtype::DType;
use vortex_array::dtype::extension::ExtDType;
use vortex_array::extension::EmptyMetadata;
use vortex_array::validity::Validity;
use vortex_buffer::BufferMut;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_ensure;
use vortex_error::vortex_err;
use vortex_layout::IntoLayout;
use vortex_layout::LayoutRef;
use vortex_layout::LayoutStrategy;
use vortex_layout::OwnedLayoutChildren;
use vortex_layout::layouts::chunked::ChunkedLayout;
use vortex_layout::segments::SegmentSinkRef;
use vortex_layout::sequence::SendableSequentialStream;
use vortex_layout::sequence::SequenceId;
use vortex_layout::sequence::SequencePointer;
use vortex_layout::sequence::SequentialArrayStreamExt;
use vortex_session::VortexSession;
use vortex_tensor::utils::cast_to_f32;
use vortex_tensor::vector::AnyVector;
use vortex_tensor::vector::Vector;

use super::DEFAULT_NPROBES;
use super::IvfLayout;
use crate::IvfBuildConfig;
use crate::IvfIndex;

/// Options for [`IvfStrategy`].
#[derive(Clone, Debug)]
pub struct IvfLayoutOptions {
    /// Number of clusters (K).
    pub num_clusters: u32,
    /// Maximum k-means iterations.
    pub max_iterations: u32,
    /// Seed for k-means++ initialization.
    pub seed: u64,
    /// Default number of clusters to probe at read time. May be overridden via runtime config.
    pub nprobes: u32,
}

impl Default for IvfLayoutOptions {
    fn default() -> Self {
        Self {
            num_clusters: 64,
            max_iterations: 20,
            seed: 42,
            nprobes: DEFAULT_NPROBES,
        }
    }
}

impl From<IvfLayoutOptions> for IvfBuildConfig {
    fn from(value: IvfLayoutOptions) -> Self {
        Self {
            num_clusters: value.num_clusters,
            max_iterations: value.max_iterations,
            seed: value.seed,
        }
    }
}

/// A [`LayoutStrategy`] that IVF-encodes a `Vector<dim, f32>` column.
pub struct IvfStrategy {
    data: Arc<dyn LayoutStrategy>,
    centroids: Arc<dyn LayoutStrategy>,
    options: IvfLayoutOptions,
}

impl IvfStrategy {
    /// Construct a new [`IvfStrategy`].
    ///
    /// - `data` is the strategy used for each per-cluster chunk. Typically a
    ///   `CompressedLayoutStrategy` wrapping a flat writer so that TurboQuant is applied inside
    ///   each chunk, or a plain `FlatLayoutStrategy` for uncompressed data.
    /// - `centroids` is the strategy used for the centroid array. Typically a
    ///   `FlatLayoutStrategy`.
    pub fn new<D: LayoutStrategy, C: LayoutStrategy>(
        data: D,
        centroids: C,
        options: IvfLayoutOptions,
    ) -> Self {
        Self {
            data: Arc::new(data),
            centroids: Arc::new(centroids),
            options,
        }
    }
}

#[async_trait]
impl LayoutStrategy for IvfStrategy {
    async fn write_stream(
        &self,
        ctx: ArrayContext,
        segment_sink: SegmentSinkRef,
        mut stream: SendableSequentialStream,
        mut eof: SequencePointer,
        session: &VortexSession,
    ) -> VortexResult<LayoutRef> {
        let dtype = stream.dtype().clone();

        // Validate: we need a Vector extension type.
        let vector_dim = vector_dim_from_dtype(&dtype)?;
        vortex_ensure!(
            vector_dim == self.options.num_clusters || vector_dim > 0,
            "dim must be > 0"
        );

        // Collect all input chunks in order.
        let mut chunks: Vec<ArrayRef> = Vec::new();
        while let Some(next) = stream.next().await {
            let (_seq_id, chunk) = next?;
            chunks.push(chunk);
        }

        if chunks.is_empty() {
            vortex_bail!("IvfStrategy requires at least one input chunk");
        }

        // Materialize a flat f32 buffer for clustering.
        let (f32_data, total_rows, dim) = materialize_flat_f32(&chunks, session).await?;

        // Clamp clusters to number of rows.
        let effective_clusters = (self.options.num_clusters as usize).min(total_rows).max(1);
        let ivf_config = IvfBuildConfig {
            num_clusters: u32::try_from(effective_clusters)?,
            max_iterations: self.options.max_iterations,
            seed: self.options.seed,
        };
        let index = IvfIndex::build(&f32_data, dim, &ivf_config)?;

        // Partition rows by cluster and build per-cluster chunk arrays.
        let per_cluster = partition_rows_by_cluster(&chunks, &index, dim, &dtype, session).await?;

        // Write each cluster chunk via the data strategy.
        let mut data_child_layouts: Vec<LayoutRef> = Vec::with_capacity(per_cluster.len());
        for chunk in per_cluster {
            let data_eof = eof.split_off();
            let ctx2 = ctx.clone();
            let sink = Arc::clone(&segment_sink);

            let (ptr, sub_eof) = SequenceId::root().split();
            let stream = chunk.to_array_stream().sequenced(ptr);

            let layout = self
                .data
                .write_stream(ctx2, sink, stream, sub_eof, session)
                .await?;
            // Unused: data_eof is kept open for potential future use.
            drop(data_eof);
            data_child_layouts.push(layout);
        }

        // Combine the cluster chunks under a ChunkedLayout.
        let data_row_count: u64 = data_child_layouts.iter().map(|l| l.row_count()).sum();
        let data_layout = ChunkedLayout::new(
            data_row_count,
            dtype.clone(),
            OwnedLayoutChildren::layout_children(data_child_layouts),
        )
        .into_layout();

        // Build the centroids array and write it via the centroids strategy.
        let centroid_array = centroids_to_vector_array(index.centroids(), dim, effective_clusters)?;
        let centroid_eof = eof.split_off();
        let centroid_sink = Arc::clone(&segment_sink);
        let (ptr, sub_eof) = SequenceId::root().split();
        let centroid_stream = centroid_array.clone().to_array_stream().sequenced(ptr);
        let centroids_layout = self
            .centroids
            .write_stream(ctx, centroid_sink, centroid_stream, sub_eof, session)
            .await?;
        drop(centroid_eof);

        IvfLayout::try_new(
            data_layout,
            centroids_layout,
            u32::try_from(dim)?,
            u32::try_from(effective_clusters)?,
            self.options.nprobes,
        )
        .map(IntoLayout::into_layout)
    }

    fn buffered_bytes(&self) -> u64 {
        self.data.buffered_bytes() + self.centroids.buffered_bytes()
    }
}

/// Materialize all chunks as a flat `[N * dim]` f32 buffer for k-means input.
async fn materialize_flat_f32(
    chunks: &[ArrayRef],
    session: &VortexSession,
) -> VortexResult<(Vec<f32>, usize, usize)> {
    let mut ctx = session.create_execution_ctx();
    let mut flat: Vec<f32> = Vec::new();
    let mut total_rows = 0usize;
    let mut dim = 0usize;

    for chunk in chunks {
        let ext = chunk
            .as_opt::<Extension>()
            .ok_or_else(|| vortex_err!("IVF expects a Vector extension column"))?;
        let meta = ext
            .dtype()
            .as_extension()
            .metadata_opt::<AnyVector>()
            .ok_or_else(|| vortex_err!("IVF expects a Vector extension type"))?;
        let chunk_dim = meta.dimensions() as usize;
        if dim == 0 {
            dim = chunk_dim;
        } else if dim != chunk_dim {
            vortex_bail!("IVF chunks must share dimensionality; got {chunk_dim}, expected {dim}");
        }

        let storage = ext.storage_array();
        let fsl: FixedSizeListArray = storage.clone().execute(&mut ctx)?;
        let elements: PrimitiveArray = fsl.elements().clone().execute(&mut ctx)?;
        let f32_buf = cast_to_f32(elements)?;
        flat.extend_from_slice(f32_buf.as_ref());
        total_rows += chunk.len();
    }

    Ok((flat, total_rows, dim))
}

/// Partition rows into per-cluster `Vector<dim, f32>` arrays according to `index.assignments()`.
///
/// Returns one array per cluster, in cluster-id order. Empty clusters produce empty arrays.
async fn partition_rows_by_cluster(
    chunks: &[ArrayRef],
    index: &IvfIndex,
    dim: usize,
    dtype: &DType,
    session: &VortexSession,
) -> VortexResult<Vec<ArrayRef>> {
    let num_clusters = index.num_clusters();

    // Gather per-cluster row vectors.
    let mut ctx = session.create_execution_ctx();
    let mut global_row_idx = 0usize;
    let mut per_cluster: Vec<Vec<f32>> = vec![Vec::new(); num_clusters];

    for chunk in chunks {
        let ext = chunk
            .as_opt::<Extension>()
            .ok_or_else(|| vortex_err!("expected Vector extension"))?;
        let storage = ext.storage_array();
        let fsl: FixedSizeListArray = storage.clone().execute(&mut ctx)?;
        let elements: PrimitiveArray = fsl.elements().clone().execute(&mut ctx)?;
        let f32_buf = cast_to_f32(elements)?;
        let chunk_len = chunk.len();

        for row in 0..chunk_len {
            let assignment = index.assignments()[global_row_idx] as usize;
            let start = row * dim;
            let end = start + dim;
            per_cluster[assignment].extend_from_slice(&f32_buf.as_ref()[start..end]);
            global_row_idx += 1;
        }
    }

    // Convert each per-cluster f32 buffer into a Vector<dim, f32> array.
    let mut arrays = Vec::with_capacity(num_clusters);
    for cluster_rows in per_cluster {
        let row_count = cluster_rows.len() / dim;
        let mut buf = BufferMut::<f32>::with_capacity(cluster_rows.len());
        for v in cluster_rows {
            buf.push(v);
        }
        let elements = PrimitiveArray::new::<f32>(buf.freeze(), Validity::NonNullable);
        let fsl = FixedSizeListArray::try_new(
            elements.into_array(),
            u32::try_from(dim)?,
            Validity::NonNullable,
            row_count,
        )?;
        let ext_dtype = dtype.clone();
        let array =
            ExtensionArray::new(ext_dtype.as_extension().clone(), fsl.into_array()).into_array();
        arrays.push(array);
    }

    Ok(arrays)
}

/// Build a `Vector<dim, f32>` extension array from the flat centroid buffer.
fn centroids_to_vector_array(
    centroids: &[f32],
    dim: usize,
    num_clusters: usize,
) -> VortexResult<ArrayRef> {
    let mut buf = BufferMut::<f32>::with_capacity(centroids.len());
    for &v in centroids {
        buf.push(v);
    }
    let elements = PrimitiveArray::new::<f32>(buf.freeze(), Validity::NonNullable);
    let fsl = FixedSizeListArray::try_new(
        elements.into_array(),
        u32::try_from(dim)?,
        Validity::NonNullable,
        num_clusters,
    )?;
    let ext_dtype = ExtDType::<Vector>::try_new(EmptyMetadata, fsl.dtype().clone())?.erased();
    Ok(ExtensionArray::new(ext_dtype, fsl.into_array()).into_array())
}

/// Extract the vector dimension from a `Vector<dim, float>` extension dtype.
fn vector_dim_from_dtype(dtype: &DType) -> VortexResult<u32> {
    let ext = dtype
        .as_extension_opt()
        .ok_or_else(|| vortex_err!("IVF expects a Vector extension dtype"))?;
    let meta = ext
        .metadata_opt::<AnyVector>()
        .ok_or_else(|| vortex_err!("IVF expects a Vector extension type"))?;
    Ok(meta.dimensions())
}
