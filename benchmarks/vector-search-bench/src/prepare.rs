// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Per-flavor on-disk ingest.
//!
//! For each `(dataset, layout, flavor)` triple, [`prepare_flavor`] streams every parquet
//! shard through the [`crate::ingest::ChunkTransform`] and writes one `.vortex` file per
//! shard. The pipeline is idempotent (existing `.vortex` files are skipped) and reports
//! end-to-end wall-clock time, summed input parquet bytes, and total output bytes.

use std::path::Path;
use std::path::PathBuf;
use std::time::Duration;
use std::time::Instant;

use anyhow::Context;
use anyhow::Result;
use futures::StreamExt;
use parquet::arrow::ParquetRecordBatchStreamBuilder;
use tokio::fs::File;
use tokio::io::AsyncWriteExt;
use tracing::info;
use tracing::warn;
use vortex::array::VortexSessionExecute;
use vortex::array::stream::ArrayStreamAdapter;
use vortex::array::stream::ArrayStreamExt;
use vortex::error::vortex_err;
use vortex_bench::conversions::parquet_to_vortex_stream;
use vortex_bench::utils::file::idempotent_async;
use vortex_bench::vector_dataset::DatasetPaths;
use vortex_bench::vector_dataset::VectorDataset;
use vortex_bench::vector_dataset::layout::TrainLayout;

use crate::compression::VortexCompression;
use crate::ingest::ChunkTransform;
use crate::paths;
use crate::session::SESSION;

/// Result of preparing one `(dataset, layout, flavor)` triple — the per-flavor `.vortex`
/// file paths, plus aggregate timing and size counters used to emit measurements.
#[derive(Debug, Clone)]
pub struct CompressionResult {
    /// Which compression flavor produced these files.
    pub flavor: VortexCompression,
    /// One `.vortex` path per train shard, in shard order.
    pub vortex_files: Vec<PathBuf>,
    /// Sum of per-shard wall-clock write time. `Duration::ZERO` when every shard was
    /// already cached.
    pub total_wall_time: Duration,
    /// Sum of input parquet shard sizes.
    pub total_input_bytes: u64,
    /// Sum of output `.vortex` shard sizes.
    pub total_output_bytes: u64,
}

/// Prepare one flavor of one dataset by writing one `.vortex` file per train shard.
///
/// Sequential by default — files are processed one at a time so the timing numbers
/// reflect a deterministic, bandwidth-friendly write pass. Concurrency is not exposed
/// here; future work can add a `--ingest-concurrency` knob if needed.
pub async fn prepare_flavor(
    ds: VectorDataset,
    layout: TrainLayout,
    flavor: VortexCompression,
    paths_for_dataset: &DatasetPaths,
) -> Result<CompressionResult> {
    let transform = ChunkTransform {
        src_ptype: ds.element_ptype(),
        include_scalar_labels: ds.has_scalar_labels(),
    };

    let mut vortex_files = Vec::with_capacity(paths_for_dataset.train_files.len());
    let mut total_wall_time = Duration::ZERO;
    let mut total_input_bytes: u64 = 0;
    let mut total_output_bytes: u64 = 0;

    for parquet_path in &paths_for_dataset.train_files {
        let vortex_path = paths::vortex_path_for_parquet(parquet_path, ds, layout, flavor)?;

        let input_bytes = tokio::fs::metadata(parquet_path)
            .await
            .with_context(|| format!("stat parquet shard {}", parquet_path.display()))?
            .len();
        total_input_bytes = total_input_bytes.saturating_add(input_bytes);

        let already_cached = vortex_path.exists();
        if already_cached {
            warn!(
                "skipping cached vortex shard {} ({} flavor)",
                vortex_path.display(),
                flavor.label()
            );
        } else {
            info!(
                "ingesting {} -> {} ({} flavor)",
                parquet_path.display(),
                vortex_path.display(),
                flavor.label(),
            );
        }

        let pp = parquet_path.clone();
        let started = Instant::now();
        let written_path = idempotent_async(vortex_path.as_path(), |tmp| async move {
            let bytes = write_shard_streaming(&pp, &tmp, flavor, transform).await?;
            tracing::debug!("wrote shard {} -> {} bytes", pp.display(), bytes,);
            Ok(())
        })
        .await?;
        if !already_cached {
            total_wall_time = total_wall_time.saturating_add(started.elapsed());
        }

        let output_bytes = tokio::fs::metadata(&written_path)
            .await
            .with_context(|| format!("stat vortex shard {}", written_path.display()))?
            .len();
        total_output_bytes = total_output_bytes.saturating_add(output_bytes);
        vortex_files.push(written_path);
    }

    Ok(CompressionResult {
        flavor,
        vortex_files,
        total_wall_time,
        total_input_bytes,
        total_output_bytes,
    })
}

/// Stream one parquet shard through the chunk transform into a Vortex file.
///
/// The output dtype is derived once from the first transformed chunk so the
/// [`ArrayStreamAdapter`] can declare it ahead of time.
async fn write_shard_streaming(
    parquet_path: &Path,
    vortex_path: &Path,
    flavor: VortexCompression,
    transform: ChunkTransform,
) -> Result<u64> {
    let session = &*SESSION;
    let file = File::open(parquet_path).await?;
    let builder = ParquetRecordBatchStreamBuilder::new(file).await?;
    let mut chunks = parquet_to_vortex_stream(builder.build()?);

    let mut ctx = session.create_execution_ctx();
    let first = match chunks.next().await {
        Some(chunk) => transform.apply(chunk?, &mut ctx)?,
        None => {
            return Err(vortex_err!(
                "ingest: parquet shard {} produced no chunks",
                parquet_path.display(),
            )
            .into());
        }
    };
    let dtype = first.dtype().clone();

    let transformed =
        futures::stream::iter(std::iter::once(Ok(first))).chain(chunks.map(move |chunk_or_err| {
            let mut local_ctx = session.create_execution_ctx();
            chunk_or_err.and_then(|chunk| {
                transform
                    .apply(chunk, &mut local_ctx)
                    .map_err(|e| vortex_err!(External: e))
            })
        }));

    let stream = ArrayStreamExt::boxed(ArrayStreamAdapter::new(dtype, transformed));

    let mut output = tokio::fs::OpenOptions::new()
        .write(true)
        .truncate(true)
        .create(true)
        .open(vortex_path)
        .await?;

    flavor
        .write_options(session)
        .write(&mut output, stream)
        .await?;
    output.flush().await?;

    Ok(tokio::fs::metadata(vortex_path).await?.len())
}

/// Drive [`prepare_flavor`] across a list of flavors, returning a [`CompressionResult`] per
/// flavor in input order.
pub async fn prepare_all(
    ds: VectorDataset,
    layout: TrainLayout,
    flavors: &[VortexCompression],
    paths_for_dataset: &DatasetPaths,
) -> Result<Vec<CompressionResult>> {
    let mut results = Vec::with_capacity(flavors.len());
    for &flavor in flavors {
        let r = prepare_flavor(ds, layout, flavor, paths_for_dataset).await?;
        results.push(r);
    }
    Ok(results)
}
