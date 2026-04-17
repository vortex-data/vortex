// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Per-flavor on-disk ingest.
//!
//! For each `(dataset, layout, flavor)` triple, [`prepare_flavor`] streams every parquet shard
//! into one per-shard output file:
//!
//! - Vortex flavors run the chunk transform + [`crate::compression::VectorFlavor`] write options
//!   and emit a `.vortex` file.
//! - The handrolled baseline flavor skips Vortex entirely and emits a flat `.f32` file via
//!   [`crate::baseline::write_shard_raw_f32`].
//!
//! The pipeline is idempotent (existing shard outputs are skipped) so repeated runs only
//! materialize new combinations.

use std::path::Path;
use std::path::PathBuf;

use anyhow::Context;
use anyhow::Result;
use futures::StreamExt;
use parquet::arrow::ParquetRecordBatchStreamBuilder;
use tokio::fs::File;
use tokio::io::AsyncWriteExt;
use tracing::info;
use tracing::warn;
use vortex::array::ArrayRef;
use vortex::array::ExecutionCtx;
use vortex::array::VortexSessionExecute;
use vortex::array::stream::ArrayStreamAdapter;
use vortex::array::stream::ArrayStreamExt;
use vortex::error::VortexResult;
use vortex::error::vortex_err;
use vortex_bench::conversions::parquet_to_vortex_stream;
use vortex_bench::data_dir;
use vortex_bench::utils::file::idempotent_async;
use vortex_bench::vector_dataset::DatasetPaths;
use vortex_bench::vector_dataset::TrainLayout;
use vortex_bench::vector_dataset::VectorDataset;

use crate::SESSION;
use crate::baseline::write_shard_raw_f32;
use crate::compression::VectorFlavor;
use crate::ingest::transform_chunk;

/// One prepared `(dataset, layout, flavor)` triple: the per-shard output files, ready to scan.
///
/// For Vortex flavors these are `.vortex` files; for the handrolled baseline they are flat `.f32`
/// files. The scan dispatcher branches on [`Self::flavor`] to pick the right reader.
#[derive(Debug, Clone)]
pub struct PreparedDataset {
    pub dataset: VectorDataset,
    pub layout: TrainLayout,
    pub flavor: VectorFlavor,
    pub shard_files: Vec<PathBuf>,
}

/// Drive [`prepare_flavor`] across a list of flavors, returning a [`PreparedDataset`] per flavor
/// in input order.
pub async fn prepare_all(
    dataset: VectorDataset,
    layout: TrainLayout,
    paths_for_dataset: &DatasetPaths,
    flavors: &[VectorFlavor],
) -> Result<Vec<PreparedDataset>> {
    let mut results = Vec::with_capacity(flavors.len());

    for &flavor in flavors {
        let r = prepare_flavor(dataset, layout, paths_for_dataset, flavor).await?;
        results.push(r);
    }

    Ok(results)
}

/// Prepare one flavor of one dataset by writing one shard output per train shard.
///
/// This function is sequential (for now).
pub async fn prepare_flavor(
    dataset: VectorDataset,
    layout: TrainLayout,
    paths_for_dataset: &DatasetPaths,
    flavor: VectorFlavor,
) -> Result<PreparedDataset> {
    let mut shard_files = Vec::with_capacity(paths_for_dataset.train_files.len());

    for parquet_path in &paths_for_dataset.train_files {
        let parquet_path = parquet_path.clone();
        let output_path = parquet_to_output_path(&parquet_path, dataset, layout, flavor)?;

        let already_cached = output_path.exists();
        if already_cached {
            warn!(
                "skipping cached shard {} ({} flavor)",
                output_path.display(),
                flavor.label()
            );
        } else {
            info!(
                "ingesting {} -> {} ({} flavor)",
                parquet_path.display(),
                output_path.display(),
                flavor.label(),
            );
        }

        let dim = dataset.dim() as usize;
        let src_ptype = dataset.element_ptype();
        let written_path = idempotent_async(output_path.as_path(), |tmp| async move {
            if flavor.is_vortex() {
                write_shard_streaming(&parquet_path, &tmp, flavor).await
            } else {
                write_shard_raw_f32(&parquet_path, &tmp, dim, src_ptype).await
            }
        })
        .await?;

        shard_files.push(written_path);
    }

    Ok(PreparedDataset {
        dataset,
        layout,
        flavor,
        shard_files,
    })
}

/// Stream one parquet shard through the chunk transform into a Vortex file.
///
/// The output dtype is derived once from the first transformed chunk so the [`ArrayStreamAdapter`]
/// can declare it ahead of time.
async fn write_shard_streaming(
    parquet_path: &Path,
    vortex_path: &Path,
    flavor: VectorFlavor,
) -> Result<()> {
    let file = File::open(parquet_path).await?;
    let builder = ParquetRecordBatchStreamBuilder::new(file).await?;
    let mut array_stream = parquet_to_vortex_stream(builder.build()?);

    let mut ctx = SESSION.create_execution_ctx();

    // We need to get the first chunk so that we know what the dtype of the file is.
    let first = match array_stream.next().await {
        Some(chunk) => transform_chunk_with_error(chunk, &mut ctx, parquet_path, 1)?,
        None => {
            return Err(vortex_err!(
                "ingest: parquet shard {} produced no chunks",
                parquet_path.display(),
            )
            .into());
        }
    };
    let dtype = first.dtype().clone();
    let shard_path = parquet_path.to_path_buf();

    let transformed =
        futures::stream::iter(std::iter::once(Ok(first))).chain(array_stream.enumerate().map(
            move |(chunk_offset, chunk_or_err)| {
                let mut local_ctx = SESSION.create_execution_ctx();
                transform_chunk_with_error(
                    chunk_or_err,
                    &mut local_ctx,
                    &shard_path,
                    chunk_offset + 2,
                )
            },
        ));

    let stream = ArrayStreamExt::boxed(ArrayStreamAdapter::new(dtype, transformed));

    let mut output = tokio::fs::OpenOptions::new()
        .write(true)
        .truncate(true)
        .create(true)
        .open(vortex_path)
        .await?;

    // This will write in parallel, using `std::thread::available_parallelism()`.
    // See `CompressingStrategy` for more details.
    flavor
        .create_write_options(&SESSION)
        .write(&mut output, stream)
        .await?;
    output.flush().await?;

    Ok(())
}

fn transform_chunk_with_error(
    chunk_or_err: VortexResult<ArrayRef>,
    ctx: &mut ExecutionCtx,
    parquet_path: &Path,
    chunk_idx: usize,
) -> VortexResult<ArrayRef> {
    let chunk = chunk_or_err.map_err(|err| {
        vortex_err!(
            "ingest: failed to read chunk {} from {}: {err:#}",
            chunk_idx,
            parquet_path.display(),
        )
    })?;

    transform_chunk(chunk, ctx).map_err(|err| {
        vortex_err!(
            "ingest: failed to transform chunk {} from {}: {err:#}",
            chunk_idx,
            parquet_path.display(),
        )
    })
}

/// Translate a parquet shard path to its output companion under the flavor directory.
///
/// Just swaps the file extension (to `flavor.output_extension()`) and rebases the file name into
/// the per-[`VectorFlavor`] flavor directory. The shard stem is preserved so a directory listing
/// pairs `00-of-10.parquet` with e.g. `00-of-10.vortex` or `00-of-10.f32`.
pub fn parquet_to_output_path(
    parquet: &Path,
    dataset: VectorDataset,
    layout: TrainLayout,
    flavor: VectorFlavor,
) -> Result<PathBuf> {
    let stem = parquet
        .file_stem()
        .with_context(|| format!("parquet path {} has no file stem", parquet.display()))?
        .to_owned();

    let mut name = stem;
    name.push(".");
    name.push(flavor.output_extension());

    Ok(flavor_dir(dataset, layout, flavor).join(name))
}

/// `vortex-bench/data/vector-search/<dataset>/<layout>/<flavor>/`.
fn flavor_dir(ds: VectorDataset, layout: TrainLayout, flavor: VectorFlavor) -> PathBuf {
    data_dir()
        .join("vector-search")
        .join(ds.name())
        .join(layout.label())
        .join(flavor.dir_name())
}
