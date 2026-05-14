// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Per-flavor on-disk ingest.
//!
//! For each `(dataset, layout, flavor)` triple, [`prepare_flavor`] streams every parquet shard
//! and writes one `.vortex` file per shard. The pipeline is idempotent (existing `.vortex` files
//! are skipped) and reports end-to-end wall-clock time, summed input parquet bytes, and total
//! output bytes.

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
use crate::compression::VectorFlavor;
use crate::ingest::transform_chunk;

/// The paths of the vortex files that result from preparing one `(dataset, layout, flavor)` triple.
#[derive(Debug, Clone)]
pub struct CompressedVortexDataset {
    pub dataset: VectorDataset,
    pub layout: TrainLayout,
    pub flavor: VectorFlavor,
    pub vortex_files: Vec<PathBuf>,
}

/// Drive [`prepare_flavor`] across a list of flavors, returning a [`CompressedVortexDataset`] per
/// flavor in input order.
pub async fn prepare_all(
    dataset: VectorDataset,
    layout: TrainLayout,
    paths_for_dataset: &DatasetPaths,
    flavors: &[VectorFlavor],
) -> Result<Vec<CompressedVortexDataset>> {
    let mut results = Vec::with_capacity(flavors.len());

    for &flavor in flavors {
        let r = prepare_flavor(dataset, layout, paths_for_dataset, flavor).await?;
        results.push(r);
    }

    Ok(results)
}

/// Prepare one flavor of one dataset by writing one `.vortex` file per train shard.
///
/// This function is sequential (for now).
pub async fn prepare_flavor(
    dataset: VectorDataset,
    layout: TrainLayout,
    paths_for_dataset: &DatasetPaths,
    flavor: VectorFlavor,
) -> Result<CompressedVortexDataset> {
    let mut vortex_files = Vec::with_capacity(paths_for_dataset.train_files.len());

    for parquet_path in &paths_for_dataset.train_files {
        let parquet_path = parquet_path.clone();
        let vortex_path = parquet_to_vortex_path(&parquet_path, dataset, layout, flavor)?;

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

        let written_path = idempotent_async(vortex_path.as_path(), |tmp| async move {
            write_shard_streaming(&parquet_path, &tmp, flavor).await
        })
        .await?;

        vortex_files.push(written_path);
    }

    Ok(CompressedVortexDataset {
        dataset,
        layout,
        flavor,
        vortex_files,
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

/// Translate a parquet shard path to its `.vortex` companion under the flavor directory.
///
/// Just swaps the file extension and rebases the file name into the per-[`VectorFlavor`] flavor
/// directory. The shard stem is preserved so a directory listing pairs `00-of-10.parquet` with
/// `00-of-10.vortex`.
pub fn parquet_to_vortex_path(
    parquet: &Path,
    dataset: VectorDataset,
    layout: TrainLayout,
    flavor: VectorFlavor,
) -> Result<PathBuf> {
    let stem = parquet
        .file_stem()
        .with_context(|| format!("parquet path {} has no file stem", parquet.display()))?
        .to_owned();

    // TODO(connor): Is there a better way to do this?
    let mut name = stem;
    name.push(".vortex");

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
