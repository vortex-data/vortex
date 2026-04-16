// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! URL builders and idempotent download driver for vector benchmark datasets.
//!
//! The upstream bucket is `https://assets.zilliz.com/benchmark/<prefix>/`. Within each
//! prefix the train split is named according to a four-way convention:
//!
//! - `Single`: `train.parquet`
//! - `SingleShuffled`: `shuffle_train.parquet`
//! - `Partitioned`: `train-NN-of-MM.parquet`
//! - `PartitionedShuffled`: `shuffle_train-NN-of-MM.parquet`
//!
//! `test.parquet` and (when present) `neighbors.parquet` live alongside the train files.

use std::path::PathBuf;

use anyhow::Context;
use anyhow::Result;

use crate::datasets::data_downloads::DEFAULT_DOWNLOAD_CONCURRENCY;
use crate::datasets::data_downloads::download_data;
use crate::datasets::data_downloads::download_many;
use crate::vector_dataset::catalog::VectorDataset;
use crate::vector_dataset::layout::LayoutSpec;
use crate::vector_dataset::layout::TrainLayout;
use crate::vector_dataset::paths;

/// Bucket root for all VectorDBBench datasets we mirror against.
const BENCHMARK_ROOT: &str = "https://assets.zilliz.com/benchmark";

/// All train-shard URLs for a `(dataset, layout)` pair. Length matches `layout.num_files()`.
pub fn train_urls(ds: VectorDataset, spec: LayoutSpec) -> Vec<String> {
    let prefix = format!("{BENCHMARK_ROOT}/{}", ds.s3_prefix());
    let layout = spec.layout();
    if layout.is_partitioned() {
        let n = spec.num_files();
        (0..n)
            .map(|i| format!("{prefix}/{}", partitioned_file_name(layout, i, n),))
            .collect()
    } else {
        let name = match layout {
            TrainLayout::Single => "train.parquet",
            TrainLayout::SingleShuffled => "shuffle_train.parquet",
            _ => unreachable!("non-partitioned guard above"),
        };
        vec![format!("{prefix}/{name}")]
    }
}

/// URL for `test.parquet`.
pub fn test_url(ds: VectorDataset) -> String {
    format!("{BENCHMARK_ROOT}/{}/test.parquet", ds.s3_prefix())
}

/// URL for `neighbors.parquet`, or `None` when the dataset doesn't host one.
pub fn neighbors_url(ds: VectorDataset) -> Option<String> {
    ds.has_neighbors()
        .then(|| format!("{BENCHMARK_ROOT}/{}/neighbors.parquet", ds.s3_prefix()))
}

fn partitioned_file_name(layout: TrainLayout, shard_idx: u32, num_files: u32) -> String {
    let prefix = match layout {
        TrainLayout::Partitioned => "train",
        TrainLayout::PartitionedShuffled => "shuffle_train",
        _ => unreachable!("partitioned guard"),
    };
    format!(
        "{prefix}-{shard_idx:0width$}-of-{num_files:0width$}.parquet",
        width = num_files_width(num_files),
    )
}

fn num_files_width(num_files: u32) -> usize {
    let digits = num_files.checked_ilog10().unwrap_or(0) as usize + 1;
    digits.max(2)
}

/// Local on-disk paths to the cached parquet files for a `(dataset, layout)` pair after
/// [`download`] returns successfully.
#[derive(Debug, Clone)]
pub struct DatasetPaths {
    /// Per-shard train parquet paths in shard order.
    pub train_files: Vec<PathBuf>,
    /// `test.parquet`.
    pub test: PathBuf,
    /// `neighbors.parquet` if the dataset hosts top-K ground truth.
    pub neighbors: Option<PathBuf>,
}

/// Download every parquet file required to run a `(dataset, layout)` benchmark, returning local
/// on-disk paths.
///
/// This has idempotent semantics, so files already present on disk are skipped, and re-runs only
/// pay for new files.
///
/// Train shards download via [`download_many`] with bounded parallelism (up to
/// [`DEFAULT_DOWNLOAD_CONCURRENCY`]); the small `test.parquet` and `neighbors.parquet` files use
/// the simple [`download_data`] helper. All HTTP requests share a single pooled client.
pub async fn download(ds: VectorDataset, layout: TrainLayout) -> Result<DatasetPaths> {
    let spec = ds.validate_layout(layout)?;
    let urls = train_urls(ds, spec);
    let train_targets = paths::train_files(ds, layout, spec.num_files());
    debug_assert_eq!(urls.len(), train_targets.len());

    let train_downloads: Vec<(PathBuf, String)> = train_targets
        .iter()
        .cloned()
        .zip(urls.into_iter())
        .collect();
    download_many(train_downloads, DEFAULT_DOWNLOAD_CONCURRENCY)
        .await
        .with_context(|| format!("download train shards for {}", ds.name()))?;

    let test = download_data(paths::test_path(ds, layout), &test_url(ds))
        .await
        .with_context(|| format!("download test.parquet for {}", ds.name()))?;

    let neighbors = if let Some(url) = neighbors_url(ds) {
        Some(
            download_data(paths::neighbors_path(ds, layout), &url)
                .await
                .with_context(|| format!("download neighbors.parquet for {}", ds.name()))?,
        )
    } else {
        None
    };

    Ok(DatasetPaths {
        train_files: train_targets,
        test,
        neighbors,
    })
}
