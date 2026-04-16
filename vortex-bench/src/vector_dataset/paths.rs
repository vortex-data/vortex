// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Local-filesystem layout for cached vector benchmark datasets.
//!
//! ```text
//! <vortex-bench/data>/vector-search/<dataset>/<layout>/
//!     train<>                  single-file: train.parquet
//!                              partitioned: 00-of-N.parquet, 01-of-N.parquet, ...
//!     test.parquet
//!     neighbors.parquet        only when ds.has_neighbors()
//!
//! + some more things
//! ```
//!
//! This module exists purely to centralize the path-construction logic used by both the downloader
//! and the ingest pipeline.

use std::path::PathBuf;

use crate::utils::file::data_dir;
use crate::vector_dataset::VectorDataset;
use crate::vector_dataset::layout::TrainLayout;

/// Top-level cache root: `<vortex-bench/data>/vector-search/`.
pub fn root() -> PathBuf {
    data_dir().join("vector-search")
}

/// Per-dataset directory: `<root>/<dataset>/<layout>/`.
pub fn dataset_dir(ds: VectorDataset, layout: TrainLayout) -> PathBuf {
    root().join(ds.name()).join(layout.label())
}

/// Train-shard directory: `<dataset_dir>/train/`.
pub fn train_dir(ds: VectorDataset, layout: TrainLayout) -> PathBuf {
    dataset_dir(ds, layout).join("train")
}

/// File name for one train shard within [`train_dir`].
///
/// Single-file layouts produce `train.parquet`; partitioned layouts produce `NN-of-MM.parquet` so a
/// directory listing sorts shards in sequence order.
pub fn train_file_name(layout: TrainLayout, shard_idx: u32, num_files: u32) -> String {
    if layout.is_partitioned() {
        format!(
            "{shard_idx:0width$}-of-{num_files:0width$}.parquet",
            width = num_files_width(num_files),
        )
    } else {
        "train.parquet".to_owned()
    }
}

/// All train-shard paths for a dataset/layout pair, in shard order.
pub fn train_files(ds: VectorDataset, layout: TrainLayout, num_files: u32) -> Vec<PathBuf> {
    let dir = train_dir(ds, layout);
    (0..num_files)
        .map(|i| dir.join(train_file_name(layout, i, num_files)))
        .collect()
}

/// Path to the cached `test.parquet` for a dataset/layout pair.
pub fn test_path(ds: VectorDataset, layout: TrainLayout) -> PathBuf {
    dataset_dir(ds, layout).join("test.parquet")
}

/// Path to the cached `neighbors.parquet` for a dataset/layout pair.
pub fn neighbors_path(ds: VectorDataset, layout: TrainLayout) -> PathBuf {
    dataset_dir(ds, layout).join("neighbors.parquet")
}

/// Width used to zero-pad shard indices in partitioned filenames. `10` shards is 2 digits, `100`
/// shards is 3 digits.
fn num_files_width(num_files: u32) -> usize {
    let digits = num_files.checked_ilog10().unwrap_or(0) as usize + 1;
    digits.max(2)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn single_layout_uses_train_parquet() {
        assert_eq!(train_file_name(TrainLayout::Single, 0, 1), "train.parquet");
    }

    #[test]
    fn partitioned_filename_zero_pads_to_match_total() {
        assert_eq!(
            train_file_name(TrainLayout::Partitioned, 0, 10),
            "00-of-10.parquet"
        );
        assert_eq!(
            train_file_name(TrainLayout::Partitioned, 9, 10),
            "09-of-10.parquet"
        );
        assert_eq!(
            train_file_name(TrainLayout::Partitioned, 99, 100),
            "099-of-100.parquet"
        );
    }

    #[test]
    fn train_files_lists_all_shards_in_order() {
        let files = train_files(VectorDataset::CohereLarge10m, TrainLayout::Partitioned, 10);
        assert_eq!(files.len(), 10);
        for (i, path) in files.iter().enumerate() {
            assert!(
                path.to_string_lossy()
                    .ends_with(&format!("{i:02}-of-10.parquet"))
            );
        }
    }
}
