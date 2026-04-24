// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Train-split layout variants for vector benchmark datasets.
//!
//! VectorDBBench corpora are published under `assets.zilliz.com/benchmark/<prefix>/` in up to four
//! different shapes:
//! - a single train file
//! - a single shuffled-rows train file
//! - a partitioned train file split into N shards
//! - the same partitioned shape in shuffled-rows order.
//!
//! Not every dataset hosts every layout. See [`VectorDataset::layouts`] for the per-dataset list.
//!
//! [`VectorDataset::layouts`]: super::VectorDataset::layouts

use std::fmt;
use std::num::NonZeroU32;

use clap::ValueEnum;
use serde::Deserialize;
use serde::Serialize;

/// Distance metric a dataset was curated for.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VectorMetric {
    /// `dot(a, b) / (||a|| * ||b||)`.
    Cosine,
    /// `sum((a - b)^2)`.
    L2,
    // TODO(connor): Do we even need this?
    /// `dot(a, b)`.
    InnerProduct,
}

/// A specific train layout published for a dataset, plus the shard count when partitioned
/// (always `1` for `Single` / `SingleShuffled`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LayoutSpec {
    layout: TrainLayout,
    num_files: NonZeroU32,
}

impl LayoutSpec {
    /// Build a single-file layout spec.
    pub const fn single() -> Self {
        Self {
            layout: TrainLayout::Single,
            num_files: NonZeroU32::MIN,
        }
    }

    /// Build a shuffled single-file layout spec.
    pub const fn single_shuffled() -> Self {
        Self {
            layout: TrainLayout::SingleShuffled,
            num_files: NonZeroU32::MIN,
        }
    }

    /// Build a partitioned layout spec with the given shard count.
    pub const fn partitioned(num_files: NonZeroU32) -> Self {
        Self {
            layout: TrainLayout::Partitioned,
            num_files,
        }
    }

    /// Build a shuffled partitioned layout spec with the given shard count.
    pub const fn partitioned_shuffled(num_files: NonZeroU32) -> Self {
        Self {
            layout: TrainLayout::PartitionedShuffled,
            num_files,
        }
    }

    /// Which of the four published shapes this entry describes.
    pub const fn layout(&self) -> TrainLayout {
        self.layout
    }

    /// Number of parquet shards on the bucket. `1` for the single-file layouts.
    pub const fn num_files(&self) -> u32 {
        self.num_files.get()
    }
}

/// One of the four published train-split shapes for a VectorDBBench corpus.
///
/// `Single` and `SingleShuffled` are one-file layouts; `Partitioned` and `PartitionedShuffled` are
/// sharded into N files. The shuffled variants randomize the row order, which is useful when you
/// want the on-disk arrangement to be representative of a query workload rather than of the
/// upstream ingest order.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, ValueEnum, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum TrainLayout {
    /// One `train.parquet` file. Row order matches the upstream curation.
    #[clap(name = "single")]
    Single,
    /// One `shuffle_train.parquet` file. Row order is randomized.
    #[clap(name = "single-shuffled")]
    SingleShuffled,
    /// Multiple `train-NN-of-N.parquet` shards. Row order matches the upstream curation.
    #[clap(name = "partitioned")]
    Partitioned,
    /// Multiple `shuffle_train-NN-of-N.parquet` shards. Row order is randomized.
    #[clap(name = "partitioned-shuffled")]
    PartitionedShuffled,
}

impl TrainLayout {
    /// Stable kebab-cased label used in CLI args, file paths, and metric names.
    pub fn label(&self) -> &'static str {
        match self {
            TrainLayout::Single => "single",
            TrainLayout::SingleShuffled => "single-shuffled",
            TrainLayout::Partitioned => "partitioned",
            TrainLayout::PartitionedShuffled => "partitioned-shuffled",
        }
    }

    /// Whether this layout is split across multiple parquet files.
    pub fn is_partitioned(&self) -> bool {
        matches!(
            self,
            TrainLayout::Partitioned | TrainLayout::PartitionedShuffled
        )
    }
}

impl fmt::Display for TrainLayout {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.label())
    }
}

#[cfg(test)]
mod tests {
    use std::num::NonZeroU32;

    use super::*;

    #[test]
    fn label_round_trips_through_value_enum() {
        for layout in [
            TrainLayout::Single,
            TrainLayout::SingleShuffled,
            TrainLayout::Partitioned,
            TrainLayout::PartitionedShuffled,
        ] {
            let parsed = TrainLayout::from_str(layout.label(), true).unwrap();
            assert_eq!(parsed, layout);
        }
    }

    #[test]
    fn is_partitioned_matches_variant() {
        assert!(!TrainLayout::Single.is_partitioned());
        assert!(!TrainLayout::SingleShuffled.is_partitioned());
        assert!(TrainLayout::Partitioned.is_partitioned());
        assert!(TrainLayout::PartitionedShuffled.is_partitioned());
    }

    #[test]
    fn layout_specs_encode_valid_shape_and_count() {
        assert_eq!(LayoutSpec::single().layout(), TrainLayout::Single);
        assert_eq!(LayoutSpec::single().num_files(), 1);
        assert_eq!(
            LayoutSpec::single_shuffled().layout(),
            TrainLayout::SingleShuffled
        );
        assert_eq!(
            LayoutSpec::partitioned(NonZeroU32::new(10).unwrap()).layout(),
            TrainLayout::Partitioned
        );
        assert_eq!(
            LayoutSpec::partitioned_shuffled(NonZeroU32::new(10).unwrap()).layout(),
            TrainLayout::PartitionedShuffled
        );
        assert_eq!(
            LayoutSpec::partitioned_shuffled(NonZeroU32::new(10).unwrap()).num_files(),
            10
        );
    }
}
