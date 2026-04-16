// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! The static catalog of hosted vector benchmark corpora.
//!
//! Every entry in [`ALL_VECTOR_DATASETS`] declares its dimensionality, row count, element type,
//! distance metric, the set of train-split layouts hosted on the upstream bucket, and whether
//! ground-truth `neighbors.parquet` and `scalar_labels` columns are available.
//! Most entries mirror VectorDBBench's dataset model; a few are extra prefixes that are present
//! on the same public bucket and use the same file layout.
//!
//! Higher-level code (downloaders, ingest pipelines, recall measurement) should be parameterized
//! over these descriptors rather than hardcoding per-dataset URLs.

use std::num::NonZeroU32;

use anyhow::Result;
use anyhow::bail;
use clap::ValueEnum;
use vortex::dtype::PType;

use super::layout::LayoutSpec;
use super::layout::TrainLayout;
use super::layout::VectorMetric;

const TEN_SHARDS: NonZeroU32 = NonZeroU32::new(10).unwrap();
const FIFTY_SHARDS: NonZeroU32 = NonZeroU32::new(50).unwrap();
const ONE_HUNDRED_SHARDS: NonZeroU32 = NonZeroU32::new(100).unwrap();

/// Every [`VectorDataset`] variant in catalog order. Useful for CLI help and metadata-consistency
/// tests.
pub const ALL_VECTOR_DATASETS: &[VectorDataset] = &[
    VectorDataset::CohereSmall100k,
    VectorDataset::CohereMedium1m,
    VectorDataset::CohereLarge10m,
    VectorDataset::OpenaiSmall50k,
    VectorDataset::OpenaiMedium500k,
    VectorDataset::OpenaiLarge5m,
    VectorDataset::BioasqMedium1m,
    VectorDataset::BioasqLarge10m,
    VectorDataset::GloveSmall100k,
    VectorDataset::GloveMedium1m,
    VectorDataset::GistSmall100k,
    VectorDataset::GistMedium1m,
    VectorDataset::SiftSmall500k,
    VectorDataset::SiftMedium5m,
    VectorDataset::SiftLarge50m,
    VectorDataset::LaionLarge100m,
];

/// The publicly hosted vector benchmark datasets.
///
/// Variants are named `<source><size><rowcount>`, kebab-cased on the CLI (e.g. `cohere-large-10m`).
///
/// The static metadata for each variant (dimensionality, row count, hosted layouts, etc.) is
/// exposed via the inherent methods below; the full table is reachable via [`ALL_VECTOR_DATASETS`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, ValueEnum)]
#[clap(rename_all = "kebab-case")]
pub enum VectorDataset {
    /// Cohere wiki-22-12, 100K × 768 f32, cosine. Single + SingleShuffled.
    CohereSmall100k,
    /// Cohere wiki-22-12, 1M × 768 f32, cosine. Single + SingleShuffled.
    CohereMedium1m,
    /// Cohere wiki-22-12, 10M × 768 f32, cosine. Partitioned + PartitionedShuffled (10 shards).
    CohereLarge10m,

    /// OpenAI embeddings on C4, 50K × 1536 f64, cosine. Single + SingleShuffled.
    OpenaiSmall50k,
    /// OpenAI embeddings on C4, 500K × 1536 f64, cosine. Single + SingleShuffled.
    OpenaiMedium500k,
    /// OpenAI embeddings on C4, 5M × 1536 f64, cosine. Partitioned + PartitionedShuffled (10
    /// shards).
    OpenaiLarge5m,

    /// Bioasq biomedical, 1M × 1024 f32, cosine. SingleShuffled only.
    BioasqMedium1m,
    /// Bioasq biomedical, 10M × 1024 f32, cosine. PartitionedShuffled only (10 shards).
    BioasqLarge10m,

    /// GloVe word vectors, 100K × 200 f32, cosine. Single only. No neighbors / labels.
    GloveSmall100k,
    /// GloVe word vectors, 1M × 200 f32, cosine. Single only. No neighbors / labels.
    GloveMedium1m,

    /// GIST image features, 100K × 960 f32, L2. Single only. No neighbors / labels.
    GistSmall100k,
    /// GIST image features, 1M × 960 f32, L2. Single only. No neighbors / labels.
    GistMedium1m,

    /// SIFT image features, 500K × 128 f32, L2. Single only. No neighbors / labels.
    SiftSmall500k,
    /// SIFT image features, 5M × 128 f32, L2. Single only. No neighbors / labels.
    SiftMedium5m,
    /// SIFT image features, 50M × 128 f32, L2. Partitioned only (50 shards). No labels.
    SiftLarge50m,

    /// LAION image embeddings, 100M × 768 f32, L2. Partitioned only (100 shards).
    /// Has `neighbors.parquet` and `scalar_labels.parquet`.
    LaionLarge100m,
}

impl VectorDataset {
    /// Stable kebab-cased label used in CLI args, file paths, and metric names.
    pub fn name(&self) -> &'static str {
        match self {
            VectorDataset::CohereSmall100k => "cohere-small-100k",
            VectorDataset::CohereMedium1m => "cohere-medium-1m",
            VectorDataset::CohereLarge10m => "cohere-large-10m",
            VectorDataset::OpenaiSmall50k => "openai-small-50k",
            VectorDataset::OpenaiMedium500k => "openai-medium-500k",
            VectorDataset::OpenaiLarge5m => "openai-large-5m",
            VectorDataset::BioasqMedium1m => "bioasq-medium-1m",
            VectorDataset::BioasqLarge10m => "bioasq-large-10m",
            VectorDataset::GloveSmall100k => "glove-small-100k",
            VectorDataset::GloveMedium1m => "glove-medium-1m",
            VectorDataset::GistSmall100k => "gist-small-100k",
            VectorDataset::GistMedium1m => "gist-medium-1m",
            VectorDataset::SiftSmall500k => "sift-small-500k",
            VectorDataset::SiftMedium5m => "sift-medium-5m",
            VectorDataset::SiftLarge50m => "sift-large-50m",
            VectorDataset::LaionLarge100m => "laion-large-100m",
        }
    }

    /// The directory name on `assets.zilliz.com/benchmark/<prefix>/`. Snake-cased to
    /// match the upstream bucket's existing naming convention.
    pub fn s3_prefix(&self) -> &'static str {
        match self {
            VectorDataset::CohereSmall100k => "cohere_small_100k",
            VectorDataset::CohereMedium1m => "cohere_medium_1m",
            VectorDataset::CohereLarge10m => "cohere_large_10m",
            VectorDataset::OpenaiSmall50k => "openai_small_50k",
            VectorDataset::OpenaiMedium500k => "openai_medium_500k",
            VectorDataset::OpenaiLarge5m => "openai_large_5m",
            VectorDataset::BioasqMedium1m => "bioasq_medium_1m",
            VectorDataset::BioasqLarge10m => "bioasq_large_10m",
            VectorDataset::GloveSmall100k => "glove_small_100k",
            VectorDataset::GloveMedium1m => "glove_medium_1m",
            VectorDataset::GistSmall100k => "gist_small_100k",
            VectorDataset::GistMedium1m => "gist_medium_1m",
            VectorDataset::SiftSmall500k => "sift_small_500k",
            VectorDataset::SiftMedium5m => "sift_medium_5m",
            VectorDataset::SiftLarge50m => "sift_large_50m",
            VectorDataset::LaionLarge100m => "laion_large_100m",
        }
    }

    /// Vector dimensionality.
    pub fn dim(&self) -> u32 {
        match self {
            VectorDataset::CohereSmall100k
            | VectorDataset::CohereMedium1m
            | VectorDataset::CohereLarge10m
            | VectorDataset::LaionLarge100m => 768,
            VectorDataset::OpenaiSmall50k
            | VectorDataset::OpenaiMedium500k
            | VectorDataset::OpenaiLarge5m => 1536,
            VectorDataset::BioasqMedium1m | VectorDataset::BioasqLarge10m => 1024,
            VectorDataset::GloveSmall100k | VectorDataset::GloveMedium1m => 200,
            VectorDataset::GistSmall100k | VectorDataset::GistMedium1m => 960,
            VectorDataset::SiftSmall500k
            | VectorDataset::SiftMedium5m
            | VectorDataset::SiftLarge50m => 128,
        }
    }

    /// Number of rows in the train split (sum across shards if partitioned).
    pub fn num_rows(&self) -> u64 {
        match self {
            VectorDataset::CohereSmall100k => 100_000,
            VectorDataset::CohereMedium1m => 1_000_000,
            VectorDataset::CohereLarge10m => 10_000_000,
            VectorDataset::OpenaiSmall50k => 50_000,
            VectorDataset::OpenaiMedium500k => 500_000,
            VectorDataset::OpenaiLarge5m => 5_000_000,
            VectorDataset::BioasqMedium1m => 1_000_000,
            VectorDataset::BioasqLarge10m => 10_000_000,
            VectorDataset::GloveSmall100k => 100_000,
            VectorDataset::GloveMedium1m => 1_000_000,
            VectorDataset::GistSmall100k => 100_000,
            VectorDataset::GistMedium1m => 1_000_000,
            VectorDataset::SiftSmall500k => 500_000,
            VectorDataset::SiftMedium5m => 5_000_000,
            VectorDataset::SiftLarge50m => 50_000_000,
            VectorDataset::LaionLarge100m => 100_000_000,
        }
    }

    /// Element scalar type as stored in the upstream parquet. The benchmark always casts to
    /// `f32` after ingest (TurboQuant + the handrolled baseline operate in f32), so this is
    /// only consulted by the parquet readers.
    pub fn element_ptype(&self) -> PType {
        match self {
            VectorDataset::OpenaiSmall50k
            | VectorDataset::OpenaiMedium500k
            | VectorDataset::OpenaiLarge5m => PType::F64,
            _ => PType::F32,
        }
    }

    /// Distance metric the upstream dataset was curated for.
    pub fn metric(&self) -> VectorMetric {
        match self {
            VectorDataset::GistSmall100k
            | VectorDataset::GistMedium1m
            | VectorDataset::SiftSmall500k
            | VectorDataset::SiftMedium5m
            | VectorDataset::SiftLarge50m
            | VectorDataset::LaionLarge100m => VectorMetric::L2,
            _ => VectorMetric::Cosine,
        }
    }

    /// The set of train-split layouts hosted on the upstream bucket for this dataset.
    ///
    /// Always non-empty since the catalog never declares a dataset with zero layouts.
    pub fn layouts(&self) -> Vec<LayoutSpec> {
        match self {
            VectorDataset::CohereSmall100k
            | VectorDataset::CohereMedium1m
            | VectorDataset::OpenaiSmall50k
            | VectorDataset::OpenaiMedium500k => {
                vec![LayoutSpec::single(), LayoutSpec::single_shuffled()]
            }
            VectorDataset::CohereLarge10m | VectorDataset::OpenaiLarge5m => vec![
                LayoutSpec::partitioned(TEN_SHARDS),
                LayoutSpec::partitioned_shuffled(TEN_SHARDS),
            ],
            VectorDataset::BioasqMedium1m => vec![LayoutSpec::single_shuffled()],
            VectorDataset::BioasqLarge10m => {
                vec![LayoutSpec::partitioned_shuffled(TEN_SHARDS)]
            }
            VectorDataset::GloveSmall100k
            | VectorDataset::GloveMedium1m
            | VectorDataset::GistSmall100k
            | VectorDataset::GistMedium1m
            | VectorDataset::SiftSmall500k
            | VectorDataset::SiftMedium5m => vec![LayoutSpec::single()],
            VectorDataset::SiftLarge50m => vec![LayoutSpec::partitioned(FIFTY_SHARDS)],
            VectorDataset::LaionLarge100m => vec![LayoutSpec::partitioned(ONE_HUNDRED_SHARDS)],
        }
    }

    /// Whether `neighbors.parquet` (top-K ground truth for recall) is hosted.
    pub fn has_neighbors(&self) -> bool {
        match self {
            VectorDataset::CohereSmall100k
            | VectorDataset::CohereMedium1m
            | VectorDataset::CohereLarge10m
            | VectorDataset::OpenaiSmall50k
            | VectorDataset::OpenaiMedium500k
            | VectorDataset::OpenaiLarge5m
            | VectorDataset::BioasqMedium1m
            | VectorDataset::BioasqLarge10m
            | VectorDataset::LaionLarge100m => true,
            VectorDataset::GloveSmall100k
            | VectorDataset::GloveMedium1m
            | VectorDataset::GistSmall100k
            | VectorDataset::GistMedium1m
            | VectorDataset::SiftSmall500k
            | VectorDataset::SiftMedium5m
            | VectorDataset::SiftLarge50m => false,
        }
    }

    /// Whether the train split carries a `scalar_labels` column (for filtered-search
    /// benchmarks). The benchmark does not exercise filtered search yet, but the ingest
    /// pipeline copies the column through when present so future filtered-recall work does
    /// not require re-ingest.
    pub fn has_scalar_labels(&self) -> bool {
        matches!(
            self,
            VectorDataset::CohereSmall100k
                | VectorDataset::CohereMedium1m
                | VectorDataset::CohereLarge10m
                | VectorDataset::OpenaiSmall50k
                | VectorDataset::OpenaiMedium500k
                | VectorDataset::OpenaiLarge5m
                | VectorDataset::BioasqMedium1m
                | VectorDataset::BioasqLarge10m
                | VectorDataset::LaionLarge100m
        )
    }

    /// Validate that `layout` is one of the layouts hosted for this dataset and return its
    /// [`LayoutSpec`].
    ///
    /// Bails with an error message that lists the allowed values.
    pub fn validate_layout(&self, layout: TrainLayout) -> Result<LayoutSpec> {
        let layouts = self.layouts();
        match layouts.iter().find(|spec| spec.layout() == layout) {
            Some(spec) => Ok(*spec),
            None => {
                let allowed = layouts
                    .iter()
                    .map(|s| s.layout().label())
                    .collect::<Vec<_>>()
                    .join(", ");
                bail!(
                    "dataset {} does not have layout '{}'; allowed layouts: [{}]",
                    self.name(),
                    layout,
                    allowed,
                );
            }
        }
    }

    /// Pick the default layout for this dataset — the first entry in [`Self::layouts`].
    /// Stable across runs since the catalog table is statically ordered.
    pub fn default_layout(&self) -> LayoutSpec {
        self.layouts()[0]
    }
}

#[cfg(test)]
mod tests {
    use vortex::utils::aliases::hash_set::HashSet;

    use super::*;

    #[test]
    fn all_datasets_have_consistent_metadata() {
        let mut seen: HashSet<&'static str> = HashSet::default();
        for &ds in ALL_VECTOR_DATASETS {
            assert!(seen.insert(ds.name()), "duplicate name {}", ds.name());
            assert!(ds.dim() > 0);
            assert!(ds.num_rows() > 0);
            assert!(!ds.layouts().is_empty(), "{} has no layouts", ds.name());
            assert_eq!(
                ds.s3_prefix().chars().filter(|c| *c == '_').count() + 1,
                ds.name().split('-').count(),
                "{}: s3_prefix '{}' shape disagrees with name '{}'",
                ds.name(),
                ds.s3_prefix(),
                ds.name(),
            );
        }
    }

    #[test]
    fn validate_layout_accepts_hosted_layout() {
        let ds = VectorDataset::CohereSmall100k;
        let spec = ds.validate_layout(TrainLayout::Single).unwrap();
        assert_eq!(spec.layout(), TrainLayout::Single);
        assert_eq!(spec.num_files(), 1);
    }

    #[test]
    fn validate_layout_rejects_unhosted_layout() {
        let ds = VectorDataset::SiftSmall500k;
        let err = ds
            .validate_layout(TrainLayout::SingleShuffled)
            .unwrap_err()
            .to_string();
        assert!(
            err.contains("does not have layout 'single-shuffled'"),
            "{err}"
        );
        assert!(err.contains("[single]"), "{err}");
    }

    #[test]
    fn partitioned_datasets_declare_shard_count() {
        let layouts = VectorDataset::CohereLarge10m.layouts();
        assert_eq!(layouts.len(), 2);
        for spec in layouts {
            assert!(spec.layout().is_partitioned());
            assert_eq!(spec.num_files(), 10);
        }
    }

    #[test]
    fn l2_datasets_match_upstream_metric() {
        for ds in [
            VectorDataset::GistSmall100k,
            VectorDataset::GistMedium1m,
            VectorDataset::SiftSmall500k,
            VectorDataset::SiftMedium5m,
            VectorDataset::SiftLarge50m,
            VectorDataset::LaionLarge100m,
        ] {
            assert_eq!(ds.metric(), VectorMetric::L2, "{} should use L2", ds.name());
        }
    }

    #[test]
    fn datasets_without_neighbors_skip_recall() {
        for ds in [
            VectorDataset::GloveSmall100k,
            VectorDataset::GloveMedium1m,
            VectorDataset::GistSmall100k,
            VectorDataset::GistMedium1m,
            VectorDataset::SiftSmall500k,
            VectorDataset::SiftMedium5m,
            VectorDataset::SiftLarge50m,
        ] {
            assert!(
                !ds.has_neighbors(),
                "{} unexpectedly has neighbors",
                ds.name()
            );
        }
    }
}
