// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Datasets used by the vector-search benchmark.
//!
//! These are a subset of the public VectorDBBench
//! (<https://github.com/zilliztech/VectorDBBench>) datasets — MIT-licensed canonical
//! embedding corpora published by Zilliz under
//! `https://assets.zilliz.com/benchmark/<dir>/`. Each dataset is distributed as one or more
//! parquet files with a `emb: list<float>` column (the raw embedding vectors) and an
//! `id: int64` column.
//!
//! The URL constants below point at the upstream Zilliz bucket. For CI runs we recommend
//! mirroring these files into an internal bucket first to avoid repeated egress charges on
//! a third-party bucket — mirror setup is a one-off manual operation and documented in the
//! vector-search-bench crate README.

use std::path::PathBuf;

use anyhow::Result;
use async_trait::async_trait;
use tokio::fs::File;
use vortex::array::ArrayRef;
use vortex::array::IntoArray;
use vortex::array::stream::ArrayStreamExt;
use vortex::file::OpenOptionsSessionExt;
use vortex::file::WriteOptionsSessionExt;

use crate::IdempotentPath;
use crate::SESSION;
use crate::conversions::parquet_to_vortex_chunks;
use crate::datasets::Dataset;
use crate::datasets::data_downloads::download_data;
use crate::idempotent_async;

/// A public embedding-vector dataset used by the vector-search benchmark.
///
/// Each variant is one of the canonical VectorDBBench corpora, distributed as parquet under
/// the Zilliz public benchmark bucket. The smaller `*Small` sizes are appropriate for CI
/// runs; the larger sizes are intended for local / on-demand experiments. Only
/// cosine-metric datasets are wired today — SIFT / GIST / LAION (L2) will follow when an
/// L2-distance scalar function lands in `vortex-tensor`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VectorDataset {
    /// Cohere wiki-22-12, 100K rows × 768 dims, cosine metric. ~307 MB raw / ~150 MB
    /// zstd-parquet — the default CI-friendly size.
    CohereSmall,
    /// Cohere wiki-22-12, 1M rows × 768 dims, cosine metric. ~3 GB raw / ~1.5 GB
    /// zstd-parquet. Local-only by default.
    CohereMedium,
    /// OpenAI embeddings on C4, 50K rows × 1536 dims, cosine metric. ~307 MB raw —
    /// the smallest OpenAI variant and comparable in size to Cohere-small, but with
    /// double the dimensionality.
    OpenAiSmall,
    /// OpenAI embeddings on C4, 500K rows × 1536 dims, cosine metric. ~3 GB raw.
    /// Local-only by default.
    OpenAiMedium,
    /// Bioasq biomedical embeddings, 1M rows × 1024 dims, cosine metric. ~4 GB raw.
    /// Local-only by default.
    BioasqMedium,
    /// Glove word embeddings, 1M rows × 200 dims, cosine metric. ~800 MB raw.
    GloveMedium,
}

/// All built-in [`VectorDataset`] variants in a fixed order. Convenient for iterating or
/// for listing choices in CLI help.
pub const ALL_VECTOR_DATASETS: &[VectorDataset] = &[
    VectorDataset::CohereSmall,
    VectorDataset::CohereMedium,
    VectorDataset::OpenAiSmall,
    VectorDataset::OpenAiMedium,
    VectorDataset::BioasqMedium,
    VectorDataset::GloveMedium,
];

impl VectorDataset {
    /// The upstream URL for this dataset's canonical train-split parquet file.
    ///
    /// **CI note**: point at an internal mirror before enabling this benchmark in CI —
    /// see `benchmarks/vector-search-bench/README.md` for the procedure.
    pub fn parquet_url(&self) -> &'static str {
        match self {
            VectorDataset::CohereSmall => {
                "https://assets.zilliz.com/benchmark/cohere_small_100k/train.parquet"
            }
            VectorDataset::CohereMedium => {
                "https://assets.zilliz.com/benchmark/cohere_medium_1m/train.parquet"
            }
            VectorDataset::OpenAiSmall => {
                "https://assets.zilliz.com/benchmark/openai_small_50k/train.parquet"
            }
            VectorDataset::OpenAiMedium => {
                "https://assets.zilliz.com/benchmark/openai_medium_500k/train.parquet"
            }
            VectorDataset::BioasqMedium => {
                "https://assets.zilliz.com/benchmark/bioasq_medium_1m/train.parquet"
            }
            VectorDataset::GloveMedium => {
                "https://assets.zilliz.com/benchmark/glove_medium_1m/train.parquet"
            }
        }
    }

    /// Fixed vector dimensionality for this dataset.
    pub fn dim(&self) -> u32 {
        match self {
            VectorDataset::CohereSmall | VectorDataset::CohereMedium => 768,
            VectorDataset::OpenAiSmall | VectorDataset::OpenAiMedium => 1536,
            VectorDataset::BioasqMedium => 1024,
            VectorDataset::GloveMedium => 200,
        }
    }

    /// Expected number of rows in the train split.
    pub fn num_rows(&self) -> usize {
        match self {
            VectorDataset::CohereSmall => 100_000,
            VectorDataset::CohereMedium => 1_000_000,
            VectorDataset::OpenAiSmall => 50_000,
            VectorDataset::OpenAiMedium => 500_000,
            VectorDataset::BioasqMedium => 1_000_000,
            VectorDataset::GloveMedium => 1_000_000,
        }
    }

    /// The distance metric the upstream dataset was curated for. v1 only wires cosine, so
    /// every built-in dataset returns [`VectorMetric::Cosine`]. The enum variant exists so
    /// that L2 / inner-product datasets can be added later without a breaking change.
    pub fn metric(&self) -> VectorMetric {
        match self {
            VectorDataset::CohereSmall
            | VectorDataset::CohereMedium
            | VectorDataset::OpenAiSmall
            | VectorDataset::OpenAiMedium
            | VectorDataset::BioasqMedium
            | VectorDataset::GloveMedium => VectorMetric::Cosine,
        }
    }
}

/// Distance metric a dataset was curated for. The vector-search benchmark only wires cosine
/// today, but having this explicit makes it obvious when a future dataset should be paired
/// with L2 or inner-product instead.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VectorMetric {
    /// Cosine similarity: `dot(a, b) / (||a|| * ||b||)`.
    Cosine,
    /// Squared L2 distance: `sum((a - b)^2)`.
    L2,
    /// Inner product: `dot(a, b)`.
    InnerProduct,
}

#[async_trait]
impl Dataset for VectorDataset {
    fn name(&self) -> &str {
        match self {
            VectorDataset::CohereSmall => "cohere-small",
            VectorDataset::CohereMedium => "cohere-medium",
            VectorDataset::OpenAiSmall => "openai-small",
            VectorDataset::OpenAiMedium => "openai-medium",
            VectorDataset::BioasqMedium => "bioasq-medium",
            VectorDataset::GloveMedium => "glove-medium",
        }
    }

    async fn to_parquet_path(&self) -> Result<PathBuf> {
        let dir = format!("{}/", self.name()).to_data_path();
        let parquet = dir.join(format!("{}.parquet", self.name()));
        download_data(parquet.clone(), self.parquet_url()).await?;
        Ok(parquet)
    }

    async fn to_vortex_array(&self) -> Result<ArrayRef> {
        let parquet = self.to_parquet_path().await?;
        let dir = format!("{}/", self.name()).to_data_path();
        let vortex = dir.join(format!("{}.vortex", self.name()));

        let data = parquet_to_vortex_chunks(parquet).await?;
        idempotent_async(&vortex, async |path| -> Result<()> {
            SESSION
                .write_options()
                .write(
                    &mut File::create(path)
                        .await
                        .map_err(|e| anyhow::anyhow!("Failed to create file: {}", e))?,
                    data.into_array().to_array_stream(),
                )
                .await
                .map_err(|e| anyhow::anyhow!("Failed to write vortex file: {}", e))?;
            Ok(())
        })
        .await?;

        Ok(SESSION
            .open_options()
            .open_path(vortex.as_path())
            .await?
            .scan()?
            .into_array_stream()?
            .read_all()
            .await?)
    }
}

#[cfg(test)]
mod tests {
    use vortex::utils::aliases::hash_set::HashSet;

    use super::ALL_VECTOR_DATASETS;
    use super::VectorDataset;
    use super::VectorMetric;
    use crate::datasets::Dataset;

    #[test]
    fn cohere_small_metadata() {
        let ds = VectorDataset::CohereSmall;
        assert_eq!(ds.name(), "cohere-small");
        assert_eq!(ds.dim(), 768);
        assert_eq!(ds.num_rows(), 100_000);
        assert_eq!(ds.metric(), VectorMetric::Cosine);
        assert!(ds.parquet_url().ends_with("/train.parquet"));
        assert!(ds.parquet_url().contains("cohere_small_100k"));
    }

    #[test]
    fn all_datasets_have_consistent_metadata() {
        // Every built-in dataset must have a unique kebab-cased name, point at a
        // `train.parquet` file under `assets.zilliz.com/benchmark/`, declare a
        // dimension ≥ `MIN_DIMENSION` for TurboQuant, a non-zero row count, and
        // (for v1) cosine metric.
        let mut seen_names: HashSet<String> = HashSet::default();
        for &ds in ALL_VECTOR_DATASETS {
            let name = ds.name();
            assert!(
                seen_names.insert(name.to_string()),
                "duplicate dataset name {name}",
            );
            assert!(
                ds.dim() >= 128,
                "{name} dim {} below TurboQuant minimum",
                ds.dim()
            );
            assert!(ds.num_rows() > 0, "{name} has zero rows");
            assert_eq!(
                ds.metric(),
                VectorMetric::Cosine,
                "{name} must be cosine for v1"
            );
            let url = ds.parquet_url();
            assert!(url.starts_with("https://assets.zilliz.com/benchmark/"));
            assert!(url.ends_with("/train.parquet"));
        }
    }
}
