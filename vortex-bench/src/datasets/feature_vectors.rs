// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::fs::File;
use std::path::PathBuf;
use std::sync::Arc;

use anyhow::Result;
use arrow_array::Int64Array;
use arrow_array::RecordBatch;
use arrow_array::builder::FixedSizeListBuilder;
use arrow_array::builder::Float32Builder;
use arrow_schema::DataType;
use arrow_schema::Field;
use arrow_schema::Schema;
use async_trait::async_trait;
use parquet::arrow::ArrowWriter;
use rand::RngExt;
use rand::SeedableRng;
use rand::rngs::StdRng;
use vortex::array::ArrayRef;
use vortex::array::ExecutionCtx;

use crate::CompactionStrategy;
use crate::Format;
use crate::conversions::parquet_to_vortex_chunks;
use crate::conversions::write_parquet_as_vortex;
use crate::datasets::Dataset;
use crate::idempotent_async;
use crate::random_access::BenchDataset;
use crate::random_access::data_path;
use crate::random_access::parquet_to_arrow_file;
use crate::vector_dataset::TrainLayout;
use crate::vector_dataset::VectorDataset;
use crate::vector_dataset::download;

/// Dataset identifier used for data path generation.
pub const DATASET: &str = "feature_vectors";

/// Number of rows in the feature vectors dataset.
pub const ROW_COUNT: usize = 1_000_000;

pub struct FeatureVectorsData;

/// A real f32 embedding dataset from the GloVe vector benchmark corpus.
pub struct GloveEmbeddingsData;

/// A real f64 embedding dataset from the OpenAI-on-C4 vector benchmark corpus.
pub struct OpenAiEmbeddingsData;

#[async_trait]
impl Dataset for GloveEmbeddingsData {
    fn name(&self) -> &str {
        "glove-embeddings-100k"
    }

    async fn to_vortex_array(&self, _ctx: &mut ExecutionCtx) -> Result<ArrayRef> {
        Ok(parquet_to_vortex_chunks(self.to_parquet_path().await?)
            .await?
            .into())
    }

    async fn to_parquet_path(&self) -> Result<PathBuf> {
        let paths = download(VectorDataset::GloveSmall100k, TrainLayout::Single).await?;
        paths
            .train_files
            .into_iter()
            .next()
            .ok_or_else(|| anyhow::anyhow!("GloVe embedding train file is missing"))
    }
}

#[async_trait]
impl Dataset for OpenAiEmbeddingsData {
    fn name(&self) -> &str {
        "openai-c4-embeddings-50k"
    }

    async fn to_vortex_array(&self, _ctx: &mut ExecutionCtx) -> Result<ArrayRef> {
        Ok(parquet_to_vortex_chunks(self.to_parquet_path().await?)
            .await?
            .into())
    }

    async fn to_parquet_path(&self) -> Result<PathBuf> {
        let paths = download(VectorDataset::OpenaiSmall50k, TrainLayout::Single).await?;
        paths
            .train_files
            .into_iter()
            .next()
            .ok_or_else(|| anyhow::anyhow!("OpenAI embedding train file is missing"))
    }
}

#[async_trait]
impl BenchDataset for FeatureVectorsData {
    fn name(&self) -> &str {
        "feature-vectors"
    }

    fn row_count(&self) -> u64 {
        ROW_COUNT as u64
    }

    async fn path(&self, format: Format) -> Result<PathBuf> {
        match format {
            Format::ArrowIpc => {
                let parquet_path = feature_vectors_parquet().await?;
                parquet_to_arrow_file(parquet_path, data_path(DATASET, Format::ArrowIpc))
            }
            Format::OnDiskVortex => feature_vectors_vortex().await,
            Format::VortexCompact => feature_vectors_vortex_compact().await,
            Format::Parquet => feature_vectors_parquet().await,
            other => unimplemented!("Random access bench not implemented for {other}"),
        }
    }
}

/// Dimensionality of each feature vector.
const VECTOR_DIM: i32 = 1024;

/// Batch size for data generation.
const BATCH_SIZE: usize = 100_000;

/// Generate a synthetic feature vectors parquet file.
///
/// Schema: `id: Int64, embedding: FixedSizeList<Float32, VECTOR_DIM>`.
/// This simulates a table of embedding vectors, common in ML workloads.
pub async fn feature_vectors_parquet() -> Result<PathBuf> {
    idempotent_async(
        data_path(DATASET, Format::Parquet),
        |temp_path| async move {
            let schema = Arc::new(Schema::new(vec![
                Field::new("id", DataType::Int64, false),
                Field::new(
                    "embedding",
                    DataType::FixedSizeList(
                        Arc::new(Field::new("item", DataType::Float32, true)),
                        VECTOR_DIM,
                    ),
                    false,
                ),
            ]));

            let file = File::create(&temp_path)?;
            let mut writer = ArrowWriter::try_new(file, Arc::clone(&schema), None)?;
            let mut rng = StdRng::seed_from_u64(42);

            for batch_start in (0..ROW_COUNT).step_by(BATCH_SIZE) {
                let batch_len = BATCH_SIZE.min(ROW_COUNT - batch_start);

                let ids = Int64Array::from_iter_values(
                    (batch_start as i64)..((batch_start + batch_len) as i64),
                );

                let mut list_builder = FixedSizeListBuilder::new(Float32Builder::new(), VECTOR_DIM);
                for _ in 0..batch_len {
                    for _ in 0..VECTOR_DIM {
                        list_builder.values().append_value(rng.random::<f32>());
                    }
                    list_builder.append(true);
                }
                let embedding = list_builder.finish();

                let batch = RecordBatch::try_new(
                    Arc::clone(&schema),
                    vec![Arc::new(ids), Arc::new(embedding)],
                )?;
                writer.write(&batch)?;
            }

            writer.close()?;
            Ok(())
        },
    )
    .await
}

/// Get the path to the feature vectors vortex file, converting from parquet if needed.
pub async fn feature_vectors_vortex() -> Result<PathBuf> {
    let parquet_path = feature_vectors_parquet().await?;
    let path = data_path(DATASET, Format::OnDiskVortex);
    write_parquet_as_vortex(parquet_path, &path, CompactionStrategy::Default).await
}

/// Get the path to the feature vectors compact vortex file, converting from parquet if needed.
pub async fn feature_vectors_vortex_compact() -> Result<PathBuf> {
    let parquet_path = feature_vectors_parquet().await?;
    let path = data_path(DATASET, Format::VortexCompact);
    write_parquet_as_vortex(parquet_path, &path, CompactionStrategy::Compact).await
}
