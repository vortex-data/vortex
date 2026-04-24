// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::path::PathBuf;
use std::sync::Arc;

use anyhow::Result;
use arrow_array::Float64Array;
use arrow_array::Int64Array;
use arrow_array::RecordBatch;
use arrow_array::StructArray;
use arrow_schema::DataType;
use arrow_schema::Field;
use arrow_schema::Fields;
use arrow_schema::Schema;
use async_trait::async_trait;
use parquet::arrow::ArrowWriter;
use rand::RngExt;
use rand::SeedableRng;
use rand::rngs::StdRng;

use crate::CompactionStrategy;
use crate::Format;
use crate::conversions::write_parquet_as_vortex;
use crate::idempotent_async;
use crate::random_access::BenchDataset;
use crate::random_access::data_path;

/// Dataset identifier used for data path generation.
pub const DATASET: &str = "nested_structs";

/// Number of rows in the nested structs dataset.
pub const ROW_COUNT: usize = 1_000_000;

pub struct NestedStructsData;

#[async_trait]
impl BenchDataset for NestedStructsData {
    fn name(&self) -> &str {
        "nested-structs"
    }

    fn row_count(&self) -> u64 {
        ROW_COUNT as u64
    }

    async fn path(&self, format: Format) -> Result<PathBuf> {
        match format {
            Format::OnDiskVortex => nested_structs_vortex().await,
            Format::VortexCompact => nested_structs_vortex_compact().await,
            Format::Parquet => nested_structs_parquet().await,
            other => unimplemented!("Random access bench not implemented for {other}"),
        }
    }
}

/// Batch size for data generation.
const BATCH_SIZE: usize = 100_000;

/// Generate a synthetic nested structs parquet file.
///
/// Schema:
/// ```text
/// id: Int64
/// metadata: Struct {
///     a: Int64,
///     b: Float64,
///     inner: Struct {
///         x: Float64,
///         y: Float64,
///         z: Float64,
///     }
/// }
/// ```
pub async fn nested_structs_parquet() -> Result<PathBuf> {
    idempotent_async(
        data_path(DATASET, Format::Parquet),
        |temp_path| async move {
            let inner_fields = Fields::from(vec![
                Field::new("x", DataType::Float64, false),
                Field::new("y", DataType::Float64, false),
                Field::new("z", DataType::Float64, false),
            ]);
            let outer_fields = Fields::from(vec![
                Field::new("a", DataType::Int64, false),
                Field::new("b", DataType::Float64, false),
                Field::new("inner", DataType::Struct(inner_fields.clone()), false),
            ]);
            let schema = Arc::new(Schema::new(vec![
                Field::new("id", DataType::Int64, false),
                Field::new("metadata", DataType::Struct(outer_fields.clone()), false),
            ]));

            let file = std::fs::File::create(&temp_path)?;
            let mut writer = ArrowWriter::try_new(file, Arc::clone(&schema), None)?;
            let mut rng = StdRng::seed_from_u64(42);

            for batch_start in (0..ROW_COUNT).step_by(BATCH_SIZE) {
                let batch_len = BATCH_SIZE.min(ROW_COUNT - batch_start);

                let ids = Int64Array::from_iter_values(
                    (batch_start as i64)..((batch_start + batch_len) as i64),
                );

                let inner_x =
                    Float64Array::from_iter_values((0..batch_len).map(|_| rng.random::<f64>()));
                let inner_y =
                    Float64Array::from_iter_values((0..batch_len).map(|_| rng.random::<f64>()));
                let inner_z =
                    Float64Array::from_iter_values((0..batch_len).map(|_| rng.random::<f64>()));
                let inner = StructArray::try_new(
                    inner_fields.clone(),
                    vec![Arc::new(inner_x), Arc::new(inner_y), Arc::new(inner_z)],
                    None,
                )?;

                let outer_a =
                    Int64Array::from_iter_values((0..batch_len).map(|_| rng.random::<i64>()));
                let outer_b =
                    Float64Array::from_iter_values((0..batch_len).map(|_| rng.random::<f64>()));
                let outer = StructArray::try_new(
                    outer_fields.clone(),
                    vec![Arc::new(outer_a), Arc::new(outer_b), Arc::new(inner)],
                    None,
                )?;

                let batch = RecordBatch::try_new(
                    Arc::clone(&schema),
                    vec![Arc::new(ids), Arc::new(outer)],
                )?;
                writer.write(&batch)?;
            }

            writer.close()?;
            Ok(())
        },
    )
    .await
}

/// Get the path to the nested structs vortex file, converting from parquet if needed.
pub async fn nested_structs_vortex() -> Result<PathBuf> {
    let parquet_path = nested_structs_parquet().await?;
    let path = data_path(DATASET, Format::OnDiskVortex);
    write_parquet_as_vortex(parquet_path, &path, CompactionStrategy::Default).await
}

/// Get the path to the nested structs compact vortex file, converting from parquet if needed.
pub async fn nested_structs_vortex_compact() -> Result<PathBuf> {
    let parquet_path = nested_structs_parquet().await?;
    let path = data_path(DATASET, Format::VortexCompact);
    write_parquet_as_vortex(parquet_path, &path, CompactionStrategy::Compact).await
}
