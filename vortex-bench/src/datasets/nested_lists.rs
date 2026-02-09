// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::path::PathBuf;
use std::sync::Arc;

use anyhow::Result;
use arrow_array::Int64Array;
use arrow_array::RecordBatch;
use arrow_array::builder::Int64Builder;
use arrow_array::builder::ListBuilder;
use arrow_schema::DataType;
use arrow_schema::Field;
use arrow_schema::Schema;
use parquet::arrow::ArrowWriter;
use rand::Rng;
use rand::SeedableRng;
use rand::rngs::StdRng;

use crate::CompactionStrategy;
use crate::conversions::write_parquet_as_vortex;
use crate::idempotent_async;

/// Number of rows in the nested lists dataset.
pub const ROW_COUNT: usize = 1_000_000;

/// Maximum number of elements in each list.
const MAX_LIST_LEN: usize = 20;

/// Batch size for data generation.
const BATCH_SIZE: usize = 100_000;

/// Generate a synthetic nested lists parquet file.
///
/// Schema: `id: Int64, values: List<Int64>`.
/// Each row contains a variable-length list of 1 to 20 random integers.
pub async fn nested_lists_parquet() -> Result<PathBuf> {
    idempotent_async(
        "nested_lists/nested_lists.parquet",
        |temp_path| async move {
            let schema = Arc::new(Schema::new(vec![
                Field::new("id", DataType::Int64, false),
                Field::new(
                    "values",
                    DataType::List(Arc::new(Field::new("item", DataType::Int64, true))),
                    false,
                ),
            ]));

            let file = std::fs::File::create(&temp_path)?;
            let mut writer = ArrowWriter::try_new(file, schema.clone(), None)?;
            let mut rng = StdRng::seed_from_u64(42);

            for batch_start in (0..ROW_COUNT).step_by(BATCH_SIZE) {
                let batch_len = BATCH_SIZE.min(ROW_COUNT - batch_start);

                let ids = Int64Array::from_iter_values(
                    (batch_start as i64)..((batch_start + batch_len) as i64),
                );

                let mut list_builder = ListBuilder::new(Int64Builder::new());
                for _ in 0..batch_len {
                    let list_len = rng.random_range(1..=MAX_LIST_LEN);
                    for _ in 0..list_len {
                        list_builder.values().append_value(rng.random::<i64>());
                    }
                    list_builder.append(true);
                }
                let values = list_builder.finish();

                let batch =
                    RecordBatch::try_new(schema.clone(), vec![Arc::new(ids), Arc::new(values)])?;
                writer.write(&batch)?;
            }

            writer.close()?;
            Ok(())
        },
    )
    .await
}

/// Get the path to the nested lists vortex file, converting from parquet if needed.
pub async fn nested_lists_vortex() -> Result<PathBuf> {
    let parquet_path = nested_lists_parquet().await?;
    write_parquet_as_vortex(
        parquet_path,
        "nested_lists/nested_lists.vortex",
        CompactionStrategy::Default,
    )
    .await
}

/// Get the path to the nested lists compact vortex file, converting from parquet if needed.
pub async fn nested_lists_vortex_compact() -> Result<PathBuf> {
    let parquet_path = nested_lists_parquet().await?;
    write_parquet_as_vortex(
        parquet_path,
        "nested_lists/nested_lists-compact.vortex",
        CompactionStrategy::Compact,
    )
    .await
}
