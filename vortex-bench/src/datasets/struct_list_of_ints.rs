// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::fs::File;
use std::path::PathBuf;

use anyhow::Result;
use async_trait::async_trait;
use parquet::arrow::ArrowWriter;
use rand::RngExt;
use rand::SeedableRng;
use rand::rngs::StdRng;
use vortex::array::ArrayRef;
use vortex::array::ExecutionCtx;
use vortex::array::IntoArray;
use vortex::array::LEGACY_SESSION;
use vortex::array::VortexSessionExecute;
use vortex::array::arrays::ChunkedArray;
use vortex::array::arrays::ListArray;
use vortex::array::arrays::PrimitiveArray;
use vortex::array::arrays::StructArray;
use vortex::array::arrays::chunked::ChunkedArrayExt;
use vortex::array::arrays::listview::recursive_list_from_list_view;
use vortex::array::arrow::ArrowArrayExecutor;
use vortex::array::validity::Validity;
use vortex::dtype::FieldNames;

use crate::IdempotentPath;
use crate::datasets::Dataset;
use crate::idempotent_async;

/// Creates a randomly generated struct array, where each field is a list of
/// i64 of size one.
pub struct StructListOfInts {
    num_columns: usize,
    row_count: usize,
    chunk_count: usize,
    name: String,
}

impl StructListOfInts {
    pub fn new(num_columns: usize, row_count: usize, chunk_count: usize) -> Self {
        Self {
            num_columns,
            row_count,
            chunk_count,
            name: format!("wide table cols={num_columns} chunks={chunk_count} rows={row_count}"),
        }
    }
}

impl StructListOfInts {
    fn parquet_filename(&self) -> String {
        format!(
            "struct_list_of_ints_cols{}_chunks{}_rows{}.parquet",
            self.num_columns, self.chunk_count, self.row_count
        )
    }
}

#[async_trait]
impl Dataset for StructListOfInts {
    fn name(&self) -> &str {
        &self.name
    }

    async fn to_vortex_array(&self, _ctx: &mut ExecutionCtx) -> Result<ArrayRef> {
        let names: FieldNames = (0..self.num_columns)
            .map(|col_idx| col_idx.to_string())
            .collect();
        let mut rng = StdRng::seed_from_u64(0);

        let rows_per_chunk = (self.row_count / self.chunk_count).max(1usize);
        let chunks: Result<Vec<_>> = (0..self.row_count)
            .step_by(rows_per_chunk)
            .map(|starting_row| rows_per_chunk.min(self.row_count - starting_row))
            .map(|chunk_row_count| {
                let fields = (0..self.num_columns)
                    .map(|_| -> Result<ArrayRef> {
                        let elements = PrimitiveArray::from_iter(
                            (0..chunk_row_count).map(|_| rng.random::<i64>()),
                        );
                        let offsets: Result<Vec<u32>> = (0..=chunk_row_count)
                            .map(|i| {
                                u32::try_from(i).map_err(|e| {
                                    anyhow::anyhow!("Failed to convert index to u32: {}", e)
                                })
                            })
                            .collect();
                        let offsets = PrimitiveArray::from_iter(offsets?);
                        Ok(ListArray::try_new(
                            elements.into_array(),
                            offsets.into_array(),
                            Validity::AllValid,
                        )?
                        .into_array())
                    })
                    .collect::<Result<Vec<_>>>()?;
                Ok(StructArray::try_new(
                    names.clone(),
                    fields,
                    chunk_row_count,
                    Validity::NonNullable,
                )?
                .into_array())
            })
            .collect();

        let chunks = chunks?;
        Ok(ChunkedArray::from_iter(chunks).into_array())
    }

    async fn to_parquet_path(&self) -> Result<PathBuf> {
        let parquet_path =
            format!("struct_list_of_ints/{}", self.parquet_filename()).to_data_path();

        idempotent_async(&parquet_path, |temp_path| async move {
            // Generate the data
            let mut ctx = LEGACY_SESSION.create_execution_ctx();
            let array = self.to_vortex_array(&mut ctx).await?;

            // Convert to Arrow RecordBatches and write to parquet
            let chunked = array.as_::<vortex::array::arrays::Chunked>();

            let file = File::create(&temp_path)?;
            let mut writer: Option<ArrowWriter<File>> = None;

            for chunk in chunked.iter_chunks() {
                let converted = recursive_list_from_list_view(chunk.clone())?;
                let schema = converted.dtype().to_arrow_schema()?;
                let batch = converted
                    .execute_record_batch(&schema, &mut LEGACY_SESSION.create_execution_ctx())?;

                if writer.is_none() {
                    writer = Some(ArrowWriter::try_new(
                        file.try_clone()?,
                        batch.schema(),
                        None,
                    )?);
                }
                writer.as_mut().unwrap().write(&batch)?;
            }

            if let Some(w) = writer {
                w.close()?;
            }

            Ok(())
        })
        .await
    }
}
