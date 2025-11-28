// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use anyhow::Result;
use async_trait::async_trait;
use rand::Rng;
use rand::SeedableRng;
use rand::rngs::StdRng;
use vortex::ArrayRef;
use vortex::IntoArray;
use vortex::arrays::ChunkedArray;
use vortex::arrays::ListArray;
use vortex::arrays::PrimitiveArray;
use vortex::arrays::StructArray;
use vortex::dtype::FieldNames;
use vortex::validity::Validity;

use crate::datasets::Dataset;

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

#[async_trait]
impl Dataset for StructListOfInts {
    fn name(&self) -> &str {
        &self.name
    }

    async fn to_vortex_array(&self) -> Result<ArrayRef> {
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
}
