use async_trait::async_trait;
use rand::{Rng, SeedableRng};
use vortex::arrays::{ChunkedArray, ListArray, PrimitiveArray, StructArray};
use vortex::dtype::FieldNames;
use vortex::error::VortexResult;
use vortex::validity::Validity;
use vortex::{Array, IntoArray};

use crate::datasets::BenchmarkDataset;

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
impl BenchmarkDataset for StructListOfInts {
    fn name(&self) -> &str {
        &self.name
    }

    async fn to_vortex_array(&self) -> Array {
        let names: FieldNames = (0..self.num_columns)
            .map(|col_idx| (col_idx.to_string().into()))
            .collect();
        let mut rng = rand::rngs::StdRng::seed_from_u64(0);

        let rows_per_chunk = (self.row_count / self.chunk_count).max(1usize);
        (0..self.row_count)
            .step_by(rows_per_chunk)
            .map(|starting_row| rows_per_chunk.min(self.row_count - starting_row))
            .map(|chunk_row_count| {
                let fields = (0..self.num_columns)
                    .map(|_| {
                        let elements = PrimitiveArray::from_iter(
                            (0..chunk_row_count).map(|_| rng.gen::<i64>()),
                        );
                        let offsets =
                            PrimitiveArray::from_iter((0..=chunk_row_count).map(|i| i as u32));
                        ListArray::try_new(
                            elements.into_array(),
                            offsets.into_array(),
                            Validity::AllValid,
                        )
                        .map(|a| a.into_array())
                    })
                    .collect::<VortexResult<Vec<_>>>()?;
                StructArray::try_new(
                    names.clone(),
                    fields,
                    chunk_row_count,
                    Validity::NonNullable,
                )
                .map(|a| a.into_array())
            })
            .collect::<VortexResult<ChunkedArray>>()
            .unwrap()
            .into_array()
    }
}
