use vortex::arrays::StructArray;
use vortex::error::VortexResult;
use vortex::validity::Validity;
use vortex::{ArrayRef, IntoArray};

use crate::convert::array::data_chunk_adaptor::SizedFlatVector;
use crate::{FromDuckDB, NamedDataChunk};

impl<'a> FromDuckDB<&'a NamedDataChunk<'a>> for ArrayRef {
    fn from_duckdb(named_chunk: &'a NamedDataChunk<'a>) -> VortexResult<ArrayRef> {
        let chunk = &named_chunk.chunk;
        let names = &named_chunk.names;
        let len = chunk.len();

        let columns = (0..chunk.num_columns())
            .map(|i| {
                let vector = chunk.flat_vector(i);
                let array = ArrayRef::from_duckdb(SizedFlatVector {
                    vector,
                    nullable: named_chunk.nullable.map(|null| null[i]).unwrap_or(true),
                    len,
                })?;

                // Figure out the column names
                Ok((
                    names
                        .as_ref()
                        .map(|names| names[i].clone())
                        .unwrap_or_else(|| i.to_string().into()),
                    array,
                ))
            })
            .collect::<VortexResult<Vec<_>>>()?;

        let (names, arrays): (Vec<_>, Vec<_>) = columns.into_iter().unzip();

        // All top level struct are non-nullable in duckdb, only inner columns can be nullable.
        StructArray::try_new(names.into(), arrays, len, Validity::NonNullable)
            .map(IntoArray::into_array)
    }
}
