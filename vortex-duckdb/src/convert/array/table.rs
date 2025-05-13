use duckdb::core::DataChunkHandle;
use vortex_array::arrays::StructArray;
use vortex_array::validity::Validity;
use vortex_array::{ArrayRef, IntoArray};
use vortex_error::VortexResult;

use crate::convert::array::array_ref::to_duckdb;
use crate::convert::array::cache::ConversionCache;
use crate::convert::array::data_chunk_adaptor::{DataChunkHandleSlice, SizedFlatVector};
use crate::{FromDuckDB, NamedDataChunk};

/// Converts a top level struct array into a duckdb data chunk, the capacity of the data chunk
/// must be larger that the len of the struct array.
pub fn to_duckdb_chunk(
    struct_array: &StructArray,
    chunk: &mut DataChunkHandle,
    cache: &mut ConversionCache,
) -> VortexResult<()> {
    if struct_array.fields().is_empty() {
        // This happens If the file result is a count(*), then there will be struct fields,
        // but a single chunk, column.
        // We just need to set the length and can ignore the values.
        assert!(chunk.num_columns() <= 1);
        chunk.set_len(struct_array.len());
        return Ok(());
    }

    assert_eq!(struct_array.fields().len(), chunk.num_columns());

    chunk.set_len(struct_array.len());
    for (idx, field) in struct_array.fields().iter().enumerate() {
        to_duckdb(field, &mut DataChunkHandleSlice::new(chunk, idx), cache)?;
    }
    Ok(())
}

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
