use duckdb::core::{DataChunkHandle, FlatVector};
use duckdb::vtab::arrow::{
    WritableVector, flat_vector_to_arrow_array, write_arrow_array_to_vector,
};
use vortex_array::arrays::StructArray;
use vortex_array::arrow::{FromArrowArray, IntoArrowArray};
use vortex_array::validity::Validity;
use vortex_array::{Array, ArrayRef};
use vortex_dtype::FieldNames;
use vortex_error::{VortexResult, vortex_err};

pub trait ToDuckDB {
    fn to_duckdb(&self, chunk: &mut dyn WritableVector) -> VortexResult<()>;
}

impl ToDuckDB for ArrayRef {
    fn to_duckdb(&self, chunk: &mut dyn WritableVector) -> VortexResult<()> {
        write_arrow_array_to_vector(&self.clone().into_arrow_preferred()?, chunk)
            .map_err(|e| vortex_err!("Failed to convert vrotex duckdb array: {}", e.to_string()))
    }
}

struct NamedDataChunk<'a> {
    pub chunk: &'a DataChunkHandle,
    pub names: FieldNames,
}

struct SizedFlatVector {
    pub vector: FlatVector,
    pub len: usize,
}

pub trait FromDuckDB<V> {
    fn from_duckdb(vector: V) -> VortexResult<ArrayRef>;
}

impl<'a> FromDuckDB<&'a NamedDataChunk<'a>> for ArrayRef {
    fn from_duckdb(named_chunk: &'a NamedDataChunk<'a>) -> VortexResult<ArrayRef> {
        let chunk = &named_chunk.chunk;
        let names = &named_chunk.names;
        let len = chunk.len();

        let columns = (0..chunk.num_columns())
            .map(|i| {
                let vector = chunk.flat_vector(i);
                let array = ArrayRef::from_duckdb(SizedFlatVector { vector, len })?;
                // Figure out the column names
                Ok((names[i].clone(), array))
            })
            .collect::<VortexResult<Vec<_>>>()?;

        let (names, arrays): (Vec<_>, Vec<_>) = columns.into_iter().unzip();

        // TODO(joe): extract validity
        StructArray::try_new(names.into(), arrays, len, Validity::AllValid)
            .map(StructArray::into_array)
    }
}

impl FromDuckDB<SizedFlatVector> for ArrayRef {
    fn from_duckdb(mut sized_vector: SizedFlatVector) -> VortexResult<ArrayRef> {
        let len = sized_vector.len;
        let arrow_arr = flat_vector_to_arrow_array(&mut sized_vector.vector, len)
            .map_err(|e| vortex_err!("Failed to convert duckdb array to vortex: {}", e))?;
        Ok(ArrayRef::from_arrow(arrow_arr, true))
    }
}
