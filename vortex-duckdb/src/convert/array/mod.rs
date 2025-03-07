use duckdb::core::{DataChunkHandle, FlatVector};
use duckdb::vtab::arrow::{
    WritableVector, flat_vector_to_arrow_array, write_arrow_array_to_vector,
};
use vortex_array::arrays::StructArray;
use vortex_array::arrow::{FromArrowArray, IntoArrowArray};
use vortex_array::builders::ArrayBuilder;
use vortex_array::validity::Validity;
use vortex_array::{Array, ArrayRef, Canonical};
use vortex_error::VortexResult;

pub trait ToDuckDB {
    fn to_duckdb(&self, chunk: &mut dyn WritableVector) -> VortexResult<()>;
}

impl ToDuckDB for ArrayRef {
    fn to_duckdb(&self, chunk: &mut dyn WritableVector) -> VortexResult<()> {
        write_arrow_array_to_vector(&self.clone().into_arrow_preferred()?, chunk)?;
    }
}

struct SizedFlatVector<'a> {
    pub vector: &'a FlatVector,
    pub len: usize,
}

pub trait FromDuckDB<V> {
    fn from_duckdb(vector: &V) -> VortexResult<ArrayRef>;
}

impl FromDuckDB<DataChunkHandle> for ArrayRef {
    fn from_duckdb(chunk: &DataChunkHandle) -> VortexResult<ArrayRef> {
        let len = chunk.len();

        let columns = (0..chunk.num_columns())
            .map(|i| {
                let vector = chunk.flat_vector(i);
                ArrayRef::from_duckdb(&SizedFlatVector {
                    vector: &vector,
                    len,
                })
            })
            .collect::<VortexResult<Vec<_>>>()?;

        let (names, arrays) = columns.iter().unzip();

        // TODO(joe): extract validity
        StructArray::try_new(names, arrays, len, Validity::AllValid).map(StructArray::to_array)
    }
}

impl FromDuckDB<SizedFlatVector> for ArrayRef {
    fn from_duckdb(vector: &SizedFlatVector) -> VortexResult<ArrayRef> {
        let arrow_arr = flat_vector_to_arrow_array(&mut vector.vector.clone(), vector.len)?;
        Ok(ArrayRef::from_arrow(arrow_arr, true))
    }
}
