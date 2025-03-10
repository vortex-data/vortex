mod data_chunk_adaptor;

use duckdb::core::DataChunkHandle;
use duckdb::vtab::arrow::{
    WritableVector, flat_vector_to_arrow_array, write_arrow_array_to_vector,
};
use vortex_array::arrays::StructArray;
use vortex_array::arrow::{FromArrowArray, IntoArrowArray};
use vortex_array::validity::Validity;
use vortex_array::{Array, ArrayRef};
use vortex_error::{VortexResult, vortex_err};

use crate::convert::array::data_chunk_adaptor::{
    DataChunkHandleSlice, NamedDataChunk, SizedFlatVector,
};

pub trait ToDuckDB {
    fn to_duckdb(&self, chunk: &mut dyn WritableVector) -> VortexResult<()>;
}

pub fn to_duckdb_chunk(
    struct_array: &StructArray,
    chunk: &mut DataChunkHandle,
) -> VortexResult<Vec<bool>> {
    let mut nullable = vec![false; struct_array.len()];
    for (idx, field) in struct_array.fields().iter().enumerate() {
        field.to_duckdb(&mut DataChunkHandleSlice::new(chunk, idx))?;
        nullable[idx] = field.dtype().is_nullable();
    }
    chunk.set_len(struct_array.len());
    Ok(nullable)
}

impl ToDuckDB for ArrayRef {
    fn to_duckdb(&self, chunk: &mut dyn WritableVector) -> VortexResult<()> {
        let arrow = &self.clone().into_arrow_preferred()?;
        write_arrow_array_to_vector(arrow, chunk)
            .map_err(|e| vortex_err!("Failed to convert vrotex duckdb array: {}", e.to_string()))
    }
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

        // all top level struct are non nullable is duckdb, only inner columns can be.
        StructArray::try_new(names.into(), arrays, len, Validity::NonNullable)
            .map(StructArray::into_array)
    }
}

impl FromDuckDB<SizedFlatVector> for ArrayRef {
    // TODO(joe): going via is slow, make it faster.
    fn from_duckdb(mut sized_vector: SizedFlatVector) -> VortexResult<ArrayRef> {
        let len = sized_vector.len;
        let arrow_arr = flat_vector_to_arrow_array(&mut sized_vector.vector, len)
            .map_err(|e| vortex_err!("Failed to convert duckdb array to vortex: {}", e))?;
        Ok(ArrayRef::from_arrow(arrow_arr, sized_vector.nullable))
    }
}

#[cfg(test)]
mod tests {
    use duckdb::core::DataChunkHandle;
    use itertools::Itertools;
    use vortex_array::arrays::{BoolArray, PrimitiveArray, StructArray, VarBinArray};
    use vortex_array::validity::Validity;
    use vortex_array::variants::StructArrayTrait;
    use vortex_array::{Array, ArrayRef, ToCanonical};
    use vortex_dtype::{DType, FieldNames, Nullability};

    use crate::convert::array::data_chunk_adaptor::NamedDataChunk;
    use crate::convert::array::to_duckdb_chunk;
    use crate::{FromDuckDB, ToDuckDBType};

    fn data() -> ArrayRef {
        let xs = PrimitiveArray::from_iter(0..5);
        let ys = VarBinArray::from_vec(
            vec!["a", "b", "c", "d", "e"],
            DType::Utf8(Nullability::NonNullable),
        );
        let zs = BoolArray::from_iter([true, true, true, false, false]);

        let struct_a = StructArray::try_new(
            FieldNames::from(["xs".into(), "ys".into(), "zs".into()]),
            vec![xs.into_array(), ys.into_array(), zs.into_array()],
            5,
            Validity::NonNullable,
        )
        .unwrap();
        struct_a.to_array()
    }

    #[test]
    fn test_vortex_to_duckdb() {
        let arr = data();
        let ddb_type = arr
            .dtype()
            .as_struct()
            .unwrap()
            .fields()
            .map(|f| f.to_duckdb_type().unwrap())
            .collect_vec();
        let struct_arr = arr.to_struct().unwrap();
        let mut output_chunk = DataChunkHandle::new(ddb_type.as_slice());
        let nullable = to_duckdb_chunk(&struct_arr, &mut output_chunk).unwrap();

        let vx_arr = ArrayRef::from_duckdb(&NamedDataChunk::new(
            &output_chunk,
            &nullable,
            FieldNames::from(["xs".into(), "ys".into(), "zs".into()]),
        ))
        .unwrap();
        assert_eq!(
            struct_arr.names(),
            vx_arr.clone().to_struct().unwrap().names()
        );
        for field in vx_arr.to_struct().unwrap().fields() {
            assert_eq!(field.len(), arr.len());
        }
        assert_eq!(vx_arr.len(), arr.len());
        assert_eq!(vx_arr.dtype(), arr.dtype());
    }
}
